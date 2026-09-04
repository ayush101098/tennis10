"""Model bridge (PRD §12, §13) — live state in, fair probability out.

THIS MODULE COMPUTES NOTHING ITSELF. THAT IS THE POINT.
    The pricing already exists in this repo and has been tuned against settled
    bets: `inplay.py` is the score-aware Markov true_p, `momentum.py` is the
    live momentum engine, `edgescore.py` is the uncertainty gate. A second
    live engine written beside them would be the THIRD implementation of the
    same idea in this codebase (there is already a TypeScript one in
    trading-terminal/src/lib), and three engines that disagree in production is
    a worse problem than the duplication we already have.

    So this file translates: MatchState -> the arguments those engines want,
    and their output -> one `Fair` result. Every probability in it comes from
    code that predates it.

TIERING (§13)
    The cheap path runs on every event: the Markov re-price is a closed-form
    walk over a small state space and costs microseconds. The expensive path —
    momentum, which folds the whole game tape — runs only on events that can
    actually move the number. Spending it on every point buys precision the
    market will not have moved on, and costs latency where latency is the
    product.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from execution.live.events import P1, P2
from execution.live.state import MatchState, is_significant


@dataclass
class Fair:
    """The model's opinion on one match at one instant."""

    p1: float
    source: str                       # which engine produced it
    tier: str                         # "cheap" | "full"
    computed_ms: int
    compute_us: int = 0
    momentum: Optional[dict] = None
    components: dict = field(default_factory=dict)   # engine name -> its p1, for §24 disagreement

    @property
    def p2(self) -> float:
        return 1.0 - self.p1

    def as_dict(self) -> dict:
        return {
            "p1": round(self.p1, 4),
            "p2": round(self.p2, 4),
            "source": self.source,
            "tier": self.tier,
            "compute_us": self.compute_us,
            "momentum": self.momentum,
        }


class ModelBridge:
    """Adapts live state onto the existing engines.

    The engines are imported lazily and defensively: `inplay.py` pulls in the
    model DB and the hierarchical Markov model, which are heavy and not always
    present (a fresh checkout, a CI box, a container without the .db files).
    A live gateway that refuses to start because a model file is missing is
    worse than one that starts and reports NO_MODEL — the former takes the
    scoreboard down with it, and the scoreboard is useful on its own.
    """

    def __init__(self, *, inplay=None, momentum=None):
        self._inplay = inplay
        self._momentum = momentum
        self._tried = inplay is not None
        self.unavailable_reason: Optional[str] = None

    def _ensure(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            from execution.inplay import InPlayModel
            from execution.momentum import LiveMomentumEngine
            self._inplay = InPlayModel()
            self._momentum = LiveMomentumEngine()
        except Exception as e:                       # pragma: no cover - env dependent
            self.unavailable_reason = f"{type(e).__name__}: {e}"[:180]

    @property
    def available(self) -> bool:
        self._ensure()
        return self._inplay is not None

    def price(self, state: MatchState, transitions: list, *,
              force_full: bool = False) -> Optional[Fair]:
        """Fair probability for player 1, or None when the model cannot speak.

        None is a first-class answer. An unranked field, a missing model file
        or a match that has not started are all cases where the honest output
        is silence — and `signals.py` treats None as "no opinion", never as
        50/50. A coin flip presented as a model output is the single easiest
        way to manufacture a fake edge.
        """
        self._ensure()
        if self._inplay is None or not state.player1 or not state.player2:
            return None

        t0 = time.perf_counter()
        full = force_full or is_significant(transitions, state)

        # Price the state THIS pipeline assembled.
        #
        # It used to call `inplay.live_true_p(name, name, surface)`, which looks
        # the match up on SofaScore itself and returns None when it cannot find
        # it live. Two things were wrong with that: the carefully-built
        # MatchState was ignored, and on any FAILOVER leg the model still
        # depended on the very source we had failed away from — so the failover
        # moved scores but not pricing. `win_prob_from_score` takes the score
        # directly, which is what makes the second provider actually useful.
        try:
            sp1 = self._inplay._serve_win(state.player1)
            sp2 = self._inplay._serve_win(state.player2)
            markov = self._inplay.markov
        except Exception as e:                       # pragma: no cover - engine dependent
            self.unavailable_reason = f"{type(e).__name__}: {e}"[:180]
            return None

        sets = state.score.sets
        games = state.score.games
        best_of = state.best_of or 3

        def priced(p1_serving: bool):
            return markov.win_prob_from_score(
                sets_p1=sets[0], sets_p2=sets[1],
                games_p1=games[0], games_p2=games[1],
                p1_serving=p1_serving, p1_point_win=sp1, p2_point_win=sp2,
                best_of=best_of)

        components = {}
        try:
            # NOTE, and it is not a small one: `win_prob_from_score` DECLARES
            # `p1_serving` and never reads it. Verified 2026-09-04 — the two
            # orientations agree to every decimal place at 5-4, 4-5 and 6-5,
            # where the set engine in this package puts the difference at ~26
            # points of set probability. So the match-level model is currently
            # blind to who is serving.
            #
            # That is a model defect, not a bridge defect, and fixing it changes
            # every live price in the product — which is a validated change, not
            # a drive-by one. Until then this passes the true server when it has
            # one and does NOT pretend the output depends on it: no fake
            # marginalisation, no invented uncertainty. The set and game rungs
            # of the ladder DO use the server correctly (see setengine.py), so
            # the ladder is where server-awareness currently lives.
            p1 = markov.win_prob_from_score(
                sets_p1=sets[0], sets_p2=sets[1],
                games_p1=games[0], games_p2=games[1],
                p1_serving=(state.server != P2),
                p1_point_win=sp1, p2_point_win=sp2, best_of=best_of)
            components["markov"] = float(p1)
            source = "markov" if state.server in (P1, P2) else "markov~no-server"
        except Exception as e:                       # pragma: no cover
            self.unavailable_reason = f"{type(e).__name__}: {e}"[:180]
            return None

        if p1 is None:
            return None

        momentum_dict = None
        # The expensive tier. Momentum is read from the engine's cached state
        # rather than recomputed — recomputing would apply the adjustment twice.
        if full:
            try:
                momentum_dict = self._inplay.last_momentum()
            except Exception:
                momentum_dict = None

        compute_us = int((time.perf_counter() - t0) * 1_000_000)
        return Fair(
            p1=float(p1),
            source=source,
            tier="full" if full else "cheap",
            computed_ms=int(time.time() * 1000),
            compute_us=compute_us,
            momentum=momentum_dict,
            components=components,
        )


@dataclass
class GameLadder:
    """Match / set / game probabilities (§14).

    The interesting claim in the PRD: a match market can be efficiently priced
    while the current-game market is not. That is plausible — far fewer people
    price a single game — and it is where a live model has the most room.

    Set and game numbers come from `momentum.hold_prob_from_score`, which
    already solves the in-game Markov chain from the current point score. This
    only assembles them; it does not re-derive the mathematics.
    """

    match_p1: Optional[float] = None
    set_p1: Optional[float] = None
    game_p1: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "match": round(self.match_p1, 4) if self.match_p1 is not None else None,
            "set": round(self.set_p1, 4) if self.set_p1 is not None else None,
            "game": round(self.game_p1, 4) if self.game_p1 is not None else None,
        }


def game_ladder(state: MatchState, fair: Optional[Fair],
                serve_win_p1: float = 0.62, serve_win_p2: float = 0.62) -> GameLadder:
    """Assemble the three market levels for the CURRENT game.

    `game_p1` is P(player 1 wins the game in progress) — which depends on who
    is serving, so it is computed from the server's hold probability and then
    oriented to player 1. Getting that orientation wrong would invert every
    game-market signal, so it is done in one place and tested both ways.
    """
    ladder = GameLadder(match_p1=fair.p1 if fair else None)
    if state.server not in (P1, P2):
        return ladder                     # no server, no game market

    try:
        from execution.momentum import hold_prob_from_score
    except Exception:                     # pragma: no cover - env dependent
        return ladder

    server_is_p1 = state.server == P1
    serve_p = serve_win_p1 if server_is_p1 else serve_win_p2
    pts = state.score.points
    server_pts = pts[0] if server_is_p1 else pts[1]
    returner_pts = pts[1] if server_is_p1 else pts[0]

    try:
        hold = hold_prob_from_score(serve_p, server_pts, returner_pts)
    except Exception:
        return ladder
    if hold is None:
        return ladder

    ladder.game_p1 = hold if server_is_p1 else 1.0 - hold

    # ── set level ──
    # `set_p1` was declared and never populated, so the middle rung of the
    # ladder has been null since it was written. It is filled from the ACTUAL
    # games score rather than converted from the match probability: at 5-2 with
    # the serve, the games already banked dominate, and a conversion from the
    # match number discards exactly that.
    try:
        from execution.live.setengine import set_win_prob
        from execution.momentum import game_win_prob
    except Exception:                     # pragma: no cover - env dependent
        return ladder

    try:
        hold_1 = game_win_prob(serve_win_p1)
        hold_2 = game_win_prob(serve_win_p2)
        ladder.set_p1 = set_win_prob(
            state.score.games[0], state.score.games[1],
            server=1 if server_is_p1 else 2,
            hold_p1=hold_1, hold_p2=hold_2,
            sp1=serve_win_p1, sp2=serve_win_p2,
            current_game_p1=ladder.game_p1,
        )
    except Exception:                     # pragma: no cover
        pass

    return ladder
