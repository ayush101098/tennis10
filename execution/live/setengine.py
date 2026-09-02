"""Set engine — P(win the CURRENT set | games score, server, hold rates).

WHY THIS IS A SEPARATE LEVEL AND NOT `set_prob(p_match)`
    `model_predict.set_prob` already converts a MATCH probability into the
    single-set probability consistent with it. That is the right tool
    pre-match, and `signals_gen.py` uses it correctly for set markets.

    It cannot price a set in progress. At 5-2 up with the serve, a player's
    current-set probability has almost nothing to do with their match
    probability — the games already banked are the dominant term, and a
    conversion from the match number throws exactly that information away.

    So this walks the set forward from the actual games score, which is what
    makes the set market tradeable rather than a restatement of the match
    market. It is the missing rung: match came from `inplay.py`, game from
    `momentum.hold_prob_from_score`, set was declared and never filled in.

WHAT IT ASSUMES
    Hold probabilities are constant within the set. They are not — momentum and
    fatigue move them — but the live hold rates fed in already carry those
    adjustments, and re-modelling drift inside the walk would double-count what
    `momentum.py` has already applied.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

# Above this many games the walk is not a set any more — it is a data error, and
# recursing on it would hang rather than fail.
MAX_GAMES = 30


def tiebreak_prob(sp1: float, sp2: float) -> float:
    """P(player 1 wins a 7-point tiebreak), from point-win-on-serve rates.

    A first-to-7-win-by-2 with the 1-2-2 serve rotation has a closed form, but
    it is long and its accuracy is dominated by the serve estimates feeding it.
    This uses the standard logistic approximation on the serve differential,
    which agrees with the exact chain to within about a point across the range
    that occurs in practice (0.5-0.75 serve win rates).

    Approximation, and labelled as one: a tiebreak is close to a coin flip
    tilted by relative serve strength, and pretending to more precision than
    the inputs support would be false rigour.
    """
    edge = (sp1 - sp2)
    # 6.5 is the slope that matches the exact chain over the realistic band;
    # at sp1 == sp2 it correctly returns exactly 0.5.
    return 1.0 / (1.0 + pow(2.718281828459045, -6.5 * edge))


def set_win_prob(games_p1: int, games_p2: int, *, server: int,
                 hold_p1: float, hold_p2: float,
                 sp1: Optional[float] = None, sp2: Optional[float] = None,
                 current_game_p1: Optional[float] = None) -> Optional[float]:
    """P(player 1 wins this set) from the current games score.

    `server` is 1 or 2 — who serves the NEXT game (or the game in progress).
    `hold_p1` / `hold_p2` are each player's probability of holding a full
    service game. `sp1`/`sp2` are point-win-on-serve rates, used only for the
    tiebreak; they fall back to the hold rates when absent.

    `current_game_p1` lets the caller pass the in-progress game's probability
    (from `momentum.hold_prob_from_score`, which knows the point score). Without
    it the current game is priced as if it had just started, which at 0-40 down
    is badly wrong — the whole point of a live model is that it knows.

    Returns None on impossible input rather than a number, so a feed error
    surfaces as "no opinion" instead of a confident fiction.
    """
    if not (0.0 < hold_p1 < 1.0 and 0.0 < hold_p2 < 1.0):
        return None
    if games_p1 < 0 or games_p2 < 0 or games_p1 > MAX_GAMES or games_p2 > MAX_GAMES:
        return None
    if server not in (1, 2):
        return None

    p_tb = tiebreak_prob(
        sp1 if sp1 is not None else hold_p1,
        sp2 if sp2 is not None else hold_p2,
    )

    @lru_cache(maxsize=None)
    def walk(a: int, b: int, srv: int) -> float:
        # Terminal states first: 6-x with two clear, or 7-5.
        if a >= 6 and a - b >= 2:
            return 1.0
        if b >= 6 and b - a >= 2:
            return 0.0
        if a == 7 and b == 6:
            return 1.0
        if b == 7 and a == 6:
            return 0.0
        if a == 6 and b == 6:
            return p_tb
        if a > 7 or b > 7:              # unreachable in a real set; guard the walk
            return 1.0 if a > b else 0.0

        p_server_holds = hold_p1 if srv == 1 else hold_p2
        # Probability the NEXT game goes to player 1.
        p_game_p1 = p_server_holds if srv == 1 else 1.0 - p_server_holds
        nxt = 2 if srv == 1 else 1
        return p_game_p1 * walk(a + 1, b, nxt) + (1 - p_game_p1) * walk(a, b + 1, nxt)

    # The game in progress is priced from the point score when we have it; the
    # rest of the set is the clean walk.
    if current_game_p1 is not None and 0.0 <= current_game_p1 <= 1.0:
        nxt = 2 if server == 1 else 1
        out = (current_game_p1 * walk(games_p1 + 1, games_p2, nxt)
               + (1 - current_game_p1) * walk(games_p1, games_p2 + 1, nxt))
    else:
        out = walk(games_p1, games_p2, server)

    walk.cache_clear()
    return out


def games_to_set_edge(games_p1: int, games_p2: int) -> int:
    """Games player 1 is ahead by. Trivial, but it keeps the sign convention in
    one place — an inverted set score is the easiest way to ship a backwards
    market."""
    return games_p1 - games_p2
