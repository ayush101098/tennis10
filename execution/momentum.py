"""
Live momentum engine (Tier 2, derive-it-ourselves).

The scoreboard is a lagging indicator: by the time the score says "5-4, break",
the momentum that produced it has been building for several games. This engine
reads the point-by-point game log we already ingest from Sofascore and derives a
forward-looking momentum read, which nudges the per-player point-win-on-serve fed
into the Markov re-pricer (hierarchical_model.win_prob_from_score). The effect is
a live true_p that leans *ahead* of the raw score instead of only reacting to it.

Signals (all oriented to player1, symmetric for player2):
  momentum        composite recency-weighted game control in [-1, 1]. Breaking
                  serve counts far more than holding (it is the surprising,
                  informative event); holds barely move it.
  serve_reg       serve regression: recent hold rate vs the hold rate implied by
                  the player's point-win-on-serve prior, plus break-point pressure
                  faced. Negative = serving worse than baseline (regressing)
                  *before* it has fully shown up as breaks.

Tier 1 bridge: a player's rally profile modulates how we read serve regression.
A high first-strike-index player (lives on the serve/first ball) is more fragile
when their serve dips, so we amplify their serve-regression nudge; a grinder's
serve dip is less predictive of collapse, so we damp it. This is exactly the
"interpret momentum relative to style" idea — the rally prior is the baseline the
live signal is measured against.

Pure and dependency-light: `compute()` takes a plain list of game dicts (see
SofascoreOdds.game_sequence) so it is unit-testable with synthetic sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import tanh
from typing import Optional

# recency: weight_i = DECAY ** i, i=0 is the most recent game. ~3-game half-life.
DECAY = 0.79
HOLD_INFO = 0.35      # a hold is expected -> low information weight
BREAK_INFO = 1.0      # a break is surprising -> full information weight
MAX_SERVE_NUDGE = 0.035   # cap on point-win-on-serve adjustment per player
MOM_SERVE_COUPLING = 0.010  # how much overall momentum bleeds into serve pt-win


def game_win_prob(p: float) -> float:
    """Probability of holding a service game given point-win-on-serve p.
    Standard independent-points model incl. the deuce geometric tail."""
    p = min(max(p, 0.01), 0.99)
    q = 1.0 - p
    # win to love / 15 / 30
    pre = p**4 + 4 * p**4 * q + 10 * p**4 * q * q
    # reach deuce (3-3), then win deuce: p^2 / (p^2 + q^2)
    deuce = 20 * p**3 * q**3
    win_deuce = (p * p) / (p * p + q * q)
    return pre + deuce * win_deuce


# Running point-score string -> index (0,15,30,40 -> 0..3; AD -> 4).
_PT_IDX = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}


def hold_prob_from_score(p: float, server_pts, returner_pts) -> float:
    """P(server holds the CURRENT game) given point-win-on-serve p and the live
    point score. Accepts running-score strings ('0'/'15'/'30'/'40'/'AD') or int
    indices (0..4). Solves the standard per-point Markov from the current state,
    collapsing AD/deuce into the geometric tail p^2/(p^2+q^2)."""
    p = min(max(p, 0.01), 0.99)
    q = 1.0 - p

    def idx(v):
        if isinstance(v, (int, float)):
            return int(v)
        return _PT_IDX.get(str(v).upper(), 0)

    a, b = idx(server_pts), idx(returner_pts)          # server, returner
    win_deuce = (p * p) / (p * p + q * q)              # server wins from deuce

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def P(sa: int, sb: int) -> float:                   # P(server wins game | state)
        # Advantage states / deuce: use the geometric tail.
        if sa >= 3 and sb >= 3:
            if sa > sb:      # server has advantage
                return p + q * win_deuce
            if sb > sa:      # returner has advantage
                return p * win_deuce
            return win_deuce  # deuce (incl. 40-40)
        if sa >= 4:
            return 1.0        # server already won
        if sb >= 4:
            return 0.0        # returner already won (break)
        return p * P(sa + 1, sb) + q * P(sa, sb + 1)

    return P(a, b)


def break_prob_from_score(p: float, server_pts, returner_pts) -> float:
    """P(returner breaks the CURRENT game) = 1 - server hold prob from score."""
    return 1.0 - hold_prob_from_score(p, server_pts, returner_pts)


@dataclass
class MomentumState:
    momentum_p1: float          # [-1, 1], + favors player1
    serve_reg_p1: float         # [-1, 1], - means p1 serving below baseline
    serve_reg_p2: float
    completed_games: int
    recent_holds_p1: int
    recent_breaks_p1: int       # games p1 won on p2's serve (recent window)
    recent_holds_p2: int
    recent_breaks_p2: int
    has_signal: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _running_pt_to_num(s) -> Optional[int]:
    return {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}.get(str(s).upper())


def _bp_faced_in_game(points, server) -> int:
    """Best-effort count of break-point situations the server faced, from running
    scores. A break point = returner is one point from winning the game."""
    if not points:
        return 0
    bp = 0
    for p1pt, p2pt in points:
        a, b = _running_pt_to_num(p1pt), _running_pt_to_num(p2pt)
        if a is None or b is None:
            continue
        returner_pt, server_pt = (b, a) if server == 1 else (a, b)
        # returner at AD, or at 40 with server < 40 -> break point
        if returner_pt == 4 or (returner_pt == 3 and server_pt < 3):
            bp += 1
    return bp


class LiveMomentumEngine:
    def __init__(self, decay: float = DECAY):
        self.decay = decay

    def compute(self, games: list, sp1_prior: float, sp2_prior: float,
                first_strike_p1: float = 0.5, first_strike_p2: float = 0.5,
                window: int = 8) -> MomentumState:
        completed = [g for g in (games or []) if g.get("winner") in (1, 2)
                     and g.get("server") in (1, 2)]
        if not completed:
            return MomentumState(0.0, 0.0, 0.0, 0, 0, 0, 0, 0, has_signal=False)

        recent = completed[-window:]
        # index i=0 is the most recent game
        rev = list(reversed(recent))

        # ── composite momentum ────────────────────────────────────────────────
        num = wsum = 0.0
        for i, g in enumerate(rev):
            w = self.decay ** i
            val = 1.0 if g["winner"] == 1 else -1.0
            info = BREAK_INFO if g["winner"] != g["server"] else HOLD_INFO
            num += w * val * info
            wsum += w
        momentum = tanh(num / wsum) if wsum else 0.0

        # ── per-player serve regression ───────────────────────────────────────
        h1 = b1 = h2 = b2 = 0
        exp_hold1, exp_hold2 = game_win_prob(sp1_prior), game_win_prob(sp2_prior)

        def serve_reg(player, exp_hold):
            # weighted recent hold rate on this player's serve, minus expectation,
            # minus a small penalty for break points faced even when held.
            hnum = hden = bp_pressure = 0.0
            n_sg = 0
            for i, g in enumerate(rev):
                if g["server"] != player:
                    continue
                n_sg += 1
                w = self.decay ** i
                held = 1.0 if g["winner"] == player else 0.0
                hnum += w * held
                hden += w
                bpf = _bp_faced_in_game(g.get("points"), player)
                # each BP faced beyond the first is latent regression pressure
                bp_pressure += w * min(bpf, 3) * 0.06
            if hden == 0:
                return 0.0, 0
            actual_hold = hnum / hden
            reg = (actual_hold - exp_hold) - (bp_pressure / hden)
            # shrink toward 0 for small service-game samples: over a full match a
            # player's hold-vs-expectation residual averages ~0, so a short run of
            # routine holds shouldn't read as strong over/under-performance.
            reg *= n_sg / (n_sg + 2.0)
            return max(-1.0, min(1.0, reg)), n_sg

        for g in recent:
            held = g["winner"] == g["server"]
            if g["winner"] == 1:
                h1 += 1 if held else 0
                b1 += 0 if held else 1
            else:
                h2 += 1 if held else 0
                b2 += 0 if held else 1

        reg1, _ = serve_reg(1, exp_hold1)
        reg2, _ = serve_reg(2, exp_hold2)

        return MomentumState(
            momentum_p1=round(momentum, 4),
            serve_reg_p1=round(reg1, 4),
            serve_reg_p2=round(reg2, 4),
            completed_games=len(completed),
            recent_holds_p1=h1, recent_breaks_p1=b1,
            recent_holds_p2=h2, recent_breaks_p2=b2,
            has_signal=True,
        )

    # ── mapping momentum -> Markov serve inputs ───────────────────────────────
    @staticmethod
    def _style_scale(first_strike: float) -> float:
        """First-strike players are more fragile when serving dips -> amplify;
        grinders -> damp. Bounded ~[0.85, 1.20] around a 0.55 tour baseline."""
        return max(0.85, min(1.20, 1.0 + 1.3 * (first_strike - 0.55)))

    def apply_to_serve(self, sp1: float, sp2: float, st: MomentumState,
                       first_strike_p1: float = 0.5,
                       first_strike_p2: float = 0.5) -> tuple[float, float]:
        """Return (sp1_adj, sp2_adj): point-win-on-serve nudged by momentum, so the
        Markov prices ahead of the score. Bounded and style-modulated."""
        if not st.has_signal:
            return sp1, sp2
        d1 = MAX_SERVE_NUDGE * st.serve_reg_p1 * self._style_scale(first_strike_p1)
        d2 = MAX_SERVE_NUDGE * st.serve_reg_p2 * self._style_scale(first_strike_p2)
        # overall momentum bleeds a little into both serves (confidence effect)
        d1 += MOM_SERVE_COUPLING * st.momentum_p1
        d2 -= MOM_SERVE_COUPLING * st.momentum_p1
        d1 = max(-MAX_SERVE_NUDGE, min(MAX_SERVE_NUDGE, d1))
        d2 = max(-MAX_SERVE_NUDGE, min(MAX_SERVE_NUDGE, d2))
        clip = lambda x: max(0.40, min(0.90, x))
        return clip(sp1 + d1), clip(sp2 + d2)
