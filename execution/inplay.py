"""Live in-play TRUE-P engine — score-aware win probability.

Combines the live scoreboard (Sofascore: sets/games/current server) with each
player's serve strength (career serve-points-won %, from the model DB) and runs
the repo's Markov score-to-win-probability model
(`HierarchicalTennisModel.win_prob_from_score`). The result is a live,
score-driven `true_p` that updates as the match unfolds — the "real engine"
that a pre-match number can't be.

    ip = InPlayModel()
    p = ip.live_true_p("Jannik Sinner", "Alexander Zverev", surface="Grass")
    #  -> P(player1 wins) given the CURRENT score + server, or None if not live

Serve strength defaults to 0.62 point-win-on-serve for players not in the DB
(typical ATP/ITF baseline), so ITF matches still get a score-aware estimate.
"""

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hierarchical_model import HierarchicalTennisModel  # noqa: E402
from execution.model_predict import MatchModel  # noqa: E402
from execution.live_odds import SofascoreOdds  # noqa: E402
from execution.momentum import LiveMomentumEngine  # noqa: E402

try:
    from execution.rally import RallyProfiler  # Tier 1 style priors
except Exception:  # pragma: no cover
    RallyProfiler = None

DEFAULT_SERVE_WIN = 0.62   # point-win-on-serve baseline for unknown players
SERVE_BLEND_RAMP = 60      # serve points before live serve% fully replaces the prior
SERVE_BLEND_MIN = 12       # ignore live serve% below this many serve points (too noisy)

# Tier 2 live momentum: on by default; set TRADING_MOMENTUM=false to price on the
# raw score alone. Bounded nudges (see execution/momentum.py) keep it conservative.
MOMENTUM_ON = os.getenv("TRADING_MOMENTUM", "true").lower() == "true"

# Opponent-adjusted serve inputs (Klaassen & Magnus combined serve/return model).
# The bare Markov was fed each player's OWN career serve% and ignored the
# opponent's return skill entirely, which makes strong servers look near-
# unbeatable and pushes match probabilities to the extremes — the exact
# overconfidence measured against settled bets (Brier 0.28 vs market 0.20).
# Adjusted point-win-on-serve for A vs B:  p = f_A - g_B + (1 - f_t)
#   f_A = A serve-points-won %, g_B = B return-points-won %, f_t = tour average.
# When B is an average returner (g_B = 1 - f_t) this collapses back to f_A.
OPP_ADJUST_ON = os.getenv("TRADING_OPP_ADJUST", "true").lower() == "true"
TOUR_AVG_SERVE = float(os.getenv("TRADING_TOUR_AVG_SERVE", "0.636"))  # DB-wide f_t


class InPlayModel:
    def __init__(self, sofa: Optional[SofascoreOdds] = None,
                 mm: Optional[MatchModel] = None):
        self.markov = HierarchicalTennisModel()
        self.sofa = sofa or SofascoreOdds()
        self.mm = mm or MatchModel()
        self.momentum = LiveMomentumEngine()
        self.rally = None
        if RallyProfiler is not None:
            try:
                r = RallyProfiler()
                r.conn.execute("SELECT 1 FROM rally_stats LIMIT 1")
                self.rally = r
            except Exception:
                self.rally = None

    def _first_strike(self, name: str, tour: str = "ATP") -> float:
        """Tier 1 first-strike-index prior for a player (0.55 neutral if unknown)."""
        if self.rally is None:
            return 0.55
        try:
            from datetime import datetime
            return self.rally.profile(name, datetime.utcnow(), tour).first_strike_index
        except Exception:
            return 0.55

    def _serve_win(self, name: str) -> float:
        """Career serve-points-won % for a player, or the baseline if unknown."""
        found = self.mm.find_player(name)
        if not found:
            return DEFAULT_SERVE_WIN
        prof = self.mm._profile(found[0])
        sp = getattr(prof, "serve_pct", None)
        return float(sp) if sp and 0.4 < sp < 0.9 else DEFAULT_SERVE_WIN

    @staticmethod
    def _opp_adjust(serve_win: float, opp_return: Optional[float]) -> float:
        """Opponent-adjusted point-win-on-serve: f_A - g_B + (1 - f_t), clamped.
        opp_return None (unknown returner) -> return the raw serve prior."""
        if opp_return is None:
            return serve_win
        adj = serve_win - opp_return + (1.0 - TOUR_AVG_SERVE)
        return max(0.40, min(0.90, adj))

    @staticmethod
    def _blend(career: float, live, ramp: int = SERVE_BLEND_RAMP) -> float:
        """Ramp from career prior toward the live serve% as serve points accumulate."""
        if not live or live[1] < SERVE_BLEND_MIN:
            return career
        live_pct, pts = live
        if not 0.3 < live_pct < 0.95:      # ignore degenerate early readings
            return career
        w = min(0.8, pts / ramp)           # cap live weight at 80% — always keep some prior
        return (1.0 - w) * career + w * live_pct

    def live_true_p(self, player1: str, player2: str, surface: str = "Hard",
                    best_of: int = 3, state: Optional[dict] = None) -> Optional[float]:
        """P(player1 wins the match) from the live score. None if not live / no server."""
        st = state or self.sofa.live_state(player1, player2)
        if not st or st.get("p1_serving") is None:
            return None
        sp1, sp2 = self._serve_win(player1), self._serve_win(player2)
        live = self.sofa.serve_stats_for(player1, player2)   # blend in in-match form
        if live:
            sp1 = self._blend(sp1, live.get("p1"))
            sp2 = self._blend(sp2, live.get("p2"))
        # Opponent-adjust each server's point-win by the RETURNER's return skill,
        # so a strong serve is discounted against a strong returner (and vice
        # versa). Only applied when we actually know the opponent's return% —
        # unknown returners fall back to the raw (average-opponent) serve prior.
        if OPP_ADJUST_ON:
            sp1 = self._opp_adjust(sp1, self.mm.return_pct(player2))
            sp2 = self._opp_adjust(sp2, self.mm.return_pct(player1))
        # Tier 2: nudge serve inputs by live momentum so the Markov prices ahead
        # of the score (bounded, style-modulated by the Tier 1 rally prior). The
        # state is cached so callers can journal it (momentum_tag / last_momentum).
        self._last_momentum = None
        if MOMENTUM_ON:
            sp1, sp2, ms = self._apply_momentum(player1, player2, sp1, sp2)
            self._last_momentum = ms
        raw = self.markov.win_prob_from_score(
            sets_p1=st["sets_p1"], sets_p2=st["sets_p2"],
            games_p1=st["games_p1"], games_p2=st["games_p2"],
            p1_serving=bool(st["p1_serving"]),
            p1_point_win=sp1, p2_point_win=sp2, best_of=best_of)
        # Apply the fitted shrinkage calibration if explicitly enabled. Default
        # OFF: the current calibration.json was fit on legacy mixed-source bets,
        # not the in-play engine, so it isn't a valid correction for this signal
        # yet. Enable (TRADING_CALIBRATION=true) once it's refit on inplay bets.
        if os.getenv("TRADING_CALIBRATION", "false").lower() == "true":
            from execution.calibrate import apply as _cal
            return _cal(raw)
        return raw

    def last_momentum(self) -> Optional[dict]:
        """Momentum state from the most recent live_true_p call, or None."""
        ms = getattr(self, "_last_momentum", None)
        return ms.as_dict() if ms and ms.has_signal else None

    def momentum_tag(self) -> str:
        """Compact journalable momentum string from the last live_true_p call."""
        ms = getattr(self, "_last_momentum", None)
        if not ms or not ms.has_signal:
            return ""
        return (f"mom={ms.momentum_p1:+.2f} sregP1={ms.serve_reg_p1:+.2f} "
                f"sregP2={ms.serve_reg_p2:+.2f}")

    def _apply_momentum(self, player1: str, player2: str, sp1: float, sp2: float):
        """Return (sp1_adj, sp2_adj, MomentumState|None) from the live game log."""
        games = self.sofa.game_sequence(player1, player2)
        if not games:
            return sp1, sp2, None
        fs1, fs2 = self._first_strike(player1), self._first_strike(player2)
        st = self.momentum.compute(games, sp1, sp2, fs1, fs2)
        if not st.has_signal:
            return sp1, sp2, st
        a1, a2 = self.momentum.apply_to_serve(sp1, sp2, st, fs1, fs2)
        return a1, a2, st

    def detail(self, player1: str, player2: str, surface: str = "Hard",
               best_of: int = 3) -> Optional[dict]:
        """live_true_p plus the inputs, for display/debug."""
        st = self.sofa.live_state(player1, player2)
        if not st or st.get("p1_serving") is None:
            return None
        p = self.live_true_p(player1, player2, surface, best_of, state=st)
        live = self.sofa.serve_stats_for(player1, player2) or {}
        sp1b = self._blend(self._serve_win(player1), live.get("p1"))
        sp2b = self._blend(self._serve_win(player2), live.get("p2"))
        mom = None
        if MOMENTUM_ON:
            _, _, ms = self._apply_momentum(player1, player2, sp1b, sp2b)
            mom = ms.as_dict() if ms else None
        return {"true_p": p,
                "serve_p1": round(sp1b, 3),
                "serve_p2": round(sp2b, 3),
                "serve_pts": [(live.get("p1") or [None, 0])[1], (live.get("p2") or [None, 0])[1]],
                "momentum": mom,
                **st}

    def close(self):
        self.mm.close()
