"""
Hierarchical Markov Chain Tennis Model — v3.0
Barnett & Clarke (2005) implementation with:
  - Tour-aware serve defaults (ATP / WTA / ITF)
  - Score-conditioned match win probability
  - clean API for ensemble integration
"""

import math
from typing import Tuple, Optional


# ── Tour defaults ──────────────────────────────────────────────────────────────
TOUR_SERVE_DEFAULTS = {
    'ATP': {
        'first_serve_pct':      0.625,
        'first_serve_win_pct':  0.715,
        'second_serve_win_pct': 0.510,
        'return_first_win_pct': 0.285,
        'return_second_win_pct':0.490,
    },
    'WTA': {
        'first_serve_pct':      0.590,
        'first_serve_win_pct':  0.640,
        'second_serve_win_pct': 0.470,
        'return_first_win_pct': 0.360,
        'return_second_win_pct':0.530,
    },
    'ITF': {
        'first_serve_pct':      0.570,
        'first_serve_win_pct':  0.610,
        'second_serve_win_pct': 0.440,
        'return_first_win_pct': 0.390,
        'return_second_win_pct':0.560,
    },
}


class HierarchicalTennisModel:
    """
    Nested Markov chain: Point → Game → Set → Match.
    All levels computed analytically — fast enough for point-by-point updates.
    """

    def __init__(self, tour: str = 'WTA'):
        self.tour = tour.upper()
        self._defaults = TOUR_SERVE_DEFAULTS.get(self.tour, TOUR_SERVE_DEFAULTS['WTA'])

    # ── Level 1: point win probability ────────────────────────────────────────

    def point_win_prob(self,
                       server_first_pct: Optional[float] = None,
                       server_first_win: Optional[float] = None,
                       server_second_win: Optional[float] = None,
                       returner_first_win: Optional[float] = None,
                       returner_second_win: Optional[float] = None) -> float:
        """
        P(server wins point) mixing server strength and returner quality.
        Falls back to tour-specific defaults for missing values.
        """
        d = self._defaults
        f   = server_first_pct    or d['first_serve_pct']
        fw  = server_first_win    or d['first_serve_win_pct']
        sw  = server_second_win   or d['second_serve_win_pct']
        rfw = returner_first_win  or d['return_first_win_pct']
        rsw = returner_second_win or d['return_second_win_pct']

        alpha = 0.60  # server's own ability weighted more

        # Matchup-adjusted first/second serve win probs
        fw_adj = alpha * fw + (1 - alpha) * (1 - rfw)
        sw_adj = alpha * sw + (1 - alpha) * (1 - rsw)

        p = f * fw_adj + (1 - f) * 0.95 * sw_adj
        return float(max(0.45, min(0.85, p)))

    # ── Level 2: game win probability ─────────────────────────────────────────

    def game_win_prob(self, p: float) -> float:
        """P(server wins game) given point win prob p."""
        q = 1.0 - p
        # Paths to 4-0, 4-1, 4-2 + deuce
        hold = (p**4
                + 4 * p**4 * q
                + 10 * p**4 * q**2
                + 20 * p**3 * q**3 * (p**2 / (p**2 + q**2)))
        return float(max(0.0, min(1.0, hold)))

    # ── Level 3: set win probability ──────────────────────────────────────────

    def set_win_prob(self, p_hold_server: float, p_hold_returner: float) -> float:
        """
        P(player wins set) given their hold probability as server and
        their opponent's hold probability as server.
        """
        # p_avg = average prob of winning a game regardless of serve
        p_avg = (p_hold_server + (1.0 - p_hold_returner)) / 2.0
        q_avg = 1.0 - p_avg

        # Sum over 6-0, 6-1, 6-2, 6-3, 6-4
        from math import comb
        p_set = 0.0
        for opp_games in range(5):       # 0 to 4
            total = 6 + opp_games - 1
            p_set += comb(total, opp_games) * (p_avg ** 6) * (q_avg ** opp_games)

        # 7-5: reach 5-5 then player wins next 2 service games
        p_55 = comb(10, 5) * (p_avg ** 5) * (q_avg ** 5)
        p_75 = p_55 * p_avg**2

        # Tiebreak from 6-6
        p_66 = comb(12, 6) * (p_avg ** 6) * (q_avg ** 6)
        p_tb = 0.5 + 0.8 * (p_avg - 0.5)  # serve advantage compressed in TB
        p_tb = max(0.0, min(1.0, p_tb))

        return float(max(0.0, min(1.0, p_set + p_75 + p_66 * p_tb)))

    # ── Level 4: match win probability ────────────────────────────────────────

    def match_win_prob_from_set_prob(self, p_set: float,
                                      best_of: int = 3) -> float:
        """P(player wins match) from set win probability."""
        q = 1.0 - p_set
        if best_of == 3:
            return float(p_set**2 * (3 - 2 * p_set))
        else:  # best_of == 5
            return float(p_set**3 * (6*p_set**2 - 15*p_set + 10))

    # ── Score-conditioned match win probability ────────────────────────────────

    def win_prob_from_score(self,
                            sets_p1: int, sets_p2: int,
                            games_p1: int, games_p2: int,
                            p1_serving: bool,
                            p1_point_win: float,
                            p2_point_win: float,
                            best_of: int = 3) -> float:
        """
        P(P1 wins match) conditioned on current score state.
        This is the key score-aware component for live trading.
        """
        sets_needed = math.ceil(best_of / 2)

        # Already won?
        if sets_p1 >= sets_needed:
            return 1.0
        if sets_p2 >= sets_needed:
            return 0.0

        # Remaining sets needed
        p1_need = sets_needed - sets_p1
        p2_need = sets_needed - sets_p2

        # Compute per-set win prob from current server
        p1_hold = self.game_win_prob(p1_point_win)
        p2_hold = self.game_win_prob(p2_point_win)

        # P1 as server vs P2 as server — alternate for each game
        # Approximate: use average set win prob
        p1_set_as_server   = self.set_win_prob(p1_hold, p2_hold)
        p1_set_as_returner = self.set_win_prob(1.0 - p2_hold, 1.0 - p1_hold)
        # Blend — if P1 serves first in remaining sets half the time
        p1_set = (p1_set_as_server + p1_set_as_returner) / 2.0

        # Account for partial current set: adjust p1_set slightly
        # If P1 leads games in current set, boost; if trailing, reduce
        game_lead = games_p1 - games_p2
        set_boost = game_lead * 0.02  # ~2% per game lead
        p1_set = max(0.05, min(0.95, p1_set + set_boost))

        # Remaining sets to play (this set hasn't finished yet — treat as 1 more)
        # Full match win from remaining sets
        p_win = 0.0
        for p1_wins in range(p1_need, p1_need + p2_need):
            # P1 wins exactly p1_wins sets out of (p1_wins + p2_need - 1) played, then wins one more
            n = p1_wins + p2_need - 1
            k = p1_wins - 1
            if k < 0:
                continue
            from math import comb
            p_win += comb(n, k) * (p1_set ** (k+1)) * ((1-p1_set) ** (p2_need-1+1-1))

        # Simpler closed-form for remaining sets
        rem_p1 = p1_need
        rem_p2 = p2_need
        p_win = self._match_prob_remaining(p1_set, rem_p1, rem_p2)
        return float(max(0.0, min(1.0, p_win)))

    def _match_prob_remaining(self, p_set: float,
                               need1: int, need2: int) -> float:
        """P(player1 wins) needing need1 sets, opponent needing need2 sets.

        Negative binomial race: P1 must take need1 sets before P2 takes need2.
            P = sum_{j=0}^{need2-1} C(need1+j-1, j) * p^need1 * (1-p)^j
        """
        from math import comb
        if need1 <= 0:
            return 1.0
        if need2 <= 0:
            return 0.0
        p = 0.0
        for j in range(need2):
            p += comb(need1 + j - 1, j) * (p_set ** need1) * ((1 - p_set) ** j)
        return min(1.0, max(0.0, p))

    # ── Convenience full-match entry point ────────────────────────────────────

    def pre_match_prob(self,
                       p1_feats: dict, p2_feats: dict,
                       best_of: int = 3) -> float:
        """
        Pre-match P(P1 wins) from player feature dictionaries.
        Uses _p1_wsp / _p1_first_serve_pct keys written by features.py.
        """
        p1_pp = self.point_win_prob(
            server_first_pct  = p1_feats.get('_p1_first_serve_pct'),
            server_first_win  = p1_feats.get('_p1_wsp'),
            server_second_win = p1_feats.get('_p1_wsp', 0.55) * 0.75,
        )
        p2_pp = self.point_win_prob(
            server_first_pct  = p2_feats.get('_p2_first_serve_pct'),
            server_first_win  = p2_feats.get('_p2_wsp'),
            server_second_win = p2_feats.get('_p2_wsp', 0.55) * 0.75,
        )
        p1_hold = self.game_win_prob(p1_pp)
        p2_hold = self.game_win_prob(p2_pp)
        p1_set  = self.set_win_prob(p1_hold, p2_hold)
        return self.match_win_prob_from_set_prob(p1_set, best_of)
