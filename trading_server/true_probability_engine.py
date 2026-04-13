"""
True Probability Engine — integrates the existing Markov model
and produces point-level, game-level, and match-level probabilities.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.special import comb
from typing import Dict, Optional
from .models import ScoreState, ProbabilityState


class TrueProbabilityEngine:
    """
    Wraps Barnett-Clarke Markov hierarchy + optional DB lookup.
    All methods are stateless/pure — no DB hit at request time.
    """

    def __init__(
        self,
        p1_serve_pct: float = 63.0,
        p2_serve_pct: float = 63.0,
        p1_return_pct: float = 35.0,
        p2_return_pct: float = 35.0,
        p1_rank: int = 50,
        p2_rank: int = 50,
    ) -> None:
        # Store as fractions (0-1)
        self.p1_serve = p1_serve_pct / 100.0
        self.p2_serve = p2_serve_pct / 100.0
        self.p1_return = p1_return_pct / 100.0
        self.p2_return = p2_return_pct / 100.0
        self.p1_rank = p1_rank
        self.p2_rank = p2_rank

        # Derived: probability server wins a point on serve
        self.p_point_p1_serves = self._point_prob(self.p1_serve, self.p2_return)
        self.p_point_p2_serves = self._point_prob(self.p2_serve, self.p1_return)

    @staticmethod
    def _point_prob(serve_pct: float, opp_return_pct: float) -> float:
        """Weighted blend of serve ability and opponent return ability."""
        alpha = 0.60
        p = alpha * serve_pct + (1 - alpha) * (1 - opp_return_pct)
        return float(np.clip(p, 0.45, 0.85))

    # ── Point → Game (Markov) ─────────────────────────────────────────────

    @staticmethod
    def p_game_from_state(sp: int, rp: int, p: float) -> float:
        """P(server wins game) from arbitrary (sp, rp) using recursion + deuce formula."""
        cache: dict = {}

        def _rec(s: int, r: int) -> float:
            if (s, r) in cache:
                return cache[(s, r)]
            if s >= 4 and s - r >= 2:
                return 1.0
            if r >= 4 and r - s >= 2:
                return 0.0
            # Deuce / advantage zone
            if s >= 3 and r >= 3:
                d = (p ** 2) / (p ** 2 + (1 - p) ** 2)  # P(win from deuce)
                if s == r:
                    return d
                elif s > r:
                    return p + (1 - p) * d  # Ad-in
                else:
                    return p * d              # Ad-out
            val = p * _rec(s + 1, r) + (1 - p) * _rec(s, r + 1)
            cache[(s, r)] = val
            return val

        return _rec(sp, rp)

    @staticmethod
    def p_game_full(p: float) -> float:
        """P(server wins game) from 0-0."""
        q = 1 - p
        before_deuce = p ** 4 * (1 + 4 * q + 10 * q ** 2)
        p_deuce = comb(6, 3, exact=True) * (p ** 3) * (q ** 3) * (p ** 2 / (p ** 2 + q ** 2))
        return before_deuce + p_deuce

    # ── Game → Set (Markov) ───────────────────────────────────────────────

    def p_set_from_state(
        self, sg: int, rg: int, p_hold: float, p_break: float
    ) -> float:
        """P(server of first game wins set) from (sg, rg) games."""
        cache: dict = {}

        def _rec(s: int, r: int, on_serve: bool) -> float:
            if (s, r, on_serve) in cache:
                return cache[(s, r, on_serve)]
            if s >= 6 and s - r >= 2:
                return 1.0
            if r >= 6 and r - s >= 2:
                return 0.0
            if s == 6 and r == 6:
                # Tiebreak: simplified as 50/50 adjusted by point-win prob
                p_tb = self.p_point_p1_serves  # approximate
                return self._p_tiebreak(p_tb)

            p_win_game = p_hold if on_serve else p_break
            val = (p_win_game * _rec(s + 1, r, not on_serve) +
                   (1 - p_win_game) * _rec(s, r + 1, not on_serve))
            cache[(s, r, on_serve)] = val
            return val

        return _rec(sg, rg, True)

    @staticmethod
    def _p_tiebreak(p_point: float) -> float:
        """Approximate tiebreak win probability."""
        q = 1 - p_point
        total = 0.0
        # Win 7-0 through 7-5
        for i in range(6):
            total += comb(6 + i, i, exact=True) * (p_point ** 7) * (q ** i)
        # 6-6 onwards: deuce logic
        p66 = comb(12, 6, exact=True) * (p_point ** 6) * (q ** 6)
        p_deuce = (p_point ** 2) / (p_point ** 2 + q ** 2)
        total += p66 * p_deuce
        return float(np.clip(total, 0.01, 0.99))

    # ── Set → Match ───────────────────────────────────────────────────────

    @staticmethod
    def p_match(p_set: float, best_of: int = 3) -> float:
        """P(player wins match) from independent set-win probability."""
        n = (best_of + 1) // 2  # sets to win
        total = 0.0
        for k in range(n, best_of + 1):
            total += comb(best_of, k, exact=True) * (p_set ** k) * ((1 - p_set) ** (best_of - k))
        # Actually need to use negative-binomial: first to n
        # P(win) = sum over k=n..2n-1 of C(k-1, n-1) * p^n * q^(k-n)
        total = 0.0
        for losses in range(n):
            total += comb(n - 1 + losses, losses, exact=True) * (p_set ** n) * ((1 - p_set) ** losses)
        return float(np.clip(total, 0.01, 0.99))

    # ── Unified evaluate ──────────────────────────────────────────────────

    def evaluate(
        self,
        score: ScoreState,
        server: int,
        market_odds_server: float = 1.80,
        market_odds_receiver: float = 2.10,
    ) -> ProbabilityState:
        """Full probability evaluation at current score state."""
        # Which player is serving?
        if server == 1:
            p_point = self.p_point_p1_serves
            opp_p_point = self.p_point_p2_serves
        else:
            p_point = self.p_point_p2_serves
            opp_p_point = self.p_point_p1_serves

        # Game-level
        hold_prob = self.p_game_from_state(
            score.server_points, score.receiver_points, p_point
        )
        break_prob = 1.0 - hold_prob

        # Set-level (from current game score)
        p_hold_full = self.p_game_full(p_point)
        # Use the *opponent's* serve-point-win prob (not 1 - p_point!)
        p_break_full = 1.0 - self.p_game_full(opp_p_point)
        p_set = self.p_set_from_state(
            score.server_games, score.receiver_games, p_hold_full, p_break_full
        )

        # Match-level
        match_p = self.p_match(p_set, best_of=3)

        # Market
        total_implied = (1.0 / market_odds_server) + (1.0 / market_odds_receiver)
        market_p = (1.0 / market_odds_server) / total_implied if total_implied > 0 else 0.5

        # True P for the server holding
        true_p = hold_prob
        edge = true_p - market_p
        ev = (true_p * market_odds_server) - 1.0

        # Confidence: higher when we have more data / stronger edge
        conf = int(min(100, max(10, abs(edge) * 1000 + 30)))

        return ProbabilityState(
            true_probability=round(true_p, 4),
            market_probability=round(market_p, 4),
            edge_pct=round(edge, 4),
            ev=round(ev, 4),
            server_hold_prob=round(hold_prob, 4),
            break_prob=round(break_prob, 4),
            match_win_prob_p1=round(match_p, 4),
            confidence=conf,
        )
