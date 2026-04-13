"""
Live Feature Engine
====================
Converts real-time match stats + pre-match profile into the exact
feature vectors the trained ML models expect.

Two model pipelines are supported:
  1. LR-trained (train_sackmann_models): 14 features
     ['rank_diff', 'rank_ratio', 'pts_ratio', 'first_serve_pct_diff',
      'first_win_diff', 'second_win_diff', 'bp_save_diff', 'ace_diff',
      'df_diff', 'win_rate_diff', 'is_clay', 'is_grass',
      'is_grand_slam', 'is_masters']

  2. LR-advanced / RF-advanced (train_proper_models): 6 features
     ['p1_serve_pct', 'p2_serve_pct', 'p1_bp_save', 'p2_bp_save',
      'surface_hard', 'surface_clay']

The engine blends *pre-match career averages* with *live in-match stats*
using a ramp:
  blend_weight = min(1.0, live_serve_points / ramp_points)
  value = (1 - w) * career + w * live

This lets the model start from career priors and smoothly shift to
in-match evidence as data accumulates.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from .live_stats import PlayerMatchStats


# ── Pre-match profile (passed in at setup) ───────────────────────────────────

@dataclass
class PlayerProfile:
    """Career / historical averages for one player (from DB or user input)."""
    rank: int = 50
    ranking_points: int = 1000
    serve_pct: float = 0.63       # career WSP (serve points won %)
    return_pct: float = 0.35      # career WRP
    first_serve_pct: float = 0.62
    first_serve_win_pct: float = 0.70
    second_serve_win_pct: float = 0.50
    bp_save_pct: float = 0.60
    aces_per_game: float = 0.50
    df_per_game: float = 0.20
    win_rate: float = 0.50
    surface_win_rate: float = 0.50
    h2h_win_rate: float = 0.50    # vs this specific opponent


# ── Blending helper ──────────────────────────────────────────────────────────

def _blend(career: float, live: float, live_points: int, ramp: int = 20) -> float:
    """Ramp from career prior to live data as points accumulate."""
    w = min(1.0, live_points / ramp) if ramp > 0 else 1.0
    return (1.0 - w) * career + w * live


# ── Feature builder ──────────────────────────────────────────────────────────

class LiveFeatureEngine:
    """
    Builds feature vectors for the ML models using a blend of
    pre-match career stats and live in-match stats.
    """

    # The 14 features expected by the Sackmann LR model
    SACKMANN_FEATURES = [
        "rank_diff", "rank_ratio", "pts_ratio",
        "first_serve_pct_diff", "first_win_diff", "second_win_diff",
        "bp_save_diff", "ace_diff", "df_diff", "win_rate_diff",
        "is_clay", "is_grass", "is_grand_slam", "is_masters",
    ]

    # The 6 features expected by the Advanced LR / RF models
    ADVANCED_FEATURES = [
        "p1_serve_pct", "p2_serve_pct",
        "p1_bp_save", "p2_bp_save",
        "surface_hard", "surface_clay",
    ]

    def __init__(
        self,
        p1_profile: PlayerProfile,
        p2_profile: PlayerProfile,
        surface: str = "Hard",
        tournament_level: str = "",  # "G" = Grand Slam, "M" = Masters
        ramp_points: int = 20,
    ) -> None:
        self.p1 = p1_profile
        self.p2 = p2_profile
        self.surface = surface.title()
        self.tournament_level = tournament_level.upper()
        self.ramp = ramp_points

    # ── Blended per-player stats ─────────────────────────────────────────

    def _blended_stats(
        self, profile: PlayerProfile, live: PlayerMatchStats
    ) -> Dict[str, float]:
        """Blend career priors with live match data."""
        sp = live.serve_points_played
        return {
            "serve_pct": _blend(profile.serve_pct, live.serve_points_won_pct, sp, self.ramp),
            "return_pct": _blend(profile.return_pct, live.return_points_won_pct, sp, self.ramp),
            "first_serve_pct": _blend(profile.first_serve_pct, live.first_serve_pct, sp, self.ramp),
            "first_serve_win_pct": _blend(profile.first_serve_win_pct, live.first_serve_win_pct, sp, self.ramp),
            "second_serve_win_pct": _blend(profile.second_serve_win_pct, live.second_serve_win_pct, sp, self.ramp),
            "bp_save_pct": _blend(profile.bp_save_pct, live.break_point_save_pct, sp, self.ramp),
            "aces_per_game": _blend(profile.aces_per_game, live.aces_per_game, sp, self.ramp),
            "df_per_game": _blend(profile.df_per_game, live.df_per_game, sp, self.ramp),
            "win_rate": _blend(profile.win_rate, live.win_rate, sp, self.ramp),
        }

    # ── Build Sackmann-14 feature vector ─────────────────────────────────

    def build_sackmann_features(
        self, live_p1: PlayerMatchStats, live_p2: PlayerMatchStats
    ) -> np.ndarray:
        """
        Build the 14-feature vector for the LR model trained by
        train_sackmann_models.py.

        Returns: np.ndarray of shape (1, 14)
        """
        s1 = self._blended_stats(self.p1, live_p1)
        s2 = self._blended_stats(self.p2, live_p2)

        rank1, rank2 = max(self.p1.rank, 1), max(self.p2.rank, 1)
        pts1, pts2 = max(self.p1.ranking_points, 1), max(self.p2.ranking_points, 1)

        features = np.array([[
            rank1 - rank2,                                    # rank_diff
            rank1 / rank2,                                    # rank_ratio
            pts1 / pts2,                                      # pts_ratio
            s1["first_serve_pct"] - s2["first_serve_pct"],    # first_serve_pct_diff
            s1["first_serve_win_pct"] - s2["first_serve_win_pct"],  # first_win_diff
            s1["second_serve_win_pct"] - s2["second_serve_win_pct"],  # second_win_diff
            s1["bp_save_pct"] - s2["bp_save_pct"],            # bp_save_diff
            s1["aces_per_game"] - s2["aces_per_game"],        # ace_diff
            s2["df_per_game"] - s1["df_per_game"],            # df_diff (reversed)
            s1["win_rate"] - s2["win_rate"],                  # win_rate_diff
            1.0 if self.surface == "Clay" else 0.0,           # is_clay
            1.0 if self.surface == "Grass" else 0.0,          # is_grass
            1.0 if self.tournament_level == "G" else 0.0,     # is_grand_slam
            1.0 if self.tournament_level == "M" else 0.0,     # is_masters
        ]], dtype=np.float64)

        return features  # shape (1, 14)

    # ── Build Advanced-6 feature vector ──────────────────────────────────

    def build_advanced_features(
        self, live_p1: PlayerMatchStats, live_p2: PlayerMatchStats
    ) -> np.ndarray:
        """
        Build the 6-feature vector for the Advanced LR / RF models.

        Returns: np.ndarray of shape (1, 6)
        """
        s1 = self._blended_stats(self.p1, live_p1)
        s2 = self._blended_stats(self.p2, live_p2)

        features = np.array([[
            s1["serve_pct"],                                  # p1_serve_pct
            s2["serve_pct"],                                  # p2_serve_pct
            s1["bp_save_pct"],                                # p1_bp_save
            s2["bp_save_pct"],                                # p2_bp_save
            1.0 if self.surface == "Hard" else 0.0,           # surface_hard
            1.0 if self.surface == "Clay" else 0.0,           # surface_clay
        ]], dtype=np.float64)

        return features  # shape (1, 6)

    # ── Build blended serve/return for Markov engine ─────────────────────

    def build_markov_inputs(
        self, live_p1: PlayerMatchStats, live_p2: PlayerMatchStats
    ) -> Dict[str, float]:
        """
        Return updated serve/return percentages for the Markov engine,
        blended with live data.
        """
        s1 = self._blended_stats(self.p1, live_p1)
        s2 = self._blended_stats(self.p2, live_p2)
        return {
            "p1_serve_pct": s1["serve_pct"] * 100.0,
            "p2_serve_pct": s2["serve_pct"] * 100.0,
            "p1_return_pct": s1["return_pct"] * 100.0,
            "p2_return_pct": s2["return_pct"] * 100.0,
        }

    # ── Summary of blended stats for the UI ──────────────────────────────

    def blended_summary(
        self, live_p1: PlayerMatchStats, live_p2: PlayerMatchStats
    ) -> Dict[str, Dict[str, float]]:
        """Return rounded blended stats for display."""
        s1 = self._blended_stats(self.p1, live_p1)
        s2 = self._blended_stats(self.p2, live_p2)
        return {
            "player1": {k: round(v, 4) for k, v in s1.items()},
            "player2": {k: round(v, 4) for k, v in s2.items()},
        }
