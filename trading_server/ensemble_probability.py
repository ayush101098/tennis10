"""
Ensemble Probability Engine
============================
Combines Markov chain True P with ML model predictions to produce
a single, data-driven match-win and game-hold probability.

Weights shift dynamically:
  - Pre-match / few points: career-heavy (Markov dominates)
  - As live stats accumulate: ML models gain weight
  - Models with higher historical calibration get more weight

Output includes per-model breakdown so the UI can show what each
model is contributing.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from .models import ScoreState, ProbabilityState
from .true_probability_engine import TrueProbabilityEngine
from .live_stats import LiveMatchStatsAccumulator, PlayerMatchStats
from .live_features import LiveFeatureEngine, PlayerProfile
from .ml_predictor import MLPredictor


class ModelContribution:
    """One model's probability + weight in the ensemble."""
    __slots__ = ("name", "probability", "weight", "available")

    def __init__(self, name: str, probability: float, weight: float, available: bool = True):
        self.name = name
        self.probability = probability
        self.weight = weight
        self.available = available

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "probability": round(self.probability, 4),
            "weight": round(self.weight, 4),
            "available": self.available,
        }


class EnsembleProbabilityEngine:
    """
    Produces a weighted-average True P from:
      1. Markov model (point → game → set → match)
      2. LR Sackmann (14-feature logistic regression)
      3. LR Advanced (6-feature logistic regression)
      4. RF Advanced (6-feature random forest)

    The Markov model always runs.  ML models require live stats
    to have accumulated ≥ min_points_for_ml before they contribute.
    """

    # Base weights when all models are available
    _BASE_WEIGHTS = {
        "markov": 0.30,
        "lr_sackmann": 0.30,
        "lr_advanced": 0.15,
        "rf_advanced": 0.25,
    }

    def __init__(
        self,
        markov: TrueProbabilityEngine,
        feature_engine: LiveFeatureEngine,
        ml_predictor: MLPredictor,
        min_points_for_ml: int = 4,
    ) -> None:
        self.markov = markov
        self.features = feature_engine
        self.ml = ml_predictor
        self.min_pts = min_points_for_ml

    # ── Main evaluate ────────────────────────────────────────────────────

    def evaluate(
        self,
        score: ScoreState,
        server: int,
        market_odds_server: float,
        market_odds_receiver: float,
        live_stats: LiveMatchStatsAccumulator,
    ) -> tuple[ProbabilityState, list[dict]]:
        """
        Compute ensemble probability.

        Returns:
            (ProbabilityState, list[model_contribution_dicts])
        """
        contributions: List[ModelContribution] = []

        # ── 1. Markov (always available) ─────────────────────────────────
        # Feed updated serve/return from live data blend
        markov_inputs = self.features.build_markov_inputs(
            live_stats.p1, live_stats.p2
        )
        # Update Markov engine with blended stats
        self.markov.p_point_p1_serves = TrueProbabilityEngine._point_prob(
            markov_inputs["p1_serve_pct"] / 100.0,
            markov_inputs["p2_return_pct"] / 100.0,
        )
        self.markov.p_point_p2_serves = TrueProbabilityEngine._point_prob(
            markov_inputs["p2_serve_pct"] / 100.0,
            markov_inputs["p1_return_pct"] / 100.0,
        )

        markov_prob_state = self.markov.evaluate(
            score, server, market_odds_server, market_odds_receiver
        )
        # Markov gives us server_hold_prob as True P for the current game
        markov_hold = markov_prob_state.server_hold_prob
        markov_match_p1 = markov_prob_state.match_win_prob_p1

        contributions.append(
            ModelContribution("markov", markov_match_p1, self._BASE_WEIGHTS["markov"])
        )

        # ── 2. ML models (only after enough points) ─────────────────────
        ml_available = live_stats.has_meaningful_data(self.min_pts)
        ml_predictions: Dict[str, Optional[float]] = {
            "lr_sackmann": None,
            "lr_advanced": None,
            "rf_advanced": None,
        }

        if ml_available:
            sackmann_vec = self.features.build_sackmann_features(
                live_stats.p1, live_stats.p2
            )
            advanced_vec = self.features.build_advanced_features(
                live_stats.p1, live_stats.p2
            )
            ml_predictions = self.ml.predict(sackmann_vec, advanced_vec)

        for model_name in ["lr_sackmann", "lr_advanced", "rf_advanced"]:
            prob_val = ml_predictions.get(model_name)
            avail = prob_val is not None
            contributions.append(
                ModelContribution(
                    model_name,
                    prob_val if avail else 0.5,
                    self._BASE_WEIGHTS[model_name] if avail else 0.0,
                    available=avail,
                )
            )

        # ── 3. Normalise weights and compute ensemble ────────────────────
        total_w = sum(c.weight for c in contributions)
        if total_w > 0:
            for c in contributions:
                c.weight = c.weight / total_w

        ensemble_match_p1 = sum(c.probability * c.weight for c in contributions)

        # ── 4. Market probability ────────────────────────────────────────
        total_implied = (1.0 / market_odds_server) + (1.0 / market_odds_receiver)
        market_p = (
            (1.0 / market_odds_server) / total_implied if total_implied > 0 else 0.5
        )

        # True P for the *server holding* is the Markov game-level output
        true_p_hold = markov_hold
        edge = ensemble_match_p1 - market_p
        ev = (ensemble_match_p1 * market_odds_server) - 1.0

        # Confidence: ramp with number of models + data points
        n_models = sum(1 for c in contributions if c.available)
        data_factor = min(1.0, live_stats.points_played / 30.0)
        conf = int(min(100, max(10, (n_models * 15) + (data_factor * 40) + abs(edge) * 500)))

        prob_state = ProbabilityState(
            true_probability=round(ensemble_match_p1, 4),
            market_probability=round(market_p, 4),
            edge_pct=round(edge, 4),
            ev=round(ev, 4),
            server_hold_prob=round(true_p_hold, 4),
            break_prob=round(1.0 - true_p_hold, 4),
            match_win_prob_p1=round(ensemble_match_p1, 4),
            confidence=conf,
        )

        return prob_state, [c.to_dict() for c in contributions]
