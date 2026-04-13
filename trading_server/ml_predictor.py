"""
ML Predictor — loads trained models and runs real-time inference.
================================================================

Loads all available saved models from ml_models/ and provides a
unified `predict()` method that returns P(player1 wins) from each model.

Supported models:
  - LR trained (CalibratedClassifierCV, 14 features)
  - LR advanced (sklearn LogisticRegression, 6 features)
  - RF advanced (RandomForestClassifier, 6 features)

The Neural Network ensemble is rebuilt from saved state_dicts if available.
"""

from __future__ import annotations

import os
import pickle
import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Paths relative to project root
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ml_models")


def _load_pickle(filename: str):
    """Safely load a pickle file, return None on failure."""
    path = os.path.join(_MODEL_DIR, filename)
    if not os.path.exists(path):
        logger.warning("Model file not found: %s", path)
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", filename, e)
        return None


class MLPredictor:
    """
    Holds all loaded models and runs inference.

    Usage:
        predictor = MLPredictor()
        results = predictor.predict(sackmann_vec, advanced_vec)
        # results = {"lr_sackmann": 0.62, "lr_advanced": 0.59, "rf_advanced": 0.61}
    """

    def __init__(self) -> None:
        self.lr_sackmann = None        # CalibratedClassifierCV
        self.lr_sackmann_scaler = None  # StandardScaler
        self.lr_sackmann_features = None

        self.lr_advanced = None        # LogisticRegression
        self.rf_advanced = None        # RandomForestClassifier
        self.advanced_scaler = None    # StandardScaler
        self.advanced_features = None  # list[str]

        self.models_loaded: Dict[str, bool] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all available models from disk."""

        # ── LR Sackmann (14 features) ────────────────────────────────────
        lr_data = _load_pickle("logistic_regression_trained.pkl")
        if lr_data and isinstance(lr_data, dict):
            self.lr_sackmann = lr_data.get("model")
            self.lr_sackmann_scaler = lr_data.get("scaler")
            self.lr_sackmann_features = lr_data.get("features")
            self.models_loaded["lr_sackmann"] = self.lr_sackmann is not None
            logger.info(
                "LR Sackmann loaded: %d features, acc=%.4f",
                len(self.lr_sackmann_features or []),
                lr_data.get("test_accuracy", 0),
            )
        else:
            self.models_loaded["lr_sackmann"] = False

        # ── LR Advanced (6 features) ────────────────────────────────────
        self.lr_advanced = _load_pickle("logistic_regression_advanced.pkl")
        self.models_loaded["lr_advanced"] = self.lr_advanced is not None

        # ── RF Advanced (6 features) ────────────────────────────────────
        self.rf_advanced = _load_pickle("random_forest_advanced.pkl")
        self.models_loaded["rf_advanced"] = self.rf_advanced is not None

        # ── Shared scaler + feature names for advanced models ───────────
        self.advanced_scaler = _load_pickle("scaler_advanced.pkl")
        self.advanced_features = _load_pickle("feature_names_advanced.pkl")

        loaded = [k for k, v in self.models_loaded.items() if v]
        logger.info("ML Predictor ready: %s", loaded or "no models loaded")

    # ── Inference ────────────────────────────────────────────────────────

    def predict(
        self,
        sackmann_vec: Optional[np.ndarray] = None,
        advanced_vec: Optional[np.ndarray] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Run all loaded models and return P(player1 wins) from each.

        Args:
            sackmann_vec: shape (1, 14) — features for LR Sackmann model
            advanced_vec: shape (1, 6) — features for LR/RF advanced models

        Returns:
            Dict[model_name → probability or None]
        """
        results: Dict[str, Optional[float]] = {
            "lr_sackmann": None,
            "lr_advanced": None,
            "rf_advanced": None,
        }

        # ── LR Sackmann ─────────────────────────────────────────────────
        if sackmann_vec is not None and self.lr_sackmann is not None:
            try:
                X = sackmann_vec.copy()
                if self.lr_sackmann_scaler is not None:
                    X = self.lr_sackmann_scaler.transform(X)
                # CalibratedClassifierCV has predict_proba
                proba = self.lr_sackmann.predict_proba(X)
                # proba shape (1, 2) — column 1 = P(class=1) = P(player1 wins)
                results["lr_sackmann"] = float(proba[0, 1])
            except Exception as e:
                logger.warning("LR Sackmann predict failed: %s", e)

        # ── LR Advanced ──────────────────────────────────────────────────
        if advanced_vec is not None and self.lr_advanced is not None:
            try:
                X = advanced_vec.copy()
                if self.advanced_scaler is not None:
                    X = self.advanced_scaler.transform(X)
                proba = self.lr_advanced.predict_proba(X)
                results["lr_advanced"] = float(proba[0, 1])
            except Exception as e:
                logger.warning("LR Advanced predict failed: %s", e)

        # ── RF Advanced ──────────────────────────────────────────────────
        if advanced_vec is not None and self.rf_advanced is not None:
            try:
                X = advanced_vec.copy()
                if self.advanced_scaler is not None:
                    X = self.advanced_scaler.transform(X)
                proba = self.rf_advanced.predict_proba(X)
                results["rf_advanced"] = float(proba[0, 1])
            except Exception as e:
                logger.warning("RF Advanced predict failed: %s", e)

        return results

    def available_models(self) -> list[str]:
        """Return names of successfully loaded models."""
        return [k for k, v in self.models_loaded.items() if v]
