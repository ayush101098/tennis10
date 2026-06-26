"""
train_proper_models.py — v3.0
Trains LR + NN ensemble with:
  - Strict temporal train/test split (no leakage)
  - Symmetry augmentation (data-level, not architecture)
  - Isotonic calibration
  - Reliability diagram output
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from features import TennisFeatureExtractor
from neural_network import SymmetricTennisNet, FeatureScaler
from true_p_ensemble import reliability_diagram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'RANK_DIFF', 'POINTS_DIFF', 'WSP_DIFF', 'WRP_DIFF', 'ACES_DIFF',
    'DF_DIFF', 'BP_SAVE_DIFF', 'FIRST_SERVE_PCT_DIFF', 'FIRST_SERVE_WIN_PCT_DIFF',
    'SECOND_SERVE_WIN_PCT_DIFF', 'WIN_RATE_DIFF', 'SURFACE_WIN_RATE_DIFF',
    'SERVEADV', 'COMPLETE_DIFF', 'FATIGUE_DIFF', 'RETIRED_DIFF',
    'DIRECT_H2H', 'MATCHES_PLAYED_DIFF', 'SURFACE_EXP_DIFF',
]

MODELS_DIR = Path('ml_models')
MODELS_DIR.mkdir(exist_ok=True)


def temporal_split(df: pd.DataFrame, train_frac: float = 0.80):
    """
    Split by date so no future data appears in training.
    Crucial: feature computation must also be isolated (done in features.py).
    """
    df = df.sort_values('match_date').reset_index(drop=True)
    cutoff = int(len(df) * train_frac)
    return df.iloc[:cutoff], df.iloc[cutoff:]


def augment_symmetry(X: np.ndarray, y: np.ndarray):
    """Add flipped samples (−X, 1−y) to enforce prediction symmetry."""
    return np.vstack([X, -X]), np.concatenate([y, 1.0 - y])


def train(db_path: str = 'tennis_betting.db', tour: str = 'WTA'):
    logger.info("Extracting features...")
    extractor = TennisFeatureExtractor(db_path, tour=tour)
    df = extractor.extract_features_batch(uncertainty_threshold=0.70)
    extractor.close()

    if df.empty:
        logger.error("No features extracted — check database path and statistics table")
        return

    # Drop rows missing key features
    df = df.dropna(subset=FEATURE_COLS)
    logger.info(f"Dataset: {len(df)} matches")

    # ── Temporal split (NO SHUFFLE before split) ───────────────────────────
    train_df, test_df = temporal_split(df, train_frac=0.80)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    X_train = train_df[FEATURE_COLS].values.astype(float)
    y_train = train_df['label'].values.astype(float)
    X_test  = test_df[FEATURE_COLS].values.astype(float)
    y_test  = test_df['label'].values.astype(float)

    # Symmetry augmentation on TRAINING set only
    X_train_aug, y_train_aug = augment_symmetry(X_train, y_train)

    # Normalise
    scaler = FeatureScaler().fit(X_train_aug)
    Xtr = scaler.transform(X_train_aug)
    Xte = scaler.transform(X_test)
    scaler.save(str(MODELS_DIR / 'scaler.pkl'))

    # ── Logistic Regression ────────────────────────────────────────────────
    logger.info("Training Logistic Regression...")
    base_lr = LogisticRegression(C=0.5, class_weight='balanced',
                                  max_iter=1000, random_state=42)
    lr_cal = CalibratedClassifierCV(base_lr, method='isotonic', cv=5)
    lr_cal.fit(Xtr, y_train_aug)

    lr_probs = lr_cal.predict_proba(Xte)[:, 1]
    lr_auc   = roc_auc_score(y_test, lr_probs)
    lr_brier = brier_score_loss(y_test, lr_probs)
    lr_acc   = ((lr_probs > 0.5) == y_test).mean()
    logger.info(f"LR — AUC={lr_auc:.4f}  Brier={lr_brier:.4f}  Acc={lr_acc:.4f}")

    with open(MODELS_DIR / 'logistic_regression.pkl', 'wb') as f:
        pickle.dump(lr_cal, f)

    # ── Random Forest ──────────────────────────────────────────────────────
    logger.info("Training Random Forest...")
    rf_base = RandomForestClassifier(n_estimators=200, max_depth=12,
                                      min_samples_leaf=10, random_state=42,
                                      n_jobs=-1)
    rf_cal = CalibratedClassifierCV(rf_base, method='isotonic', cv=5)
    rf_cal.fit(Xtr, y_train_aug)

    rf_probs = rf_cal.predict_proba(Xte)[:, 1]
    rf_auc   = roc_auc_score(y_test, rf_probs)
    rf_brier = brier_score_loss(y_test, rf_probs)
    rf_acc   = ((rf_probs > 0.5) == y_test).mean()
    logger.info(f"RF — AUC={rf_auc:.4f}  Brier={rf_brier:.4f}  Acc={rf_acc:.4f}")

    with open(MODELS_DIR / 'random_forest.pkl', 'wb') as f:
        pickle.dump(rf_cal, f)

    # ── Neural Network Ensemble ────────────────────────────────────────────
    logger.info("Training Neural Network ensemble (20 models)...")
    # Val split from training data (last 10%)
    val_n = int(0.10 * len(X_train))
    Xtr_nn, ytr_nn = Xtr[:-val_n], y_train_aug[:-val_n]
    Xvl_nn, yvl_nn = Xtr[-val_n:], y_train_aug[-val_n:]

    nn = SymmetricTennisNet(n_models=20, input_dim=len(FEATURE_COLS),
                             hidden_dim=100, lr=0.0004, momentum=0.55,
                             weight_decay=0.002, dropout=0.10)
    nn.fit(Xtr_nn, ytr_nn, Xvl_nn, yvl_nn, max_epochs=50, patience=10)
    nn.save(str(MODELS_DIR / 'neural_network.pkl'))

    nn_probs = nn.predict_proba(Xte)
    nn_auc   = roc_auc_score(y_test, nn_probs)
    nn_brier = brier_score_loss(y_test, nn_probs)
    nn_acc   = ((nn_probs > 0.5) == y_test).mean()
    logger.info(f"NN — AUC={nn_auc:.4f}  Brier={nn_brier:.4f}  Acc={nn_acc:.4f}")

    # ── ML Ensemble (LR + RF weighted) ────────────────────────────────────
    ensemble_probs = 0.40 * lr_probs + 0.60 * rf_probs
    ens_brier = brier_score_loss(y_test, ensemble_probs)
    ens_acc   = ((ensemble_probs > 0.5) == y_test).mean()
    logger.info(f"ML Ensemble (LR+RF) — Brier={ens_brier:.4f}  Acc={ens_acc:.4f}")

    # ── Reliability Diagram ────────────────────────────────────────────────
    rd = reliability_diagram(ensemble_probs.tolist(), y_test.tolist(), n_bins=10)
    logger.info(f"\nReliability Diagram (ML Ensemble):")
    logger.info(f"  Brier Score: {rd['brier_score']:.5f}")
    for mid, mp, ma, cnt in zip(rd['midpoints'], rd['mean_pred'],
                                  rd['mean_actual'], rd['counts']):
        gap = ma - mp
        flag = " ← MISCALIBRATED" if abs(gap) > 0.05 else ""
        logger.info(f"  P̂={mp:.2f}  Actual={ma:.2f}  N={cnt:4d}{flag}")

    # Save reliability data
    pd.DataFrame(rd).to_csv(MODELS_DIR / 'calibration_report.csv', index=False)
    logger.info("\nAll models saved to ml_models/")
    logger.info("NOTE: These accuracy figures reflect held-out TEMPORAL test set.")
    logger.info("Expected range for well-calibrated tennis models: 65-72% accuracy.")


if __name__ == '__main__':
    import sys
    tour = sys.argv[1] if len(sys.argv) > 1 else 'WTA'
    train(tour=tour)
