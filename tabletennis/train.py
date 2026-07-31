"""Train + validate the match-winner model. Time-based split, never random.

Two models, exactly as the plan prescribes:
  • LogisticRegression — the sanity-check baseline
  • HistGradientBoosting — sklearn's LightGBM-equivalent (native-dep-free)

Validation is walk-forward: train on the first ~80% of matches BY TIME,
evaluate on the last ~20%. Reported: accuracy, log-loss (probability quality)
and a calibration table (is a 70% prediction right 70% of the time?).
The better model (by held-out log-loss) is refit on ALL data and pickled.

    python -m tabletennis.train
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabletennis.features import FEATURE_NAMES, build_dataset

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.pkl"
METRICS_PATH = HERE / "site" / "metrics.json"


def calibration_table(p: np.ndarray, y: np.ndarray, bins=((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01))):
    """Fold predictions to the favoured side, then bin: pred vs actual."""
    conf = np.where(p >= 0.5, p, 1 - p)
    hit = np.where(p >= 0.5, y == 1, y == 0)
    out = []
    for lo, hi in bins:
        m = (conf >= lo) & (conf < hi)
        if m.sum() < 10:
            continue
        out.append({"bin": f"{int(lo*100)}-{int(min(hi,1.0)*100)}%",
                    "n": int(m.sum()),
                    "pred": round(float(conf[m].mean()), 3),
                    "actual": round(float(hit[m].mean()), 3)})
    return out


def train(verbose: bool = True) -> dict:
    X, y, ts, _ = build_dataset()
    X, y, ts = np.array(X), np.array(y), np.array(ts)
    if len(y) < 400:
        raise SystemExit(f"only {len(y)//2} matches ingested — run tabletennis.ingest first")

    # time split on the underlying match order (X holds fwd/rev pairs, so a cut
    # at an even index keeps both orientations of a match on the same side)
    cut = (int(len(y) * 0.8) // 2) * 2
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "gbdt": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                               max_depth=4, min_samples_leaf=40,
                                               validation_fraction=None, random_state=7),
    }
    report = {"n_train_rows": int(cut), "n_test_rows": int(len(y) - cut),
              "n_matches_total": int(len(y) // 2), "features": FEATURE_NAMES, "models": {}}
    best_name, best_ll = None, 1e9
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xte)[:, 1]
        acc = accuracy_score(yte, (p >= 0.5).astype(int))
        ll = log_loss(yte, p)
        report["models"][name] = {
            "accuracy": round(float(acc), 4), "log_loss": round(float(ll), 4),
            "calibration": calibration_table(p, yte),
        }
        if ll < best_ll:
            best_ll, best_name = ll, name
        if verbose:
            print(f"{name:9s} acc={acc:.3f}  logloss={ll:.4f}")

    # naive baselines for honest context
    base_rate = max(yte.mean(), 1 - yte.mean())
    elo_p = 1.0 / (1.0 + 10 ** (-Xte[:, 0] / 400.0))          # Elo diff alone
    report["baselines"] = {
        "majority_accuracy": round(float(base_rate), 4),
        "elo_only_accuracy": round(float(accuracy_score(yte, (elo_p >= 0.5).astype(int))), 4),
        "elo_only_log_loss": round(float(log_loss(yte, np.clip(elo_p, 1e-6, 1 - 1e-6))), 4),
        "coin_flip_log_loss": 0.6931,
    }
    report["best_model"] = best_name

    # refit best on everything, persist
    final = models[best_name]
    final.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final, "features": FEATURE_NAMES, "name": best_name}, f)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    if verbose:
        print(f"\nbest={best_name}  → saved {MODEL_PATH.name}, metrics → site/metrics.json")
        print(f"baselines: majority {report['baselines']['majority_accuracy']}, "
              f"elo-only acc {report['baselines']['elo_only_accuracy']}")
    return report


if __name__ == "__main__":
    train()
