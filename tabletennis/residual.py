"""§3 Neural residual — bounded character correction on the analytic baseline.

    final = clip(P_analytic + tanh(raw) * MAX_ADJ)

Trained exactly as the spec says: on the baseline's ERROR (outcome − baseline),
not the raw outcome — a smaller learning problem than predicting the match.
Training rows are every intermediate game state of historical matches (traits
and Elo snapshotted causally by traits.compute, so no leakage), time-split for
honest evaluation: the residual ships only if it beats the baseline's log-loss
out of sample; otherwise predict-time falls back to pure analytics.

Point-level residuals (momentum, serve pressure) start training once live.py
has logged enough real point sequences — this file's feature layout already
reserves the state slots.

    python -m tabletennis.residual
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPRegressor

from tabletennis.traits import compute

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "residual.pkl"
MAX_ADJ = 0.15          # spec: cap the live adjustment at ±15 points


def _xy(rows):
    X = np.array([[r["base"], r["pre"], r["ga"], r["gb"], r["best_of"],
                   r["elo_diff"] / 100.0, *r["feats"], *r["traits1"], *r["traits2"]]
                  for r in rows])
    y = np.array([r["y"] for r in rows], float)
    base = np.array([r["base"] for r in rows])
    return X, y, base


def train(verbose: bool = True) -> dict:
    _, rows = compute(emit_states=True)
    rows.sort(key=lambda r: r["ts"])
    if len(rows) < 2000:
        raise SystemExit(f"only {len(rows)} states — ingest more history first")
    cut = int(len(rows) * 0.8)
    Xtr, ytr, btr = _xy(rows[:cut])
    Xte, yte, bte = _xy(rows[cut:])

    mdl = MLPRegressor(hidden_layer_sizes=(64,), activation="relu", alpha=1e-3,
                       max_iter=60, early_stopping=True, random_state=7)
    mdl.fit(Xtr, ytr - btr)                      # learn the baseline's error

    def blended(X, base):
        return np.clip(base + np.tanh(mdl.predict(X)) * MAX_ADJ, 0.01, 0.99)

    ll_base = log_loss(yte, np.clip(bte, 0.01, 0.99))
    ll_blend = log_loss(yte, blended(Xte, bte))
    improved = ll_blend < ll_base
    report = {"n_states": len(rows), "log_loss_analytic": round(float(ll_base), 4),
              "log_loss_blended": round(float(ll_blend), 4),
              "improvement": round(float(ll_base - ll_blend), 4), "shipped": bool(improved)}
    if verbose:
        print(f"states={len(rows)}  analytic LL={ll_base:.4f}  +residual LL={ll_blend:.4f}"
              f"  Δ={ll_base - ll_blend:+.4f}  → {'SHIP' if improved else 'FALL BACK to analytic'}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": mdl if improved else None, "max_adj": MAX_ADJ,
                     "report": report}, f)
    return report


def load():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {"model": None, "max_adj": MAX_ADJ, "report": {}}


def adjust(base: float, features: list[float], bundle) -> tuple[float, float]:
    """(final_prob, applied_residual). Falls back to base if no shipped model."""
    mdl = bundle.get("model")
    if mdl is None:
        return base, 0.0
    raw = float(mdl.predict([features])[0])
    adj = float(np.tanh(raw) * bundle["max_adj"])
    return float(np.clip(base + adj, 0.01, 0.99)), adj


if __name__ == "__main__":
    train()
