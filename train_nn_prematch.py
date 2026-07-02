"""
Train the pre-match neural network for the trading terminal.

Data:    tennis_data.db (Sackmann ATP 2010-2024, ~42k matches)
Model:   MLP on symmetric rank/physical/surface features, trained on both
         player orientations so P(p1) + P(p2) == 1 by construction.
Output:  trading-terminal/public/nn_model.json — weights + scaler + Platt
         calibration, consumed by src/lib/nnModel.ts for in-browser inference.

Validation is time-split: train < 2024, test on 2024 — printed metrics are
out-of-sample and compared against the Elo formula the terminal used before.
"""

import json
import sqlite3

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.neural_network import MLPClassifier

DB = "tennis_data.db"
OUT = "trading-terminal/public/nn_model.json"

LEVEL_BIG = {"G", "M", "F"}  # slam / masters / finals


def load_matches():
    con = sqlite3.connect(DB)
    rows = con.execute(
        """
        SELECT tournament_date, surface, best_of, tourney_level,
               winner_rank, loser_rank, winner_rank_points, loser_rank_points,
               winner_age, loser_age, winner_ht, loser_ht
        FROM matches
        WHERE winner_rank IS NOT NULL AND loser_rank IS NOT NULL
          AND winner_rank > 0 AND loser_rank > 0
        ORDER BY tournament_date
        """
    ).fetchall()
    con.close()
    return rows


def feats(r_a, r_b, pts_a, pts_b, age_a, age_b, ht_a, ht_b, surface, best_of, level):
    """Feature vector for orientation (A vs B). Difference terms flip sign on swap."""
    lr = np.log(r_b) - np.log(r_a)                      # + means A better ranked
    lp = np.log1p(pts_a or 0.0) - np.log1p(pts_b or 0.0)
    age_d = ((age_a or 27.0) - (age_b or 27.0)) / 10.0
    ht_d = ((ht_a or 185.0) - (ht_b or 185.0)) / 10.0
    s_clay = 1.0 if surface == "Clay" else 0.0
    s_grass = 1.0 if surface == "Grass" else 0.0
    s_hard = 1.0 if surface == "Hard" else 0.0
    bo5 = 1.0 if best_of == 5 else 0.0
    big = 1.0 if level in LEVEL_BIG else 0.0
    return [lr, lp, age_d, ht_d, s_clay, s_grass, s_hard, bo5, big,
            lr * bo5, lr * s_clay, lr * s_grass]


FEATURE_NAMES = [
    "log_rank_diff", "log_pts_diff", "age_diff", "ht_diff",
    "clay", "grass", "hard", "bo5", "big_event",
    "log_rank_diff*bo5", "log_rank_diff*clay", "log_rank_diff*grass",
]


def build_dataset(rows):
    X, y, dates = [], [], []
    for (date, surface, best_of, level,
         wr, lr_, wp, lp, wa, la, wh, lh) in rows:
        # winner as A (label 1) and loser as A (label 0) — full symmetry
        X.append(feats(wr, lr_, wp, lp, wa, la, wh, lh, surface, best_of, level))
        y.append(1)
        dates.append(date)
        X.append(feats(lr_, wr, lp, wp, la, wa, lh, wh, surface, best_of, level))
        y.append(0)
        dates.append(date)
    return np.array(X, dtype=float), np.array(y, dtype=float), np.array(dates)


def elo_baseline_prob(rank_a, rank_b):
    """The old formula from scheduleService.ts, for comparison."""
    def elo(rank):
        if rank <= 0:
            return 1700.0
        if rank == 1:
            return 2400.0
        return max(1300.0, 2400.0 - 180.0 * np.log(rank))
    return 1.0 / (1.0 + 10 ** ((elo(rank_b) - elo(rank_a)) / 400.0))


def make_nn():
    return MLPClassifier(
        hidden_layer_sizes=(24, 12),
        activation="relu",
        alpha=1e-3,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=12,
        random_state=42,
    )


def fit_platt(nn, Xs, y):
    """Platt-calibrate the network's output logit."""
    raw = nn.predict_proba(Xs)[:, 1]
    logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / np.clip(1 - raw, 1e-6, 1 - 1e-6))
    platt = LogisticRegression()
    platt.fit(logit.reshape(-1, 1), y)
    return float(platt.coef_[0][0]), float(platt.intercept_[0])


def apply_platt(p_raw, A, B):
    z = np.log(np.clip(p_raw, 1e-6, 1 - 1e-6) / np.clip(1 - p_raw, 1e-6, 1 - 1e-6))
    return 1.0 / (1.0 + np.exp(-(A * z + B)))


def main():
    rows = load_matches()
    print(f"matches with both ranks: {len(rows)}")

    X, y, dates = build_dataset(rows)
    train_mask = dates < "2024-01-01"
    test_mask = ~train_mask
    print(f"train samples: {train_mask.sum()}, test samples (2024): {test_mask.sum()}")

    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std

    nn = make_nn()
    nn.fit(Xs[train_mask], y[train_mask])
    A, B = fit_platt(nn, Xs[train_mask], y[train_mask])

    # ── Out-of-sample evaluation (2024) ──────────────────────────────────
    p_test = apply_platt(nn.predict_proba(Xs[test_mask])[:, 1], A, B)
    y_test = y[test_mask]
    acc = float(((p_test > 0.5) == y_test).mean())
    ll = float(log_loss(y_test, p_test))
    br = float(brier_score_loss(y_test, p_test))
    print(f"\nNN  2024 out-of-sample:  acc={acc:.4f}  logloss={ll:.4f}  brier={br:.4f}")

    elo_p, elo_y = [], []
    for (date, surface, best_of, level, wr, lr_, *_r) in rows:
        if date >= "2024-01-01":
            elo_p.append(elo_baseline_prob(wr, lr_)); elo_y.append(1)
            elo_p.append(elo_baseline_prob(lr_, wr)); elo_y.append(0)
    elo_p = np.array(elo_p); elo_y = np.array(elo_y)
    print(f"Elo 2024 baseline:       acc={((elo_p > 0.5) == elo_y).mean():.4f}  "
          f"logloss={log_loss(elo_y, elo_p):.4f}  brier={brier_score_loss(elo_y, elo_p):.4f}")

    print("\ncalibration (bin → mean_pred / mean_actual / n):")
    bins = np.linspace(0, 1, 11)
    for i in range(10):
        m = (p_test >= bins[i]) & (p_test < bins[i + 1])
        if m.sum() > 50:
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {p_test[m].mean():.3f} / {y_test[m].mean():.3f} / {m.sum()}")

    # ── Retrain on ALL data for the shipped model ─────────────────────────
    mean_all = X.mean(axis=0)
    std_all = X.std(axis=0)
    std_all[std_all == 0] = 1.0
    Xs_all = (X - mean_all) / std_all
    nn_final = make_nn()
    nn_final.fit(Xs_all, y)
    A_f, B_f = fit_platt(nn_final, Xs_all, y)

    model = {
        "feature_names": FEATURE_NAMES,
        "scaler_mean": mean_all.tolist(),
        "scaler_std": std_all.tolist(),
        "layers": [
            {"W": w.tolist(), "b": b.tolist()}
            for w, b in zip(nn_final.coefs_, nn_final.intercepts_)
        ],
        "platt": {"A": A_f, "B": B_f},
        "meta": {
            "trained_on": f"{len(rows)} ATP matches 2010-2024 (both orientations)",
            "oos_2024": {"acc": round(acc, 4), "logloss": round(ll, 4), "brier": round(br, 4)},
        },
    }
    with open(OUT, "w") as f:
        json.dump(model, f)
    print(f"\nwrote {OUT} ({len(json.dumps(model)) // 1024} KB)")


if __name__ == "__main__":
    main()
