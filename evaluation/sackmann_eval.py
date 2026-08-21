"""
Leak-free evaluation harness for the saved Sackmann-lineage models.

WHAT THIS IS FOR
  `ml_models/logistic_regression_trained.pkl` and `neural_network_ensemble.pkl`
  both expect the same 14 features. In production those are built by
  `trading_server/live_features.py` from a player's *career profile* blended with
  live in-match stats — i.e. from information available BEFORE the match starts.

  To evaluate the models honestly on history we have to reproduce that same
  pre-match view. The obvious shortcut — reading w_ace / w_1stIn / w_svpt off the
  match row — is the stats OF the match being predicted, which is target leakage:
  a model told how many break points a player saved has largely been told who
  won. `train_sackmann_models.py` does exactly that, which is why its reported
  numbers should not be read as out-of-sample skill.

HOW LEAKAGE IS AVOIDED
  Matches are walked in date order while per-player accumulators are carried
  forward. Each match is scored from the accumulator state as it stands BEFORE
  that match, and only then are its stats folded in. One pass, O(n), and no row
  can ever see itself or anything later.

FEATURE SEMANTICS — established from the artefacts, not from a sibling script
  Neither `train_sackmann_models.py` nor `trading_server/live_features.py` matches
  the pickles exactly: the training script's 10th feature is `serve_pts_diff`,
  while the pickles want `win_rate_diff`; the live engine has the right names but
  builds plain ratios. Guessing between them puts the model in a different
  feature space while still "running", which is the worst failure mode available
  because it looks like a result.

  So the conventions below were read off the saved objects themselves:

  * `scaler.mean_` / `scaler.scale_` record the TRAINING distribution.
    `rank_ratio` and `pts_ratio` are centred at ~0.01 with sd ~1.3-1.5 — that is
    a LOG ratio. Plain ratios land at ~3.0/~2.1, i.e. several sd off-centre, and
    the resulting bias is visible as the model predicting p1 at ~0.62 on a
    balanced set.
  * Sign conventions come from the fitted coefficients. `rank_diff` carries a
    POSITIVE weight, which is only coherent as `rank2 - rank1` (lower rank number
    is the better player, so a positive difference favours p1). `win_rate_diff`
    (+0.72, the dominant term) and `pts_ratio` (+0.52) confirm p1-minus-p2.
  * `df_diff`'s weight is +0.0005 — indistinguishable from zero, so its
    orientation cannot be recovered and does not matter. It follows the p1-p2
    convention for consistency with every other differential.

  Every remaining feature reproduces the training distribution closely
  (e.g. first_serve_pct_diff 0.0000 vs 0.0002, is_clay 0.3079 vs 0.3068).

USAGE
    from evaluation.sackmann_eval import build_dataset, load_models, predict_all
    ds = build_dataset('tennis_data.db', start='2023-01-01', end='2025-01-01')
    models = load_models()
    preds = predict_all(models, ds)
"""

from __future__ import annotations

import pickle
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Exactly the order the pickles were fit on. Do not reorder.
SACKMANN_FEATURES = [
    "rank_diff", "rank_ratio", "pts_ratio",
    "first_serve_pct_diff", "first_win_diff", "second_win_diff",
    "bp_save_diff", "ace_diff", "df_diff", "win_rate_diff",
    "is_clay", "is_grass", "is_grand_slam", "is_masters",
]

# Priors for a player we have never seen. Same values train_sackmann_models.py
# falls back to, so an unseen player lands in the same region of feature space
# the model was fit over rather than at zero.
PRIOR = {
    "first_serve_pct": 0.62,
    "first_serve_win_pct": 0.70,
    "second_serve_win_pct": 0.50,
    "bp_save_pct": 0.65,
    "aces_per_game": 0.5,
    "df_per_game": 0.2,
    "win_rate": 0.50,
}


@dataclass
class _Acc:
    """Running career totals for one player, as of 'so far'."""
    svpt: float = 0.0
    first_in: float = 0.0
    first_won: float = 0.0
    second_won: float = 0.0
    sv_gms: float = 0.0
    ace: float = 0.0
    df: float = 0.0
    bp_saved: float = 0.0
    bp_faced: float = 0.0
    wins: int = 0
    losses: int = 0

    @property
    def n(self) -> int:
        return self.wins + self.losses

    def profile(self) -> Dict[str, float]:
        """Career rates. Falls back to the prior for any denominator we lack."""
        second_pts = self.svpt - self.first_in
        return {
            "first_serve_pct": self.first_in / self.svpt if self.svpt > 0 else PRIOR["first_serve_pct"],
            "first_serve_win_pct": self.first_won / self.first_in if self.first_in > 0 else PRIOR["first_serve_win_pct"],
            "second_serve_win_pct": self.second_won / second_pts if second_pts > 0 else PRIOR["second_serve_win_pct"],
            "bp_save_pct": self.bp_saved / self.bp_faced if self.bp_faced > 0 else PRIOR["bp_save_pct"],
            "aces_per_game": self.ace / self.sv_gms if self.sv_gms > 0 else PRIOR["aces_per_game"],
            "df_per_game": self.df / self.sv_gms if self.sv_gms > 0 else PRIOR["df_per_game"],
            "win_rate": self.wins / self.n if self.n > 0 else PRIOR["win_rate"],
        }

    def add(self, r: pd.Series, won: bool) -> None:
        p = "w_" if won else "l_"
        def g(col, default=0.0):
            v = r.get(p + col)
            return float(v) if v is not None and not pd.isna(v) else default
        self.svpt += g("svpt"); self.first_in += g("1stIn")
        self.first_won += g("1stWon"); self.second_won += g("2ndWon")
        self.sv_gms += g("SvGms"); self.ace += g("ace"); self.df += g("df")
        self.bp_saved += g("bpSaved"); self.bp_faced += g("bpFaced")
        if won:
            self.wins += 1
        else:
            self.losses += 1


def _feature_row(p1: Dict[str, float], p2: Dict[str, float],
                 rank1: float, rank2: float, pts1: float, pts2: float,
                 surface: str, level: str) -> List[float]:
    """One 14-vector in the space the pickles were fit on (see module docstring)."""
    rank1, rank2 = max(rank1, 1.0), max(rank2, 1.0)
    pts1, pts2 = max(pts1, 1.0), max(pts2, 1.0)
    return [
        rank2 - rank1,                                                  # rank_diff (+ve favours p1)
        float(np.log(rank2 / rank1)),                                   # rank_ratio (log)
        float(np.log(pts1 / pts2)),                                     # pts_ratio (log)
        p1["first_serve_pct"] - p2["first_serve_pct"],                  # first_serve_pct_diff
        p1["first_serve_win_pct"] - p2["first_serve_win_pct"],          # first_win_diff
        p1["second_serve_win_pct"] - p2["second_serve_win_pct"],        # second_win_diff
        p1["bp_save_pct"] - p2["bp_save_pct"],                          # bp_save_diff
        p1["aces_per_game"] - p2["aces_per_game"],                      # ace_diff
        p1["df_per_game"] - p2["df_per_game"],                          # df_diff (weight ~0; see docstring)
        p1["win_rate"] - p2["win_rate"],                                # win_rate_diff
        1.0 if surface == "Clay" else 0.0,
        1.0 if surface == "Grass" else 0.0,
        1.0 if level == "G" else 0.0,
        1.0 if level == "M" else 0.0,
    ]


def build_dataset(db_path: str = "tennis_data.db",
                  start: str = "2023-01-01",
                  end: str = "2025-01-01",
                  min_history: int = 10,
                  warmup_from: str = "2010-01-01",
                  seed: int = 42) -> pd.DataFrame:
    """Pre-match features + outcome for every match in [start, end).

    Accumulators are warmed from `warmup_from` so that a match early in the test
    window is still scored against a full career history — otherwise every
    January would be judged on players who look brand new.

    `min_history` drops matches where either player has fewer than that many
    prior matches; their career rates are mostly prior, not signal.

    Player 1 is assigned by a coin flip per match, so the label is balanced and
    the model cannot profit from a positional convention.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT match_id, tournament_date, surface, tourney_level,
               winner_id, loser_id, winner_rank, loser_rank,
               winner_rank_points, loser_rank_points,
               w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms,
               w_bpSaved, w_bpFaced,
               l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms,
               l_bpSaved, l_bpFaced
        FROM matches
        WHERE tournament_date >= ? AND tournament_date < ?
        ORDER BY tournament_date, match_id
        """,
        conn, params=(warmup_from, end),
    )
    conn.close()

    rng = np.random.default_rng(seed)
    accs: Dict[int, _Acc] = {}
    rows: List[dict] = []

    for _, r in df.iterrows():
        w, l = int(r["winner_id"]), int(r["loser_id"])
        aw = accs.setdefault(w, _Acc())
        al = accs.setdefault(l, _Acc())

        in_window = str(r["tournament_date"])[:10] >= start
        if in_window and aw.n >= min_history and al.n >= min_history:
            # Snapshot BEFORE folding this match in — this is the leak barrier.
            pw, pl = aw.profile(), al.profile()
            surface = r["surface"] or "Hard"
            level = r["tourney_level"] or "A"

            if rng.random() < 0.5:
                p1, p2, y = pw, pl, 1          # winner is player 1
                r1, r2 = r["winner_rank"], r["loser_rank"]
                q1, q2 = r["winner_rank_points"], r["loser_rank_points"]
            else:
                p1, p2, y = pl, pw, 0          # loser is player 1
                r1, r2 = r["loser_rank"], r["winner_rank"]
                q1, q2 = r["loser_rank_points"], r["winner_rank_points"]

            def num(v, d):
                return float(v) if v is not None and not pd.isna(v) else d

            feats = _feature_row(p1, p2, num(r1, 100), num(r2, 100),
                                 num(q1, 1000), num(q2, 1000), surface, level)
            rows.append({
                "match_id": r["match_id"],
                "tournament_date": r["tournament_date"],
                "surface": surface,
                "tourney_level": level,
                **dict(zip(SACKMANN_FEATURES, feats)),
                "actual": y,
            })

        aw.add(r, won=True)
        al.add(r, won=False)

    return pd.DataFrame(rows)


# ─── Model loading ───────────────────────────────────────────────────────────

class TennisNN:
    """Rebuilt so the pickled ensemble can be unpickled.

    The ensemble was saved from a notebook, so torch recorded its class as
    `__main__.TennisNN`. Without an identically-shaped class registered under
    that name, `pickle.load` raises AttributeError and the models look lost when
    they are merely unaddressable. Architecture matches train_sackmann_models.py.
    """
    def __new__(cls, *a, **k):
        import torch.nn as nn

        class _TennisNN(nn.Module):
            def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.3):
                super().__init__()
                layers, d = [], input_dim
                for h in hidden_dims:
                    layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                    d = h
                layers.append(nn.Linear(d, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)

        return _TennisNN(*a, **k)


def _register_nn_class() -> None:
    import torch.nn as nn

    class TennisNN(nn.Module):                      # noqa: D401 - pickle target
        def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.3):
            super().__init__()
            layers, d = [], input_dim
            for h in hidden_dims:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    setattr(sys.modules["__main__"], "TennisNN", TennisNN)


def load_models(models_dir: str = "ml_models") -> Dict[str, dict]:
    """Load every saved model that speaks the Sackmann-14 feature set."""
    out: Dict[str, dict] = {}

    try:
        with open(f"{models_dir}/logistic_regression_trained.pkl", "rb") as f:
            d = pickle.load(f)
        out["Logistic Regression"] = {
            "kind": "sklearn", "model": d["model"], "scaler": d.get("scaler"),
            "features": list(d["features"]),
        }
    except Exception as e:
        print(f"  LR unavailable: {str(e)[:90]}")

    try:
        _register_nn_class()
        with open(f"{models_dir}/neural_network_ensemble.pkl", "rb") as f:
            d = pickle.load(f)
        out["Neural Network"] = {
            "kind": "torch_ensemble", "models": d["models"], "scaler": d.get("scaler"),
            "features": list(d["features"]),
        }
    except Exception as e:
        print(f"  NN unavailable: {str(e)[:90]}")

    return out


def predict_all(models: Dict[str, dict], ds: pd.DataFrame) -> Dict[str, np.ndarray]:
    """P(player 1 wins) from each model, plus two reference baselines."""
    preds: Dict[str, np.ndarray] = {}

    for name, m in models.items():
        X = ds[m["features"]].fillna(0).to_numpy(dtype=np.float64)
        Xs = m["scaler"].transform(X) if m.get("scaler") is not None else X

        if m["kind"] == "sklearn":
            preds[name] = m["model"].predict_proba(Xs)[:, 1]

        elif m["kind"] == "torch_ensemble":
            import torch
            t = torch.tensor(Xs, dtype=torch.float32)
            members = []
            for net in m["models"]:
                net.eval()
                with torch.no_grad():
                    members.append(torch.sigmoid(net(t)).squeeze(-1).numpy())
            # Average member probabilities — this is what makes it an ensemble.
            preds[name] = np.mean(members, axis=0)

    # Baselines. Without these a Brier score has no scale: 0.21 sounds fine until
    # the "always 50%" line comes in at 0.25 and rank alone matches you.
    n = len(ds)
    preds["Baseline (coin flip)"] = np.full(n, 0.5)
    # rank_diff is rank2 - rank1, so it is POSITIVE when p1 is the better-ranked
    # player — the logit therefore takes it with a positive sign. Getting this
    # backwards makes the baseline score worse than a coin flip, which is a
    # useful tell that the sign is wrong rather than that ranking is useless.
    rank_edge = 0.02 * ds["rank_diff"].to_numpy(dtype=np.float64)
    preds["Baseline (rank only)"] = 1.0 / (1.0 + np.exp(-np.clip(rank_edge, -6, 6)))
    return preds
