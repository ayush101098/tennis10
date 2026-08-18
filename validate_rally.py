#!/usr/bin/env python3
"""
Tier 1 validation harness for rally intelligence.

Answers four questions honestly:
  1. COVERAGE      what fraction of recent ATP matches have BOTH players charted
                   (as-of the match date)? This is the real usable-signal rate.
  2. DISCRIMINATION do the features actually separate playing styles?
                   (big servers/first-strikers vs grinders)
  3. ISOLATION     is the profile strictly pre-match (no future leakage)?
  4. LIFT          does adding the 5 rally diffs on top of a rank baseline
                   improve OUT-OF-SAMPLE match prediction (time-split)?

Run: python3 validate_rally.py
"""

import sqlite3
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from execution.rally import RallyProfiler

DB = "tennis_data.db"
RALLY_KEYS = ["RALLY_SHORT_WIN_DIFF", "RALLY_MID_WIN_DIFF", "RALLY_LONG_WIN_DIFF",
              "FIRST_STRIKE_DIFF", "RALLY_AGGRESSION_DIFF"]


def line(c="-"):
    print(c * 64)


def load_matches(since="2018-01-01"):
    con = sqlite3.connect(DB)
    rows = con.execute(
        """SELECT m.tournament_date, pw.player_name, pl.player_name,
                  m.winner_rank, m.loser_rank
           FROM matches m
           JOIN players pw ON m.winner_id = pw.player_id
           JOIN players pl ON m.loser_id  = pl.player_id
           WHERE m.tournament_date >= ?
             AND m.winner_rank > 0 AND m.loser_rank > 0
           ORDER BY m.tournament_date""", (since,)).fetchall()
    con.close()
    return rows


# ── 1 + 4: coverage and predictive lift ─────────────────────────────────────
def coverage_and_lift(prof: RallyProfiler):
    rows = load_matches("2018-01-01")
    print(f"ATP matches since 2018 with ranks: {len(rows):,}")

    X_base, X_full, y, dates = [], [], [], []
    both = 0
    for (date, wname, lname, wrank, lrank) in rows:
        d = prof.diff_features(wname, lname, date, tour="ATP")
        has_both = d["RALLY_DATA_BOTH"] == 1
        if has_both:
            both += 1
        else:
            continue  # lift is measured only where we actually have signal

        rank_feat = np.log(lrank) - np.log(wrank)   # + => winner higher ranked
        rally_feat = [d[k] for k in RALLY_KEYS]

        # symmetric: add winner-orientation (y=1) and mirrored loser-orientation (y=0)
        X_base.append([rank_feat]);            X_full.append([rank_feat] + rally_feat);            y.append(1); dates.append(date)
        X_base.append([-rank_feat]);           X_full.append([-rank_feat] + [-v for v in rally_feat]); y.append(0); dates.append(date)

    line("=")
    print("1. COVERAGE")
    line()
    print(f"  matches with BOTH players charted (as-of date): {both:,}/{len(rows):,} "
          f"= {both/max(len(rows),1):.1%}")
    recent = [r for r in rows if r[0] >= "2023-01-01"]
    recent_both = sum(1 for (date, w, l, wr, lr) in recent
                      if prof.diff_features(w, l, date, "ATP")["RALLY_DATA_BOTH"] == 1)
    print(f"  of those since 2023:                            {recent_both:,}/{len(recent):,} "
          f"= {recent_both/max(len(recent),1):.1%}")

    X_base, X_full, y = np.array(X_base), np.array(X_full), np.array(y)
    dates = np.array(dates)
    train = dates < "2023-01-01"
    test = ~train
    if test.sum() < 50:
        print("\n  (not enough post-2023 charted matches for a lift test)")
        return

    line("=")
    print("4. OUT-OF-SAMPLE PREDICTIVE LIFT  (train<2023, test>=2023)")
    line()
    print(f"  train samples: {train.sum():,}   test samples: {test.sum():,}")

    def fit_eval(X):
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(X[train], y[train])
        p = m.predict_proba(X[test])[:, 1]
        return log_loss(y[test], p), roc_auc_score(y[test], p), m

    ll_b, auc_b, _ = fit_eval(X_base)
    ll_f, auc_f, mf = fit_eval(X_full)
    print(f"  baseline (rank only)   logloss {ll_b:.4f}   AUC {auc_b:.4f}")
    print(f"  rank + rally features  logloss {ll_f:.4f}   AUC {auc_f:.4f}")
    print(f"  Δ logloss {ll_f-ll_b:+.4f} (lower=better)   Δ AUC {auc_f-auc_b:+.4f} (higher=better)")
    print("\n  rally feature coefficients (on top of rank):")
    for k, c in zip(RALLY_KEYS, mf.coef_[0][1:]):
        print(f"    {k:24s} {c:+.3f}")


# ── 2: discrimination ────────────────────────────────────────────────────────
def discrimination(prof: RallyProfiler):
    line("=")
    print("2. STYLE DISCRIMINATION  (profiles as of 2024-01-01)")
    line()
    players = ["John Isner", "Reilly Opelka", "Nick Kyrgios",   # first-strike servers
               "Rafael Nadal", "Diego Schwartzman", "Casper Ruud",  # grinders
               "Novak Djokovic", "Jannik Sinner"]
    print(f"  {'player':22s} {'short':>6s} {'mid':>6s} {'long':>6s} "
          f"{'1stStrk':>8s} {'aggr':>7s} {'pts':>7s}")
    for name in players:
        p = prof.profile(name, "2024-01-01", "ATP")
        flag = "" if p.has_data else "  (fallback)"
        print(f"  {name:22s} {p.short_win_pct:6.3f} {p.mid_win_pct:6.3f} "
              f"{p.long_win_pct:6.3f} {p.first_strike_index:8.3f} "
              f"{p.aggression:+7.3f} {p.n_points:7d}{flag}")
    print("\n  Expect: servers -> high 1stStrk, short>=long ; grinders -> long>=short.")


# ── 3: temporal isolation ────────────────────────────────────────────────────
def isolation(prof: RallyProfiler):
    line("=")
    print("3. TEMPORAL ISOLATION  (Carlos Alcaraz, sampled forward in time)")
    line()
    for asof in ["2020-01-01", "2021-06-01", "2022-06-01", "2023-06-01", "2024-06-01"]:
        p = prof.profile("Carlos Alcaraz", asof, "ATP")
        tag = "fallback (no prior data)" if not p.has_data else f"{p.n_matches} prior matches"
        print(f"  as-of {asof}:  short {p.short_win_pct:.3f}  long {p.long_win_pct:.3f}  "
              f"pts {p.n_points:5d}   [{tag}]")
    print("\n  Expect: fallback early (before he was charted), then growing sample.")


if __name__ == "__main__":
    prof = RallyProfiler()
    discrimination(prof)
    print()
    isolation(prof)
    print()
    coverage_and_lift(prof)
    prof.close()
