"""
Walk-forward backtest of the three SX strategy tiers' PRICING.

WHAT THIS MEASURES (and what it cannot)
  It scores the probabilities the strategies bet on — match winner, set winner,
  total games — against 42k historical matches, walk-forward and leakage-safe:
  every serve input for a match comes ONLY from that player's prior matches.

  It does NOT produce ROI. tennis_data.db's `odds` table is empty (0 rows) and
  SX publishes no historical order book, so there are no historical prices to
  price an edge against. Any ROI number here would be invented. What this DOES
  answer is the question that decides whether the edges are real at all:
  *is our pricing calibrated, and does it beat a naive baseline?*
  An uncalibrated model cannot have a real edge, whatever the book says.

SCORING
  Brier score (lower = better) and calibration bins (predicted vs actual) per
  tier, each against a baseline:
    MATCH  → vs always-50% and vs a rank-based logistic prior
    SET    → vs always-50%
    GAME   → vs the empirical base rate of going over the line

    python -m execution.sx_backtest                    # default 4000 matches
    python -m execution.sx_backtest --n 8000 --sims 600
    python -m execution.sx_backtest --min-prior 30     # stricter serve prior
"""

import argparse
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from execution.sx_strategies import simulate, parse_state   # noqa: E402

DB = REPO_ROOT / "tennis_data.db"
TOTAL_GAMES_LINE = 22.5          # the common SX best-of-3 line


# ── score parsing ────────────────────────────────────────────────────────────

def parse_score(score: str):
    """'6-3 4-6 6-4' → [(6,3),(4,6),(6,4)] oriented to the WINNER.
    None for retirements/walkovers/unparseable."""
    if not score:
        return None
    s = score.strip()
    if re.search(r"RET|W/O|DEF|Walkover|ABN|Def\.", s, re.I):
        return None
    sets = []
    for tok in s.split():
        tok = re.sub(r"\(\d+\)", "", tok)          # strip tiebreak detail
        m = re.match(r"^(\d{1,2})-(\d{1,2})$", tok)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        if a > 30 or b > 30:
            return None
        sets.append((a, b))
    return sets or None


# ── walk-forward serve priors ────────────────────────────────────────────────

class ServeTracker:
    """Career-to-date serve point-win % per player, updated only AFTER a match
    is scored — so every prediction uses strictly prior information."""

    def __init__(self):
        self.won = defaultdict(float)
        self.pts = defaultdict(float)
        self.n = defaultdict(int)

    def spw(self, pid: int):
        if self.pts[pid] <= 0:
            return None
        return self.won[pid] / self.pts[pid]

    def count(self, pid: int) -> int:
        return self.n[pid]

    def update(self, pid: int, won: float, pts: float):
        if pts and pts > 0:
            self.won[pid] += won
            self.pts[pid] += pts
            self.n[pid] += 1


# ── scoring helpers ──────────────────────────────────────────────────────────

class Scorer:
    """Brier + calibration bins for one binary forecast stream."""

    def __init__(self, name: str):
        self.name = name
        self.preds, self.outcomes = [], []

    def add(self, p: float, outcome: bool):
        self.preds.append(p)
        self.outcomes.append(1 if outcome else 0)

    @property
    def n(self):
        return len(self.preds)

    def brier(self):
        if not self.preds:
            return None
        return sum((p - o) ** 2 for p, o in zip(self.preds, self.outcomes)) / self.n

    def brier_const(self, c: float):
        if not self.preds:
            return None
        return sum((c - o) ** 2 for o in self.outcomes) / self.n

    def accuracy(self):
        if not self.preds:
            return None
        return sum(1 for p, o in zip(self.preds, self.outcomes)
                   if (p >= 0.5) == (o == 1)) / self.n

    def base_rate(self):
        return sum(self.outcomes) / self.n if self.preds else None

    def calibration(self, edges=(0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01)):
        rows = []
        for lo, hi in zip(edges, edges[1:]):
            idx = [i for i, p in enumerate(self.preds) if lo <= p < hi]
            if len(idx) < 25:
                continue
            pred = statistics.mean(self.preds[i] for i in idx)
            act = statistics.mean(self.outcomes[i] for i in idx)
            rows.append((f"{lo:.0%}-{hi:.0%}", len(idx), pred, act))
        return rows

    def report(self, baseline_label: str, baseline_brier: float | None):
        if not self.n:
            print(f"\n{self.name}: no samples")
            return
        b = self.brier()
        print(f"\n── {self.name}  (n={self.n:,}) ──")
        print(f"   Brier      {b:.4f}     {baseline_label} {baseline_brier:.4f}"
              f"   → {'BETTER' if b < baseline_brier else 'WORSE'} by {abs(b - baseline_brier):.4f}")
        print(f"   accuracy   {self.accuracy():.4f}     base rate {self.base_rate():.4f}")
        cal = self.calibration()
        if cal:
            print("   calibration:  bucket        n     predicted   actual   gap")
            for lbl, n, pred, act in cal:
                flag = "  ⚠" if abs(pred - act) > 0.05 else ""
                print(f"                 {lbl:10s} {n:6,}   {pred:8.3f} {act:8.3f}  "
                      f"{pred - act:+.3f}{flag}")


# ── backtest ─────────────────────────────────────────────────────────────────

def run(limit: int, sims: int, min_prior: int, seed: int):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT match_id, tournament_date, best_of, winner_id, loser_id, score,
               w_svpt, w_1stWon, w_2ndWon, l_svpt, l_1stWon, l_2ndWon,
               winner_rank, loser_rank
        FROM matches
        WHERE score IS NOT NULL AND best_of = 3
          AND w_svpt > 0 AND l_svpt > 0
        ORDER BY tournament_date, match_id
    """).fetchall()
    print(f"loaded {len(rows):,} best-of-3 matches with serve stats")

    tracker = ServeTracker()
    s_match = Scorer("MATCH  — P(player one wins the match)")
    s_set1 = Scorer("SET    — P(player one wins set 1)")
    s_games = Scorer(f"GAME   — P(total games > {TOTAL_GAMES_LINE})")
    scored = 0
    skipped_prior = 0

    for r in rows:
        sets = parse_score(r["score"])
        w_pts, l_pts = r["w_svpt"], r["l_svpt"]
        w_won = (r["w_1stWon"] or 0) + (r["w_2ndWon"] or 0)
        l_won = (r["l_1stWon"] or 0) + (r["l_2ndWon"] or 0)

        if sets and scored < limit:
            wid, lid = r["winner_id"], r["loser_id"]
            sp_w, sp_l = tracker.spw(wid), tracker.spw(lid)
            enough = (tracker.count(wid) >= min_prior and tracker.count(lid) >= min_prior)
            if sp_w and sp_l and enough:
                # Orient to "player one" = the LOWER player_id, so the label is
                # independent of who won (no outcome leakage into orientation).
                p1_is_winner = wid < lid
                sp1 = sp_w if p1_is_winner else sp_l
                sp2 = sp_l if p1_is_winner else sp_w

                sim = simulate(sp1, sp2, parse_state(None), n=sims,
                               seed=seed + r["match_id"])

                # MATCH
                s_match.add(sim["p1_match"], p1_is_winner)

                # SET 1 — score is oriented to the winner
                set1 = sets[0]
                set1_won_by_winner = set1[0] > set1[1]
                set1_won_by_p1 = (set1_won_by_winner == p1_is_winner)
                p_set1 = sim["set_winner"].get(1)
                if p_set1 is not None:
                    s_set1.add(p_set1, set1_won_by_p1)

                # GAME — total games over the line
                total = sum(a + b for a, b in sets)
                s_games.add(sim["p_over_games"](TOTAL_GAMES_LINE), total > TOTAL_GAMES_LINE)

                scored += 1
                if scored % 500 == 0:
                    print(f"  scored {scored:,}…", flush=True)
            elif sp_w and sp_l:
                skipped_prior += 1

        # update AFTER scoring — strictly walk-forward
        tracker.update(r["winner_id"], w_won, w_pts)
        tracker.update(r["loser_id"], l_won, l_pts)
        if scored >= limit:
            break

    print(f"\nscored {scored:,} matches ({skipped_prior:,} skipped: "
          f"< {min_prior} prior matches for a player)")
    print("=" * 74)
    print(f"WALK-FORWARD BACKTEST — {sims} sims/match, serve priors from prior matches only")
    print("=" * 74)

    s_match.report("vs coin-flip 0.5:", s_match.brier_const(0.5))
    s_set1.report("vs coin-flip 0.5:", s_set1.brier_const(0.5))
    s_games.report(f"vs base rate {s_games.base_rate():.3f}:",
                   s_games.brier_const(s_games.base_rate()))

    print("\n" + "=" * 74)
    print("NOTE: no ROI here by design — tennis_data.db has 0 historical odds rows")
    print("and SX publishes no historical book, so edge/ROI cannot be computed")
    print("without inventing prices. Calibration is the precondition for a real edge.")
    print("=" * 74)
    conn.close()


def main():
    ap = argparse.ArgumentParser(description="Walk-forward backtest of SX strategy pricing")
    ap.add_argument("--n", type=int, default=4000, help="matches to score")
    ap.add_argument("--sims", type=int, default=400, help="Monte-Carlo paths per match")
    ap.add_argument("--min-prior", type=int, default=20,
                    help="min prior matches per player for a usable serve prior")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    run(args.n, args.sims, args.min_prior, args.seed)


if __name__ == "__main__":
    main()
