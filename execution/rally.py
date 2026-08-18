"""
Rally intelligence profiler (Tier 1).

Turns the raw MCP rally counts in `rally_stats` (populated by
fetch_charting_data.py) into pre-match rally-construction signals for a player,
as of a given date:

  short_win_pct / mid_win_pct / long_win_pct
        win rate in 1-3 / 4-6 / 7+ shot rallies. Short-rally dominance is
        first-strike (serve/return) power; long-rally dominance is grinding /
        consistency / stamina.
  first_strike_index
        share of a player's points that come in short (<=3 shot) rallies.
        High = the player lives and dies on the first strike.
  aggression
        (winners - unforced errors) per point. A shot-quality proxy.

Design mirrors features.py: strict pre-match temporal isolation (only matches
before as_of) and exponential time-decay. Raw point counts are pooled
(sum numerators / sum denominators) rather than averaging per-match rates, so
small samples degrade gracefully. When a player has no charted history, the
tour-average profile is returned with has_data=False so downstream diffs are
neutral and the uncertainty layer can down-weight.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "tennis_data.db"

# Minimum pooled points in a bucket before we trust its rate; below this we
# blend toward the tour average (shrinkage) to avoid noisy small-sample rates.
MIN_PTS = 30
SHRINK_K = 40.0          # pseudo-count strength for shrinkage
HALF_LIFE_YEARS = 1.5    # rally style is fairly stable -> slow decay


def norm_name(name: str) -> str:
    """Must match fetch_charting_data.norm_name exactly (join key)."""
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().replace(".", " ").replace("-", " ").replace("'", " ")
    return " ".join(n.split())


@dataclass
class RallyProfile:
    short_win_pct: float
    mid_win_pct: float
    long_win_pct: float
    first_strike_index: float
    aggression: float
    n_matches: int
    n_points: int
    has_data: bool

    def as_dict(self) -> dict:
        return asdict(self)


class RallyProfiler:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.conn = sqlite3.connect(str(db_path))
        self._tour_avg: dict[str, dict] = {}

    def close(self):
        self.conn.close()

    # ── tour-average fallback (computed once, pooled over all data) ──────────
    def _tour_average(self, tour: str) -> dict:
        tour = (tour or "ATP").upper()
        if tour not in ("ATP", "WTA"):
            tour = "ATP"
        if tour in self._tour_avg:
            return self._tour_avg[tour]
        row = self.conn.execute(
            """SELECT SUM(short_won), SUM(short_pts), SUM(mid_won), SUM(mid_pts),
                      SUM(long_won), SUM(long_pts), SUM(total_won), SUM(total_pts),
                      SUM(winners), SUM(unforced)
               FROM rally_stats WHERE tour=?""", (tour,)).fetchone()
        sw, sp, mw, mp, lw, lp, tw, tp, win, unf = (v or 0 for v in row)
        avg = {
            "short": sw / sp if sp else 0.5,
            "mid": mw / mp if mp else 0.5,
            "long": lw / lp if lp else 0.5,
            "first_strike": sw / tw if tw else 0.5,
            "aggression": (win - unf) / tp if tp else 0.0,
        }
        self._tour_avg[tour] = avg
        return avg

    @staticmethod
    def _decay(as_of: datetime, past: datetime) -> float:
        years = (as_of - past).days / 365.25
        if years < 0:
            return 0.0
        return 0.5 ** (years / HALF_LIFE_YEARS)

    def _shrink(self, won: float, pts: float, prior: float) -> float:
        """Empirical-Bayes shrinkage toward the tour prior for thin samples."""
        return (won + SHRINK_K * prior) / (pts + SHRINK_K)

    def profile(self, player_name: str, as_of: datetime | str,
                tour: str = "ATP") -> RallyProfile:
        if isinstance(as_of, str):
            as_of = datetime.strptime(as_of[:10], "%Y-%m-%d")
        key = norm_name(player_name)
        avg = self._tour_average(tour)

        if not key:
            return self._fallback(avg)

        rows = self.conn.execute(
            """SELECT match_date, short_won, short_pts, mid_won, mid_pts,
                      long_won, long_pts, total_won, total_pts, winners, unforced
               FROM rally_stats
               WHERE player_key=? AND match_date < ?
               ORDER BY match_date DESC""",
            (key, as_of.strftime("%Y-%m-%d"))).fetchall()
        if not rows:
            return self._fallback(avg)

        acc = {k: 0.0 for k in ("sw", "sp", "mw", "mp", "lw", "lp",
                                "tw", "tp", "win", "unf")}
        n = 0
        for (d, sw, sp, mw, mp, lw, lp, tw, tp, win, unf) in rows:
            try:
                pd_ = datetime.strptime(d[:10], "%Y-%m-%d")
            except (TypeError, ValueError):
                continue
            w = self._decay(as_of, pd_)
            if w <= 1e-4:
                continue
            acc["sw"] += w * (sw or 0); acc["sp"] += w * (sp or 0)
            acc["mw"] += w * (mw or 0); acc["mp"] += w * (mp or 0)
            acc["lw"] += w * (lw or 0); acc["lp"] += w * (lp or 0)
            acc["tw"] += w * (tw or 0); acc["tp"] += w * (tp or 0)
            acc["win"] += w * (win or 0); acc["unf"] += w * (unf or 0)
            n += 1

        raw_pts = int(round(acc["tp"]))
        if acc["tp"] < 1:
            return self._fallback(avg)

        short = self._shrink(acc["sw"], acc["sp"], avg["short"])
        mid = self._shrink(acc["mw"], acc["mp"], avg["mid"])
        long_ = self._shrink(acc["lw"], acc["lp"], avg["long"])
        first_strike = (acc["sw"] / acc["tw"]) if acc["tw"] > 0 else avg["first_strike"]
        aggression = ((acc["win"] - acc["unf"]) / acc["tp"]) if acc["tp"] > 0 else avg["aggression"]

        return RallyProfile(
            short_win_pct=round(short, 4),
            mid_win_pct=round(mid, 4),
            long_win_pct=round(long_, 4),
            first_strike_index=round(first_strike, 4),
            aggression=round(aggression, 4),
            n_matches=n,
            n_points=raw_pts,
            has_data=(raw_pts >= MIN_PTS),
        )

    def _fallback(self, avg: dict) -> RallyProfile:
        return RallyProfile(
            short_win_pct=round(avg["short"], 4),
            mid_win_pct=round(avg["mid"], 4),
            long_win_pct=round(avg["long"], 4),
            first_strike_index=round(avg["first_strike"], 4),
            aggression=round(avg["aggression"], 4),
            n_matches=0, n_points=0, has_data=False,
        )

    def diff_features(self, name1: str, name2: str, as_of: datetime | str,
                      tour: str = "ATP") -> dict:
        """Player1-minus-player2 rally features + raw values for the Markov layer."""
        p1 = self.profile(name1, as_of, tour)
        p2 = self.profile(name2, as_of, tour)
        return {
            "RALLY_SHORT_WIN_DIFF": p1.short_win_pct - p2.short_win_pct,
            "RALLY_MID_WIN_DIFF":   p1.mid_win_pct - p2.mid_win_pct,
            "RALLY_LONG_WIN_DIFF":  p1.long_win_pct - p2.long_win_pct,
            "FIRST_STRIKE_DIFF":    p1.first_strike_index - p2.first_strike_index,
            "RALLY_AGGRESSION_DIFF": p1.aggression - p2.aggression,
            "RALLY_DATA_BOTH": 1 if (p1.has_data and p2.has_data) else 0,
            "_p1_rally": p1.as_dict(),
            "_p2_rally": p2.as_dict(),
        }


if __name__ == "__main__":
    import sys
    prof = RallyProfiler()
    a = sys.argv[1] if len(sys.argv) > 1 else "Jannik Sinner"
    b = sys.argv[2] if len(sys.argv) > 2 else "Rafael Nadal"
    asof = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
    import json
    print(json.dumps(prof.diff_features(a, b, asof), indent=2))
    prof.close()
