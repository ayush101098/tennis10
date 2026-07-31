"""Walk-forward feature engine: Elo, form, H2H, rest — leakage-safe by design.

One chronological pass over all matches. For each match we FIRST snapshot the
features from current state (which only reflects earlier matches), THEN update
state with the result. That ordering makes look-ahead leakage structurally
impossible — the property the plan calls critical.

Elo is keyed (player, category): Setka Cup and Czech Liga Pro are different
strength pools, and cross-category play is rare, so per-pool ratings are cleaner
than one global rating with a tier offset.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "tabletennis.db"

ELO_START = 1500.0
ELO_K = 32.0

FEATURE_NAMES = [
    "elo_diff", "form10_diff", "form20_diff", "gwr10_diff", "streak_diff",
    "h2h_n", "h2h_p1_rate", "rest_hrs_diff", "load24_diff", "exp_diff",
]


@dataclass
class PlayerState:
    elo: float = ELO_START
    results: deque = field(default_factory=lambda: deque(maxlen=20))   # 1=win 0=loss
    gw: deque = field(default_factory=lambda: deque(maxlen=10))        # (games_won, games_played)
    streak: int = 0
    last_ts: int = 0
    recent_ts: deque = field(default_factory=lambda: deque(maxlen=50)) # match start times
    n_matches: int = 0


class FeatureEngine:
    def __init__(self):
        self.players: dict[tuple, PlayerState] = defaultdict(PlayerState)   # (cat, pid)
        self.h2h: dict[tuple, list] = defaultdict(lambda: [0, 0])           # (a,b) sorted -> [wins_lo, wins_hi]

    @staticmethod
    def _expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def _form(self, st: PlayerState, n: int) -> float:
        r = list(st.results)[-n:]
        if not r:
            return 0.5
        # shrink toward 0.5 for tiny samples
        w = len(r) / (len(r) + 3.0)
        return w * (sum(r) / len(r)) + (1 - w) * 0.5

    def _gwr(self, st: PlayerState) -> float:
        won = sum(g for g, _ in st.gw)
        tot = sum(t for _, t in st.gw)
        return won / tot if tot else 0.5

    def snapshot(self, cat: int, p1: int, p2: int, ts: int) -> list[float]:
        """Feature vector for p1-vs-p2 BEFORE the result is known."""
        a, b = self.players[(cat, p1)], self.players[(cat, p2)]
        lo, hi = min(p1, p2), max(p1, p2)
        h = self.h2h[(lo, hi)]
        h2h_n = h[0] + h[1]
        p1_wins = h[0] if p1 == lo else h[1]
        h2h_rate = (p1_wins / h2h_n) if h2h_n else 0.5

        def rest_hrs(st: PlayerState) -> float:
            return min((ts - st.last_ts) / 3600.0, 72.0) if st.last_ts else 72.0

        def load24(st: PlayerState) -> int:
            return sum(1 for t in st.recent_ts if ts - t < 86400)

        return [
            a.elo - b.elo,
            self._form(a, 10) - self._form(b, 10),
            self._form(a, 20) - self._form(b, 20),
            self._gwr(a) - self._gwr(b),
            float(max(-5, min(5, a.streak)) - max(-5, min(5, b.streak))),
            float(min(h2h_n, 10)),
            h2h_rate,
            (rest_hrs(a) - rest_hrs(b)) / 24.0,
            float(load24(a) - load24(b)),
            float(min(a.n_matches, 200) - min(b.n_matches, 200)) / 50.0,
        ]

    def update(self, cat: int, p1: int, p2: int, ts: int, winner: int,
               g1: int, g2: int) -> None:
        a, b = self.players[(cat, p1)], self.players[(cat, p2)]
        exp = self._expected(a.elo, b.elo)
        s = 1.0 if winner == 1 else 0.0
        a.elo += ELO_K * (s - exp)
        b.elo += ELO_K * ((1 - s) - (1 - exp))
        for st, won, gw, gl in ((a, s, g1, g2), (b, 1 - s, g2, g1)):
            st.results.append(int(won))
            st.gw.append((gw, gw + gl))
            st.streak = st.streak + 1 if (won and st.streak >= 0) else (
                st.streak - 1 if (not won and st.streak <= 0) else (1 if won else -1))
            st.last_ts = ts
            st.recent_ts.append(ts)
            st.n_matches += 1
        lo, hi = min(p1, p2), max(p1, p2)
        w_is_lo = (p1 == lo) == (winner == 1)
        self.h2h[(lo, hi)][0 if w_is_lo else 1] += 1


def build_dataset(db_path: Path = DB_PATH):
    """Chronological pass → (X, y, ts, meta) with symmetric duplication:
    each match is emitted as (p1,p2,y) AND (p2,p1,1-y) so the model can't learn
    a home/away slot bias (Sofascore's ordering is arbitrary in these leagues)."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT id, category_id, start_ts, p1_id, p2_id, winner FROM matches "
        "ORDER BY start_ts").fetchall()
    games = defaultdict(lambda: [0, 0])
    for mid, g1, g2 in conn.execute(
            "SELECT match_id, SUM(p1_pts > p2_pts), SUM(p2_pts > p1_pts) "
            "FROM games GROUP BY match_id"):
        games[mid] = [int(g1 or 0), int(g2 or 0)]
    conn.close()

    eng = FeatureEngine()
    X, y, ts_list = [], [], []
    for mid, cat, ts, p1, p2, winner in rows:
        f_fwd = eng.snapshot(cat, p1, p2, ts)
        f_rev = eng.snapshot(cat, p2, p1, ts)
        X.append(f_fwd); y.append(1 if winner == 1 else 0); ts_list.append(ts)
        X.append(f_rev); y.append(0 if winner == 1 else 1); ts_list.append(ts)
        g1, g2 = games[mid]
        eng.update(cat, p1, p2, ts, winner, g1, g2)
    return X, y, ts_list, eng


def engine_from_history(db_path: Path = DB_PATH) -> FeatureEngine:
    """Replay all history to get the CURRENT state (for live prediction)."""
    _, _, _, eng = build_dataset(db_path)
    return eng
