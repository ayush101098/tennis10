"""
Point-level corpus: persist every observed point so a sequence model can exist later.

WHY THIS IS PHASE 0
  The architecture's whole left column — the observation vector, the point
  probability model, the HMM's latent momentum states, the Transformer over
  point tokens — consumes a *sequence of points*. None of it can be trained
  without a point corpus, and there is currently no way to obtain one:

    · SofaScore is the only true point-by-point source and it challenges this IP
      on /api/v1/*, so the feed is dark.
    · Flashscore, the working fallback, publishes NO point-by-point feed at all
      (df_pbp_1 returns empty on every live match, verified across a full slate).

  So the corpus cannot be back-filled. It can only be accumulated forward, which
  means the cost of not starting today is paid in months. This module is the
  cheapest possible version of starting today: poll the live score, diff it, and
  write down every point transition we see.

WHAT IS AND IS NOT RECOVERABLE
  Honest limits, recorded per-row so nothing downstream mistakes a gap for data:

    · Points are RECONSTRUCTED from polling, not read from a feed. Points before
      the poller attaches to a match are lost, and two points inside one polling
      interval collapse into one observation.
    · `winner` is inferred from how the score moved and can be NULL — a deuce
      reset (40-A -> 40-40) says someone scored but not who.
    · `server` is NULL. Flashscore renders it as an icon, not a field. It is left
      unknown rather than guessed, because a wrong server silently corrupts every
      serve-conditioned statistic built on top.
    · `gap` marks a row where the previous observation was too old to trust as
      adjacent, so a sequence model can break the sequence there instead of
      learning across a hole.

  Deliberately not stored: serve speed, first/second serve, rally length,
  winner/error. The design's state vector wants them; this source does not carry
  them. Empty columns invite someone to train on zeros and call it a feature.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = Path(os.getenv("POINTSTORE_DB", REPO_ROOT / "tennis_points.db"))

# Two observations further apart than this cannot be assumed adjacent, so the
# row is flagged rather than silently joined to the previous one.
MAX_ADJACENT_GAP_S = float(os.getenv("POINTSTORE_MAX_GAP", "45"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    fs_id        TEXT PRIMARY KEY,
    tour         TEXT,
    tournament   TEXT,
    surface      TEXT,
    p1           TEXT,
    p2           TEXT,
    start_ts     INTEGER,
    first_seen   REAL,
    last_seen    REAL,
    last_status  TEXT,
    final_sets_p1 INTEGER,
    final_sets_p2 INTEGER,
    n_points     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS points (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fs_id      TEXT NOT NULL,
    ts         REAL NOT NULL,
    set_index  INTEGER,
    games_p1   INTEGER,
    games_p2   INTEGER,
    sets_p1    INTEGER,
    sets_p2    INTEGER,
    point_p1   TEXT,
    point_p2   TEXT,
    winner     INTEGER,      -- 1 = p1 took the point, 2 = p2, NULL = ambiguous
    server     INTEGER,      -- always NULL from Flashscore; see module docstring
    gap        INTEGER DEFAULT 0,   -- 1 = preceding observation too old to chain
    source     TEXT
    -- NO uniqueness on game state, deliberately. A game can pass through 40-40
    -- repeatedly (deuce, advantage, back to deuce) and each pass is a distinct
    -- point; a UNIQUE(fs_id, set, games, points) would silently discard every
    -- repeat and quietly shorten exactly the longest, most informative games.
    -- Duplicate suppression belongs upstream instead: PointStream.observe()
    -- emits only when the observed state actually changed, so a quiet poll
    -- never reaches this table.
);

CREATE INDEX IF NOT EXISTS idx_points_match ON points(fs_id, id);
CREATE INDEX IF NOT EXISTS idx_points_ts    ON points(ts);
CREATE INDEX IF NOT EXISTS idx_matches_seen ON matches(last_seen);
"""


def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    # WAL so the collector can write while a notebook reads the corpus.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


class PointStore:
    """Writes observed points.

    Every call to `record_point` inserts. De-duplication is upstream's job —
    `PointStream.observe()` returns None when the polled state has not moved —
    because only the caller can tell a quiet poll apart from a genuine return to
    a state the game has already visited (deuce).
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = str(db_path)
        self.conn = connect(db_path)
        self._last_ts: dict[str, float] = {}

    # ── matches ──────────────────────────────────────────────────────────────

    def upsert_match(self, m) -> None:
        """Record/refresh a match. `m` is an execution.flashscore.Match."""
        now = time.time()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO matches (fs_id, tour, tournament, surface, p1, p2,
                                     start_ts, first_seen, last_seen, last_status,
                                     final_sets_p1, final_sets_p2)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(fs_id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    last_status = excluded.last_status,
                    final_sets_p1 = excluded.final_sets_p1,
                    final_sets_p2 = excluded.final_sets_p2
                """,
                (m.fs_id, m.tour, m.tournament, m.surface, m.home, m.away,
                 m.start_ts, now, now, m.status, m.home_sets, m.away_sets),
            )

    # ── points ───────────────────────────────────────────────────────────────

    def record_point(self, m, point, source: str = "flashscore") -> bool:
        """Persist one reconstructed point. Returns True if it was new."""
        now = getattr(point, "ts", None) or time.time()
        prev = self._last_ts.get(m.fs_id)
        gap = 1 if (prev is not None and now - prev > MAX_ADJACENT_GAP_S) else 0
        self._last_ts[m.fs_id] = now

        try:
            with self.conn:
                cur = self.conn.execute(
                    """
                    INSERT INTO points
                        (fs_id, ts, set_index, games_p1, games_p2, sets_p1, sets_p2,
                         point_p1, point_p2, winner, server, gap, source)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (m.fs_id, now, point.set_index, point.home_games, point.away_games,
                     m.home_sets, m.away_sets, point.point_home, point.point_away,
                     point.winner, m.server, gap, source),
                )
                if cur.rowcount:
                    self.conn.execute(
                        "UPDATE matches SET n_points = n_points + 1, last_seen = ? "
                        "WHERE fs_id = ?", (now, m.fs_id))
                    return True
        except sqlite3.Error as e:
            print(f"[pointstore] write failed for {m.fs_id}: {str(e)[:90]}")
        return False

    # ── reads ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        q = self.conn.execute
        row = q("""SELECT
                     (SELECT COUNT(*) FROM matches)                       AS matches,
                     (SELECT COUNT(*) FROM points)                        AS points,
                     (SELECT COUNT(*) FROM points WHERE winner IS NULL)   AS ambiguous,
                     (SELECT COUNT(*) FROM points WHERE gap = 1)          AS gaps,
                     (SELECT COUNT(*) FROM matches WHERE n_points >= 20)  AS usable
                """).fetchone()
        out = dict(row)
        span = q("SELECT MIN(ts), MAX(ts) FROM points").fetchone()
        out["first_point"] = span[0]
        out["last_point"] = span[1]
        by_tour = q("""SELECT tour, COUNT(*) n FROM matches
                       WHERE n_points > 0 GROUP BY tour ORDER BY n DESC""").fetchall()
        out["by_tour"] = {r["tour"]: r["n"] for r in by_tour}
        return out

    def sequence(self, fs_id: str) -> list[sqlite3.Row]:
        """Every point for one match, in observation order."""
        return self.conn.execute(
            "SELECT * FROM points WHERE fs_id = ? ORDER BY id", (fs_id,)).fetchall()

    def close(self) -> None:
        self.conn.close()


# ─── Collector ───────────────────────────────────────────────────────────────

def collect(poll_s: float = 8.0, db_path: Path | str = DEFAULT_DB,
            iterations: Optional[int] = None, verbose: bool = True) -> None:
    """Run the reconstructed point stream and persist everything it sees."""
    from execution.flashscore import FlashscoreClient, PointStream

    store = PointStore(db_path)
    client = FlashscoreClient()
    stream = PointStream(client, poll_s=poll_s)

    if verbose:
        print(f"[pointstore] collecting -> {store.db_path}")
        print(f"[pointstore] polling every {poll_s}s — Ctrl-C to stop")

    n = 0
    written = 0
    try:
        while iterations is None or n < iterations:
            try:
                # live_matches() drives the diff; upsert first so a match exists
                # in the corpus even before it yields its first point.
                for m in client.live_matches(with_points=True):
                    store.upsert_match(m)
                    p = stream.observe(m)
                    if p and store.record_point(m, p):
                        written += 1
                        if verbose:
                            who = {1: m.home, 2: m.away, None: "?"}[p.winner]
                            print(f"  {m.tour:<10} {m.home[:15]:<15} v {m.away[:15]:<15} "
                                  f"s{p.set_index} {p.home_games}-{p.away_games} "
                                  f"[{p.point_home}-{p.point_away}] -> {who}")
            except Exception as e:
                print(f"[pointstore] poll error: {str(e)[:110]}")

            n += 1
            if iterations is None or n < iterations:
                time.sleep(poll_s)
    except KeyboardInterrupt:
        pass
    finally:
        s = store.stats()
        print(f"\n[pointstore] {written} point(s) written this run")
        print(f"[pointstore] corpus: {s['points']:,} points across "
              f"{s['matches']:,} matches ({s['usable']:,} with 20+ points)")
        store.close()


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Point-level corpus collector")
    ap.add_argument("--collect", action="store_true", help="run the collector")
    ap.add_argument("--stats", action="store_true", help="show corpus stats")
    ap.add_argument("--match", metavar="FS_ID", help="dump one match's sequence")
    ap.add_argument("--interval", type=float, default=8.0)
    ap.add_argument("--polls", type=int, default=None, help="stop after N polls")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    args = ap.parse_args()

    if args.collect:
        collect(poll_s=args.interval, db_path=args.db, iterations=args.polls)
        return

    store = PointStore(args.db)
    if args.match:
        rows = store.sequence(args.match)
        print(f"{len(rows)} point(s) for {args.match}\n")
        for r in rows:
            flag = " GAP" if r["gap"] else ""
            print(f"  s{r['set_index']} {r['games_p1']}-{r['games_p2']} "
                  f"[{r['point_p1']}-{r['point_p2']}] winner={r['winner']}{flag}")
    else:
        s = store.stats()
        print(f"corpus      : {args.db}")
        print(f"matches     : {s['matches']:,}  ({s['usable']:,} with 20+ points)")
        print(f"points      : {s['points']:,}")
        if s["points"]:
            amb = 100.0 * s["ambiguous"] / s["points"]
            gap = 100.0 * s["gaps"] / s["points"]
            print(f"  ambiguous : {s['ambiguous']:,} ({amb:.1f}%) — winner unrecoverable")
            print(f"  gapped    : {s['gaps']:,} ({gap:.1f}%) — sequence break before row")
            print(f"  span      : {time.strftime('%Y-%m-%d %H:%M', time.localtime(s['first_point']))}"
                  f" -> {time.strftime('%Y-%m-%d %H:%M', time.localtime(s['last_point']))}")
        if s["by_tour"]:
            print(f"by tour     : {s['by_tour']}")
    store.close()


if __name__ == "__main__":
    _main()
