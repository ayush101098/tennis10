"""Table-tennis data ingestion — Sofascore via the local sofa proxy.

Replaces the project plan's Flashscore/Playwright scraping layer entirely: the
repo's `sofa_proxy.py` (port 3001, Chrome-TLS impersonation) already serves
table tennis. One request per category-day returns EVERY event in that league
that day (Czech Liga Pro alone is ~600 finished matches/day), so a two-week
backfill is ~70 polite requests, not thousands of page scrapes.

Schema (SQLite, per the plan, games kept granular):
    players(id, name, category_id, first_seen)
    matches(id, category_id, tournament, start_ts, best_of,
            p1_id, p2_id, winner)          -- winner: 1|2
    games(match_id, game_no, p1_pts, p2_pts)

    python -m tabletennis.ingest --days 14
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

PROXY = "http://127.0.0.1:3001"
DB_PATH = Path(__file__).resolve().parent / "tabletennis.db"

# High-volume pro/semi-pro TT leagues (Sofascore category ids), per the plan.
CATEGORIES = {
    1369: "Czech Liga Pro",
    1713: "Setka Cup (UKR)",
    1340: "TT Cup (POL)",
    1554: "Liga Pro (RUS)",
    88: "International (WTT)",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS players(
    id INTEGER PRIMARY KEY, name TEXT, category_id INTEGER, first_seen INTEGER);
CREATE TABLE IF NOT EXISTS matches(
    id INTEGER PRIMARY KEY, category_id INTEGER, tournament TEXT,
    start_ts INTEGER, best_of INTEGER, p1_id INTEGER, p2_id INTEGER,
    winner INTEGER);
CREATE TABLE IF NOT EXISTS games(
    match_id INTEGER, game_no INTEGER, p1_pts INTEGER, p2_pts INTEGER,
    PRIMARY KEY(match_id, game_no));
CREATE INDEX IF NOT EXISTS idx_matches_ts ON matches(start_ts);
CREATE INDEX IF NOT EXISTS idx_matches_p1 ON matches(p1_id);
CREATE INDEX IF NOT EXISTS idx_matches_p2 ON matches(p2_id);
"""


def _get(path: str, timeout: int = 25) -> dict:
    req = urllib.request.Request(f"{PROXY}/{path}",
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    return conn


def _store_event(conn: sqlite3.Connection, ev: dict, cat_id: int) -> bool:
    """Insert one FINISHED singles event. Returns True if stored."""
    st = (ev.get("status") or {}).get("type")
    if st != "finished":
        return False
    winner = ev.get("winnerCode")
    if winner not in (1, 2):
        return False
    home, away = ev.get("homeTeam") or {}, ev.get("awayTeam") or {}
    # doubles pairs come through with "/" in the name — singles only
    if not home.get("id") or not away.get("id") or "/" in home.get("name", "") or "/" in away.get("name", ""):
        return False
    hs, as_ = ev.get("homeScore") or {}, ev.get("awayScore") or {}
    games = []
    for n in range(1, 8):
        a, b = hs.get(f"period{n}"), as_.get(f"period{n}")
        if a is None or b is None:
            break
        games.append((n, int(a), int(b)))
    if not games:
        return False
    best_of = 2 * max(int(hs.get("current") or 0), int(as_.get("current") or 0)) - 1
    ts = int(ev.get("startTimestamp") or 0)
    for pid, name in ((home["id"], home.get("name", "")), (away["id"], away.get("name", ""))):
        conn.execute("INSERT OR IGNORE INTO players VALUES(?,?,?,?)",
                     (pid, name, cat_id, ts))
    conn.execute("INSERT OR REPLACE INTO matches VALUES(?,?,?,?,?,?,?,?)",
                 (ev["id"], cat_id, (ev.get("tournament") or {}).get("name", ""),
                  ts, max(best_of, len(games)), home["id"], away["id"], winner))
    for n, a, b in games:
        conn.execute("INSERT OR REPLACE INTO games VALUES(?,?,?,?)",
                     (ev["id"], n, a, b))
    return True


def backfill(days: int = 14, db_path: Path = DB_PATH) -> None:
    conn = connect(db_path)
    total = 0
    for d in range(1, days + 1):
        day = (date.today() - timedelta(days=d)).isoformat()
        for cat_id, cat_name in CATEGORIES.items():
            try:
                data = _get(f"category/{cat_id}/scheduled-events/{day}")
            except Exception as e:
                print(f"  {day} {cat_name}: fetch failed ({e})")
                continue
            n = sum(_store_event(conn, ev, cat_id) for ev in data.get("events", []))
            total += n
            if n:
                print(f"  {day} {cat_name}: +{n}")
            time.sleep(0.3)  # be polite to the proxy/upstream
        conn.commit()
    n_m = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    n_p = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    print(f"\ningested {total} new · DB now {n_m} matches, {n_p} players")
    conn.close()


def upcoming(for_date: str | None = None) -> list[dict]:
    """Today's not-yet-finished singles fixtures across all categories."""
    day = for_date or date.today().isoformat()
    out = []
    for cat_id, cat_name in CATEGORIES.items():
        try:
            data = _get(f"category/{cat_id}/scheduled-events/{day}")
        except Exception:
            continue
        for ev in data.get("events", []):
            st = (ev.get("status") or {}).get("type")
            if st == "finished":
                continue
            home, away = ev.get("homeTeam") or {}, ev.get("awayTeam") or {}
            if not home.get("id") or not away.get("id") or "/" in home.get("name", "") or "/" in away.get("name", ""):
                continue
            out.append({
                "event_id": ev["id"], "category_id": cat_id, "category": cat_name,
                "tournament": (ev.get("tournament") or {}).get("name", ""),
                "start_ts": int(ev.get("startTimestamp") or 0),
                "status": st,
                "p1_id": home["id"], "p1": home.get("name", ""),
                "p2_id": away["id"], "p2": away.get("name", ""),
            })
        time.sleep(0.3)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backfill TT history from Sofascore")
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()
    backfill(days=args.days)
