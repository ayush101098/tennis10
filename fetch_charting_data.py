#!/usr/bin/env python3
"""
Tier 1 rally intelligence — ingest Jeff Sackmann's Match Charting Project.

Source:  https://github.com/JeffSackmann/tennis_MatchChartingProject
Unlike the ATP/WTA match-summary CSVs (which give scoreboard-level serve/return
stats), the MCP is shot-by-shot: it exposes rally-length breakdowns and
winner/unforced counts. This is the raw material for rally-construction features
(first-strike dependency, grinding ability, shot quality) that move ahead of the
scoreboard.

We aggregate three MCP files per tour (m = ATP, w = WTA):
  - charting-{m,w}-matches.csv         match metadata (players, date, surface)
  - charting-{m,w}-stats-Rally.csv     per-match points won by rally-length bucket
  - charting-{m,w}-stats-Overview.csv  per-match winners / unforced / serve pts

Output: a `rally_stats` table in tennis_data.db, one row per (match, player),
holding raw point counts. Rates are computed downstream in execution/rally.py so
that time-decay and strict pre-match temporal isolation stay in the feature layer
(mirroring features.py). Storing raw counts also lets us pool small samples
correctly (sum numerators/denominators rather than averaging noisy per-match rates).
"""

import sqlite3
import ssl
import unicodedata
import urllib.request
from io import StringIO

import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
DB_PATH = "tennis_data.db"


def norm_name(name: str) -> str:
    """Normalize a player name for cross-source joining.

    MCP uses 'First Last' with accents; our players table is mostly ASCII.
    Lowercase, strip accents, drop punctuation, collapse whitespace so
    'Stan Wawrinka' == 'stan wawrinka' == 'Stanislas  Wawrinka'? (no — see note).

    NOTE: this does NOT reconcile nickname/initial differences (e.g. 'Stan' vs
    'Stanislas'); it only canonicalizes case/accents/spacing. Genuine alias
    mismatches surface as low join-rate in the coverage report, which is the
    honest signal to add an alias map later.
    """
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().replace(".", " ").replace("-", " ").replace("'", " ")
    return " ".join(n.split())


def fetch_csv(fname: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{fname}"
    print(f"  fetching {fname} ...", end=" ", flush=True)
    data = urllib.request.urlopen(url).read().decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(data), low_memory=False)
    print(f"{len(df):,} rows")
    return df


def parse_date(v) -> str | None:
    """MCP Date is a YYYYMMDD integer -> 'YYYY-MM-DD'."""
    try:
        s = str(int(v))
        if len(s) == 8:
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    except (ValueError, TypeError):
        pass
    return None


def build_tour(tour_code: str) -> pd.DataFrame:
    """Return one row per (match_id, player) with raw rally + overview counts."""
    tour = "ATP" if tour_code == "m" else "WTA"
    print(f"\n[{tour}] loading MCP files")
    matches = fetch_csv(f"charting-{tour_code}-matches.csv")
    rally = fetch_csv(f"charting-{tour_code}-stats-Rally.csv")
    overview = fetch_csv(f"charting-{tour_code}-stats-Overview.csv")

    # match_id -> (player1, player2, date, surface)
    meta = {}
    for _, r in matches.iterrows():
        meta[r["match_id"]] = {
            "p1": r["Player 1"],
            "p2": r["Player 2"],
            "date": parse_date(r["Date"]),
            "surface": r.get("Surface"),
        }

    # --- Rally: plain buckets only (combined servers). Suffixed rows (-1/-2)
    #     split by server and would double-count, so we exclude them. ---
    BUCKETS = {"1-3": "short", "4-6": "mid", "7-9": "long", "10": "long"}
    # accumulator keyed by (match_id, player_name)
    acc: dict[tuple, dict] = {}

    def slot(mid, pname, date, surface):
        key = (mid, pname)
        if key not in acc:
            acc[key] = {
                "match_id": mid, "player": pname, "date": date, "surface": surface,
                "short_won": 0, "short_pts": 0, "mid_won": 0, "mid_pts": 0,
                "long_won": 0, "long_pts": 0, "total_won": 0, "total_pts": 0,
                "winners": 0, "unforced": 0, "serve_pts": 0, "return_pts": 0,
            }
        return acc[key]

    rally_matched = 0
    for _, r in rally.iterrows():
        mid = r["match_id"]
        m = meta.get(mid)
        if not m:
            continue
        row = str(r["row"])
        if row == "Total":
            bucket = "total"
        elif row in BUCKETS:
            bucket = BUCKETS[row]
        else:
            continue  # skip -1/-2 server splits
        pts = int(r["pts"]) if pd.notna(r["pts"]) else 0
        p1w = int(r["pl1_won"]) if pd.notna(r["pl1_won"]) else 0
        p2w = int(r["pl2_won"]) if pd.notna(r["pl2_won"]) else 0
        a1 = slot(mid, m["p1"], m["date"], m["surface"])
        a2 = slot(mid, m["p2"], m["date"], m["surface"])
        a1[f"{bucket}_won"] += p1w
        a1[f"{bucket}_pts"] += pts
        a2[f"{bucket}_won"] += p2w
        a2[f"{bucket}_pts"] += pts
        rally_matched += 1

    # --- Overview: Total row per (match, player) for winners/unforced/serve pts ---
    for _, r in overview.iterrows():
        if str(r.get("set")) != "Total":
            continue
        mid = r["match_id"]
        m = meta.get(mid)
        if not m:
            continue
        pname = r["player"]
        # only attach if this player is a known side of the match
        if pname not in (m["p1"], m["p2"]):
            continue
        a = slot(mid, pname, m["date"], m["surface"])
        a["winners"] += int(r["winners"]) if pd.notna(r.get("winners")) else 0
        a["unforced"] += int(r["unforced"]) if pd.notna(r.get("unforced")) else 0
        a["serve_pts"] += int(r["serve_pts"]) if pd.notna(r.get("serve_pts")) else 0
        a["return_pts"] += int(r["return_pts"]) if pd.notna(r.get("return_pts")) else 0

    df = pd.DataFrame(acc.values())
    df["tour"] = tour
    df["player_key"] = df["player"].map(norm_name)
    df = df[df["date"].notna() & (df["player_key"] != "")]
    print(f"[{tour}] {len(df):,} (match,player) rows  |  {rally_matched:,} rally rows matched to metadata")
    return df


def main():
    frames = [build_tour("m"), build_tour("w")]
    allrows = pd.concat(frames, ignore_index=True)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS rally_stats")
    cur.execute("""
        CREATE TABLE rally_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mcp_match_id TEXT,
            player_name  TEXT,
            player_key   TEXT,
            match_date   DATE,
            surface      TEXT,
            tour         TEXT,
            short_won INTEGER, short_pts INTEGER,
            mid_won   INTEGER, mid_pts   INTEGER,
            long_won  INTEGER, long_pts  INTEGER,
            total_won INTEGER, total_pts INTEGER,
            winners   INTEGER, unforced  INTEGER,
            serve_pts INTEGER, return_pts INTEGER
        )
    """)
    cur.execute("CREATE INDEX idx_rally_key ON rally_stats(player_key, match_date)")

    cols = ["mcp_match_id", "player_name", "player_key", "match_date", "surface", "tour",
            "short_won", "short_pts", "mid_won", "mid_pts", "long_won", "long_pts",
            "total_won", "total_pts", "winners", "unforced", "serve_pts", "return_pts"]
    for _, r in allrows.iterrows():
        cur.execute(
            f"INSERT INTO rally_stats ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
            (r["match_id"], r["player"], r["player_key"], r["date"], r["surface"], r["tour"],
             r["short_won"], r["short_pts"], r["mid_won"], r["mid_pts"],
             r["long_won"], r["long_pts"], r["total_won"], r["total_pts"],
             r["winners"], r["unforced"], r["serve_pts"], r["return_pts"]))
    con.commit()

    # --- coverage report ---
    print("\n" + "=" * 60)
    print("RALLY_STATS COVERAGE")
    print("=" * 60)
    total = cur.execute("SELECT COUNT(*) FROM rally_stats").fetchone()[0]
    players = cur.execute("SELECT COUNT(DISTINCT player_key) FROM rally_stats").fetchone()[0]
    dmin, dmax = cur.execute("SELECT MIN(match_date), MAX(match_date) FROM rally_stats").fetchone()
    print(f"rows: {total:,}   distinct players: {players:,}   dates: {dmin} .. {dmax}")
    for tour in ("ATP", "WTA"):
        n = cur.execute("SELECT COUNT(*) FROM rally_stats WHERE tour=?", (tour,)).fetchone()[0]
        print(f"  {tour}: {n:,} (match,player) rows")

    # join-rate against the model DB's players table (normalized name match)
    dbnames = cur.execute("SELECT player_name FROM players").fetchall()
    db_keys = {norm_name(n[0]) for n in dbnames}
    rk = cur.execute("SELECT DISTINCT player_key FROM rally_stats WHERE tour='ATP'").fetchall()
    rally_keys = {r[0] for r in rk}
    matched = len(rally_keys & db_keys)
    print(f"\nATP name join-rate vs players table: "
          f"{matched}/{len(rally_keys)} = {matched/max(len(rally_keys),1):.1%}")
    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
