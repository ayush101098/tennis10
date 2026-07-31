"""§4 Live intel pipeline — poller, state machine, transition log, live feed.

Polls the Sofascore live endpoint (via the local sofa proxy) every POLL_S
seconds, diffs each match against its last known state, logs every transition
to SQLite (the future training set for point-level character features), and
regenerates site/live_predictions.json:

    analytic P_match_live  --> +bounded character residual --> merged live prob
                                                                + prob history
                                                                + annotation

Heavy state (feature engine replay + traits + models) is built ONCE at startup;
each poll cycle is then just diffs + cached table lookups.

    python -m tabletennis.live            # runs until ctrl-c
    python -m tabletennis.live --once     # single poll (testing)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from tabletennis.analytic import p_from_match_prob, p_match_live
from tabletennis.features import FEATURE_NAMES  # noqa: F401 (doc)
from tabletennis.ingest import CATEGORIES, DB_PATH, _get, connect
from tabletennis.residual import adjust, load as load_residual
from tabletennis.traits import TRAIT_NAMES, compute

HERE = Path(__file__).resolve().parent
OUT = HERE / "site" / "live_predictions.json"
POLL_S = 8

LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_states(
    event_id INTEGER, ts INTEGER, ga INTEGER, gb INTEGER,
    pa INTEGER, pb INTEGER, prob REAL, adj REAL,
    PRIMARY KEY(event_id, ts));
"""


class LiveEngine:
    def __init__(self):
        print("building state (feature replay + traits + models)…")
        self.traits, _ = compute()
        # rebuild the feature engine at current state (traits.compute consumed one)
        from tabletennis.features import engine_from_history
        self.eng = engine_from_history()
        with open(HERE / "model.pkl", "rb") as f:
            self.prematch = pickle.load(f)
        self.residual = load_residual()
        self.conn = connect(DB_PATH)
        self.conn.executescript(LOG_SCHEMA)
        self.last: dict[int, tuple] = {}       # event_id -> (ga, gb, pa, pb)
        print(f"ready — residual {'ACTIVE' if self.residual.get('model') else 'off (analytic only)'}")

    # ── pre-match anchor ──────────────────────────────────────────────────
    @lru_cache(maxsize=4096)
    def _pre(self, cat: int, p1: int, p2: int, day: int) -> float:
        f_fwd = self.eng.snapshot(cat, p1, p2, day * 86400)
        f_rev = self.eng.snapshot(cat, p2, p1, day * 86400)
        m = self.prematch["model"]
        return (float(m.predict_proba([f_fwd])[0][1])
                + 1.0 - float(m.predict_proba([f_rev])[0][1])) / 2.0

    def _traits(self, pid: int) -> list[float]:
        t = self.traits.get(pid, {})
        return [t.get(k, 0.0) for k in TRAIT_NAMES]

    # ── one poll cycle ────────────────────────────────────────────────────
    def poll(self) -> list[dict]:
        try:
            data = _get("sport/table-tennis/events/live")
        except Exception as e:
            print("poll failed:", e)
            return []
        now = int(time.time())
        out = []
        for ev in data.get("events", []):
            row = self._process(ev, now)
            if row:
                out.append(row)
        self.conn.commit()
        return out

    def _process(self, ev: dict, now: int):
        home, away = ev.get("homeTeam") or {}, ev.get("awayTeam") or {}
        cat = ((ev.get("tournament") or {}).get("category") or {}).get("id")
        if not home.get("id") or not away.get("id") or "/" in home.get("name", ""):
            return None
        hs, as_ = ev.get("homeScore") or {}, ev.get("awayScore") or {}
        ga, gb = int(hs.get("current") or 0), int(as_.get("current") or 0)
        lp = ev.get("lastPeriod") or "period1"
        pa, pb = int(hs.get(lp) or 0), int(as_.get(lp) or 0)
        best_of = 7 if ga + gb >= 5 and max(ga, gb) < 4 else 5
        eid = ev["id"]
        p1, p2 = home["id"], away["id"]

        # analytic baseline anchored to the pre-match model
        pre = self._pre(cat or 0, p1, p2, now // 86400)
        p_pt = p_from_match_prob(pre, best_of)
        base = p_match_live(ga, gb, pa, pb, p_pt, best_of)

        # character residual
        t1, t2 = self._traits(p1), self._traits(p2)
        feats = self.eng.snapshot(cat or 0, p1, p2, now)
        x = [base, pre, ga, gb, best_of, feats[0] / 100.0, *feats, *t1, *t2]
        prob, adj = adjust(base, x, self.residual)

        # log only on state change
        state = (ga, gb, pa, pb)
        if self.last.get(eid) != state:
            self.last[eid] = state
            self.conn.execute("INSERT OR REPLACE INTO live_states VALUES(?,?,?,?,?,?,?,?)",
                              (eid, now, ga, gb, pa, pb, round(prob, 4), round(adj, 4)))

        hist = self.conn.execute(
            "SELECT ts, prob FROM live_states WHERE event_id=? ORDER BY ts", (eid,)).fetchall()

        annotation = ""
        if abs(adj) >= 0.02:
            diffs = sorted(zip(TRAIT_NAMES, [a - b for a, b in zip(t1, t2)]),
                           key=lambda kv: -abs(kv[1]))
            top, d = diffs[0]
            who = home["name"] if (d > 0) == (adj > 0) else away["name"]
            annotation = f"character {adj:+.0%} — {who} {top} trait ({d:+.2f})"

        return {
            "event_id": eid, "category": CATEGORIES.get(cat, "Other"),
            "tournament": (ev.get("tournament") or {}).get("name", ""),
            "p1": home["name"], "p2": away["name"],
            "games": [ga, gb], "points": [pa, pb], "best_of": best_of,
            "pre_match_p1": round(pre, 4), "analytic_p1": round(base, 4),
            "residual": round(adj, 4), "p1_win": round(prob, 4),
            "annotation": annotation,
            "history": [[t, p] for t, p in hist[-120:]],
        }

    def write(self, rows: list[dict]) -> None:
        OUT.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generated_ts": int(time.time()), "n": len(rows), "matches": rows,
        }, indent=1))


def main():
    ap = argparse.ArgumentParser(description="TT live intelligence poller")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=POLL_S)
    args = ap.parse_args()
    le = LiveEngine()
    while True:
        t0 = time.time()
        rows = le.poll()
        le.write(rows)
        print(f"[{time.strftime('%H:%M:%S')}] {len(rows)} live · "
              + " · ".join(f"{r['p1'][:10]} {r['p1_win']:.0%}" for r in rows[:4]))
        if args.once:
            break
        time.sleep(max(1.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
