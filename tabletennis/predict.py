"""Predict today's table-tennis fixtures → site/predictions.json.

The plan's "cheapest architecture": no live backend — a cron-able script writes
a static JSON the dashboard fetches. Rebuilds the feature state by replaying
history, scores every not-yet-finished fixture, and emits win probabilities +
the feature context (Elo diff, form, H2H) the dashboard displays.

    python -m tabletennis.predict
"""

from __future__ import annotations

import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

from tabletennis.features import engine_from_history
from tabletennis.ingest import upcoming

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.pkl"
OUT_PATH = HERE / "site" / "predictions.json"


def run() -> dict:
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    print("replaying history for current player state…")
    eng = engine_from_history()
    fixtures = upcoming()
    print(f"{len(fixtures)} upcoming fixtures")

    preds = []
    for fx in fixtures:
        cat, p1, p2, ts = fx["category_id"], fx["p1_id"], fx["p2_id"], fx["start_ts"]
        f_fwd = eng.snapshot(cat, p1, p2, ts)
        f_rev = eng.snapshot(cat, p2, p1, ts)
        # average the two orientations — enforces symmetry at inference too
        p_fwd = float(model.predict_proba([f_fwd])[0][1])
        p_rev = float(model.predict_proba([f_rev])[0][1])
        p1_win = (p_fwd + (1.0 - p_rev)) / 2.0
        a = eng.players[(cat, p1)]
        b = eng.players[(cat, p2)]
        known = a.n_matches + b.n_matches
        preds.append({
            **{k: fx[k] for k in ("event_id", "category", "tournament", "start_ts", "status", "p1", "p2")},
            "p1_win": round(p1_win, 4),
            "p2_win": round(1 - p1_win, 4),
            "elo_p1": round(a.elo, 1), "elo_p2": round(b.elo, 1),
            "matches_known": known,
            "confidence": "high" if known >= 30 else ("medium" if known >= 10 else "low"),
        })
    preds.sort(key=lambda r: (r["start_ts"], -max(r["p1_win"], r["p2_win"])))

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_ts": int(time.time()),
        "model": bundle.get("name", "?"),
        "n": len(preds),
        "predictions": preds,
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_PATH.relative_to(HERE.parent)} ({len(preds)} predictions)")
    return out


if __name__ == "__main__":
    run()
