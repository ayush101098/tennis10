"""One-command pipeline: ingest → train → predict (→ serve).

    python -m tabletennis.pipeline                # refresh yesterday + predict today
    python -m tabletennis.pipeline --days 45      # deeper backfill first
    python -m tabletennis.pipeline --serve        # also serve site/ on :3222

Cron this (GitHub Actions or launchd) every 15 min during play: step 1 pulls the
latest finished results, step 2 retrains weekly (cheap enough to do every run for
now), step 3 rewrites site/predictions.json — the dashboard is static after that.
"""

from __future__ import annotations

import argparse

from tabletennis import ingest, predict, train


def main() -> None:
    ap = argparse.ArgumentParser(description="TT predictor end-to-end pipeline")
    ap.add_argument("--days", type=int, default=1, help="days of history to (re)ingest")
    ap.add_argument("--skip-train", action="store_true", help="reuse the existing model.pkl")
    ap.add_argument("--serve", action="store_true", help="serve site/ on http://localhost:3222")
    args = ap.parse_args()

    print(f"[1/3] ingest — last {args.days} day(s)")
    ingest.backfill(days=args.days)
    if not args.skip_train:
        print("[2/3] train")
        train.train()
    else:
        print("[2/3] train — skipped")
    print("[3/3] predict")
    predict.run()

    if args.serve:
        import functools
        import http.server
        from pathlib import Path
        site = Path(__file__).resolve().parent / "site"
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
        print("serving http://localhost:3222 (ctrl-c to stop)")
        http.server.ThreadingHTTPServer(("", 3222), handler).serve_forever()


if __name__ == "__main__":
    main()
