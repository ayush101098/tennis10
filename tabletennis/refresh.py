"""
TT pre-match refresher — keeps site/predictions.json from going stale.

WHY THIS EXISTS
  The 8-second live poller (tabletennis.live) runs continuously, so
  live_predictions.json is always fresh. But the PRE-MATCH fixture list is only
  written when ingest+predict are run by hand, and nothing was running them.
  Result: predictions.json silently aged out (measured 57h old) while the live
  feed looked healthy — the match centre filled with dead fixtures and appeared
  broken, with no obvious cause. The terminal's ">12h old" banner flagged it,
  but nothing fixed it.

  This daemon closes that loop: ingest recent days, re-predict, sleep, repeat.
  Retrain is optional and off by default (the model moves far more slowly than
  the fixture list, and training is the expensive step).

    python -m tabletennis.refresh                 # loop every 3h, no retrain
    python -m tabletennis.refresh --hours 6       # slower cadence
    python -m tabletennis.refresh --retrain       # retrain each cycle
    python -m tabletennis.refresh --once          # single refresh, then exit

  Run it alongside the live poller and the sofa proxy:
    python sofa_proxy.py &
    python -m tabletennis.live &
    python -m tabletennis.refresh &
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE = Path(__file__).resolve().parent / "site"


def _run(args: list[str]) -> tuple[bool, str]:
    """Run a pipeline step; never raise — a failed cycle must not kill the loop."""
    try:
        p = subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=1800)
        tail = (p.stdout or p.stderr or "").strip().splitlines()
        return p.returncode == 0, (tail[-1] if tail else "")
    except Exception as e:
        return False, str(e)[:200]


def file_age_s(name: str) -> float | None:
    try:
        d = json.loads((SITE / name).read_text())
        return time.time() - float(d["generated_ts"])
    except Exception:
        return None


def refresh_once(days: int, retrain: bool) -> None:
    stamp = time.strftime("%H:%M:%S")
    ok, msg = _run(["tabletennis.ingest", "--days", str(days)])
    print(f"[{stamp}] ingest  {'ok ' if ok else 'FAIL'} {msg}", flush=True)
    if retrain:
        ok, msg = _run(["tabletennis.train"])
        print(f"[{stamp}] train   {'ok ' if ok else 'FAIL'} {msg}", flush=True)
    ok, msg = _run(["tabletennis.predict"])
    print(f"[{stamp}] predict {'ok ' if ok else 'FAIL'} {msg}", flush=True)

    pre, live = file_age_s("predictions.json"), file_age_s("live_predictions.json")
    def age(x):
        return "missing" if x is None else f"{x / 3600:.1f}h" if x > 3600 else f"{int(x)}s"
    warn = ""
    if live is not None and live > 300:
        warn = "   ⚠ live poller looks dead — start `python -m tabletennis.live`"
    print(f"[{stamp}] ages: pre-match {age(pre)} · live {age(live)}{warn}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Keep TT pre-match predictions fresh")
    ap.add_argument("--hours", type=float, default=3.0, help="cycle interval (default 3h)")
    ap.add_argument("--days", type=int, default=2, help="days of history to re-ingest per cycle")
    ap.add_argument("--retrain", action="store_true", help="retrain the model every cycle")
    ap.add_argument("--once", action="store_true", help="one refresh then exit")
    args = ap.parse_args()

    print("=" * 64)
    print(f"TT REFRESHER — every {args.hours}h · ingest {args.days}d · "
          f"retrain={'yes' if args.retrain else 'no'}")
    print("=" * 64, flush=True)
    try:
        while True:
            t0 = time.time()
            refresh_once(args.days, args.retrain)
            if args.once:
                break
            time.sleep(max(60.0, args.hours * 3600 - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
