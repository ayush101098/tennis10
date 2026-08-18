"""Calibration report — does our predicted probability match reality?

Over settled bets in the journal, bins the predicted `true_p` (for the side we
backed) against the actual win rate, and breaks it down by signal source
(inplay / sxbet / model / market fallback). This is how we tell whether the
in-play engine is *sharp* (well-calibrated + profitable) or just *confident*.

    python -m execution.calibration            # full report
    python -m execution.calibration --source inplay

Needs settled bets — run `python -m execution.settle` as matches finish so
outcomes accrue. With few settled bets the numbers are noisy; treat as
directional until n is large per bucket.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from execution import trade_log  # noqa: E402

# Map a journal row to its signal source. Older rows without a tag -> "other".
_SOURCE_TAGS = ("inplay", "sxbet", "sofascore", "model", "market", "sx_auto")


def _source(row: dict) -> str:
    detail = (row.get("detail") or "").lower()
    side = (row.get("side") or "").lower()
    if side == "sx_auto" or row.get("venue") == "sxbet":
        return "sxbet"
    for tag in _SOURCE_TAGS:
        if tag in detail:
            return tag
    return "other"


def _rows(source=None):
    out = []
    for r in trade_log.all_trades():
        if r["status"] not in ("settled_win", "settled_loss"):
            continue
        tp, stake = r.get("true_p"), r.get("stake_usd")
        if tp is None or not stake:
            continue
        src = _source(r)
        if source and src != source:
            continue
        out.append({"true_p": float(tp), "won": r["status"] == "settled_win",
                    "pnl": float(r.get("pnl_usd") or 0), "stake": float(stake),
                    "source": src})
    return out


def _line(label, n, pred, actual, pnl, stake):
    roi = (pnl / stake) if stake else 0.0
    gap = pred - actual
    flag = "" if n < 10 else ("  ✓" if actual >= pred - 0.03 else "  ✗ over-confident")
    return (f"  {label:16} n={n:>4}  pred={pred:5.1%}  actual={actual:5.1%}  "
            f"gap={gap:+5.1%}  PnL=${pnl:+8.2f}  ROI={roi:+6.1%}{flag}")


def build_report(source=None) -> str:
    rows = _rows(source)
    L = ["=" * 74, "  CALIBRATION REPORT" + (f"  (source: {source})" if source else ""),
         "=" * 74]
    if not rows:
        L.append("\n  No settled bets yet. Run: python -m execution.settle")
        return "\n".join(L)

    n = len(rows)
    brier = sum((r["true_p"] - (1.0 if r["won"] else 0.0)) ** 2 for r in rows) / n
    pnl = sum(r["pnl"] for r in rows)
    stake = sum(r["stake"] for r in rows)
    wins = sum(r["won"] for r in rows)
    L.append(f"\n  settled bets: {n}  ({wins}W-{n - wins}L)   "
             f"Brier={brier:.4f}  (lower=better; 0.25=coinflip)")
    L.append(f"  realized PnL ${pnl:+.2f}  on ${stake:.2f} staked  (ROI {pnl / stake:+.1%})")

    # by probability bucket
    L.append("\n  by predicted-probability bucket:")
    buckets = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    for lo, hi in buckets:
        b = [r for r in rows if lo <= r["true_p"] < hi]
        if not b:
            continue
        bn = len(b)
        pred = sum(r["true_p"] for r in b) / bn
        actual = sum(r["won"] for r in b) / bn
        bp = sum(r["pnl"] for r in b)
        bs = sum(r["stake"] for r in b)
        L.append(_line(f"{lo:.0%}-{hi:.0%}", bn, pred, actual, bp, bs))

    # by source
    L.append("\n  by signal source:")
    bysrc = defaultdict(list)
    for r in rows:
        bysrc[r["source"]].append(r)
    for src, b in sorted(bysrc.items(), key=lambda kv: -len(kv[1])):
        bn = len(b)
        pred = sum(r["true_p"] for r in b) / bn
        actual = sum(r["won"] for r in b) / bn
        bp = sum(r["pnl"] for r in b)
        bs = sum(r["stake"] for r in b)
        L.append(_line(src, bn, pred, actual, bp, bs))

    L.append("\n" + "=" * 74)
    L.append("  ✓ = calibrated (actual ≈ predicted). ✗ = over-confident (bet less here).")
    L.append("  Trust a source/bucket only once n is large AND ROI is positive.")
    L.append("=" * 74)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration report over settled bets")
    ap.add_argument("--source", help="filter to one signal source (inplay/sxbet/model/...)")
    args = ap.parse_args()
    print(build_report(source=args.source))


if __name__ == "__main__":
    main()
