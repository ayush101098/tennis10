"""Fit a calibration correction that shrinks over-confident probabilities.

The report showed our `true_p` is over-confident (predicts 80%+, wins ~60%). A
shrinkage calibrator pulls probabilities toward 0.5:

    p_cal = 0.5 + k * (p_raw - 0.5)        # k in [0,1]; k<1 = less confident

`k` is fit on settled bets to minimize the Brier score. Fitting and evaluating
on the SAME bets is circular, so `fit_and_save` also does a time-based
train/test split and a **re-selection backtest** (which bets would still clear
the 2% edge gate under calibrated probs) so you see the honest, out-of-sample
effect on ROI — not an in-sample mirage.

    python -m execution.agent calibrate     # fit, save, and show the backtest

⚠ Calibration corrects confidence, not skill. If the signal has no edge, the
best a calibrator can do is stop you over-betting — ROI moves toward 0, not
necessarily positive.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from execution import trade_log  # noqa: E402

CALIB_PATH = REPO_ROOT / "calibration.json"
EDGE_GATE = 0.02   # same min edge the pipeline uses


def apply(p: float, params: dict | None = None) -> float:
    params = params if params is not None else load()
    if not params:
        return p
    k = params.get("k", 1.0)
    return 0.5 + k * (p - 0.5)


def load() -> dict | None:
    if CALIB_PATH.exists():
        try:
            return json.loads(CALIB_PATH.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _settled():
    rows = []
    for r in trade_log.all_trades():
        if r["status"] not in ("settled_win", "settled_loss"):
            continue
        tp, price, stake = r.get("true_p"), r.get("market_price"), r.get("stake_usd")
        shares = r.get("shares")
        if tp is None or price is None or not stake:
            continue
        rows.append({"true_p": float(tp), "price": float(price),
                     "won": r["status"] == "settled_win", "stake": float(stake),
                     "shares": float(shares) if shares else float(stake) / float(price),
                     "id": r["trade_id"]})
    return sorted(rows, key=lambda r: r["id"])


def _brier(rows, k):
    if not rows:
        return 0.0
    return sum((apply(r["true_p"], {"k": k}) - (1.0 if r["won"] else 0.0)) ** 2
               for r in rows) / len(rows)


def _fit_k(rows):
    """Grid-search k in [0,1] minimizing Brier."""
    best_k, best_b = 1.0, float("inf")
    for i in range(0, 101):
        k = i / 100.0
        b = _brier(rows, k)
        if b < best_b:
            best_b, best_k = b, k
    return best_k, best_b


def _reselect_roi(rows, k):
    """ROI over bets that STILL clear the edge gate under calibrated probs."""
    kept_pnl = kept_stake = kept = wins = 0.0
    for r in rows:
        cal = apply(r["true_p"], {"k": k})
        if cal - r["price"] < EDGE_GATE:      # wouldn't bet under calibration
            continue
        pnl = (r["shares"] - r["stake"]) if r["won"] else -r["stake"]
        kept_pnl += pnl
        kept_stake += r["stake"]
        kept += 1
        wins += int(r["won"])
    roi = (kept_pnl / kept_stake) if kept_stake else 0.0
    return {"kept": int(kept), "wins": int(wins), "pnl": kept_pnl,
            "stake": kept_stake, "roi": roi}


def fit_and_save() -> dict:
    rows = _settled()
    L = ["=" * 70, "  CALIBRATION FIT (shrinkage toward 0.5)", "=" * 70]
    if len(rows) < 20:
        L.append(f"\n  Only {len(rows)} settled bets — need ~20+ to fit. "
                 f"Run more paper + settle first.")
        print("\n".join(L))
        return {}

    # time-based split: older 70% train, newer 30% test
    cut = int(len(rows) * 0.7)
    train, test = rows[:cut], rows[cut:]
    k, _ = _fit_k(train)

    b_before_tr, b_after_tr = _brier(train, 1.0), _brier(train, k)
    b_before_te, b_after_te = _brier(test, 1.0), _brier(test, k)
    base_before = _reselect_roi(test, 1.0)     # raw selection on test
    base_after = _reselect_roi(test, k)         # calibrated selection on test

    L.append(f"\n  fit on {len(train)} train bets -> k = {k:.2f}  "
             f"({'shrinks' if k < 1 else 'no shrink'}; 0=all→50%, 1=unchanged)")
    L.append(f"\n  Brier (lower=better):")
    L.append(f"    train:  raw {b_before_tr:.4f}  ->  calibrated {b_after_tr:.4f}")
    L.append(f"    TEST :  raw {b_before_te:.4f}  ->  calibrated {b_after_te:.4f}   (out-of-sample)")
    L.append(f"\n  Re-selection backtest on {len(test)} held-out bets "
             f"(which bets still clear the {EDGE_GATE:.0%} edge gate):")
    L.append(f"    raw        : {base_before['kept']:>3} bets, "
             f"ROI {base_before['roi']:+.1%}  (PnL ${base_before['pnl']:+.2f})")
    L.append(f"    calibrated : {base_after['kept']:>3} bets, "
             f"ROI {base_after['roi']:+.1%}  (PnL ${base_after['pnl']:+.2f})")

    params = {"k": k, "n_train": len(train), "n_test": len(test),
              "brier_test_raw": round(b_before_te, 4),
              "brier_test_cal": round(b_after_te, 4),
              "test_roi_raw": round(base_before["roi"], 4),
              "test_roi_cal": round(base_after["roi"], 4)}
    CALIB_PATH.write_text(json.dumps(params, indent=2))
    L.append(f"\n  saved -> {CALIB_PATH.name}  (in-play engine will apply k={k:.2f})")
    verdict = ("calibration improves out-of-sample ROI"
               if base_after["roi"] > base_before["roi"]
               else "calibration does NOT create edge here (see warning)")
    L.append(f"  verdict: {verdict}")
    L.append("=" * 70)
    print("\n".join(L))
    return params


if __name__ == "__main__":
    fit_and_save()
