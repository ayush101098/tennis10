#!/usr/bin/env python3
"""
Base-engine backtest on settled bets (execution/trade_log).

IMPORTANT SCOPE: this measures the *base* true_p engine that produced the logged
bets — calibration, Brier, and realized ROI. It does NOT isolate the Tier 2
momentum signal: momentum wasn't recorded on past bets and the live
point-by-point needed to reconstruct it isn't archived. Momentum's marginal edge
can only be measured going forward, once instrumented (see trade_log momentum
fields). This backtest establishes the foundation momentum will sit on: if the
base true_p is miscalibrated, that's the first thing to fix.

Win/loss truth: status settled_win -> 1, settled_loss -> 0. true_p is P(the side
we bought wins); market_price is the price paid.
"""

import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent / "tennis_betting.db"


def load():
    con = sqlite3.connect(DB)
    rows = con.execute(
        """SELECT true_p, market_price, edge, stake_usd, pnl_usd, market_type, status
           FROM trade_log
           WHERE status IN ('settled_win','settled_loss')
             AND true_p IS NOT NULL AND market_price IS NOT NULL""").fetchall()
    con.close()
    out = []
    for tp, px, edge, stake, pnl, mkt, status in rows:
        out.append({"true_p": tp, "price": px, "edge": edge,
                    "stake": stake or 0.0, "pnl": pnl,
                    "mkt": mkt, "won": 1 if status == "settled_win" else 0})
    return out


def calib_table(bets, key="true_p"):
    edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    print(f"  {'prob bin':>12s} {'n':>5s} {'pred':>7s} {'actual':>7s} {'gap':>7s}")
    for lo, hi in zip(edges, edges[1:]):
        b = [x for x in bets if lo <= x[key] < hi]
        if not b:
            continue
        pred = sum(x[key] for x in b) / len(b)
        act = sum(x["won"] for x in b) / len(b)
        flag = "  <-- overconfident" if pred - act > 0.06 else ""
        print(f"  [{lo:.2f},{hi:.2f}) {len(b):5d} {pred:7.3f} {act:7.3f} {pred-act:+7.3f}{flag}")


def brier(bets):
    return sum((x["true_p"] - x["won"]) ** 2 for x in bets) / len(bets)


def summarize(bets, label):
    if not bets:
        print(f"\n{label}: (none)")
        return
    n = len(bets)
    wr = sum(x["won"] for x in bets) / n
    mean_tp = sum(x["true_p"] for x in bets) / n
    mean_edge = sum(x["edge"] for x in bets) / n
    pnl = sum(x["pnl"] for x in bets if x["pnl"] is not None)
    staked = sum(x["stake"] for x in bets)
    roi = pnl / staked if staked else 0.0
    print(f"\n{label}")
    print(f"  bets {n}   win rate {wr:.3f}   mean true_p {mean_tp:.3f}   "
          f"(pred-actual {mean_tp-wr:+.3f})")
    print(f"  mean edge claimed {mean_edge:+.3f}   Brier {brier(bets):.4f}   "
          f"baseline(mean) {sum((mean_tp-x['won'])**2 for x in bets)/n:.4f}")
    print(f"  PnL ${pnl:+.2f} on ${staked:.0f} staked   ROI {roi:+.1%}")
    calib_table(bets)


if __name__ == "__main__":
    bets = load()
    print("=" * 60)
    print("BASE-ENGINE BACKTEST (settled bets)")
    print("=" * 60)
    summarize(bets, "ALL SETTLED")
    summarize([b for b in bets if b["mkt"] == "match"], "MATCH markets")
    summarize([b for b in bets if b["mkt"] in ("set1", "set2", "set3")], "SET markets")
    print("\nReading: 'pred-actual' > 0 means true_p is systematically higher than")
    print("the realized win rate -> the engine is OVERCONFIDENT and edges are inflated.")
