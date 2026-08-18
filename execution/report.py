"""Performance report for the trading agent's journal.

    python -m execution.report            # full report
    python -m execution.report --user bob # scope to one user

Summarizes exposure, edges taken, Kelly sizing, hedging, and — once bets are
settled (see execution.settle) — realized PnL, ROI, win rate, and a rough
calibration check (model true_p vs actual win rate).
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from execution import trade_log  # noqa: E402

_OPEN = ("dry_run", "placed")
_SETTLED = ("settled_win", "settled_loss")


def _is_hedge(r: dict) -> bool:
    return (r.get("detail") or "").startswith("HEDGE") or r.get("side") == "hedge"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def build_report(user: str | None = None) -> str:
    rows = trade_log.all_trades(user_name=user)
    if not rows:
        return "No trades in the journal yet."

    primary = [r for r in rows if not _is_hedge(r)]
    hedges = [r for r in rows if _is_hedge(r)]
    settled = [r for r in rows if r["status"] in _SETTLED]
    open_rows = [r for r in rows if r["status"] in _OPEN]

    L = []
    add = L.append
    scope = f" (user: {user})" if user else ""
    add("=" * 68)
    add(f"  TENNIS AGENT — PERFORMANCE REPORT{scope}")
    add("=" * 68)

    # ── exposure ──────────────────────────────────────────────────────────
    total_staked = sum(float(r["stake_usd"] or 0) for r in rows)
    open_stake = sum(float(r["stake_usd"] or 0) for r in open_rows)
    add("\n[ EXPOSURE ]")
    add(f"  bets logged .......... {len(rows)}  "
        f"({len(primary)} primary, {len(hedges)} hedge)")
    add(f"  open / settled ....... {len(open_rows)} open, {len(settled)} settled")
    add(f"  total staked ......... ${total_staked:,.2f}")
    add(f"  open exposure ........ ${open_stake:,.2f}")

    # by market type
    by_mkt = defaultdict(lambda: [0, 0.0])
    for r in primary:
        by_mkt[r["market_type"]][0] += 1
        by_mkt[r["market_type"]][1] += float(r["stake_usd"] or 0)
    add("  by market:           " + "  ".join(
        f"{k}={v[0]} (${v[1]:.0f})" for k, v in sorted(by_mkt.items())))

    # ── signal quality / sizing ───────────────────────────────────────────
    add("\n[ SIGNALS & KELLY SIZING ] (primary bets)")
    add(f"  avg model true_p ..... {_mean([r['true_p'] for r in primary]):.3f}")
    add(f"  avg entry price ...... {_mean([r['market_price'] for r in primary]):.3f}")
    add(f"  avg edge taken ....... {_mean([r['edge'] for r in primary]):+.3f}")
    add(f"  avg Kelly fraction ... {_mean([r['kelly_frac'] for r in primary]):.3%}")
    add(f"  avg stake ............ ${_mean([r['stake_usd'] for r in primary]):.2f}")

    # ── hedging ───────────────────────────────────────────────────────────
    add("\n[ HEDGING ] (full-lock on adverse move)")
    if hedges:
        hedge_cost = sum(float(r["stake_usd"] or 0) for r in hedges)
        add(f"  hedges placed ........ {len(hedges)}")
        add(f"  hedge cost ........... ${hedge_cost:,.2f}")
        for r in hedges[:6]:
            add(f"    #{r['trade_id']} {(r['match_name'] or '')[:30]:30} "
                f"{r['market_type']:5} {(r.get('detail') or '')[:46]}")
    else:
        add("  none triggered yet (no open position has dropped past the threshold)")

    # ── realized performance ──────────────────────────────────────────────
    add("\n[ REALIZED PERFORMANCE ]")
    if settled:
        wins = [r for r in settled if r["status"] == "settled_win"]
        pnl = sum(float(r["pnl_usd"] or 0) for r in settled)
        settled_stake = sum(float(r["stake_usd"] or 0) for r in settled)
        roi = (pnl / settled_stake) if settled_stake else 0.0
        add(f"  settled bets ......... {len(settled)}  "
            f"({len(wins)}W-{len(settled) - len(wins)}L, "
            f"win rate {len(wins) / len(settled):.1%})")
        add(f"  realized PnL ......... ${pnl:+,.2f}")
        add(f"  ROI on settled ....... {roi:+.1%}")
        best = max(settled, key=lambda r: float(r["pnl_usd"] or 0))
        worst = min(settled, key=lambda r: float(r["pnl_usd"] or 0))
        add(f"  best  ................ #{best['trade_id']} "
            f"{(best['match_name'] or '')[:28]} ${float(best['pnl_usd'] or 0):+.2f}")
        add(f"  worst ................ #{worst['trade_id']} "
            f"{(worst['match_name'] or '')[:28]} ${float(worst['pnl_usd'] or 0):+.2f}")

        # rough calibration: model true_p vs realized win rate (primary only)
        sp = [r for r in settled if not _is_hedge(r)]
        if len(sp) >= 3:
            pred = _mean([r["true_p"] for r in sp])
            actual = _mean([1.0 if r["status"] == "settled_win" else 0.0 for r in sp])
            add(f"  calibration .......... model {pred:.1%} vs actual {actual:.1%} "
                f"(n={len(sp)}; gap {pred - actual:+.1%})")
    else:
        add("  nothing settled yet — run:  python -m execution.settle")

    add("\n" + "=" * 68)
    add("  paper journal — run 'python -m execution.settle' as matches finish")
    add("=" * 68)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent performance report")
    ap.add_argument("--user", help="scope to one user")
    args = ap.parse_args()
    print(build_report(user=args.user))


if __name__ == "__main__":
    main()
