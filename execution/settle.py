"""Settle pass: resolve finished bets against Polymarket and book PnL.

For every unsettled journal row (primary bets and hedges), look up the market's
resolution by condition_id on the CLOB. If resolved, mark the row win/loss so
realized PnL lands in the journal. Markets still open are left untouched.

    python -m execution.settle              # settle everything that's finished
    python -m execution.settle --user alice # only that user's bets
    python -m execution.settle --dry        # show what would settle, write nothing
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from execution.pipeline import load_env  # noqa: E402
from execution.polymarket import PolymarketClient  # noqa: E402
from execution import trade_log  # noqa: E402


def settle_open(user: str | None = None, dry: bool = False,
                verbose: bool = True) -> dict:
    load_env()
    client = PolymarketClient()
    rows = trade_log.unsettled_trades(user_name=user)
    resolved_cache: dict[str, dict | None] = {}
    settled = wins = losses = 0
    pnl = 0.0

    for r in rows:
        cond, tok = r.get("condition_id"), r.get("token_id")
        if not cond or not tok:
            continue
        if cond not in resolved_cache:
            resolved_cache[cond] = client.market_resolution(cond)
        res = resolved_cache[cond]
        if not res:
            continue  # not resolved yet
        info = res["tokens"].get(tok)
        if info is None:
            continue
        won = info["winner"]
        stake, shares = float(r["stake_usd"] or 0), float(r["shares"] or 0)
        row_pnl = (shares - stake) if won else -stake
        tag = ("HEDGE " if (r.get("detail") or "").startswith("HEDGE") else "")
        if verbose:
            print(f"  {'WIN ' if won else 'LOSS'}  #{r['trade_id']:>3} {tag}"
                  f"{(r['match_name'] or '')[:34]:34} {r['market_type']:5} "
                  f"'{(r['outcome'] or '')[:18]:18}' -> ${row_pnl:+.2f}"
                  + ("  (dry)" if dry else ""))
        if not dry:
            trade_log.settle_trade(r["trade_id"], won=won)
        settled += 1
        pnl += row_pnl
        wins += int(won)
        losses += int(not won)

    if verbose:
        verb = "would settle" if dry else "settled"
        print(f"\n{verb} {settled} bet(s): {wins}W-{losses}L | "
              f"realized PnL ${pnl:+.2f}")
        if settled == 0:
            print("  (nothing resolved yet — check back once matches finish)")
    return {"settled": settled, "wins": wins, "losses": losses, "pnl": pnl}


def main() -> None:
    ap = argparse.ArgumentParser(description="Settle finished Polymarket bets")
    ap.add_argument("--user", help="only settle this user's bets")
    ap.add_argument("--dry", action="store_true", help="preview; write nothing")
    args = ap.parse_args()
    settle_open(user=args.user, dry=args.dry)


if __name__ == "__main__":
    main()
