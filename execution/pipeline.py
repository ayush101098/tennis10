"""Signal -> Polymarket order pipeline.

Usage:
    python -m execution.pipeline --signals signals.json          # run bets
    python -m execution.pipeline --signals signals.json --live   # real orders
    python -m execution.pipeline --log                           # show journal

Signal file format (JSON list), one entry per bet decision:
    [
      {
        "player1": "Cameron Norrie",       # fixture, model frame
        "player2": "Mariano Navone",
        "market":  "match",                # match | set1 | set2 | set3
        "side":    "player1",              # who the model backs
        "true_p":  0.62                    # model probability that side wins
      }
    ]

Live trading additionally requires POLYMARKET_PRIVATE_KEY (and
POLYMARKET_PROXY_ADDRESS for Polymarket-profile wallets) in .env, plus
TRADING_DRY_RUN=false. Otherwise every order is simulated and journaled.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from true_p_ensemble import kelly_stake  # noqa: E402
from execution.polymarket import PolymarketClient  # noqa: E402
from execution import trade_log  # noqa: E402

VALID_MARKETS = {"match", "set1", "set2", "set3"}


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env loader (no python-dotenv dependency)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def load_signals(path: str) -> list[dict]:
    signals = json.loads(Path(path).read_text())
    for s in signals:
        missing = {"player1", "player2", "market", "side", "true_p"} - set(s)
        if missing:
            raise ValueError(f"signal missing fields {missing}: {s}")
        if s["market"] not in VALID_MARKETS:
            raise ValueError(f"bad market {s['market']!r} (use {sorted(VALID_MARKETS)})")
        if s["side"] not in ("player1", "player2"):
            raise ValueError(f"bad side {s['side']!r}")
        if not 0.0 < float(s["true_p"]) < 1.0:
            raise ValueError(f"true_p out of range: {s}")
    return signals


def run_signals(signals: list[dict], live: bool = False) -> list[dict]:
    load_env()
    dry_env = os.getenv("TRADING_DRY_RUN", "true").lower() != "false"
    bankroll = float(os.getenv("TRADING_BANKROLL", "1000"))
    max_stake = float(os.getenv("TRADING_MAX_STAKE", "25"))

    client = PolymarketClient()
    dry_run = (not live) or dry_env or not client.can_trade_live
    mode = "LIVE" if not dry_run else "DRY-RUN"
    if live and dry_run:
        why = ("TRADING_DRY_RUN=true in .env" if dry_env
               else "POLYMARKET_PRIVATE_KEY not set")
        print(f"--live requested but forced to DRY-RUN: {why}")
    print(f"mode={mode}  bankroll=${bankroll:.0f}  max_stake=${max_stake:.0f}  "
          f"signals={len(signals)}\n")

    print("Fetching open tennis events from Polymarket Gamma...")
    events = client.fetch_tennis_events()
    print(f"  {len(events)} open tennis events\n")

    results = []
    for sig in signals:
        fixture = f"{sig['player1']} vs {sig['player2']}"
        tag = f"[{fixture} | {sig['market']} | {sig['side']} @ {sig['true_p']:.3f}]"

        markets = client.find_match_markets(sig["player1"], sig["player2"], events)
        market = next((m for m in markets if m.market_type == sig["market"]), None)
        if market is None:
            have = sorted({m.market_type for m in markets}) or "none"
            print(f"SKIP {tag} no {sig['market']} market on Polymarket (found: {have})")
            results.append({"signal": sig, "status": "no_market"})
            continue

        idx = market.side_index(sig["player1"], sig["player2"], sig["side"])
        if idx is None:
            print(f"SKIP {tag} could not map side to outcomes {market.outcomes}")
            results.append({"signal": sig, "status": "no_outcome"})
            continue

        token_id = market.token_ids[idx]
        outcome = market.outcomes[idx]

        live_only = os.getenv("TRADING_LIVE_ONLY", "true").lower() != "false"
        if live_only and not market.is_live():
            print(f"SKIP {tag} pre-match (waiting for live) — {market.question}")
            results.append({"signal": sig, "status": "pre_match"})
            continue

        if trade_log.already_traded(token_id):
            print(f"SKIP {tag} already have an open bet on {outcome} ({market.question})")
            results.append({"signal": sig, "status": "duplicate"})
            continue

        price = client.best_ask(token_id)
        if price is None or not 0.0 < price < 1.0:
            snap = market.prices[idx] if idx < len(market.prices) else None
            if snap and 0.0 < snap < 1.0:
                price = snap
                print(f"     {tag} no CLOB ask; using gamma snapshot price {price:.3f}")
            else:
                print(f"SKIP {tag} no tradeable price for {outcome}")
                results.append({"signal": sig, "status": "no_price"})
                continue

        true_p = float(sig["true_p"])
        stake_frac, kelly_dbg = kelly_stake(true_p, decimal_odds=1.0 / price)
        edge = true_p - price
        if stake_frac <= 0:
            print(f"PASS {tag} edge {edge:+.3f} below threshold "
                  f"(ask {price:.3f}) -> no bet")
            results.append({"signal": sig, "status": "no_edge", "price": price})
            continue

        stake = min(stake_frac * bankroll, max_stake)
        shares = stake / price
        exec_result = client.place_buy(token_id, price, shares, dry_run=dry_run)

        # Journal the live momentum behind an in-play bet so its marginal edge can
        # be backtested later (momentum can't be reconstructed after the fact).
        detail = exec_result.get("detail") or ""
        mom = sig.get("_momentum")
        if mom:
            detail = f"{detail} | {mom}".strip(" |")

        trade_id = trade_log.record_trade({
            "match_name": fixture,
            "market_type": market.market_type,
            "question": market.question,
            "condition_id": market.condition_id,
            "token_id": token_id,
            "outcome": outcome,
            "side": sig["side"],
            "true_p": true_p,
            "market_price": price,
            "edge": edge,
            "kelly_frac": stake_frac,
            "stake_usd": round(stake, 2),
            "shares": round(shares, 2),
            "order_id": exec_result.get("order_id"),
            "status": exec_result["status"],
            "detail": detail,
        })
        print(f"BET  {tag} -> BUY {shares:.2f} sh '{outcome}' @ {price:.3f} "
              f"(${stake:.2f}, edge {edge:+.3f}, kelly {stake_frac:.3%}) "
              f"[{exec_result['status']}] trade #{trade_id}")
        results.append({"signal": sig, "status": exec_result["status"],
                        "trade_id": trade_id, "price": price, "stake": stake})
    return results


def show_log(limit: int = 25, user: str | None = None) -> None:
    rows = trade_log.recent_trades(limit, user_name=user)
    if not rows:
        print("No trades logged yet." + (f" (user: {user})" if user else ""))
        return
    print(f"{'id':>4} {'ts (utc)':19} {'user':10} {'match':38} {'mkt':5} {'outcome':22} "
          f"{'price':>6} {'edge':>6} {'stake':>7} {'status':10}")
    print("-" * 136)
    for r in rows:
        print(f"{r['trade_id']:>4} {(r['ts_utc'] or '')[:19]:19} "
              f"{(r.get('user_name') or 'local')[:10]:10} "
              f"{(r['match_name'] or '')[:38]:38} {r['market_type']:5} "
              f"{(r['outcome'] or '')[:22]:22} {r['market_price']:>6.3f} "
              f"{r['edge']:>+6.3f} {r['stake_usd']:>7.2f} {r['status']:10}")
    s = trade_log.summary(user_name=user)
    print(f"\n{s['trades']} trades | open stake ${s['open_stake_usd']:.2f} | "
          f"settled PnL ${s['settled_pnl_usd']:+.2f} "
          f"({s['wins']}W-{s['losses']}L)")
    print(f"CSV mirror: {trade_log.CSV_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal -> Polymarket pipeline")
    parser.add_argument("--signals", help="path to signals JSON file")
    parser.add_argument("--live", action="store_true",
                        help="place real orders (needs keys + TRADING_DRY_RUN=false)")
    parser.add_argument("--watch", type=int, nargs="?", const=30, metavar="SECONDS",
                        help="live terminal mode: repoll every N seconds (default 30)")
    parser.add_argument("--log", action="store_true", help="show the trade journal")
    parser.add_argument("--settle", nargs=2, metavar=("TRADE_ID", "WIN|LOSS"),
                        help="settle a trade, e.g. --settle 3 win")
    parser.add_argument("--user", help="stamp trades with this user name and "
                        "filter --log to it (default: $TRADING_USER or 'local')")
    args = parser.parse_args()

    if args.user:
        os.environ["TRADING_USER"] = args.user

    if args.settle:
        trade_id, outcome = int(args.settle[0]), args.settle[1].lower()
        trade_log.settle_trade(trade_id, won=outcome in ("win", "w", "won"))
        print(f"Trade {trade_id} settled as {'WIN' if outcome.startswith('w') else 'LOSS'}.")
        return
    if args.log:
        show_log(user=args.user)
        return
    if not args.signals:
        parser.error("provide --signals <file.json>, --log, or --settle")
    signals = load_signals(args.signals)
    if args.watch:
        load_env()
        from execution.watch import watch
        dry_env = os.getenv("TRADING_DRY_RUN", "true").lower() != "false"
        client_can_trade = bool(os.getenv("POLYMARKET_PRIVATE_KEY"))
        dry_run = (not args.live) or dry_env or not client_can_trade
        watch(signals, interval=max(10, args.watch), dry_run=dry_run,
              bankroll=float(os.getenv("TRADING_BANKROLL", "1000")),
              max_stake=float(os.getenv("TRADING_MAX_STAKE", "25")))
        return
    run_signals(signals, live=args.live)
    print("\nJournal: python -m execution.pipeline --log")


if __name__ == "__main__":
    main()
