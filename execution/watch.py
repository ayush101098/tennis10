"""Live terminal view: poll Polymarket, show edges, auto-bet on threshold.

    python -m execution.pipeline --signals signals.json --watch 30

Each cycle re-fetches open tennis events and the CLOB best ask for every
signal, prints a live table, and places a bet (dry-run unless live mode is
fully unlocked) the moment edge clears the 2% Kelly threshold. Dedupe via the
trade journal guarantees one bet per market token across cycles and restarts.
"""

import os
import sys
import time
from datetime import datetime, timezone

from true_p_ensemble import kelly_stake
from execution.polymarket import PolymarketClient
from execution import trade_log

CLEAR = "\x1b[2J\x1b[H"


def _evaluate(client: PolymarketClient, events: list[dict], sig: dict,
              dry_run: bool, bankroll: float, max_stake: float) -> dict:
    """One signal, one cycle: map -> price -> decide -> maybe bet. Returns a row."""
    fixture = f"{sig['player1'].split()[-1]} vs {sig['player2'].split()[-1]}"
    row = {"fixture": fixture, "mkt": sig["market"], "side": "?",
           "ask": None, "true_p": float(sig["true_p"]), "edge": None, "action": ""}

    markets = client.find_match_markets(sig["player1"], sig["player2"], events)
    market = next((m for m in markets if m.market_type == sig["market"]), None)
    if market is None:
        row["action"] = "no market"
        return row

    idx = market.side_index(sig["player1"], sig["player2"], sig["side"])
    if idx is None:
        row["action"] = f"side not in {market.outcomes}"
        return row
    token_id = market.token_ids[idx]
    row["side"] = market.outcomes[idx]

    price = client.best_ask(token_id)
    if price is None or not 0.0 < price < 1.0:
        snap = market.prices[idx] if idx < len(market.prices) else None
        price = snap if snap and 0.0 < snap < 1.0 else None
    if price is None:
        row["action"] = "no price"
        return row
    row["ask"] = price
    row["edge"] = row["true_p"] - price

    if trade_log.already_traded(token_id):
        row["action"] = "bet open"
        return row

    stake_frac, _ = kelly_stake(row["true_p"], decimal_odds=1.0 / price)
    if stake_frac <= 0:
        row["action"] = "pass (edge<2%)"
        return row

    stake = min(stake_frac * bankroll, max_stake)
    shares = stake / price
    result = client.place_buy(token_id, price, shares, dry_run=dry_run)
    trade_id = trade_log.record_trade({
        "match_name": f"{sig['player1']} vs {sig['player2']}",
        "market_type": market.market_type, "question": market.question,
        "condition_id": market.condition_id, "token_id": token_id,
        "outcome": market.outcomes[idx], "side": sig["side"],
        "true_p": row["true_p"], "market_price": price, "edge": row["edge"],
        "kelly_frac": stake_frac, "stake_usd": round(stake, 2),
        "shares": round(shares, 2), "order_id": result.get("order_id"),
        "status": result["status"], "detail": result.get("detail"),
    })
    row["action"] = f"BET #{trade_id} ${stake:.2f} @ {price:.3f} [{result['status']}]"
    return row


def watch(signals: list[dict], interval: int, dry_run: bool,
          bankroll: float, max_stake: float) -> None:
    client = PolymarketClient()
    mode = "DRY-RUN" if dry_run else "LIVE"
    is_tty = sys.stdout.isatty()
    cycle = 0
    try:
        while True:
            cycle += 1
            try:
                events = client.fetch_tennis_events()
                rows = [_evaluate(client, events, s, dry_run, bankroll, max_stake)
                        for s in signals]
                err = None
            except Exception as e:
                rows, err = [], f"{type(e).__name__}: {e}"

            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            out = [CLEAR.rstrip("\n")] if is_tty else [f"\n===== cycle {cycle} ====="]
            out.append(f"POLYMARKET LIVE  {now} UTC   cycle {cycle}  "
                       f"every {interval}s   [{mode}]")
            out.append("-" * 96)
            out.append(f"{'fixture':30} {'mkt':5} {'side':20} {'ask':>6} "
                       f"{'trueP':>6} {'edge':>7}  action")
            for r in rows:
                ask = f"{r['ask']:.3f}" if r["ask"] is not None else "--"
                edge = f"{r['edge']:+.3f}" if r["edge"] is not None else "--"
                out.append(f"{r['fixture'][:30]:30} {r['mkt']:5} {r['side'][:20]:20} "
                           f"{ask:>6} {r['true_p']:>6.3f} {edge:>7}  {r['action']}")
            if err:
                out.append(f"cycle error (will retry): {err}")
            out.append("-" * 96)
            s = trade_log.summary()
            out.append(f"journal: {s['trades']} bets | open stake "
                       f"${s['open_stake_usd']:.2f} | settled PnL "
                       f"${s['settled_pnl_usd']:+.2f} ({s['wins']}W-{s['losses']}L)"
                       f"   Ctrl-C to stop")
            print("\n".join(out), flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped. Journal: python -m execution.pipeline --log")
