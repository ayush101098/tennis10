"""Live terminal view: poll Polymarket, show edges, auto-bet on threshold.

    python -m execution.pipeline --signals signals.json --watch 30

Each cycle re-fetches open tennis events and the CLOB best ask for every
signal, prints a live table, and places a bet (dry-run unless live mode is
fully unlocked) the moment edge clears the 2% Kelly threshold. Dedupe via the
trade journal guarantees one bet per market token across cycles and restarts.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

from true_p_ensemble import kelly_stake
from execution.polymarket import PolymarketClient
from execution import trade_log

CLEAR = "\x1b[2J\x1b[H"

# Price drop (in probability points) on our side that triggers a full-lock hedge.
HEDGE_DROP = float(os.getenv("TRADING_HEDGE_DROP", "0.15"))

# Only enter positions on matches that are in-play (started). Set false to also
# allow pre-match entries.
LIVE_ONLY = os.getenv("TRADING_LIVE_ONLY", "true").lower() != "false"

# Re-generate signals every N cycles so newly-started matches and refreshed
# market-fallback favorites are picked up without a manual restart.
REGEN_EVERY = int(os.getenv("TRADING_REGEN_EVERY", "15"))

# Auto-fire SX Bet taker fills when a signal beats SX's live price. OFF by
# default — see execution/sx_autobet.py for the (-EV) caveat.
SX_AUTOBET = os.getenv("TRADING_SX_AUTOBET", "false").lower() == "true"


def _opposite_leg(events: list[dict], condition_id: str, our_token: str):
    """Locate the other outcome of our market. -> (opp_token, opp_outcome) | None."""
    for event in events:
        for m in event.get("markets", []):
            if m.get("conditionId") != condition_id:
                continue
            try:
                outcomes = json.loads(m.get("outcomes") or "[]")
                tokens = json.loads(m.get("clobTokenIds") or "[]")
            except (json.JSONDecodeError, TypeError):
                return None
            if our_token not in tokens or len(tokens) != 2:
                return None
            j = 1 - tokens.index(our_token)
            return tokens[j], (outcomes[j] if j < len(outcomes) else "opponent")
    return None


def _hedge_positions(client: PolymarketClient, events: list[dict],
                     dry_run: bool, drop: float = HEDGE_DROP) -> list[str]:
    """Full-lock any open position whose side has dropped >= `drop` in price.

    Buys enough of the opposite outcome to equalize payout across results,
    locking in a fixed (capped) loss. Dedupe: once the opposite token is held,
    already_traded() prevents re-hedging.
    """
    notes = []
    for pos in trade_log.open_positions():
        our_token = pos.get("token_id")
        cond = pos.get("condition_id")
        entry = pos.get("market_price")
        shares = pos.get("shares")
        if not (our_token and cond and entry and shares):
            continue
        now_price = client.best_ask(our_token)
        if now_price is None or not 0.0 < now_price < 1.0:
            continue
        if entry - now_price < drop:
            continue  # not adverse enough yet
        leg = _opposite_leg(events, cond, our_token)
        if not leg:
            continue
        opp_token, opp_outcome = leg
        if trade_log.already_traded(opp_token):
            continue  # already hedged (or independently held)
        opp_ask = client.best_ask(opp_token)
        if opp_ask is None or not 0.0 < opp_ask < 1.0:
            opp_ask = max(0.01, min(0.99, 1.0 - now_price))
        hedge_shares = round(float(shares), 2)   # equalize payout to `shares`
        cost = hedge_shares * opp_ask
        locked = round(float(shares) - (float(pos["stake_usd"]) + cost), 2)
        result = client.place_buy(opp_token, opp_ask, hedge_shares, dry_run=dry_run)
        trade_log.record_trade({
            "match_name": pos["match_name"], "market_type": pos["market_type"],
            "question": pos["question"], "condition_id": cond,
            "token_id": opp_token, "outcome": opp_outcome,
            "side": "hedge", "true_p": 1.0 - float(pos.get("true_p") or 0.5),
            "market_price": opp_ask, "edge": 0.0, "kelly_frac": 0.0,
            "stake_usd": round(cost, 2), "shares": hedge_shares,
            "order_id": result.get("order_id"), "status": result["status"],
            "detail": f"HEDGE full-lock for #{pos['trade_id']} "
                      f"(entry {entry:.3f}->{now_price:.3f}); locked ${locked:+.2f}",
        })
        notes.append(f"HEDGE #{pos['trade_id']} {pos['match_name'][:24]} "
                     f"{pos['market_type']}: bought {hedge_shares:.1f} '{opp_outcome[:16]}' "
                     f"@ {opp_ask:.3f} (drop {entry-now_price:+.3f}, lock ${locked:+.2f})")
    return notes


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

    # Live-only gate: hold pre-match fixtures until the match is in-play.
    if LIVE_ONLY and not market.is_live():
        row["action"] = "pre-match (waiting for live)"
        return row

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
                if REGEN_EVERY > 0 and cycle % REGEN_EVERY == 0:
                    try:
                        from execution.signals_gen import generate
                        fresh = generate(verbose=False)
                        if fresh:
                            signals = fresh
                    except Exception:
                        pass
                events = client.fetch_tennis_events()
                rows = [_evaluate(client, events, s, dry_run, bankroll, max_stake)
                        for s in signals]
                hedges = _hedge_positions(client, events, dry_run)
                if SX_AUTOBET:
                    try:
                        from execution.sx_autobet import autobet_pass
                        hedges += autobet_pass(signals, None, dry_run,
                                               bankroll, max_stake)
                    except Exception as e:
                        hedges.append(f"sx-autobet error: {type(e).__name__}")
                # Auto-settle finished bets periodically (every ~10 cycles) so
                # realized PnL keeps flowing without a manual pass.
                if cycle % 10 == 1:
                    try:
                        from execution.settle import settle_open
                        s_res = settle_open(verbose=False)
                        if s_res["settled"]:
                            hedges.append(f"SETTLED {s_res['settled']} bet(s): "
                                          f"{s_res['wins']}W-{s_res['losses']}L "
                                          f"${s_res['pnl']:+.2f}")
                    except Exception:
                        pass
                err = None
            except Exception as e:
                rows, hedges, err = [], [], f"{type(e).__name__}: {e}"

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
            for h in hedges:
                out.append(h)
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
