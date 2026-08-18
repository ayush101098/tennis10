"""Auto-fire SX Bet taker fills when a signal beats SX's live price.

Runs as an optional pass inside the agent watch loop (env TRADING_SX_AUTOBET).
For each signal it compares our model/odds true_p against SX Bet's live taker
price for that side; if the edge clears the threshold it fires a taker fill
(paper unless the SX live guards are open) and journals it under venue=sxbet.

    ⚠ HONEST WARNING: SX Bet is a sharp, de-vigged LIVE exchange. Our signals
    (stale model / pre-match Sofascore / market fallback) are weaker than SX's
    line, so an "edge" here usually means OUR number is wrong, not that SX is
    mispriced. This path is -EV unless/until the signal is genuinely sharper
    than SX. It ships OFF by default with a high threshold; treat it as plumbing.
"""

import os
from typing import Optional

from true_p_ensemble import kelly_stake
from execution import trade_log
from execution.sxbet import SXBetClient


def _already_bet(match_name: str, outcome: str) -> bool:
    for r in trade_log.all_trades():
        if (r.get("venue") == "sxbet" and r.get("match_name") == match_name
                and r.get("outcome") == outcome
                and r.get("status") in ("dry_run", "placed")):
            return True
    return False


def autobet_pass(signals: list[dict], sx: Optional[SXBetClient], dry_run: bool,
                 bankroll: float, max_stake: float) -> list[str]:
    if not signals:
        return []
    min_edge = float(os.getenv("TRADING_SX_MIN_EDGE", "0.08"))
    sx = sx or SXBetClient()
    notes = []
    for s in signals:
        p1, p2, side = s["player1"], s["player2"], s["side"]
        true_p = float(s["true_p"])
        q = sx.quote(p1, p2)
        if not q:
            continue
        target = p1 if side == "player1" else p2
        taker = q["back_p1"] if side == "player1" else q["back_p2"]
        if not taker or not 0.0 < taker < 1.0:
            continue
        edge = true_p - taker            # our prob vs SX price for this side
        if edge < min_edge:
            continue
        if _already_bet(f"{p1} vs {p2}", target):
            continue
        frac, _ = kelly_stake(true_p, decimal_odds=1.0 / taker)
        if frac <= 0:
            continue
        stake = round(min(frac * bankroll, max_stake), 2)
        res = sx.place_bet(p1, p2, target, stake, dry_run=dry_run)
        trade_log.record_trade({
            "venue": "sxbet", "match_name": f"{p1} vs {p2}", "market_type": "match",
            "question": q.get("league") or "SX Bet",
            "condition_id": res.get("fill", {}).get("marketHash"),
            "token_id": None, "outcome": target, "side": "sx_auto",
            "true_p": round(true_p, 4), "market_price": round(taker, 4),
            "edge": round(edge, 4), "kelly_frac": round(frac, 4),
            "stake_usd": stake, "shares": None, "order_id": None,
            "status": res["status"],
            "detail": f"SX auto {'LIVE' if not dry_run else 'paper'} "
                      f"edge {edge:+.3f}: {res.get('detail', '')[:150]}",
        })
        notes.append(f"SX-AUTO {target[:18]} ${stake} @ {taker:.3f} "
                     f"(edge {edge:+.3f}) [{res['status']}]")
    return notes
