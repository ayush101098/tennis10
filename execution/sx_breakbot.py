"""
SX Bet break-momentum bot — driven by the live intelligence terminal.

STRATEGY
  For every live match we read the point-by-point feed and compute, from the
  live point score + serve strength, P(returner breaks the CURRENT service game):

    • BREAK play — if break_prob >= BREAK_THRESH (default 0.70) the returner is
      "about to break": back the RETURNER in SX's match-winner market, getting in
      ahead of the price move the break will cause.
    • HOLD play — else if the server's hold prob >= HOLD_THRESH: back the SERVER
      (the "rest on the servers").

  Either way we only fire when our momentum-adjusted live match True P beats SX's
  live taker price by >= MIN_EDGE, and we stake ¼-Kelly (capped). Open positions
  are HEDGED (back the opposite player on SX) if our True P for the held side
  drops by >= HEDGE_DROP versus entry.

SAFETY  (read this)
  • SX Bet lists ONLY match-winner markets, so a single likely break is a weak
    match-level signal — treat this as plumbing/rehearsal, not proven +EV.
  • Nothing is POSTED unless ALL of: SXBET_PRIVATE_KEY set, a funded wallet,
    TRADING_DRY_RUN=false, and --arm passed. Default = dry-run rehearsal.
  • Point SXBET_API at the toronto testnet (https://api.toronto.sx.bet) for a
    pure sandbox; leave it at api.sx.bet to rehearse against REAL live books
    (still dry-run — builds + signs the fill, never sends it).

    python -m execution.sx_breakbot --once            # one dry-run scan
    python -m execution.sx_breakbot --interval 20     # loop, dry-run
    python -m execution.sx_breakbot --interval 20 --arm   # POST (needs key+funds)
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from execution.live_odds import SofascoreOdds          # noqa: E402
from execution.inplay import InPlayModel               # noqa: E402
from execution.sxbet import SXBetClient                # noqa: E402
from execution.momentum import break_prob_from_score   # noqa: E402
from execution import trade_log                         # noqa: E402
from execution.pipeline import load_env                 # noqa: E402
from true_p_ensemble import kelly_stake                 # noqa: E402

BREAK_THRESH = float(os.getenv("SX_BREAK_THRESH", "0.70"))
HOLD_THRESH = float(os.getenv("SX_HOLD_THRESH", "0.80"))
MIN_EDGE = float(os.getenv("SX_MIN_EDGE", "0.03"))
HEDGE_DROP = float(os.getenv("SX_HEDGE_DROP", "0.15"))
BANKROLL = float(os.getenv("TRADING_BANKROLL", "1000"))
MAX_STAKE = float(os.getenv("TRADING_MAX_STAKE", "25"))
KELLY_CAP = float(os.getenv("TRADING_KELLY_CAP", "0.05"))


def _live_fixtures(sofa: SofascoreOdds, with_meta: bool = False):
    """(home, away) full-name pairs for in-progress singles matches.
    with_meta=True yields (home, away, league) instead."""
    data = sofa._get("sport/tennis/events/live") or {}
    out = []
    for e in data.get("events", []):
        if (e.get("status") or {}).get("type") != "inprogress":
            continue
        h = (e.get("homeTeam") or {}).get("name", "")
        a = (e.get("awayTeam") or {}).get("name", "")
        if not h or not a or "/" in h or "/" in a:
            continue
        if with_meta:
            league = ((e.get("tournament") or {}).get("name")
                      or (e.get("tournament") or {}).get("uniqueTournament", {}).get("name") or "")
            out.append((h, a, league))
        else:
            out.append((h, a))
    return out


def scan_board(sofa, ip) -> list[dict]:
    """Live board for EVERY in-progress match: favored side + live win %, break
    watch, momentum, score. SX-independent (pure intelligence, no betting)."""
    board = []
    for h, a, league in _live_fixtures(sofa, with_meta=True):
        tp1 = ip.live_true_p(h, a)               # momentum-adjusted P(home wins)
        if tp1 is None:
            continue
        mom = ip.last_momentum()
        sc = sofa.state(h, a)                     # sets/games/point
        brk = None
        cg = _current_game(sofa, h, a)
        if cg:
            server, srv_pts, ret_pts, returner, server_name = cg
            bp = break_prob_from_score(ip._serve_win(server_name), srv_pts, ret_pts)
            brk = {"prob": round(bp, 3), "returner": returner,
                   "server": server_name, "pts": f"{srv_pts}-{ret_pts}"}
        side = 1 if tp1 >= 0.5 else 2
        board.append({
            "p1": h, "p2": a, "league": league[:28],
            "win": round(max(tp1, 1.0 - tp1), 3),
            "favored": h if side == 1 else a, "side": side,
            "momentum": mom, "break": brk,
            "score": {"sets": (sc or {}).get("sets"), "games": (sc or {}).get("games"),
                      "point": (sc or {}).get("point")} if sc else None,
        })
    # breaks (>=0.5) and near-sure (>=0.8) first, then by win prob
    board.sort(key=lambda r: max((r["break"] or {}).get("prob", 0), r["win"]), reverse=True)
    return board


def _current_game(sofa: SofascoreOdds, p1: str, p2: str):
    """(server, server_pts, returner_pts, returner_name, server_name) for the
    in-progress game, oriented via the point-by-point feed. None if between games."""
    games = sofa.game_sequence(p1, p2)
    if not games:
        return None
    cur = games[-1]
    if cur["winner"] is not None or cur["server"] not in (1, 2):
        return None  # no live in-progress game to price
    pts = cur["points"]
    p1pt, p2pt = (pts[-1] if pts else ("0", "0"))
    p1pt, p2pt = (p1pt or "0"), (p2pt or "0")
    server = cur["server"]
    if server == 1:
        return server, p1pt, p2pt, p2, p1   # p1 serves -> p2 returns
    return server, p2pt, p1pt, p1, p2       # p2 serves -> p1 returns


def _open_sx_positions():
    return [r for r in trade_log.all_trades()
            if r.get("venue") == "sxbet" and r.get("status") in ("dry_run", "placed")
            and r.get("side") != "hedge"]


def _already_open(match_name: str, outcome: str) -> bool:
    return any(r["match_name"] == match_name and r["outcome"] == outcome
              for r in _open_sx_positions())


def _bet(sx, ip, p1, p2, target, kind, true_p_tgt, price_tgt, dry_run, notes):
    edge = true_p_tgt - price_tgt
    if edge < MIN_EDGE:
        return
    match_name = f"{p1} vs {p2}"
    if _already_open(match_name, target):
        return
    frac, _ = kelly_stake(true_p_tgt, decimal_odds=1.0 / price_tgt)
    if frac <= 0:
        return
    stake = round(min(frac, KELLY_CAP) * BANKROLL, 2)
    stake = min(stake, MAX_STAKE)
    if stake < 1:
        return
    res = sx.place_bet(p1, p2, target, stake, dry_run=dry_run)
    trade_log.record_trade({
        "venue": "sxbet", "match_name": match_name, "market_type": "match",
        "question": f"SX {kind.upper()}: back {target}",
        "outcome": target, "side": "player1" if target == p1 else "player2",
        "true_p": round(true_p_tgt, 4), "market_price": round(price_tgt, 4),
        "edge": round(edge, 4), "kelly_frac": round(frac, 4), "stake_usd": stake,
        "status": res["status"], "detail": f"[{kind}] {res.get('detail', '')}"[:300],
    })
    notes.append(f"  {kind.upper():5s} back {target[:20]:20s} @ {price_tgt:.3f} "
                 f"trueP {true_p_tgt:.3f} edge +{edge:.3f} ${stake:.0f} [{res['status']}]")


def _hedge(sx, ip, dry_run, notes):
    """Back the opposite player if a held side's live True P has dropped enough."""
    for pos in _open_sx_positions():
        try:
            p1, p2 = pos["match_name"].split(" vs ", 1)
        except ValueError:
            continue
        cur_p1 = ip.live_true_p(p1, p2)
        if cur_p1 is None:
            continue
        held = pos["outcome"]
        cur_side = cur_p1 if held == p1 else (1.0 - cur_p1)
        drop = float(pos.get("true_p") or 0.5) - cur_side
        if drop < HEDGE_DROP:
            continue
        opp = p2 if held == p1 else p1
        q = sx.quote(p1, p2)
        if not q:
            continue
        opp_price = q["back_p1"] if opp == p1 else q["back_p2"]
        if not opp_price or not 0 < opp_price < 1:
            continue
        if _already_open(pos["match_name"], opp):
            continue
        stake = round(min(float(pos.get("stake_usd") or 0), MAX_STAKE), 2)
        if stake < 1:
            continue
        res = sx.place_bet(p1, p2, opp, stake, dry_run=dry_run)
        trade_log.record_trade({
            "venue": "sxbet", "match_name": pos["match_name"], "market_type": "match",
            "question": f"SX HEDGE for #{pos['trade_id']}", "outcome": opp,
            "side": "hedge", "true_p": round(cur_side, 4), "market_price": round(opp_price, 4),
            "edge": 0.0, "stake_usd": stake, "status": res["status"],
            "detail": f"HEDGE (drop {drop:.2f}) {res.get('detail', '')}"[:300],
        })
        notes.append(f"  HEDGE #{pos['trade_id']} back {opp[:20]} @ {opp_price:.3f} "
                     f"(True P dropped {drop:.2f}) ${stake:.0f} [{res['status']}]")


def _match_signal(sofa, ip, sx, p1, p2) -> dict | None:
    """Break/hold signal for one live match (no side effects). None if no signal.
    Includes the SX taker price + edge when SX books the fixture."""
    cg = _current_game(sofa, p1, p2)
    if not cg:
        return None
    server, srv_pts, ret_pts, returner, server_name = cg
    p_serve = ip._serve_win(server_name)
    bprob = break_prob_from_score(p_serve, srv_pts, ret_pts)

    if bprob >= BREAK_THRESH:
        target, kind = returner, "break"
    elif (1.0 - bprob) >= HOLD_THRESH:
        target, kind = server_name, "hold"
    else:
        return None

    true_p1 = ip.live_true_p(p1, p2)         # momentum-adjusted P(p1 wins)
    if true_p1 is None:
        return None
    true_p_tgt = true_p1 if target == p1 else (1.0 - true_p1)

    q = sx.quote(p1, p2)
    price_tgt = None
    if q:
        pt = q["back_p1"] if target == p1 else q["back_p2"]
        if pt and 0.02 < pt < 0.98:
            price_tgt = pt
    edge = (true_p_tgt - price_tgt) if price_tgt is not None else None
    return {
        "p1": p1, "p2": p2, "kind": kind, "target": target,
        "score": f"{srv_pts}-{ret_pts}", "server": server_name,
        "break_prob": round(bprob, 3), "true_p": round(true_p_tgt, 3),
        "sx_price": None if price_tgt is None else round(price_tgt, 3),
        "edge": None if edge is None else round(edge, 3),
        "has_market": q is not None,
        "tradeable": edge is not None and edge >= MIN_EDGE,
    }


def scan_signals(sofa, ip, sx) -> list[dict]:
    """All current break/hold signals for display (best edge first). No betting."""
    sigs = []
    for p1, p2 in _live_fixtures(sofa):
        s = _match_signal(sofa, ip, sx, p1, p2)
        if s:
            sigs.append(s)
    sigs.sort(key=lambda s: (s["edge"] if s["edge"] is not None else -9), reverse=True)
    return sigs


def open_sx_positions() -> list[dict]:
    """Open SX positions from the journal, for dashboard display."""
    out = []
    for r in trade_log.all_trades():
        if r.get("venue") == "sxbet" and r.get("status") in ("dry_run", "placed"):
            out.append({
                "trade_id": r["trade_id"], "match": r["match_name"],
                "outcome": r["outcome"], "side": r.get("side"),
                "true_p": r.get("true_p"), "price": r.get("market_price"),
                "edge": r.get("edge"), "stake": r.get("stake_usd"),
                "status": r.get("status"), "detail": (r.get("detail") or "")[:80],
            })
    return out[::-1]


def scan_once(sofa, ip, sx, dry_run) -> list[str]:
    notes: list[str] = []
    for p1, p2 in _live_fixtures(sofa):
        s = _match_signal(sofa, ip, sx, p1, p2)
        if s and s["tradeable"]:
            _bet(sx, ip, p1, p2, s["target"], s["kind"],
                 s["true_p"], s["sx_price"], dry_run, notes)
    _hedge(sx, ip, dry_run, notes)  # hedge existing positions
    return notes


def main():
    load_env()
    ap = argparse.ArgumentParser(description="SX Bet break-momentum bot (dry-run by default)")
    ap.add_argument("--once", action="store_true", help="single scan then exit")
    ap.add_argument("--interval", type=int, default=20, help="loop seconds (ignored with --once)")
    ap.add_argument("--arm", action="store_true",
                    help="allow REAL posts (still needs SXBET_PRIVATE_KEY + TRADING_DRY_RUN=false)")
    args = ap.parse_args()

    env_dry = os.getenv("TRADING_DRY_RUN", "true").lower() == "true"
    dry_run = env_dry or not args.arm
    sx = SXBetClient()
    live_ok = (not dry_run) and sx.can_trade_live
    mode = "LIVE POST" if live_ok else ("DRY-RUN (armed, no key/funds)" if args.arm else "DRY-RUN")
    print("=" * 66)
    print(f"SX BREAK BOT — {mode}   endpoint={os.getenv('SXBET_API', 'https://api.sx.bet')}")
    print(f"break>={BREAK_THRESH}  hold>={HOLD_THRESH}  min_edge={MIN_EDGE}  "
          f"hedge_drop={HEDGE_DROP}  bankroll=${BANKROLL:.0f}  max=${MAX_STAKE:.0f}")
    if live_ok:
        print("⚠ LIVE POSTING ENABLED — real fills will be sent to SX.")
    print("=" * 66)

    sofa = SofascoreOdds()
    ip = InPlayModel(sofa=sofa)
    try:
        while True:
            t0 = time.time()
            notes = scan_once(sofa, ip, sx, dry_run)
            stamp = time.strftime("%H:%M:%S")
            if notes:
                print(f"[{stamp}] {len(notes)} action(s):")
                print("\n".join(notes))
            else:
                print(f"[{stamp}] no break/hold signal cleared edge on any live match")
            if args.once:
                break
            time.sleep(max(2, args.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        ip.close()


if __name__ == "__main__":
    main()
