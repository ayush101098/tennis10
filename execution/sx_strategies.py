"""
SX Bet three-tier strategy runner — GAME / SET / MATCH markets, driven by the
live intelligence engine.

WHAT IT TRADES (all on the SX order book, taker fills)
  MATCH  (type 52)        back a player when the momentum-adjusted live True P
                          beats SX's taker price by MATCH_MIN_EDGE.
  SET    (types 202-204,  back the winner of the set in play (or a future set),
          865/866, 165)    plus set spread (-1.5) and total sets, priced by the
                          Monte-Carlo engine below.
  GAME   (types 201, 166) game spread and total-games O/U, priced from the same
                          simulation of every remaining game.
                          ⚠ OFF BY DEFAULT — backtested WORSE than a constant
                          (see execution/sx_backtest.py); the sim systematically
                          over-predicts total games by ~12-19pp.

HOW IT PRICES
  The Markov/momentum engine (execution.inplay) supplies each player's serve
  point-win probability and the live match True P. From the current score
  (sets/games via Sofascore) a Monte-Carlo simulation plays out every remaining
  game — analytic hold probability per service game, point-by-point tiebreaks —
  giving full distributions over set winners, final game counts and set counts.
  Derivative-market edges use those distributions; the match market uses the
  engine's True P directly (it carries momentum, which the sim does not).

SAFETY (same contract as sx_breakbot — read it)
  • DRY-RUN by default: builds + (with a key) signs fills, never posts.
    A real post requires ALL of: --arm, TRADING_DRY_RUN=false, SXBET_PRIVATE_KEY,
    a funded SX wallet. Point SXBET_API at https://api.toronto.sx.bet to rehearse.
  • The engine-calibration finding applies: base true_p has run overconfident
    (see the journal). Floors here are deliberately higher than Polymarket's,
    and derivative markets use their own (higher) floors. Rehearse, journal,
    verify the edge is real BEFORE arming.
  • One open position per (fixture, market-type, outcome); ¼-Kelly, capped.

    python -m execution.sx_strategies --once                 # one dry-run scan
    python -m execution.sx_strategies --interval 30          # loop (dry-run)
    python -m execution.sx_strategies --strategies match,set # subset
    python -m execution.sx_strategies --interval 30 --arm    # real (needs key+funds)
"""

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from execution.live_odds import SofascoreOdds                    # noqa: E402
from execution.inplay import InPlayModel                         # noqa: E402
from execution.sxbet import (                                    # noqa: E402
    SXBetClient, MATCH_WINNER_TYPE, SET_WINNER_TYPES,
    GAME_SPREAD_TYPE, TOTAL_GAMES_TYPE, SET_SPREAD_TYPE, TOTAL_SETS_TYPE,
)
from execution.polymarket import _surname                        # noqa: E402
from execution import trade_log                                  # noqa: E402
from execution.pipeline import load_env                          # noqa: E402
from true_p_ensemble import kelly_stake                          # noqa: E402

# ── knobs (env-overridable) ──────────────────────────────────────────────────
MATCH_MIN_EDGE = float(os.getenv("SX_MATCH_MIN_EDGE", "0.04"))
SET_MIN_EDGE = float(os.getenv("SX_SET_MIN_EDGE", "0.06"))
GAME_MIN_EDGE = float(os.getenv("SX_GAME_MIN_EDGE", "0.06"))
MAX_EDGE = float(os.getenv("SX_MAX_EDGE", "0.20"))       # bigger = data problem, not alpha
SIM_N = int(os.getenv("SX_SIM_N", "4000"))
BANKROLL = float(os.getenv("TRADING_BANKROLL", "1000"))
MAX_STAKE = float(os.getenv("TRADING_MAX_STAKE", "25"))
KELLY_CAP = float(os.getenv("TRADING_KELLY_CAP", "0.05"))


# ═════════════════════════════════════════════════════════════════════════════
# Monte-Carlo pricer: every remaining game of a best-of-3 match
# ═════════════════════════════════════════════════════════════════════════════

def hold_prob(p: float) -> float:
    """P(server wins a standard game) from serve point-win prob p (analytic,
    deuce solved in closed form)."""
    p = min(max(p, 0.01), 0.99)
    q = 1.0 - p
    # win to love/15/30
    w = p**4 * (1 + 4 * q + 10 * q * q)
    # reach deuce (3-3 in points), then win from deuce = p²/(p²+q²)
    deuce = 20 * (p**3) * (q**3)
    return w + deuce * (p * p / (p * p + q * q))


def _sim_tiebreak(rng, sp_srv: float, sp_ret: float, first_to: int = 7) -> bool:
    """Simulate a tiebreak point-by-point. True if the player serving FIRST wins."""
    a = b = 0
    serving_first = True   # point 1 by first server, then pairs alternate
    n = 0
    while True:
        p = sp_srv if serving_first else 1.0 - sp_ret  # P(first-server wins this point)
        if rng.random() < p:
            a += 1
        else:
            b += 1
        n += 1
        if (a >= first_to or b >= first_to) and abs(a - b) >= 2:
            return a > b
        if n % 2 == 1:                 # after point 1, swap every 2 points
            serving_first = not serving_first


def parse_state(sc: dict | None) -> dict:
    """Sofascore state → {sets:(a,b), cur:(g1,g2), done_games:int, set_no:int}."""
    sets = (0, 0)
    cur = (0, 0)
    done = 0
    if sc:
        m = re.match(r"(\d+)-(\d+)", sc.get("sets") or "")
        if m:
            sets = (int(m.group(1)), int(m.group(2)))
        parts = (sc.get("games") or "").split()
        pairs = [tuple(int(x) for x in p.split("-")) for p in parts
                 if re.match(r"^\d+-\d+$", p)]
        if pairs:
            cur = pairs[-1]
            done = sum(a + b for a, b in pairs[:-1]) + 0  # completed sets' games
    return {"sets": sets, "cur": cur, "done_games": done,
            "set_no": sets[0] + sets[1] + 1}


def simulate(sp1: float, sp2: float, state: dict, n: int = SIM_N,
             seed: int | None = None) -> dict:
    """Play out the remaining match n times from the current score.

    Returns distributions the SX derivative markets need:
      p1_match                 P(p1 wins the match)
      set_winner[k]            P(p1 wins set k) (only sets not yet finished)
      p_over_games(line)       P(total games in the match > line)
      p_spread_games(line)     P(p1 games + line > p2 games)
      p_three_sets             P(match goes 3 sets)
      p1_two_zero              P(p1 wins 2-0)   (set-spread -1.5)
    """
    rng = random.Random(seed)
    h1, h2 = hold_prob(sp1), hold_prob(sp2)
    totals, spreads = [], []
    p1_wins = 0
    three_sets = 0
    two_zero = two_zero2 = 0
    set_win_counts: dict[int, int] = {}
    set_played_counts: dict[int, int] = {}
    s0 = state["sets"]
    c0 = state["cur"]
    base_games = state["done_games"]

    for _ in range(n):
        sets1, sets2 = s0
        g_tot1 = g_tot2 = 0
        set_no = sets1 + sets2 + 1
        g1, g2 = c0
        # who serves the next game of the live set is unknown from the feed —
        # randomise it (the bias from this washes out over a set)
        p1_serves = rng.random() < 0.5
        while sets1 < 2 and sets2 < 2:
            # play out current set
            while True:
                if g1 >= 6 and g1 - g2 >= 2:
                    winner = 1
                    break
                if g2 >= 6 and g2 - g1 >= 2:
                    winner = 2
                    break
                if g1 == 6 and g2 == 6:
                    first_is_p1 = p1_serves
                    tb_first_wins = _sim_tiebreak(
                        rng, sp1 if first_is_p1 else sp2, sp2 if first_is_p1 else sp1)
                    winner = 1 if (tb_first_wins == first_is_p1) else 2
                    g1, g2 = (7, 6) if winner == 1 else (6, 7)
                    p1_serves = not p1_serves
                    break
                hold = h1 if p1_serves else h2
                srv_won = rng.random() < hold
                if (p1_serves and srv_won) or (not p1_serves and not srv_won):
                    g1 += 1
                else:
                    g2 += 1
                p1_serves = not p1_serves
            set_played_counts[set_no] = set_played_counts.get(set_no, 0) + 1
            if winner == 1:
                set_win_counts[set_no] = set_win_counts.get(set_no, 0) + 1
                sets1 += 1
            else:
                sets2 += 1
            g_tot1 += g1
            g_tot2 += g2
            g1 = g2 = 0
            set_no += 1
        if sets1 == 2:
            p1_wins += 1
            if sets2 == 0 and s0 == (0, 0):
                two_zero += 1
        elif sets1 == 0 and s0 == (0, 0):
            two_zero2 += 1
        if sets1 + sets2 == 3:
            three_sets += 1
        totals.append(base_games + g_tot1 + g_tot2)
        spreads.append(g_tot1 - g_tot2)   # simulated remainder only; base split unknown

    # completed-set games split for the spread: recover from the score string
    def p_over_games(line: float) -> float:
        return sum(1 for t in totals if t > line) / n

    def p_spread_games(line: float, base_diff: int) -> float:
        return sum(1 for s in spreads if (s + base_diff + line) > 0) / n

    return {
        "p1_match": p1_wins / n,
        "set_winner": {k: set_win_counts.get(k, 0) / max(set_played_counts.get(k, 1), 1)
                       for k in set_played_counts},
        "p_over_games": p_over_games,
        "p_spread_games": p_spread_games,
        "p_three_sets": three_sets / n,
        "p1_two_zero": two_zero / n,
        "p2_two_zero": two_zero2 / n,
    }


def base_game_diff(sc: dict | None) -> int:
    """p1 games − p2 games across ALL games already on the board."""
    if not sc:
        return 0
    diff = 0
    for part in (sc.get("games") or "").split():
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            diff += int(m.group(1)) - int(m.group(2))
    return diff


# ═════════════════════════════════════════════════════════════════════════════
# strategy engine
# ═════════════════════════════════════════════════════════════════════════════

def sx_fixtures(sx) -> list:
    """Every fixture SX currently books, as (teamOneName, teamTwoName) full names.

    The VENUE drives the scan, not Sofascore's live list: SX books mostly
    pre-match main-tour matches, so iterating only live fixtures finds nothing
    (measured: 14 SX fixtures with open books, 0 overlap with the live list).
    """
    sx._refresh_markets()
    out, seen = [], set()
    for types in getattr(sx, "_all_markets", {}).values():
        for ms in types.values():
            for m in ms:
                a, b = m.get("teamOneName"), m.get("teamTwoName")
                if not a or not b:
                    continue
                key = frozenset((_surname(a), _surname(b)))
                if key in seen:
                    continue
                seen.add(key)
                out.append((a, b))
    return out


def true_p_for(ip, sofa, p1, p2):
    """(P(p1 wins), sofa_state, is_live) using the live engine when the match is
    in progress, else the same Markov engine priced from 0-0 (pre-match).
    (None, None, False) when we can't price the fixture at all."""
    live_p = ip.live_true_p(p1, p2)
    if live_p is not None:
        return live_p, sofa.state(p1, p2), True
    # pre-match: identical serve inputs, Markov from a clean scoreboard
    try:
        sp1, sp2 = ip._serve_win(p1), ip._serve_win(p2)
        pre = ip.markov.win_prob_from_score(
            sets_p1=0, sets_p2=0, games_p1=0, games_p2=0, p1_serving=True,
            p1_point_win=sp1, p2_point_win=sp2, best_of=3)
        return pre, None, False
    except Exception:
        return None, None, False


def serve_inputs(ip, p1, p2):
    """Serve point-win inputs for the simulator, blended with LIVE in-match serve
    stats exactly as InPlayModel.live_true_p does internally.

    This matters enormously for the GAME tier. Career-average serve carries
    essentially no information about how long a match runs (measured correlation
    between career serve gap and total games: r = +0.02, i.e. nothing), whereas
    the ACTUAL in-match serve gap is the dominant driver (r = -0.67). The live
    blend is the only route to that signal, so passing raw career numbers here
    silently reduced the game markets to noise.

    Returns (sp1, sp2, has_live_signal).
    """
    sp1, sp2 = ip._serve_win(p1), ip._serve_win(p2)
    live = None
    try:
        live = ip.sofa.serve_stats_for(p1, p2)
    except Exception:
        live = None
    if live:
        b1, b2 = live.get("p1"), live.get("p2")
        if b1:
            sp1 = ip._blend(sp1, b1)
        if b2:
            sp2 = ip._blend(sp2, b2)
    return sp1, sp2, bool(live and (live.get("p1") or live.get("p2")))


def _open_positions() -> set:
    return {(r["match_name"], r.get("market_type"), r["outcome"])
            for r in trade_log.all_trades()
            if r.get("venue") == "sxbet" and r.get("status") in ("dry_run", "placed")}


def _stake(true_p: float, price: float) -> float:
    frac, _ = kelly_stake(true_p, decimal_odds=1.0 / price)
    if frac <= 0:
        return 0.0
    return min(round(min(frac, KELLY_CAP) * BANKROLL, 2), MAX_STAKE)


def _fire(sx, market, outcome_one: bool, true_p: float, price: float,
          strategy: str, match_name: str, desc: str, dry_run: bool,
          open_pos: set, notes: list):
    edge = true_p - price
    outcome = market.get("outcomeOneName") if outcome_one else market.get("outcomeTwoName")
    key = (match_name, strategy, outcome)
    if key in open_pos:
        return
    if edge > MAX_EDGE:      # too good to be true = bad inputs
        notes.append(f"  SKIP  {strategy.upper():5s} {desc[:44]:44s} edge +{edge:.3f} > {MAX_EDGE} (suspect)")
        return
    stake = _stake(true_p, price)
    if stake < 1:
        return
    res = sx.place_fill(market, outcome_one, stake,
                        label=f"[{strategy}] {desc}", dry_run=dry_run)
    if res["status"] in ("no_book", "no_price"):
        return
    open_pos.add(key)
    trade_log.record_trade({
        "venue": "sxbet", "match_name": match_name, "market_type": strategy,
        "question": f"SX {strategy.upper()}: {desc}",
        "outcome": outcome, "side": "outcome1" if outcome_one else "outcome2",
        "true_p": round(true_p, 4), "market_price": round(price, 4),
        "edge": round(edge, 4), "stake_usd": stake,
        "status": res["status"], "detail": res.get("detail", "")[:300],
    })
    notes.append(f"  {strategy.upper():5s} {desc[:44]:44s} @ {price:.3f} "
                 f"trueP {true_p:.3f} edge +{edge:.3f} ${stake:.0f} [{res['status']}]")


def _price_for(sx, market) -> tuple | None:
    """(back1, back2) taker prices, sane, else None."""
    q = sx.taker_prices(market)
    if not q:
        return None
    b1, b2 = q.get("back1"), q.get("back2")
    if not b1 or not b2 or not 0.02 < b1 < 0.98 or not 0.02 < b2 < 0.98:
        return None
    return b1, b2


def scan_fixture(sx, ip, sofa, p1, p2, strategies, dry_run, open_pos, notes):
    mkts = sx.markets_for(p1, p2)
    if not mkts:
        return
    match_name = f"{p1} vs {p2}"
    true_p1, sc, _is_live = true_p_for(ip, sofa, p1, p2)
    if true_p1 is None:
        return
    state = parse_state(sc)

    # orientation: SX teamOne may be either player
    any_m = next(iter(next(iter(mkts.values()))))
    one_is_p1 = _surname(any_m.get("teamOneName", "")) == _surname(p1)

    # ── MATCH: engine True P vs SX moneyline ─────────────────────────────────
    if "match" in strategies:
        for m in mkts.get(MATCH_WINNER_TYPE, []):
            pr = _price_for(sx, m)
            if not pr:
                continue
            t_one = true_p1 if one_is_p1 else 1.0 - true_p1
            for outcome_one, tp, price in ((True, t_one, pr[0]), (False, 1.0 - t_one, pr[1])):
                if tp - price >= MATCH_MIN_EDGE:
                    who = m["outcomeOneName"] if outcome_one else m["outcomeTwoName"]
                    _fire(sx, m, outcome_one, tp, price, "match", match_name,
                          f"back {who}", dry_run, open_pos, notes)

    needs_sim = ("set" in strategies) or ("game" in strategies)
    if not needs_sim:
        return
    sp1, sp2, has_live_serve = serve_inputs(ip, p1, p2)
    sim = simulate(sp1, sp2, state)
    # anchor the sim's match prob to the engine's (momentum-aware) True P so
    # derivative prices inherit the same view of who's better
    shift = true_p1 - sim["p1_match"]

    # ── SET: set winner / set spread / total sets ────────────────────────────
    if "set" in strategies:
        for t, set_no in SET_WINNER_TYPES.items():
            for m in mkts.get(t, []):
                if set_no < state["set_no"]:
                    continue                       # that set is already done
                psim = sim["set_winner"].get(set_no)
                if psim is None:
                    continue
                pr = _price_for(sx, m)
                if not pr:
                    continue
                p_one = (psim if one_is_p1 else 1.0 - psim) + shift * (1 if one_is_p1 else -1) * 0.5
                p_one = min(max(p_one, 0.02), 0.98)
                for outcome_one, tp, price in ((True, p_one, pr[0]), (False, 1.0 - p_one, pr[1])):
                    if tp - price >= SET_MIN_EDGE:
                        who = m["outcomeOneName"] if outcome_one else m["outcomeTwoName"]
                        _fire(sx, m, outcome_one, tp, price, "set", match_name,
                              f"set {set_no}: {who}", dry_run, open_pos, notes)
        for m in mkts.get(TOTAL_SETS_TYPE, []):
            pr = _price_for(sx, m)
            if not pr:
                continue
            p_over = sim["p_three_sets"]           # over 2.5 sets = 3 sets played
            for outcome_one, tp, price in ((True, p_over, pr[0]), (False, 1.0 - p_over, pr[1])):
                if tp - price >= SET_MIN_EDGE:
                    who = m["outcomeOneName"] if outcome_one else m["outcomeTwoName"]
                    _fire(sx, m, outcome_one, tp, price, "set", match_name,
                          f"total sets: {who}", dry_run, open_pos, notes)
        for m in mkts.get(SET_SPREAD_TYPE, []):
            if state["sets"] != (0, 0):
                continue                            # -1.5 sets only clean pre 1st-set end
            pr = _price_for(sx, m)
            if not pr:
                continue
            p20 = sim["p1_two_zero"] if one_is_p1 else sim["p2_two_zero"]
            p_one = min(max(p20, 0.02), 0.98)
            if p_one - pr[0] >= SET_MIN_EDGE:
                _fire(sx, m, True, p_one, pr[0], "set", match_name,
                      f"{m['outcomeOneName']}", dry_run, open_pos, notes)

    # ── GAME: game spread + total games ──────────────────────────────────────
    # Gated on LIVE serve data. Pre-match, the only available serve estimate is
    # the career average, whose gap correlates r=+0.02 with total games — no
    # signal at all, which is why the pre-match game tier backtested worse than
    # a constant. Fed the true in-match serve split the same simulator scores
    # Brier 0.175 vs a 0.249 baseline, so the market is only worth trading once
    # live serve stats have started revealing that split.
    if "game" in strategies and has_live_serve:
        bdiff = base_game_diff(sc)
        for m in mkts.get(GAME_SPREAD_TYPE, []):
            line = m.get("line")
            if line is None:
                continue
            pr = _price_for(sx, m)
            if not pr:
                continue
            d = bdiff if one_is_p1 else -bdiff
            p_cover = (sim["p_spread_games"](float(line), d) if one_is_p1
                       else 1.0 - sim["p_spread_games"](-float(line), bdiff))
            p_cover = min(max(p_cover, 0.02), 0.98)
            for outcome_one, tp, price in ((True, p_cover, pr[0]), (False, 1.0 - p_cover, pr[1])):
                if tp - price >= GAME_MIN_EDGE:
                    who = m["outcomeOneName"] if outcome_one else m["outcomeTwoName"]
                    _fire(sx, m, outcome_one, tp, price, "game", match_name,
                          f"spread: {who}", dry_run, open_pos, notes)
        for m in mkts.get(TOTAL_GAMES_TYPE, []):
            line = m.get("line")
            if line is None:
                continue
            pr = _price_for(sx, m)
            if not pr:
                continue
            p_over = min(max(sim["p_over_games"](float(line)), 0.02), 0.98)
            for outcome_one, tp, price in ((True, p_over, pr[0]), (False, 1.0 - p_over, pr[1])):
                if tp - price >= GAME_MIN_EDGE:
                    who = m["outcomeOneName"] if outcome_one else m["outcomeTwoName"]
                    _fire(sx, m, outcome_one, tp, price, "game", match_name,
                          f"games {line}: {who}", dry_run, open_pos, notes)


def scan_once(sx, ip, sofa, strategies, dry_run) -> list[str]:
    notes: list[str] = []
    open_pos = _open_positions()
    for p1, p2 in sx_fixtures(sx):
        try:
            scan_fixture(sx, ip, sofa, p1, p2, strategies, dry_run, open_pos, notes)
        except Exception as e:                      # one bad fixture must not kill the loop
            notes.append(f"  ERR   {p1} vs {p2}: {str(e)[:80]}")
    return notes


def main():
    load_env()
    ap = argparse.ArgumentParser(description="SX three-tier strategies (dry-run by default)")
    ap.add_argument("--once", action="store_true", help="single scan then exit")
    ap.add_argument("--interval", type=int, default=30, help="loop seconds")
    ap.add_argument("--strategies", default="match,set",
                    help="comma list from: match,set,game. GAME is OFF by default: "
                         "the walk-forward backtest scored it WORSE than a constant "
                         "(Brier 0.2680 vs 0.2466 base rate over 6,000 matches) — "
                         "run `python -m execution.sx_backtest` before enabling it")
    ap.add_argument("--arm", action="store_true",
                    help="allow REAL posts (still needs SXBET_PRIVATE_KEY + TRADING_DRY_RUN=false)")
    args = ap.parse_args()
    strategies = {s.strip() for s in args.strategies.split(",") if s.strip()}

    env_dry = os.getenv("TRADING_DRY_RUN", "true").lower() == "true"
    dry_run = env_dry or not args.arm
    sx = SXBetClient()
    live_ok = (not dry_run) and sx.can_trade_live
    mode = "LIVE POST" if live_ok else ("DRY-RUN (armed, no key/funds)" if args.arm else "DRY-RUN")
    print("=" * 70)
    print(f"SX STRATEGIES [{','.join(sorted(strategies))}] — {mode}   "
          f"endpoint={os.getenv('SXBET_API', 'https://api.sx.bet')}")
    print(f"edges: match>={MATCH_MIN_EDGE} set>={SET_MIN_EDGE} game>={GAME_MIN_EDGE} "
          f"(suspect>{MAX_EDGE})  sims={SIM_N}  bankroll=${BANKROLL:.0f} max=${MAX_STAKE:.0f}")
    if live_ok:
        print("⚠ LIVE POSTING ENABLED — real fills will be sent to SX.")
    print("=" * 70)

    sofa = SofascoreOdds()
    ip = InPlayModel(sofa=sofa)
    try:
        while True:
            t0 = time.time()
            notes = scan_once(sx, ip, sofa, strategies, dry_run)
            stamp = time.strftime("%H:%M:%S")
            if notes:
                print(f"[{stamp}] {len(notes)} action(s):")
                print("\n".join(notes))
            else:
                print(f"[{stamp}] no edge cleared a floor on any live SX market")
            if args.once:
                break
            time.sleep(max(2, args.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        ip.close()


if __name__ == "__main__":
    main()
