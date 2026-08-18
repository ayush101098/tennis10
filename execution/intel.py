"""Manual betting-intelligence table for every LIVE tennis fixture on Polymarket.

For each in-play singles fixture (ITF included) it joins:
  - Polymarket live prices (best ask per side)
  - model probability (if both players are in the model DB)
  - Sofascore de-vigged fair probability (if the match is live there)
and computes the edge + a suggested side. Read-only decision support — it does
NOT place bets (that's the agent). Used by the /intel dashboard page.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from execution.pipeline import load_env  # noqa: E402
from execution.polymarket import PolymarketClient  # noqa: E402
from execution.model_predict import MatchModel  # noqa: E402
from execution.live_odds import SofascoreOdds  # noqa: E402
from execution.sxbet import SXBetClient  # noqa: E402
from execution.inplay import InPlayModel  # noqa: E402
from execution.signals_gen import _split_fixture, _surface, _level  # noqa: E402
from true_p_ensemble import kelly_stake  # noqa: E402


def _score_contradicts(score: dict | None, backing_p1: bool) -> bool:
    """True if the live scoreboard shows the backed side trailing (odds likely stale)."""
    if not score:
        return False
    try:
        s1, s2 = (int(x) for x in score["sets"].split("-"))
    except (ValueError, KeyError):
        return False
    games = (score.get("games") or "").split()
    g1 = g2 = 0
    if games:
        try:
            g1, g2 = (int(x) for x in games[-1].split("-"))
        except ValueError:
            pass
    # trailing for player1?
    p1_trailing = s1 < s2 or (s1 == s2 and g1 <= g2 - 2)
    p2_trailing = s2 < s1 or (s1 == s2 and g2 <= g1 - 2)
    return p1_trailing if backing_p1 else p2_trailing


def compute_intel() -> list[dict]:
    load_env()
    bankroll = float(os.getenv("TRADING_BANKROLL", "1000"))
    max_stake = float(os.getenv("TRADING_MAX_STAKE", "25"))
    client = PolymarketClient()
    mm = MatchModel()
    odds = SofascoreOdds()
    odds.refresh(force=True)
    sx = SXBetClient()
    ip = InPlayModel(sofa=odds, mm=mm)   # share the Sofascore feed + model DB
    events = client.fetch_tennis_events()

    # Gather live singles fixtures with a match market.
    fixtures, seen = [], set()
    for e in events:
        title = e.get("title", "")
        pair = _split_fixture(title)
        if not pair or pair in seen:
            continue
        seen.add(pair)
        p1, p2 = pair
        markets = client.find_match_markets(p1, p2, events)
        match_mkt = next((m for m in markets
                          if m.market_type == "match" and m.is_live()), None)
        if match_mkt is None:
            continue
        fixtures.append((p1, p2, title, match_mkt))

    # Live asks for both outcomes, concurrently.
    def _prices(fx):
        p1, p2, title, m = fx
        asks = {}
        for tok in m.token_ids:
            asks[tok] = client.best_ask(tok)
        return fx, asks

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        priced = list(ex.map(_prices, fixtures))

    for (p1, p2, title, m), asks in priced:
        i1 = m.side_index(p1, p2, "player1")
        i2 = m.side_index(p1, p2, "player2")
        if i1 is None or i2 is None:
            continue
        pm1 = asks.get(m.token_ids[i1]) or (m.prices[i1] if i1 < len(m.prices) else None)
        pm2 = asks.get(m.token_ids[i2]) or (m.prices[i2] if i2 < len(m.prices) else None)
        surface = _surface(title)
        model_p1 = mm.predict_match(p1, p2, surface, _level(title))
        fair_p1 = odds.fair_prob(p1, p2)          # Sofascore (pre-match line)
        score = odds.state(p1, p2)
        sxq = sx.quote(p1, p2)                     # SX Bet (live exchange, de-vigged)
        sx_p1 = sxq["fair_p1"] if sxq else None
        inplay_p1 = ip.live_true_p(p1, p2, surface)  # our live score-aware engine
        momentum = ip.last_momentum() if inplay_p1 is not None else None

        # Live break watch: P(returner breaks the CURRENT service game).
        break_info = None
        try:
            from execution.momentum import break_prob_from_score
            games = odds.game_sequence(p1, p2)
            cur = games[-1] if games else None
            if cur and cur["winner"] is None and cur["server"] in (1, 2):
                pts = cur["points"]
                a, b = (pts[-1] if pts else ("0", "0"))
                a, b = (a or "0"), (b or "0")
                srv_pts, ret_pts = (a, b) if cur["server"] == 1 else (b, a)
                server_name = p1 if cur["server"] == 1 else p2
                returner = p2 if cur["server"] == 1 else p1
                bp = break_prob_from_score(ip._serve_win(server_name), srv_pts, ret_pts)
                break_info = {"prob": round(bp, 3), "returner": returner,
                              "server": server_name, "pts": f"{srv_pts}-{ret_pts}"}
        except Exception:
            break_info = None

        # Tier 1 rally style per player (first-strike vs grinder, shot quality).
        tour_guess = "WTA" if "WTA" in title.upper() else "ATP"

        def _rally(name):
            if ip.rally is None:
                return None
            pr = ip.rally.profile(name, datetime.utcnow(), tour_guess)
            return {"fs": round(pr.first_strike_index, 3),
                    "short": round(pr.short_win_pct, 3),
                    "long": round(pr.long_win_pct, 3),
                    "aggr": round(pr.aggression, 3),
                    "n": pr.n_points, "has": pr.has_data}

        # Preferred signal: our live in-play engine (score-aware, independent) >
        # SX Bet live exchange fair > Sofascore pre-match line > internal model.
        if inplay_p1 is not None:
            sig_p1, source = inplay_p1, "inplay"
        elif sx_p1 is not None:
            sig_p1, source = sx_p1, "sxbet"
        elif fair_p1 is not None:
            sig_p1, source = fair_p1, "sofascore"
        elif model_p1 is not None:
            sig_p1, source = model_p1, "model"
        else:
            sig_p1, source = None, None

        # Only suggest against a sane, tradeable price. Near-0/near-1 asks are
        # empty/collapsed books (often a near-resolved match) and pairing them
        # with a pre-match model prob yields fake edges.
        def _sane(px):
            return px is not None and 0.03 < px < 0.97

        suggest = None
        if sig_p1 is not None:
            e1 = (sig_p1 - pm1) if _sane(pm1) else None
            e2 = ((1.0 - sig_p1) - pm2) if _sane(pm2) else None
            best = max([(e, s, px, pr) for e, s, px, pr in
                        [(e1, p1, pm1, sig_p1), (e2, p2, pm2, 1.0 - sig_p1)]
                        if e is not None and e > 0], default=None)
            if best:
                e, s, px, pr = best
                frac, _ = kelly_stake(pr, decimal_odds=1.0 / px)
                stake = round(min(frac * bankroll, max_stake), 2) if frac > 0 else 0.0
                # Live sources (in-play engine, SX exchange) already reflect the
                # score. Only pre-match sources (Sofascore line / model) get the
                # stale guard: flag when the scoreboard contradicts the pick.
                stale = source not in ("inplay", "sxbet") and _score_contradicts(score, s == p1)
                suggest = {"side": s, "edge": round(e, 3), "price": px,
                           "prob": round(pr, 3), "stake": stake, "stale": stale}

        rows.append({
            "match": f"{p1} vs {p2}", "tournament": title.split(":")[0],
            "surface": surface, "p1": p1, "p2": p2,
            "pm1": pm1, "pm2": pm2,
            "model_p1": None if model_p1 is None else round(model_p1, 3),
            "fair_p1": None if fair_p1 is None else round(fair_p1, 3),
            "inplay_p1": None if inplay_p1 is None else round(inplay_p1, 3),
            "sx_p1": None if sx_p1 is None else round(sx_p1, 3),
            "sx_score": sxq["score"] if sxq else None,
            "source": source,
            "score": score,
            "suggest": suggest,
            "momentum": momentum,
            "rally_p1": _rally(p1),
            "rally_p2": _rally(p2),
            "break": break_info,
        })

    mm.close()
    # Best edges first; unsuggested rows after.
    rows.sort(key=lambda r: (r["suggest"] or {}).get("edge", -1), reverse=True)
    return rows
