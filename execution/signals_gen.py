"""Generate model signals for currently-open Polymarket tennis fixtures.

For every open singles fixture that has a tradeable match/set market AND whose
players both exist in the model DB, emit one signal per available market
(match + any set markets), backing the model-favored side with its true_p.

    python -m execution.signals_gen                 # -> signals.auto.json
    python -m execution.signals_gen --out my.json --min-edge 0.03

The downstream pipeline still applies the Kelly / min-edge gate, so emitting a
signal is only a candidate, not a guaranteed bet.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from execution.pipeline import load_env  # noqa: E402
from execution.polymarket import PolymarketClient  # noqa: E402
from execution.model_predict import MatchModel  # noqa: E402
from execution.live_odds import SofascoreOdds  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "signals.auto.json"

# Market types the downstream pipeline accepts.
VALID_MARKETS = {"match", "set1", "set2", "set3"}


def _fallback_signal(p1: str, p2: str, markets, surface: str,
                     edge: float, min_fav: float) -> dict | None:
    """Back the match-market favorite by its live price (no model). None if no clear fav.

    No real predictive edge — price/momentum trading, only sensible in-play
    (the downstream live gate ensures these only fire on started matches).
    """
    mm_market = next((m for m in markets if m.market_type == "match"), None)
    if mm_market is None or len(mm_market.prices) != 2:
        return None
    pr = mm_market.prices
    if not all(0.0 < x < 1.0 for x in pr):
        return None
    fav_idx = 0 if pr[0] >= pr[1] else 1
    if pr[fav_idx] < min_fav:
        return None  # no clear favorite — skip coin-flips
    side = "player1" if mm_market.side_index(p1, p2, "player1") == fav_idx else "player2"
    tp = max(0.02, min(0.98, pr[fav_idx] + edge))
    return {"player1": p1, "player2": p2, "market": "match", "side": side,
            "true_p": round(tp, 4), "_surface": surface, "_source": "market"}

# Tournament-name keywords -> court surface. Default Hard.
_CLAY = ("roland garros", "french", "bastad", "gstaad", "hamburg", "kitzbuhel",
         "umag", "bucharest", "geneva", "madrid", "rome", "monte carlo",
         "monte-carlo", "barcelona", "munich", "estoril", "houston",
         "contrexeville", "iasi", "braunschweig", "liege", "kzoo")
_GRASS = ("wimbledon", "newport", "mallorca", "halle", "queen", "eastbourne",
          "nottingham", "hertogenbosch", "bad homburg", "birmingham")
_GRAND_SLAM = ("wimbledon", "roland garros", "french open", "us open",
               "australian open")
_MASTERS = ("masters", "atp 1000", "wta 1000", "indian wells", "miami open",
            "cincinnati", "canadian open", "shanghai", "paris masters")


def _surface(title: str) -> str:
    t = title.lower()
    if any(k in t for k in _GRASS):
        return "Grass"
    if any(k in t for k in _CLAY):
        return "Clay"
    return "Hard"


def _level(title: str) -> str:
    t = title.lower()
    if any(k in t for k in _GRAND_SLAM):
        return "G"
    if any(k in t for k in _MASTERS):
        return "M"
    return ""


def _split_fixture(title: str):
    """('Tournament: A vs B') -> ('A', 'B') or None. Skips doubles."""
    name = title.split(":", 1)[1].strip() if ":" in title else title
    if "Doubles" in title or "/" in name:
        return None
    if " vs " not in name:
        return None
    a, b = [x.strip() for x in name.split(" vs ", 1)]
    return (a, b) if a and b else None


def generate(min_edge: float = 0.0, verbose: bool = True) -> list[dict]:
    load_env()
    market_fallback = os.getenv("TRADING_MARKET_FALLBACK", "false").lower() == "true"
    fallback_edge = float(os.getenv("TRADING_FALLBACK_EDGE", "0.05"))
    fallback_min_fav = float(os.getenv("TRADING_FALLBACK_MIN_FAV", "0.55"))
    use_live_odds = os.getenv("TRADING_LIVE_ODDS", "true").lower() == "true"
    use_inplay = os.getenv("TRADING_INPLAY", "true").lower() == "true"
    # Only bet an in-play signal when it disagrees with SX's sharp line by at
    # least this (edge we can see that the market doesn't). 0 = gate off.
    inplay_min_disagree = float(os.getenv("TRADING_INPLAY_MIN_DISAGREE", "0"))
    odds = SofascoreOdds() if use_live_odds else None
    client = PolymarketClient()
    mm = MatchModel()
    ip = None
    if use_inplay:
        from execution.inplay import InPlayModel
        ip = InPlayModel(sofa=odds, mm=mm)
    sx_gate = None
    if inplay_min_disagree > 0:
        from execution.sxbet import SXBetClient
        sx_gate = SXBetClient()
    events = client.fetch_tennis_events()
    if verbose:
        print(f"Scanning {len(events)} open tennis events...")

    signals, seen = [], set()
    n_fixtures = n_priced = n_unknown = n_fallback = n_odds = n_inplay = 0
    for event in events:
        title = event.get("title", "")
        pair = _split_fixture(title)
        if not pair:
            continue
        p1, p2 = pair
        if (p1, p2) in seen:
            continue
        seen.add((p1, p2))
        n_fixtures += 1

        markets = client.find_match_markets(p1, p2, events)
        if not markets:
            continue
        surface, level = _surface(title), _level(title)

        # Live in-play engine first: a score-aware true_p for matches in progress.
        inplay_p = ip.live_true_p(p1, p2, surface) if ip is not None else None
        if inplay_p is not None:
            mom_tag = ip.momentum_tag()   # journal the live momentum behind this bet
            # Disagreement gate: skip if our engine agrees with SX's sharp line
            # (no edge the market hasn't already priced).
            if sx_gate is not None:
                sq = sx_gate.quote(p1, p2)
                if sq and abs(inplay_p - sq["fair_p1"]) < inplay_min_disagree:
                    continue
            side = "player1" if inplay_p >= 0.5 else "player2"
            p_side = inplay_p if side == "player1" else 1.0 - inplay_p
            for m in markets:
                if m.market_type not in VALID_MARKETS:
                    continue
                tp = p_side if m.market_type == "match" else mm.set_prob(p_side, 3)
                signals.append({"player1": p1, "player2": p2,
                                "market": m.market_type, "side": side,
                                "true_p": round(max(0.02, min(0.98, tp)), 4),
                                "_surface": surface, "_source": "inplay",
                                "_momentum": mom_tag})
            n_inplay += 1
            if verbose:
                fav = p1 if side == "player1" else p2
                print(f"  {p1} vs {p2} [{surface}] -> {fav} IN-PLAY true_p {p_side:.3f}")
            continue

        p_match = mm.predict_match(p1, p2, surface, level)
        if p_match is None:
            n_unknown += 1
            # Preferred: de-vigged live odds (a real fair probability -> a real
            # edge vs the Polymarket price). Match market only.
            fp = odds.fair_prob(p1, p2) if odds is not None else None
            if fp is not None and any(m.market_type == "match" for m in markets):
                side = "player1" if fp >= 0.5 else "player2"
                tp = max(0.02, min(0.98, fp if side == "player1" else 1.0 - fp))
                signals.append({"player1": p1, "player2": p2, "market": "match",
                                "side": side, "true_p": round(tp, 4),
                                "_surface": surface, "_source": "sofascore"})
                n_odds += 1
                if verbose:
                    fav = p1 if side == "player1" else p2
                    print(f"  {p1} vs {p2} [{surface}] -> {fav} LIVE-ODDS fair {tp:.3f}")
                continue
            # Last resort: back the market favorite (no real edge).
            if market_fallback:
                fb = _fallback_signal(p1, p2, markets, surface,
                                      fallback_edge, fallback_min_fav)
                if fb:
                    signals.append(fb)
                    n_fallback += 1
                    if verbose:
                        fav = p1 if fb["side"] == "player1" else p2
                        print(f"  {p1} vs {p2} [{surface}] -> {fav} "
                              f"MARKET-fallback true_p {fb['true_p']:.3f}")
            continue
        n_priced += 1

        # Favored side in the signal frame; true_p is that side's win prob.
        side = "player1" if p_match >= 0.5 else "player2"
        p_side_match = p_match if side == "player1" else 1.0 - p_match

        for m in markets:
            if m.market_type not in VALID_MARKETS:
                continue
            if m.market_type == "match":
                tp = p_side_match
            else:  # set market — single-set win prob for the same side
                tp = mm.set_prob(p_side_match, best_of=3)
            tp = max(0.02, min(0.98, tp))
            signals.append({
                "player1": p1, "player2": p2,
                "market": m.market_type, "side": side,
                "true_p": round(tp, 4),
                "_surface": surface, "_level": level or "ATP/WTA",
            })
        if verbose:
            fav = p1 if side == "player1" else p2
            print(f"  {p1} vs {p2} [{surface}] -> {fav} "
                  f"match {p_side_match:.3f} ({len(markets)} mkts)")

    mm.close()
    if verbose:
        print(f"\n{n_fixtures} singles fixtures | {n_inplay} in-play | {n_priced} model | "
              f"{n_odds} live-odds | {n_fallback} market-fallback | "
              f"{n_unknown} unknown | {len(signals)} signals")
    return signals


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate model signals for open fixtures")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output signals JSON path")
    ap.add_argument("--min-edge", type=float, default=0.0,
                    help="(reserved) minimum edge; pipeline enforces the real gate")
    args = ap.parse_args()
    signals = generate(min_edge=args.min_edge)
    Path(args.out).write_text(json.dumps(signals, indent=2))
    print(f"Wrote {len(signals)} signals -> {args.out}")


if __name__ == "__main__":
    main()
