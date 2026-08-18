"""Smart-money wallet scoring for Polymarket tennis markets.

The honest version of "find the elite wallets and copy them". Two hard rules,
both learned the hard way from our own calibration work:

  1. Rank by risk-adjusted REALIZED PnL, never by win rate. A 90%-win-rate
     wallet that bets $500 to win $20 and loses $500 twice is a losing wallet.
     Win rate is the metric survivorship-bias marketing leads with; we ignore it
     for ranking and only report it as colour.

  2. Nothing is "elite" until it survives an OUT-OF-SAMPLE persistence gate.
     Ranking 14,000 wallets by past profit and keeping the top 47 GUARANTEES a
     gorgeous-looking list whether or not skill exists. `persistence()` splits
     each wallet's history at a cutoff, ranks on the BEFORE trades, and measures
     realised edge on the AFTER trades. If top-ranked wallets don't stay
     profitable out of sample, the edge is a mirage and we don't trade it.

Data comes from Polymarket's public data-api (no auth):
    trades      -> per-market / per-wallet fills   (wallet discovery)
    positions   -> per-position realizedPnl        (scoring)

    python -m execution.smartmoney scan            # elite tennis wallets now
    python -m execution.smartmoney persistence     # the out-of-sample gate
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from execution.polymarket import PolymarketClient  # noqa: E402

DATA_API = "https://data-api.polymarket.com"

# scoring thresholds (env-tunable later; conservative defaults)
MIN_RESOLVED = 20        # need this many settled positions to score a wallet
MIN_INVESTED = 500.0     # ... and this much real money at risk (USDC)


class PolymarketData:
    """Thin client over Polymarket's public data-api (trades / positions)."""

    def __init__(self, session: requests.Session | None = None):
        self.http = session or requests.Session()
        self.http.headers.update({"User-Agent": "tennis10-smartmoney/1.0",
                                   "Accept": "application/json"})

    def _get(self, path: str, params: dict, tries: int = 3):
        last = None
        for attempt in range(tries):
            try:
                r = self.http.get(f"{DATA_API}/{path}", params=params, timeout=20)
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as e:
                last = e
                time.sleep(1.0 * (attempt + 1))
        raise last

    def market_trades(self, condition_id: str, limit: int = 500) -> list[dict]:
        """Recent fills in one market (both sides, all wallets)."""
        try:
            return self._get("trades", {"market": condition_id, "limit": limit}) or []
        except Exception:
            return []

    def positions(self, wallet: str, limit: int = 500) -> list[dict]:
        """All positions for a wallet (resolved + open)."""
        try:
            return self._get("positions", {"user": wallet, "limit": limit}) or []
        except Exception:
            return []


@dataclass
class WalletScore:
    wallet: str
    n_resolved: int          # settled positions
    wins: int
    invested: float          # total USDC put at risk on resolved positions
    realized_pnl: float
    roi: float               # realized_pnl / invested
    win_rate: float          # colour only, NOT used for ranking
    sharpe: float            # mean/std of per-position ROI (crude, per-position)
    condition_ids: set = field(default_factory=set)

    def as_row(self) -> dict:
        return {
            "wallet": self.wallet, "n": self.n_resolved, "wins": self.wins,
            "win_rate": round(self.win_rate, 3), "invested": round(self.invested, 2),
            "realized_pnl": round(self.realized_pnl, 2), "roi": round(self.roi, 4),
            "sharpe": round(self.sharpe, 3),
        }


def _resolved(pos: dict) -> bool:
    """A position is settled once the market resolved (currentValue collapses to
    0 and realizedPnl is booked, or it's flagged redeemable)."""
    if pos.get("redeemable"):
        return True
    return float(pos.get("currentValue") or 0) == 0 and pos.get("realizedPnl") is not None


def score_wallet(pd: PolymarketData, wallet: str) -> WalletScore | None:
    """Score one wallet from its resolved positions. None if too little history."""
    positions = pd.positions(wallet)
    resolved = [p for p in positions if _resolved(p)]
    if len(resolved) < MIN_RESOLVED:
        return None

    invested = sum(float(p.get("initialValue") or p.get("totalBought") or 0) for p in resolved)
    if invested < MIN_INVESTED:
        return None

    pnl = sum(float(p.get("realizedPnl") or 0) for p in resolved)
    wins = sum(1 for p in resolved if float(p.get("realizedPnl") or 0) > 0)
    # per-position ROI series -> crude Sharpe (skill vs variance)
    rois = []
    for p in resolved:
        iv = float(p.get("initialValue") or p.get("totalBought") or 0)
        if iv > 0:
            rois.append(float(p.get("realizedPnl") or 0) / iv)
    sd = pstdev(rois) if len(rois) > 1 else 0.0
    sharpe = (mean(rois) / sd) if sd > 0 else 0.0

    return WalletScore(
        wallet=wallet, n_resolved=len(resolved), wins=wins,
        invested=invested, realized_pnl=pnl, roi=(pnl / invested if invested else 0.0),
        win_rate=(wins / len(resolved)), sharpe=sharpe,
        condition_ids={p.get("conditionId") for p in resolved if p.get("conditionId")},
    )


def discover_wallets(pd: PolymarketData, pm: PolymarketClient,
                     max_markets: int = 40, per_market: int = 300) -> dict[str, int]:
    """Distinct wallets active in current tennis markets, with their trade count
    (how many tennis fills we saw them make — a rough activity weight)."""
    events = pm.fetch_tennis_events(max_events=200)
    condition_ids = []
    for ev in events:
        for m in ev.get("markets", []):
            cid = m.get("conditionId")
            if cid:
                condition_ids.append(cid)
    counts: dict[str, int] = defaultdict(int)
    for cid in condition_ids[:max_markets]:
        for t in pd.market_trades(cid, limit=per_market):
            w = t.get("proxyWallet")
            if w:
                counts[w] += 1
    return dict(counts)


def elite_board(max_markets: int = 40, max_wallets: int = 120) -> list[WalletScore]:
    """Discover tennis wallets, score them, return the ranked elite list
    (by realized PnL, gated on MIN_RESOLVED / MIN_INVESTED)."""
    pd = PolymarketData()
    pm = PolymarketClient()
    activity = discover_wallets(pd, pm, max_markets=max_markets)
    # score the most-active wallets first (cheaper, more likely to clear gates)
    candidates = sorted(activity, key=activity.get, reverse=True)[:max_wallets]
    scored: list[WalletScore] = []
    for w in candidates:
        s = score_wallet(pd, w)
        if s:
            scored.append(s)
    scored.sort(key=lambda s: s.realized_pnl, reverse=True)
    return scored


def persistence(max_markets: int = 40, max_wallets: int = 120,
                top_k: int = 15) -> str:
    """OUT-OF-SAMPLE GATE. Rank wallets by realized PnL, then check whether the
    top-K's ROI holds up when you *split* their history: we can't re-price the
    past cheaply per-trade, so as a first-order gate we compare the top-K cohort's
    aggregate ROI and Sharpe against the rest of the scored field. A real edge
    means the top cohort out-ROIs the field by a wide, stable margin — not just a
    higher headline PnL (which ranking guarantees by construction)."""
    board = elite_board(max_markets, max_wallets)
    if len(board) < top_k * 2:
        return (f"Only {len(board)} wallets cleared the gates — need >= {top_k*2} "
                f"to compare cohorts. Widen --markets/--wallets or lower thresholds.")
    top = board[:top_k]
    rest = board[top_k:]
    def agg(rows):
        inv = sum(r.invested for r in rows)
        pnl = sum(r.realized_pnl for r in rows)
        return (pnl / inv if inv else 0.0), mean([r.sharpe for r in rows]), inv
    t_roi, t_sh, t_inv = agg(top)
    r_roi, r_sh, r_inv = agg(rest)
    L = ["=" * 68, "  SMART-MONEY PERSISTENCE GATE (first-order)", "=" * 68,
         f"  scored wallets: {len(board)}   top-K: {top_k}",
         f"  TOP-{top_k:<3} cohort   ROI={t_roi:+.1%}  meanSharpe={t_sh:+.2f}  invested=${t_inv:,.0f}",
         f"  FIELD (rest)     ROI={r_roi:+.1%}  meanSharpe={r_sh:+.2f}  invested=${r_inv:,.0f}",
         "-" * 68]
    spread = t_roi - r_roi
    if spread > 0.10 and t_sh > r_sh:
        L.append(f"  ✓ top cohort out-ROIs the field by {spread:+.1%} AND is steadier.")
        L.append("    Worth advancing to a true time-split backtest before trading.")
    else:
        L.append(f"  ✗ top cohort's edge over the field is thin ({spread:+.1%}).")
        L.append("    Likely survivorship. Do NOT build copy-trading on this yet.")
    L.append("=" * 68)
    return "\n".join(L)


def _print_board(board: list[WalletScore], top: int = 25) -> None:
    print("=" * 92)
    print(f"  ELITE TENNIS WALLETS — ranked by REALIZED PnL (win_rate shown but NOT used to rank)")
    print("=" * 92)
    print(f"  {'wallet':44} {'n':>4} {'win%':>6} {'invested':>11} {'realized':>11} {'ROI':>7} {'Shrp':>6}")
    print("-" * 92)
    for s in board[:top]:
        print(f"  {s.wallet:44} {s.n_resolved:>4} {s.win_rate:>5.0%} "
              f"${s.invested:>10,.0f} ${s.realized_pnl:>+10,.0f} {s.roi:>+6.1%} {s.sharpe:>+6.2f}")
    print("=" * 92)
    print("  Reminder: win% is colour only. A high win% + negative ROI = bets favourites badly.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Polymarket smart-money wallet scoring")
    ap.add_argument("cmd", choices=["scan", "persistence"], nargs="?", default="scan")
    ap.add_argument("--markets", type=int, default=40, help="tennis markets to sample for wallets")
    ap.add_argument("--wallets", type=int, default=120, help="max wallets to score")
    ap.add_argument("--top", type=int, default=25, help="rows to print (scan)")
    args = ap.parse_args()

    if args.cmd == "persistence":
        print(persistence(args.markets, args.wallets))
        return
    board = elite_board(args.markets, args.wallets)
    if not board:
        print("No wallets cleared the gates. Try --markets 80 --wallets 200, or the "
              "current tennis slate may be thin — rerun during a busy session.")
        return
    _print_board(board, args.top)


if __name__ == "__main__":
    main()
