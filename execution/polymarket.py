"""Polymarket integration: market discovery, live pricing, and order placement.

Discovery uses the Gamma API (public, no auth). Pricing uses the CLOB public
endpoints. Order placement uses py-clob-client and requires:

    POLYMARKET_PRIVATE_KEY    - Polygon wallet private key (0x...)
    POLYMARKET_PROXY_ADDRESS  - funder address shown in Polymarket profile
                                (omit if trading from an EOA directly)

Without credentials — or with TRADING_DRY_RUN=true — orders are simulated at
the current best ask and logged, never sent.
"""

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

# Endpoints + chain are env-overridable so the client can be pointed at a
# testnet / proxy / mock without code changes. Defaults are Polymarket
# production (Polygon mainnet, chain 137). NOTE: Polymarket does not run a
# public testnet with populated markets — override only if you have a real
# alternate host, otherwise leave these at the production defaults.
GAMMA_URL_DEFAULT = "https://gamma-api.polymarket.com"
CLOB_URL_DEFAULT = "https://clob.polymarket.com"

# Market types we trade. "match" = match winner, "setN" = winner of set N.
SET_QUESTION_RE = re.compile(r"set\s+(\d)\s+winner", re.IGNORECASE)

# Questions containing these are side markets we never trade here.
EXCLUDED_PHRASES = ("o/u", "over", "under", "completed match", "total sets",
                    "tie break", "tiebreak", "any set", "win a set")


def _norm(text: str) -> str:
    """Lowercase, strip accents/punctuation for fuzzy name comparison."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def _surname(full_name: str) -> str:
    parts = _norm(full_name).split()
    return parts[-1] if parts else ""


@dataclass
class VenueMarket:
    """One tradeable Polymarket market, normalized."""
    event_title: str
    event_slug: str
    question: str
    market_type: str              # "match" | "set1" | "set2" | "set3"
    condition_id: str
    outcomes: list[str] = field(default_factory=list)
    token_ids: list[str] = field(default_factory=list)
    prices: list[float] = field(default_factory=list)   # gamma snapshot
    game_start_time: Optional[str] = None               # ISO match start (in-play marker)

    def is_live(self, now=None) -> bool:
        """True if the match has started (in-play). Missing start time -> not live."""
        if not self.game_start_time:
            return False
        try:
            start = datetime.fromisoformat(self.game_start_time.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return False
        now = now or datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        return now >= start

    def side_index(self, signal_p1: str, signal_p2: str, side: str) -> Optional[int]:
        """Map a signal side ('player1'/'player2') to this market's outcome index."""
        target = signal_p1 if side == "player1" else signal_p2
        sn = _surname(target)
        for i, outcome in enumerate(self.outcomes):
            if sn and sn in _norm(outcome):
                return i
        return None


class PolymarketClient:
    def __init__(self, session: Optional[requests.Session] = None):
        self.http = session or requests.Session()
        self.http.headers.update({"User-Agent": "tennis10-exec/1.0"})
        self._clob = None  # lazy py-clob-client
        # Read at construction so a .env loaded by the pipeline is honored.
        self.gamma_url = os.getenv("POLYMARKET_GAMMA_URL", GAMMA_URL_DEFAULT).rstrip("/")
        self.clob_url = os.getenv("POLYMARKET_CLOB_URL", CLOB_URL_DEFAULT).rstrip("/")
        self.chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", "137"))

    # ── discovery ────────────────────────────────────────────────────────────

    def _get(self, url: str, params: dict, tries: int = 3):
        last = None
        for attempt in range(tries):
            try:
                r = self.http.get(url, params=params, timeout=15)
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise last

    def fetch_tennis_events(self, max_events: int = 500) -> list[dict]:
        """All open tennis events from Gamma (paginated)."""
        events, offset = [], 0
        while len(events) < max_events:
            page = self._get(f"{self.gamma_url}/events", {
                "tag_slug": "tennis", "closed": "false",
                "limit": 100, "offset": offset,
            })
            if not page:
                break
            events.extend(page)
            if len(page) < 100:
                break
            offset += 100
        return events[:max_events]

    @staticmethod
    def _classify(question: str, event_title: str) -> Optional[str]:
        qn = _norm(question)
        if any(p in qn for p in EXCLUDED_PHRASES):
            return None
        m = SET_QUESTION_RE.search(question)
        if m:
            return f"set{m.group(1)}"
        # Match-winner market: the question is the event title itself
        # ("Tournament: A vs B") and has no set/other qualifiers.
        if "set" not in qn and " vs" in _norm(event_title) and qn == _norm(event_title):
            return "match"
        return None

    def find_match_markets(self, player1: str, player2: str,
                           events: Optional[list[dict]] = None) -> list[VenueMarket]:
        """Return tradeable match-winner and set-winner markets for a fixture."""
        events = events if events is not None else self.fetch_tennis_events()
        s1, s2 = _surname(player1), _surname(player2)
        found: list[VenueMarket] = []
        for event in events:
            title_n = _norm(event.get("title", ""))
            if not (s1 and s2 and s1 in title_n and s2 in title_n):
                continue
            for m in event.get("markets", []):
                if not m.get("active") or m.get("closed"):
                    continue
                mtype = self._classify(m.get("question", ""), event.get("title", ""))
                if not mtype:
                    continue
                try:
                    outcomes = json.loads(m.get("outcomes") or "[]")
                    token_ids = json.loads(m.get("clobTokenIds") or "[]")
                    prices = [float(p) for p in json.loads(m.get("outcomePrices") or "[]")]
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if len(outcomes) != 2 or len(token_ids) != 2:
                    continue
                found.append(VenueMarket(
                    event_title=event.get("title", ""),
                    event_slug=event.get("slug", ""),
                    question=m.get("question", ""),
                    market_type=mtype,
                    condition_id=m.get("conditionId", ""),
                    outcomes=outcomes, token_ids=token_ids, prices=prices,
                    game_start_time=m.get("gameStartTime"),
                ))
        return found

    # ── pricing ──────────────────────────────────────────────────────────────

    def best_ask(self, token_id: str) -> Optional[float]:
        """Best ask from the CLOB book — the price a market BUY actually fills at."""
        try:
            book = self._get(f"{self.clob_url}/book", {"token_id": token_id})
            asks = book.get("asks") or []
            if not asks:
                return None
            return min(float(a["price"]) for a in asks)
        except Exception:
            return None

    def best_bid(self, token_id: str) -> Optional[float]:
        """Best bid from the CLOB book — the price a market SELL fills at."""
        try:
            book = self._get(f"{self.clob_url}/book", {"token_id": token_id})
            bids = book.get("bids") or []
            if not bids:
                return None
            return max(float(b["price"]) for b in bids)
        except Exception:
            return None

    # ── resolution ─────────────────────────────────────────────────────────

    def market_resolution(self, condition_id: str) -> Optional[dict]:
        """Resolution status of a market by condition_id (works after event close).

        Returns None if unknown or not yet resolved. Otherwise:
            {"closed": True, "tokens": {token_id: {"winner": bool,
                                                    "price": float,
                                                    "outcome": str}}}
        """
        if not condition_id:
            return None
        try:
            r = self._get(f"{self.clob_url}/markets/{condition_id}", {})
        except Exception:
            return None
        if not r or not r.get("closed"):
            return None
        tokens = {}
        for t in r.get("tokens", []):
            tid = t.get("token_id")
            if tid:
                tokens[tid] = {"winner": bool(t.get("winner")),
                               "price": float(t.get("price") or 0.0),
                               "outcome": t.get("outcome")}
        if not tokens:
            return None
        return {"closed": True, "tokens": tokens}

    # ── execution ────────────────────────────────────────────────────────────

    @property
    def can_trade_live(self) -> bool:
        return bool(os.getenv("POLYMARKET_PRIVATE_KEY"))

    def _clob_client(self):
        if self._clob is None:
            from py_clob_client.client import ClobClient  # optional dep
            key = os.environ["POLYMARKET_PRIVATE_KEY"]
            funder = os.getenv("POLYMARKET_PROXY_ADDRESS")
            kwargs = {"key": key, "chain_id": self.chain_id}
            if funder:
                kwargs.update({"signature_type": 1, "funder": funder})
            client = ClobClient(self.clob_url, **kwargs)
            client.set_api_creds(client.create_or_derive_api_creds())
            self._clob = client
        return self._clob

    def place_buy(self, token_id: str, price: float, shares: float,
                  dry_run: bool = True) -> dict:
        """Place (or simulate) a limit BUY at `price` for `shares` shares.

        Returns {"status": "dry_run"|"placed"|"failed", "order_id", "detail"}.
        """
        if dry_run:
            return {"status": "dry_run", "order_id": None,
                    "detail": f"simulated buy {shares:.2f} sh @ {price:.3f}"}
        try:
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY
            client = self._clob_client()
            order = client.create_order(OrderArgs(
                token_id=token_id, price=round(price, 3),
                size=round(shares, 2), side=BUY,
            ))
            resp = client.post_order(order)
            ok = bool(resp.get("success"))
            return {"status": "placed" if ok else "failed",
                    "order_id": resp.get("orderID"), "detail": json.dumps(resp)[:500]}
        except Exception as e:
            return {"status": "failed", "order_id": None, "detail": str(e)[:500]}
