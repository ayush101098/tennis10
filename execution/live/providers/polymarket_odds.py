"""Polymarket as the exchange price source (PRD §9, §10).

WHY POLYMARKET IS THE PRICE OF RECORD HERE
    Not preference — availability. SofaScore's per-event odds endpoint answers
    403 and its daily bulk feed returns an event-id space with zero overlap
    with the schedule, so no bookmaker price can currently be joined to a
    match. Polymarket answers, and it is the venue this project executes on.
    `odds.py` keeps the bookmaker path intact for when one comes back.

ONE BOOK FETCH, NOT TWO
    `PolymarketClient.best_ask` and `.best_bid` each fetch the whole order book
    and throw away half of it, so pricing one side costs two HTTP calls and
    pricing a two-way market costs four. This adapter fetches each book once
    and reads both sides plus their DEPTH out of it — depth being the thing
    neither existing accessor returns, and the thing that separates a real
    price from a mid with twelve dollars behind it.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from execution.live.odds import ExchangeQuote, MarketView

VENUE = "polymarket"


def _side_of_book(levels, *, best: str) -> tuple[Optional[float], float]:
    """Best price and the size resting at it.

    Depth is taken at the TOP LEVEL only, deliberately. Summing the whole book
    would count size at prices nobody would accept and make a thin market look
    deep — which is the exact misreading `ExchangeQuote.tradeable` exists to
    prevent.
    """
    if not levels:
        return None, 0.0
    try:
        priced = [(float(l["price"]), float(l.get("size") or 0.0)) for l in levels]
    except (KeyError, TypeError, ValueError):
        return None, 0.0
    if not priced:
        return None, 0.0
    px = min(p for p, _ in priced) if best == "ask" else max(p for p, _ in priced)
    size = sum(s for p, s in priced if p == px)
    return px, size


def quote_from_book(book: dict, *, market: str, side: str,
                    ts_ms: Optional[int] = None) -> ExchangeQuote:
    """One CLOB book to one `ExchangeQuote`.

    A malformed or empty book yields a quote with no prices rather than an
    exception: an unpriced market is an ordinary state on an exchange, and it
    must reach the gate as "no price" rather than as a crash.
    """
    bid, bid_size = _side_of_book((book or {}).get("bids"), best="bid")
    ask, ask_size = _side_of_book((book or {}).get("asks"), best="ask")
    return ExchangeQuote(
        venue=VENUE, market=market, side=side,
        bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
        ts_ms=ts_ms if ts_ms is not None else int(time.time() * 1000),
    )


class PolymarketOddsSource:
    """Fetches two-sided exchange prices and folds them into a `MarketView`.

    `client` and `fetch_book` are injectable so this is testable without the
    network — the interesting cases here are a one-sided book, an empty book
    and a crossed one, none of which can be requested from a live venue.
    """

    def __init__(self, *, client=None, fetch_book: Optional[Callable] = None):
        self._client = client
        self._fetch_book = fetch_book
        self.requests = 0
        self.last_error: Optional[str] = None

    def _ensure_client(self):
        if self._client is None and self._fetch_book is None:
            from execution.polymarket import PolymarketClient
            self._client = PolymarketClient()
        return self._client

    def _book(self, token_id: str) -> dict:
        self.requests += 1
        if self._fetch_book is not None:
            return self._fetch_book(token_id) or {}
        client = self._ensure_client()
        # Reach past best_ask/best_bid to the raw book: they each refetch it,
        # and neither returns size.
        return client._get(f"{client.clob_url}/book", {"token_id": token_id}) or {}

    def update(self, view: MarketView, *, token_p1: str, token_p2: Optional[str] = None,
               market: str = "match_winner") -> MarketView:
        """Price both sides into `view`.

        The second token is optional: a two-outcome market's other side is the
        complement, and `MarketView.exchange_consensus` infers it. Fetching it
        anyway is one more request per match per refresh, which is worth it
        because an inferred side carries no depth — and depth is what decides
        whether the price is real.
        """
        try:
            view.add_exchange(quote_from_book(self._book(token_p1),
                                              market=market, side="p1"))
            if token_p2:
                view.add_exchange(quote_from_book(self._book(token_p2),
                                                  market=market, side="p2"))
        except Exception as e:
            # A price source that raises takes the scoreboard down with it. The
            # gate already refuses to signal without a price, so failing quiet
            # here degrades the product by exactly the right amount.
            self.last_error = f"{type(e).__name__}: {e}"[:200]
        return view

    def find_tokens(self, player1: str, player2: str) -> Optional[tuple[str, str, str]]:
        """Locate a match market, returning (token_p1, token_p2, condition_id).

        Reuses `PolymarketClient.find_match_markets`, which already handles the
        surname matching and market classification — reimplementing that here
        would be a second place for the fixture-matching rules to drift.
        """
        client = self._ensure_client()
        if client is None:
            return None
        try:
            markets = client.find_match_markets(player1, player2)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            return None
        for m in markets or []:
            if getattr(m, "market_type", None) != "match":
                continue
            idx1 = m.side_index(player1, player2, "player1")
            idx2 = m.side_index(player1, player2, "player2")
            if idx1 is None or idx2 is None:
                continue
            if len(m.token_ids) < 2:
                continue
            return m.token_ids[idx1], m.token_ids[idx2], m.condition_id
        return None
