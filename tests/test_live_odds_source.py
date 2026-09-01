"""Polymarket odds source.

The cases worth pinning are the ones a live venue will not produce on request:
an empty book, a one-sided book, a crossed book, and a top level so thin the
mid is fiction. Each of those must reach the signal gate as "not a price"
rather than as a number.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.live.odds import MarketView  # noqa: E402
from execution.live.providers.polymarket_odds import (  # noqa: E402
    PolymarketOddsSource, quote_from_book,
)


def book(bids=None, asks=None):
    return {"bids": bids or [], "asks": asks or []}


def lvl(price, size):
    return {"price": str(price), "size": str(size)}


def test_book_becomes_a_two_sided_quote_with_depth():
    q = quote_from_book(book(bids=[lvl(0.60, 800)], asks=[lvl(0.62, 900)]),
                        market="match_winner", side="p1")
    assert q.bid == 0.60 and q.ask == 0.62
    assert q.bid_size == 800 and q.ask_size == 900
    assert q.mid == pytest.approx(0.61)
    assert q.tradeable


def test_depth_is_top_level_only_not_the_whole_book():
    # Summing every level would make a thin market look deep by counting size
    # at prices nobody would accept.
    q = quote_from_book(book(bids=[lvl(0.60, 10), lvl(0.30, 10_000)],
                             asks=[lvl(0.62, 10), lvl(0.95, 10_000)]),
                        market="m", side="p1")
    assert q.bid_size == 10 and q.ask_size == 10
    assert not q.tradeable, "20 dollars at the top is not a tradeable market"


def test_size_at_the_same_best_price_is_summed():
    q = quote_from_book(book(bids=[lvl(0.60, 300), lvl(0.60, 400)],
                             asks=[lvl(0.62, 900)]), market="m", side="p1")
    assert q.bid_size == 700


def test_empty_book_is_a_quote_with_no_price_not_an_error():
    q = quote_from_book(book(), market="m", side="p1")
    assert q.bid is None and q.ask is None
    assert q.mid is None
    assert not q.tradeable


def test_one_sided_book_keeps_the_side_it_has():
    q = quote_from_book(book(bids=[lvl(0.55, 900)]), market="m", side="p1")
    assert q.mid == 0.55           # information, not None
    assert not q.tradeable         # but not something you can trade against


def test_malformed_levels_do_not_raise():
    q = quote_from_book({"bids": [{"nope": 1}], "asks": "garbage"},
                        market="m", side="p1")
    assert q.bid is None and q.ask is None


def test_source_fetches_each_book_exactly_once():
    # best_ask + best_bid would be two fetches per side, four per market.
    calls = []

    def fake(token):
        calls.append(token)
        return book(bids=[lvl(0.60, 900)], asks=[lvl(0.62, 900)])

    src = PolymarketOddsSource(fetch_book=fake)
    mv = MarketView(match_id="m1", market="match_winner")
    src.update(mv, token_p1="t1", token_p2="t2")
    assert calls == ["t1", "t2"]
    assert src.requests == 2


def test_source_prices_both_sides_into_the_view():
    def fake(token):
        return (book(bids=[lvl(0.60, 900)], asks=[lvl(0.62, 900)]) if token == "t1"
                else book(bids=[lvl(0.37, 900)], asks=[lvl(0.39, 900)]))

    mv = MarketView(match_id="m1", market="match_winner")
    PolymarketOddsSource(fetch_book=fake).update(mv, token_p1="t1", token_p2="t2")

    assert mv.has_price
    p1, p2, source = mv.fair()
    assert source == "exchange"
    assert p1 + p2 == pytest.approx(1.0)
    assert p1 == pytest.approx(0.61 / (0.61 + 0.38), abs=0.01)


def test_a_failing_fetch_leaves_the_view_unpriced_rather_than_raising():
    # A price source that raises takes the scoreboard down with it. The gate
    # already refuses to signal without a price, so failing quiet degrades the
    # product by exactly the right amount.
    def boom(_token):
        raise ConnectionError("clob unreachable")

    src = PolymarketOddsSource(fetch_book=boom)
    mv = MarketView(match_id="m1", market="match_winner")
    src.update(mv, token_p1="t1")
    assert not mv.has_price
    assert "ConnectionError" in (src.last_error or "")


def test_thin_market_reaches_the_gate_as_unpriced_for_trading():
    def fake(_t):
        return book(bids=[lvl(0.60, 2)], asks=[lvl(0.62, 2)])

    mv = MarketView(match_id="m1", market="match_winner")
    PolymarketOddsSource(fetch_book=fake).update(mv, token_p1="t1", token_p2="t2")
    # There IS a price, and it is explicitly labelled as thin so the caller can
    # widen uncertainty rather than trusting it like a deep book.
    assert mv.fair()[2] == "exchange_thin"
