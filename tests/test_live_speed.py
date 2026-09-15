"""The board's responsiveness, as behaviour rather than a stopwatch.

These are not timing tests — timing tests are flaky and prove nothing on a
loaded machine. Each one pins the STRUCTURAL property that made the board slow:
work repeated per match that should happen once, and blocking calls issued
serially that have no dependency on each other.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.live.providers.polymarket_odds import PolymarketOddsSource  # noqa: E402


class FakeClient:
    """Counts what the catalogue actually costs."""

    def __init__(self):
        self.event_fetches = 0
        self.lookups = []

    def fetch_tennis_events(self):
        self.event_fetches += 1
        return [{"id": "e1"}]

    def find_match_markets(self, p1, p2, events=None):
        self.lookups.append(events)
        return []


# ── the catalogue is fetched once, not once per match ────────────────────────

def test_many_lookups_share_one_catalogue_fetch():
    """The bug: eight live matches meant eight full paginated fetches.

    The catalogue is identical for every match, and fetching it was the
    expensive half of a lookup — so this is the difference between a price
    cycle costing hundreds of milliseconds and costing seconds.
    """
    c = FakeClient()
    src = PolymarketOddsSource(client=c)
    for pair in [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]:
        src.find_tokens(*pair)
    assert c.event_fetches == 1, "catalogue refetched per match"
    assert len(c.lookups) == 4


def test_the_lookup_actually_receives_the_cached_catalogue():
    # Passing None would make find_match_markets refetch internally, which is
    # the original bug wearing a cache in front of it.
    c = FakeClient()
    PolymarketOddsSource(client=c).find_tokens("A", "B")
    assert c.lookups == [[{"id": "e1"}]]


def test_the_catalogue_goes_stale_so_new_markets_are_seen():
    clock = [1000.0]
    c = FakeClient()
    src = PolymarketOddsSource(client=c, events_ttl_s=60)
    src._events_at = 0.0
    src.events()
    assert c.event_fetches == 1
    src.events()
    assert c.event_fetches == 1, "refetched inside the TTL"
    src._events_at -= 61                       # age it past the TTL
    src.events()
    assert c.event_fetches == 2, "never refreshed: new markets stay invisible"
    assert clock


def test_a_failing_catalogue_serves_the_stale_one_rather_than_nothing():
    """An outage must degrade to yesterday's list, not to an unpriced board."""
    c = FakeClient()
    src = PolymarketOddsSource(client=c)
    assert src.events() == [{"id": "e1"}]

    def boom():
        raise RuntimeError("gamma down")

    c.fetch_tennis_events = boom
    src._events_at -= 999
    assert src.events() == [{"id": "e1"}]
    assert "gamma down" in (src.last_error or "")


# ── statistics are fetched together, not one match at a time ─────────────────

def _provider():
    from execution.live.providers.sofaproxy import SofaProxyProvider
    return SofaProxyProvider(poll_s=3.0)


def test_statistics_for_many_matches_are_fetched_concurrently():
    """Serially these were the whole poll; nothing here depends on anything
    else here, so they must go out together."""
    prov = _provider()
    calls = []

    def slow(mid):
        calls.append(mid)
        time.sleep(0.15)
        return {"mid": mid}

    prov._stats = slow
    mids = [str(i) for i in range(6)]
    t0 = time.perf_counter()
    got = prov._stats_many(mids)
    elapsed = time.perf_counter() - t0

    assert set(got) == set(mids), "every match must get its answer"
    assert got["3"] == {"mid": "3"}
    # Serial would be >=0.9s. Generous bound: this asserts concurrency, not speed.
    assert elapsed < 0.6, f"statistics still serial ({elapsed:.2f}s)"


def test_no_matches_needing_an_anchor_costs_nothing():
    prov = _provider()
    prov._stats = lambda mid: pytest_fail(mid)
    assert prov._stats_many([]) == {}


def pytest_fail(mid):
    raise AssertionError(f"fetched statistics for {mid} with nothing to anchor")


def test_a_single_match_still_works():
    prov = _provider()
    prov._stats = lambda mid: {"mid": mid}
    assert prov._stats_many(["7"]) == {"7": {"mid": "7"}}


# ── the proxy's background refresh ───────────────────────────────────────────

def test_background_refresh_falls_back_like_the_synchronous_path():
    """The bug that silently disabled server detection.

    `_bg_refresh` could only succeed against an origin that answers 403 to this
    IP, so it always failed and the entry was served stale until it aged out.
    Two statistics reads inside that window returned byte-identical bodies —
    and the server is inferred from the DELTA between two reads, so identical
    bodies mean it can never be anchored. Measured: 0/6 matches anchored
    before, 4/6 within a minute after.
    """
    import sofa_proxy as sp

    url, path = "https://x/event/1/statistics", "event/1/statistics"
    sp._cache[url] = {"ts": 0.0, "data": b"old", "refreshing": True}
    orig_up, orig_fb = sp._fetch_upstream, sp._fallback
    try:
        sp._fetch_upstream = lambda u: (403, b"")
        sp._fallback = lambda p: (200, b"fresh")
        sp._bg_refresh(url, path)
        assert sp._cache[url]["data"] == b"fresh", "refresh never escapes the 403"
        assert sp._cache[url]["refreshing"] is False
    finally:
        sp._fetch_upstream, sp._fallback = orig_up, orig_fb
        sp._cache.pop(url, None)


def test_a_refresh_with_no_fallback_leaves_the_stale_entry_serveable():
    """Both sources down must not blank the cache — stale beats empty."""
    import sofa_proxy as sp

    url, path = "https://x/event/2/statistics", "event/2/statistics"
    sp._cache[url] = {"ts": 0.0, "data": b"old", "refreshing": True}
    orig_up, orig_fb = sp._fetch_upstream, sp._fallback
    try:
        sp._fetch_upstream = lambda u: (403, b"")
        sp._fallback = lambda p: None
        sp._bg_refresh(url, path)
        assert sp._cache[url]["data"] == b"old"
        assert sp._cache[url]["refreshing"] is False, "entry would never refresh again"
    finally:
        sp._fetch_upstream, sp._fallback = orig_up, orig_fb
        sp._cache.pop(url, None)


# ── the gateway socket ───────────────────────────────────────────────────────

def test_the_match_socket_accepts_rather_than_closing():
    """The gateway's only real endpoint, which was silently unreachable.

    `from __future__ import annotations` makes every annotation a string, and
    FastAPI resolves those against MODULE globals. `WebSocket` was imported
    inside create_app, so the name was invisible there, FastAPI treated the
    `ws` parameter as a QUERY PARAMETER, and every connection was closed with
    1008 "field required" before accept — which uvicorn reports as a bare 403.

    /health stayed green throughout, which is why this needs a test: nothing
    else in the system could tell you the socket was dead. Driven at the ASGI
    level so it needs no server, no client library and no network.
    """
    import asyncio

    import pytest
    pytest.importorskip("fastapi")
    from execution.live.gateway import RoomRegistry, create_app

    app = create_app(RoomRegistry())
    scope = {
        "type": "websocket", "path": "/match/123", "raw_path": b"/match/123",
        "headers": [], "query_string": b"", "scheme": "ws", "http_version": "1.1",
        "asgi": {"version": "3.0"}, "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8080), "subprotocols": [], "root_path": "",
    }
    sent = []

    async def receive():
        return ({"type": "websocket.connect"} if not sent
                else {"type": "websocket.disconnect", "code": 1000})

    async def send(m):
        sent.append(m)

    asyncio.run(asyncio.wait_for(app(scope, receive, send), timeout=10))

    kinds = [m["type"] for m in sent]
    assert "websocket.accept" in kinds, f"socket refused the connection: {sent}"
    assert "websocket.close" not in kinds[:1]
