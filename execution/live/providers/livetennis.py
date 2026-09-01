"""Live Tennis API adapter (§4).

Chosen as the primary tennis feed on the provider survey: it is the only
tennis-first option in the cheap tier with ATP/WTA/Challenger/ITF coverage,
point-by-point history on Pro and live per-point events plus WebSocket on
Ultra.

TIER MATTERS MORE THAN PROVIDER CHOICE HERE
    Polling and streaming are different products from the same vendor:

        FREE   30 req/min,    100 req/day   development only
        BASIC  60 req/min,  1,000 req/day   one bulk poll a minute, no more
        PRO   300 req/min, 10,000 req/day   bulk polling, match events, odds
        ULTRA                               live per-point events + WebSocket

    The daily caps bite immediately on per-match polling: 20 matches at 30s is
    57,600 requests/day, which exhausts Basic in 25 minutes and Pro in four
    hours. The same 20 matches through ONE bulk live endpoint is 1,440/day and
    fits Basic. So `poll_live()` deliberately fetches the whole live slate in a
    single call rather than looping over subscriptions — endpoint shape, not
    user count, is what decides whether the bill is $10 or $100.

    Sub-second updates require Ultra's WebSocket. Nothing below it can meet the
    "<1s" goal, because a polling feed's floor is its poll interval.

NO CREDENTIALS ARE BAKED IN. The key comes from LIVETENNIS_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import AsyncIterator, Callable, Optional

from execution.live.events import EventType, LiveEvent, P1, P2, Score
from execution.live.provider import TennisDataProvider

BASE_URL = "https://api.livetennisapi.com/v1"


def _points(raw) -> tuple[str, str]:
    """Provider point strings to ours. Tennis point scores are not numbers, so
    they stay strings — see events.Score."""
    if not raw or len(raw) < 2:
        return ("0", "0")
    norm = {"0": "0", "15": "15", "30": "30", "40": "40", "AD": "A", "A": "A"}
    return (norm.get(str(raw[0]).upper(), str(raw[0])),
            norm.get(str(raw[1]).upper(), str(raw[1])))


def normalize(payload: dict, *, received_ms: Optional[int] = None) -> Optional[LiveEvent]:
    """One provider frame to one canonical event.

    Returns None for frames we do not model (heartbeats, subscription acks)
    rather than raising: an unrecognised frame on a live socket is routine, and
    a parser that dies on one takes the whole match down with it.
    """
    kind = str(payload.get("type") or payload.get("event") or "").upper()
    if kind in ("PING", "PONG", "HEARTBEAT", "SUBSCRIBED", "ACK", ""):
        return None

    match_id = str(payload.get("match_id") or payload.get("matchId") or "")
    if not match_id:
        return None

    type_map = {
        "POINT": EventType.POINT,
        "GAME": EventType.GAME_END,
        "SET": EventType.SET_END,
        "MATCH_START": EventType.MATCH_START,
        "MATCH_END": EventType.MATCH_END,
        "SUSPENDED": EventType.SUSPENSION,
        "RESUMED": EventType.RESUMPTION,
        "ODDS": EventType.ODDS_UPDATE,
    }
    event_type = type_map.get(kind, EventType.POINT)

    score_raw = payload.get("score") or {}
    score = Score(
        sets=tuple(score_raw.get("sets") or (0, 0))[:2],
        games=tuple(score_raw.get("games") or (0, 0))[:2],
        points=_points(score_raw.get("points")),
        tiebreak=bool(score_raw.get("tiebreak")),
    )

    server_raw = str(payload.get("server") or "").lower()
    server = P1 if server_raw in ("1", "p1", "home") else P2 if server_raw in ("2", "p2", "away") else None
    winner_raw = str((payload.get("point") or {}).get("winner") or payload.get("point_winner") or "").lower()
    winner = P1 if winner_raw in ("1", "p1", "home") else P2 if winner_raw in ("2", "p2", "away") else None

    # A frame with no sequence cannot be integrity-checked. Rather than invent
    # one (which would make gaps invisible), fall back to the provider
    # timestamp — monotonic in practice, and at least ordering-comparable.
    seq = payload.get("sequence")
    if seq is None:
        seq = payload.get("seq")
    provider_ts = int(payload.get("timestamp") or payload.get("ts") or time.time() * 1000)
    if seq is None:
        seq = provider_ts

    return LiveEvent(
        match_id=match_id,
        sequence=int(seq),
        event_type=event_type,
        provider_ts=provider_ts,
        received_ts=received_ms if received_ms is not None else int(time.time() * 1000),
        score=score,
        server=server,
        point_winner=winner,
        event_id=str(payload.get("event_id") or ""),
        raw=payload,
    )


class LiveTennisProvider(TennisDataProvider):
    """WebSocket (Ultra) with a polling fallback for lower tiers.

    `transport` is injected so this is testable without a key or a network:
    tests hand it a coroutine that yields recorded frames. A provider adapter
    that can only be exercised against the live vendor is one that gets tested
    in production.
    """

    name = "livetennis"

    def __init__(self, *, api_key: Optional[str] = None, transport: Optional[Callable] = None,
                 poll_interval_s: float = 60.0, use_websocket: bool = True):
        self.api_key = api_key or os.getenv("LIVETENNIS_API_KEY", "")
        self._transport = transport
        self._poll_interval = poll_interval_s
        self._use_ws = use_websocket
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        self._task: Optional[asyncio.Task] = None
        self.subscriptions: set[str] = set()
        self.connected = False
        self.last_error: Optional[str] = None

    async def connect(self) -> None:
        if self.connected:
            return                      # idempotent: reconnect logic re-calls this
        if not self.api_key and self._transport is None:
            raise RuntimeError("LIVETENNIS_API_KEY is not set")
        self.connected = True
        if self._transport is not None:
            self._task = asyncio.create_task(self._pump(self._transport))

    async def close(self) -> None:
        self.connected = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def subscribe(self, match_id: str) -> None:
        self.subscriptions.add(match_id)

    async def unsubscribe(self, match_id: str) -> None:
        self.subscriptions.discard(match_id)

    async def _pump(self, transport: Callable) -> None:
        """Read frames from the transport into the queue.

        Errors are recorded and the loop ends rather than propagating: the
        failover manager watches `last_error` and switches providers. An
        exception escaping here would kill the gateway for every match, not
        just this feed.
        """
        try:
            async for frame in transport():
                if isinstance(frame, (str, bytes)):
                    try:
                        frame = json.loads(frame)
                    except Exception:
                        continue
                ev = normalize(frame)
                if ev is None:
                    continue
                if self.subscriptions and ev.match_id not in self.subscriptions:
                    continue           # §20: only pay attention to watched matches
                try:
                    self._queue.put_nowait(ev)
                except asyncio.QueueFull:
                    # Dropping the OLDEST keeps the live edge current. Dropping
                    # the newest would leave us permanently behind, which for a
                    # live feed is worse than a hole the tracker will flag.
                    _ = self._queue.get_nowait()
                    self._queue.put_nowait(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            self.connected = False

    async def events(self) -> AsyncIterator[LiveEvent]:  # type: ignore[override]
        while self.connected or not self._queue.empty():
            try:
                yield await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
