"""WebSocket gateway (PRD §17-§21) — one upstream, many viewers.

THE ECONOMIC POINT, IN CODE
    `RoomRegistry` is where "one live connection, fanned out" actually happens.
    A match room is created when its first viewer arrives and torn down when
    its last one leaves, and those two transitions are the ONLY things that
    subscribe and unsubscribe upstream. Viewer 1,000 costs a dict entry and a
    socket write; it costs the provider nothing.

WHAT THE BROWSER IS TOLD
    A compact delta (§17): score, probability, market, edge, health. Not the
    match state, not the model internals, not the event tape. The browser is
    not a database client — it needs what changed.

WHY BROADCASTS ARE THROTTLED BY FIELD AND NOT GLOBALLY
    Suppressing an update when the model probability has barely moved is free
    accuracy. Applying the same rule to the PRICE is not: a user acts on price,
    and a suppressed price update is a stale price rendered as live — the most
    expensive bug this product can ship, and one its own UI docs already single
    out. So the model number has a movement threshold and the price does not.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from execution.live.feed import Health, LatencyBreakdown

# Model probability must move by at least this to justify a broadcast.
MODEL_EPSILON = 0.005

# Anything the market or the signal engine says is broadcast regardless.
ALWAYS_BROADCAST = frozenset({"price", "signal", "health", "score"})


@dataclass
class Viewer:
    """One connected client."""

    id: str
    send: Callable[[dict], Awaitable[None]]
    joined_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class Room:
    """One match, and everyone watching it."""

    match_id: str
    viewers: dict = field(default_factory=dict)
    last_payload: Optional[dict] = None
    last_model_p1: Optional[float] = None
    broadcasts: int = 0
    suppressed: int = 0

    @property
    def viewer_count(self) -> int:
        return len(self.viewers)


class RoomRegistry:
    """Match rooms, and the upstream subscriptions they imply (§19, §20)."""

    def __init__(self, *, on_first_viewer=None, on_last_viewer=None):
        # Callbacks rather than a provider reference: the registry should not
        # know what a provider is, and tests should not need one.
        self._on_first = on_first_viewer
        self._on_last = on_last_viewer
        self.rooms: dict[str, Room] = {}

    async def join(self, match_id: str, viewer: Viewer) -> Room:
        room = self.rooms.get(match_id)
        first = room is None
        if room is None:
            room = Room(match_id=match_id)
            self.rooms[match_id] = room
        room.viewers[viewer.id] = viewer

        if first and self._on_first:
            # Subscribing upstream only on the FIRST viewer is the whole saving.
            await self._on_first(match_id)

        # A joiner mid-match must not stare at an empty panel until the next
        # point — which in tennis can be 30+ seconds away.
        if room.last_payload:
            await self._safe_send(room, viewer, room.last_payload)
        return room

    async def leave(self, match_id: str, viewer_id: str) -> None:
        room = self.rooms.get(match_id)
        if room is None:
            return
        room.viewers.pop(viewer_id, None)
        if not room.viewers:
            del self.rooms[match_id]
            if self._on_last:
                await self._on_last(match_id)

    async def broadcast(self, match_id: str, payload: dict, *,
                        kind: str = "model") -> int:
        """Send to every viewer. Returns how many received it.

        Suppression is decided here rather than by the caller so the rule lives
        in one place and cannot be applied inconsistently per call site.
        """
        room = self.rooms.get(match_id)
        if room is None:
            return 0

        if kind not in ALWAYS_BROADCAST and self._suppress(room, payload):
            room.suppressed += 1
            return 0

        room.last_payload = payload
        p = (payload.get("probability") or {}).get("p1")
        if p is not None:
            room.last_model_p1 = p
        room.broadcasts += 1

        sent = 0
        for viewer in list(room.viewers.values()):
            if await self._safe_send(room, viewer, payload):
                sent += 1
        return sent

    def _suppress(self, room: Room, payload: dict) -> bool:
        p = (payload.get("probability") or {}).get("p1")
        if p is None or room.last_model_p1 is None:
            return False
        return abs(p - room.last_model_p1) < MODEL_EPSILON

    async def _safe_send(self, room: Room, viewer: Viewer, payload: dict) -> bool:
        """A dead socket removes itself.

        Without this a disconnected viewer accumulates in the room forever and
        keeps the upstream subscription alive — a slow leak that shows up as an
        unexplained provider bill.
        """
        try:
            await viewer.send(payload)
            return True
        except Exception:
            room.viewers.pop(viewer.id, None)
            return False

    @property
    def total_viewers(self) -> int:
        return sum(r.viewer_count for r in self.rooms.values())

    def watched_matches(self) -> list:
        return list(self.rooms)


def build_payload(*, match_id: str, state, fair=None, market=None,
                  ladder=None, signal=None, health: Health = Health.LIVE,
                  latency: Optional[LatencyBreakdown] = None,
                  sequence: int = -1) -> dict:
    """The §17 message. Small, flat, and honest about freshness.

    `health` is included on every message on purpose. §22 says do not generate
    signals from stale data; the client also has to be able to SHOW that the
    data is stale, and it cannot do that if freshness only exists server-side.
    """
    payload: dict[str, Any] = {
        "type": "MATCH_UPDATE",
        "match_id": match_id,
        "sequence": sequence,
        "health": health.value,
        "ts": int(time.time() * 1000),
        "state": {
            "sets": list(state.score.sets),
            "games": list(state.score.games),
            "points": list(state.score.points),
            "server": state.server,
        },
    }
    if fair is not None:
        payload["probability"] = {"p1": round(fair.p1, 4), "p2": round(fair.p2, 4)}
        payload["tier"] = fair.tier
    if ladder is not None:
        payload["ladder"] = ladder.as_dict()
    if market is not None:
        payload["market"] = market
    if signal is not None:
        payload["signal"] = signal.as_dict() if hasattr(signal, "as_dict") else signal
    if latency is not None:
        payload["latency"] = latency.as_dict()
    return payload


def create_app(registry: Optional[RoomRegistry] = None, *, runtime=None):
    """FastAPI app exposing the rooms.

    Imported lazily so this module can be used — and tested — without FastAPI
    installed. The registry and payload logic are the parts worth testing, and
    they have no web dependency at all.
    """
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    reg = registry or RoomRegistry()
    app = FastAPI(title="TennisAlpha Live Market Engine")

    @app.get("/health")
    async def health():
        out = {
            "rooms": len(reg.rooms),
            "viewers": reg.total_viewers,
            "matches": reg.watched_matches(),
        }
        if runtime is not None:
            out.update(runtime.health())
        return out

    @app.websocket("/match/{match_id}")
    async def match_socket(ws: WebSocket, match_id: str):
        await ws.accept()
        viewer = Viewer(id=f"{id(ws)}-{int(time.time()*1000)}", send=ws.send_json)
        await reg.join(match_id, viewer)
        try:
            while True:
                # The client sends nothing meaningful; this is the disconnect
                # detector. Without a read the server never learns the socket
                # closed and the room never empties.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await reg.leave(match_id, viewer.id)

    return app
