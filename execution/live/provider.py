"""Provider abstraction (PRD §4) — one interface, swappable feeds.

WHY THE INTERFACE COMES BEFORE THE PROVIDER
    This repo has already been burned once by binding directly to a feed: the
    production board depends on scraped SofaScore, and when its odds endpoints
    started answering 403 the pricing path had nowhere to fall back to. The
    fix is not "pick a better provider", it is "make the provider replaceable".

    So the model never imports a provider. It consumes `LiveEvent`s from
    whatever implements `TennisDataProvider`.

WHY A REPLAY PROVIDER SHIPS WITH IT
    A live-data system that can only be tested against live data cannot be
    tested: matches are not on demand, and the interesting cases (a dropped
    packet, a reorder, a five-minute silence) cannot be requested from a real
    feed. `ReplayProvider` makes those cases ordinary unit tests, which is the
    difference between knowing the gap detector works and hoping it does.
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional

from execution.live.events import EventType, LiveEvent, Score


class TennisDataProvider(abc.ABC):
    """The whole contract. Anything a provider needs beyond this belongs in
    that provider's own module, not in the interface."""

    name: str = "abstract"

    @abc.abstractmethod
    async def connect(self) -> None:
        """Open the upstream connection. Must be idempotent — reconnect logic
        calls it again without tearing the object down."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release the upstream connection. Must be safe to call when never
        connected, so shutdown paths need no special-casing."""

    @abc.abstractmethod
    async def subscribe(self, match_id: str) -> None:
        """Start receiving events for a match.

        §20: this is what makes the upstream cost track the number of MATCHES
        being watched rather than the number of viewers. The gateway calls it
        when a match room gains its first viewer.
        """

    @abc.abstractmethod
    async def unsubscribe(self, match_id: str) -> None:
        """Stop receiving events for a match.

        NOTE, and it matters: dropping a subscription discards the point tape.
        `momentum.py` is a function of recent history, so a match resubscribed
        after a gap has no momentum until enough points accumulate. Either keep
        matches you intend to price subscribed for their duration, or backfill
        the tape on resubscribe — which needs a point-history endpoint, and
        that is a paid tier on every provider surveyed.
        """

    @abc.abstractmethod
    def events(self) -> AsyncIterator[LiveEvent]:
        """Normalized events, in arrival order.

        Arrival order, NOT sequence order: reordering is a fact of the
        transport and hiding it here would defeat `SequenceTracker`, which is
        the thing that can actually tell a reorder from a loss.
        """

    async def resync(self, match_id: str) -> Optional[LiveEvent]:
        """Authoritative current state after a detected gap.

        Optional: a provider with no snapshot endpoint returns None and the
        match stays DEGRADED until the hole ages out of relevance. Returning a
        guess would be worse than staying degraded.
        """
        return None


@dataclass
class ScriptedEvent:
    """One step in a replay script. `delay_ms` is the wait BEFORE emitting,
    which is how a test writes 'then nothing happened for six seconds'."""

    sequence: int
    event_type: EventType = EventType.POINT
    score: Score = None            # type: ignore[assignment]
    server: Optional[str] = None
    point_winner: Optional[str] = None
    delay_ms: int = 0
    provider_ts: Optional[int] = None


class ReplayProvider(TennisDataProvider):
    """Deterministic provider for tests and offline development.

    Emits exactly the script it is given — including out-of-order sequences,
    duplicates and silences — so the failure modes the live system must survive
    can be asserted on rather than waited for.
    """

    name = "replay"

    def __init__(self, script: Iterable[ScriptedEvent], *, match_id: str = "test-match",
                 clock_ms: int = 1_788_000_000_000, real_time: bool = False):
        self._script = list(script)
        self._match_id = match_id
        self._clock = clock_ms
        self._real_time = real_time
        self.connected = False
        self.subscriptions: set[str] = set()
        self.connect_calls = 0

    async def connect(self) -> None:
        self.connected = True
        self.connect_calls += 1

    async def close(self) -> None:
        self.connected = False

    async def subscribe(self, match_id: str) -> None:
        self.subscriptions.add(match_id)

    async def unsubscribe(self, match_id: str) -> None:
        self.subscriptions.discard(match_id)

    async def events(self) -> AsyncIterator[LiveEvent]:  # type: ignore[override]
        for step in self._script:
            # A virtual clock by default: a test asserting OFFLINE after 15
            # seconds must not take 15 seconds to run.
            self._clock += step.delay_ms
            if self._real_time and step.delay_ms:
                await asyncio.sleep(step.delay_ms / 1000)
            yield LiveEvent(
                match_id=self._match_id,
                sequence=step.sequence,
                event_type=step.event_type,
                provider_ts=step.provider_ts if step.provider_ts is not None else self._clock,
                received_ts=self._clock,
                score=step.score or Score(),
                server=step.server,
                point_winner=step.point_winner,
            )

    def now_ms(self) -> int:
        """The replay's virtual clock, for injecting into SequenceTracker so
        freshness is evaluated on script time rather than wall time."""
        return self._clock
