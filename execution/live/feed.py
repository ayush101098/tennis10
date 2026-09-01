"""Feed integrity — sequence validation and freshness (PRD §7, §22, §24).

THE RULE THIS MODULE EXISTS TO ENFORCE
    A pricing engine must never be handed a match state it cannot vouch for.
    Two failures look identical from inside the model and are both silent:

      1. We missed events.  The scoreboard we hold is stale but *plausible*,
         so every downstream number is confidently wrong.
      2. The feed went quiet. Nothing arrives, the last state stays on screen,
         and a price from four minutes ago renders as live.

    Both must be visible as state, not inferred from an absence.

WHY THERE IS A REORDER BUFFER
    The PRD says: on seeing 101 -> 103, declare a gap and resync. Taken
    literally that is too eager — UDP-ish transports and load-balanced
    WebSocket fan-out routinely deliver 100, 101, 103, 102 with 102 arriving
    tens of milliseconds later. Resyncing on every transient reorder would mean
    resyncing constantly under normal conditions, and a resync storm is itself
    an outage.

    So an out-of-order arrival opens a short grace window. If the missing
    sequence lands inside it, nothing happened. If the window closes with a
    hole, THEN the feed is degraded and a resync is warranted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

from execution.live.events import LiveEvent


class Health(str, Enum):
    """Freshness bands from PRD §22. Ordered worst-last so comparisons read
    naturally and the worst of several inputs is `max()`."""

    LIVE = "LIVE"           # < 2s since last event
    DELAYED = "DELAYED"     # 2-5s
    STALE = "STALE"         # 5-15s
    OFFLINE = "OFFLINE"     # > 15s
    DEGRADED = "DEGRADED"   # sequence hole — freshness is irrelevant, we know we're wrong


_ORDER = [Health.LIVE, Health.DELAYED, Health.STALE, Health.OFFLINE, Health.DEGRADED]


def worst(*states: Health) -> Health:
    """The worst of several health states. Used to combine score-feed health
    with odds-feed health: a live score against a stale price is not a live
    signal, and reporting it as one is the most expensive bug this product
    can ship."""
    return max(states, key=_ORDER.index)


# §22 thresholds, in milliseconds.
LIVE_MS = 2_000
DELAYED_MS = 5_000
STALE_MS = 15_000

# How long a hole may stay open before we call it a real gap rather than a
# reorder. Generous enough to absorb network jitter, short enough that a true
# loss is caught within one point.
REORDER_GRACE_MS = 750


def health_for_age(age_ms: int) -> Health:
    if age_ms < LIVE_MS:
        return Health.LIVE
    if age_ms < DELAYED_MS:
        return Health.DELAYED
    if age_ms < STALE_MS:
        return Health.STALE
    return Health.OFFLINE


@dataclass
class FeedStatus:
    """What we are willing to say about this match's data right now."""

    health: Health = Health.OFFLINE
    last_sequence: int = -1
    last_event_ms: int = 0
    missing: tuple[int, ...] = ()
    duplicates: int = 0
    reordered: int = 0

    @property
    def tradeable(self) -> bool:
        """§24: the gate. A signal may only be published from data we can
        vouch for. DELAYED is allowed — a two-second-old score is still a
        score — but a hole in the sequence is not, at any freshness."""
        return self.health in (Health.LIVE, Health.DELAYED)


class SequenceTracker:
    """Per-match sequence and freshness state.

    Not thread-safe by design: one match is owned by one consumer task. Sharing
    one tracker across tasks would reintroduce exactly the interleaving it
    exists to detect.
    """

    def __init__(self, *, grace_ms: int = REORDER_GRACE_MS, now_ms=None):
        self._grace_ms = grace_ms
        self._now = now_ms or (lambda: int(time.time() * 1000))
        self.last_sequence: int = -1
        self.last_event_ms: int = 0
        self._pending: dict[int, int] = {}   # missing sequence -> first noticed (ms)
        self._seen: set[int] = set()
        self.duplicates = 0
        self.reordered = 0

    def observe(self, event: LiveEvent) -> FeedStatus:
        """Record an event and return the resulting status.

        Returns rather than raises: a gap is a normal operating condition on a
        live feed, and an exception here would mean one lost packet kills the
        consumer for a match that is otherwise fine.
        """
        now = self._now()
        seq = event.sequence

        if seq in self._seen:
            # Providers replay on reconnect. A duplicate is not an error, but
            # it must not be priced twice — the caller checks `duplicates`.
            self.duplicates += 1
            return self.status(now=now)

        self._seen.add(seq)
        self.last_event_ms = max(self.last_event_ms, event.received_ts or now)

        if self.last_sequence < 0:
            self.last_sequence = seq
            return self.status(now=now)

        if seq == self.last_sequence + 1:
            self.last_sequence = seq
        elif seq > self.last_sequence + 1:
            # Forward jump: everything between is missing, provisionally.
            for missing in range(self.last_sequence + 1, seq):
                self._pending.setdefault(missing, now)
            self.last_sequence = seq
        else:
            # Late arrival. If it fills a hole we were waiting on, the feed was
            # never actually broken — just reordered.
            if self._pending.pop(seq, None) is not None:
                self.reordered += 1

        return self.status(now=now)

    def status(self, *, now: Optional[int] = None) -> FeedStatus:
        now = now if now is not None else self._now()

        # Holes still inside the grace window are not yet gaps.
        overdue = tuple(sorted(
            s for s, first_seen in self._pending.items()
            if now - first_seen > self._grace_ms
        ))

        age = now - self.last_event_ms if self.last_event_ms else STALE_MS + 1
        health = Health.DEGRADED if overdue else health_for_age(age)

        return FeedStatus(
            health=health,
            last_sequence=self.last_sequence,
            last_event_ms=self.last_event_ms,
            missing=overdue,
            duplicates=self.duplicates,
            reordered=self.reordered,
        )

    def resynced(self, up_to_sequence: int) -> None:
        """Called after a successful resync: the provider has told us the
        authoritative state, so previously-missing events are no longer
        outstanding. Without this a single early gap would pin the match to
        DEGRADED for its entire duration."""
        for s in list(self._pending):
            if s <= up_to_sequence:
                del self._pending[s]
        self.last_sequence = max(self.last_sequence, up_to_sequence)


@dataclass
class LatencyBreakdown:
    """§21. Each stage measured, so a slow product can be blamed on the right
    component instead of argued about."""

    provider_ms: int = 0
    processing_ms: int = 0
    model_ms: int = 0
    broadcast_ms: int = 0

    @property
    def total_ms(self) -> int:
        return self.provider_ms + self.processing_ms + self.model_ms + self.broadcast_ms

    def as_dict(self) -> dict:
        return {
            "provider": self.provider_ms,
            "processing": self.processing_ms,
            "model": self.model_ms,
            "broadcast": self.broadcast_ms,
            "total": self.total_ms,
        }


def summarize(events: Iterable[LiveEvent]) -> LatencyBreakdown:
    """Provider latency across a batch. The other stages are filled in by the
    pipeline that performs them; this only knows the one term it can see."""
    lat = [e.provider_latency_ms for e in events]
    return LatencyBreakdown(provider_ms=int(sum(lat) / len(lat)) if lat else 0)
