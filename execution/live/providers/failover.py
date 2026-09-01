"""Provider health and failover (PRD §23).

WHAT FAILOVER ACTUALLY HAS TO SOLVE
    Not "provider returns an error" — that case is easy and rare. The dangerous
    failure is a provider that stays connected and goes quiet, or one that
    keeps answering with data several minutes old. Both look healthy to a
    liveness check and are worthless to a live model.

    So health here is measured on EVENT FLOW and LATENCY, not on whether the
    socket is open:

        healthy    events arriving, provider latency within budget
        lagging    events arriving but consistently late
        silent     connected, nothing arriving for longer than expected
        failed     transport error

WHY IT DOES NOT FLAP
    Switching feeds mid-match costs the momentum tape (see the note on
    `unsubscribe`), so a switch has to be worth it. A provider must be bad for
    a sustained window before we move, and — more importantly — must be good
    for a sustained window before we move back. Without the second rule a
    marginal primary oscillates and every switch throws away history.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ProviderHealth(str, Enum):
    HEALTHY = "healthy"
    LAGGING = "lagging"
    SILENT = "silent"
    FAILED = "failed"


# SILENCE DETECTS, DEBOUNCE CONFIRMS. These compose rather than overlap: a feed
# is not called silent until SILENCE_MS of nothing, and is not switched away
# from until DEBOUNCE_MS after that. Total time to fail over is therefore
# SILENCE_MS + DEBOUNCE_MS, which is deliberate — switching costs the momentum
# tape, so the bar is a sustained outage rather than a gap between points.
# (A tennis point can be 30s apart; SILENCE_MS alone would fail over mid-game.)

# A feed with nothing on it for this long is silent, whatever the socket says.
SILENCE_MS = 20_000
# Sustained provider latency above this is "lagging" — usable, not preferred.
LAG_MS = 3_000
# How long a condition must persist before it changes the decision.
DEBOUNCE_MS = 15_000
# How long a recovered provider must stay healthy before we fail back to it.
RECOVERY_MS = 60_000


@dataclass
class ProviderStats:
    name: str
    events: int = 0
    errors: int = 0
    last_event_ms: int = 0
    latency_samples: list = field(default_factory=list)
    health: ProviderHealth = ProviderHealth.SILENT
    since_ms: int = 0
    last_error: Optional[str] = None

    @property
    def latency_ms(self) -> int:
        """Median provider latency over the recent window. Median because one
        long GC pause or a single reconnect should not redefine the feed."""
        if not self.latency_samples:
            return 0
        s = sorted(self.latency_samples)
        return int(s[len(s) // 2])

    def as_dict(self) -> dict:
        return {"status": self.health.value, "latency_ms": self.latency_ms,
                "events": self.events, "errors": self.errors,
                "last_error": self.last_error}


class FailoverManager:
    """Tracks several providers and names the one to use.

    Deliberately does not own the providers or do the switching itself. It
    reports which feed should be primary; the runtime performs the change,
    because the change has side effects (subscriptions, tape) that a health
    tracker should not be reaching into.
    """

    def __init__(self, order: list, *, now_ms=None, silence_ms: int = SILENCE_MS,
                 lag_ms: int = LAG_MS, debounce_ms: int = DEBOUNCE_MS,
                 recovery_ms: int = RECOVERY_MS):
        if not order:
            raise ValueError("failover needs at least one provider name")
        self.order = list(order)
        self._now = now_ms or (lambda: int(time.time() * 1000))
        self.silence_ms, self.lag_ms = silence_ms, lag_ms
        self.debounce_ms, self.recovery_ms = debounce_ms, recovery_ms
        self.stats: dict[str, ProviderStats] = {
            n: ProviderStats(name=n, since_ms=self._now()) for n in self.order
        }
        self._active = self.order[0]
        self._active_since = self._now()

    @property
    def active(self) -> str:
        return self._active

    def record_event(self, provider: str, latency_ms: int = 0) -> None:
        st = self.stats.get(provider)
        if st is None:
            return
        st.events += 1
        st.last_event_ms = self._now()
        st.latency_samples.append(max(0, latency_ms))
        if len(st.latency_samples) > 200:
            del st.latency_samples[:100]

    def record_error(self, provider: str, error: str) -> None:
        st = self.stats.get(provider)
        if st is None:
            return
        st.errors += 1
        st.last_error = error[:200]
        self._set(st, ProviderHealth.FAILED)

    def _set(self, st: ProviderStats, h: ProviderHealth) -> None:
        if st.health is not h:
            st.health = h
            st.since_ms = self._now()

    def refresh(self) -> None:
        """Recompute health from flow and latency. Call on a timer."""
        now = self._now()
        for st in self.stats.values():
            if st.health is ProviderHealth.FAILED:
                # Only an event clears a failure — an error with no traffic
                # after it is still a failure.
                if st.last_event_ms and now - st.last_event_ms < self.silence_ms:
                    self._set(st, ProviderHealth.HEALTHY)
                continue
            if not st.last_event_ms or now - st.last_event_ms > self.silence_ms:
                self._set(st, ProviderHealth.SILENT)
            elif st.latency_ms > self.lag_ms:
                self._set(st, ProviderHealth.LAGGING)
            else:
                self._set(st, ProviderHealth.HEALTHY)

    def decide(self) -> str:
        """Which provider should be primary now.

        Switching away needs the current one to have been bad for `debounce_ms`.
        Switching BACK needs the preferred one to have been healthy for
        `recovery_ms` — a longer bar, because failing back to a flapping
        provider costs the tape twice.
        """
        self.refresh()
        now = self._now()
        cur = self.stats[self._active]

        preferred = self.order[0]
        if self._active != preferred:
            pref = self.stats[preferred]
            if (pref.health is ProviderHealth.HEALTHY
                    and now - pref.since_ms >= self.recovery_ms):
                self._switch(preferred, now)
                return self._active

        if cur.health is ProviderHealth.HEALTHY:
            return self._active
        if now - cur.since_ms < self.debounce_ms:
            return self._active          # bad, but not yet reliably bad

        for name in self.order:
            if name == self._active:
                continue
            st = self.stats[name]
            if st.health is ProviderHealth.HEALTHY:
                self._switch(name, now)
                break
            # A lagging feed still beats a silent or failed one.
            if st.health is ProviderHealth.LAGGING and cur.health in (
                    ProviderHealth.SILENT, ProviderHealth.FAILED):
                self._switch(name, now)
                break
        return self._active

    def _switch(self, name: str, now: int) -> None:
        self._active = name
        self._active_since = now

    def as_dict(self) -> dict:
        return {"active": self._active,
                "providers": {n: s.as_dict() for n, s in self.stats.items()}}
