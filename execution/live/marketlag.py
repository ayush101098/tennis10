"""Market-lag detection — has the market repriced the point yet?

THE IDEA
    A point is played. Our model reprices immediately. The market reprices when
    someone notices. The window between those two is a price-discovery edge,
    and unlike an edge derived from the model being *right*, it does not depend
    on the model being calibrated:

        |ΔP_model| large  AND  |ΔP_market| ≈ 0   →  the market has not moved yet

    That distinction matters here. Every other signal this product emits rests
    on the model's absolute probability being correct, and it measurably is not
    (~13pp from the market, no calibration dataset yet). A LAG signal rests only
    on the model's *direction* and on timestamps. It is the one edge in this
    codebase that survives the calibration problem, which is why it is worth
    building before the calibration is fixed rather than after.

WHAT IT DOES NOT CLAIM
    A market that has not moved is not necessarily wrong. It may have priced the
    point already and disagreed with us about its importance, or the point may
    genuinely not matter. So this reports a DIVERGENCE and its age; it does not
    assert the market is mispriced. Deciding that is the signal engine's job,
    with the confidence machinery attached.

STEAM is the mirror image: the market moves hard and we did not. That is
    information arriving somewhere else first — an injury, a medical timeout, a
    feed we do not have. It is a reason to STOP, not to trade.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional


class Divergence(str, Enum):
    NONE = "NONE"
    LAG = "LAG"          # we moved, the market has not — potential edge
    STEAM = "STEAM"      # the market moved, we did not — information we lack
    AGREE = "AGREE"      # both moved the same way; the market has caught up


# A model move smaller than this is noise, not news.
MIN_MODEL_MOVE = 0.015
# A market move smaller than this counts as "has not moved".
MARKET_FLAT = 0.005
# Market moves beyond this in the window are steam.
STEAM_MOVE = 0.03
# How long a lag stays interesting. Past this the market has decided it
# disagrees, which is a different (and much weaker) claim than "not yet".
LAG_TTL_MS = 30_000
# Rolling window kept per match for reaction-time statistics.
HISTORY = 200


@dataclass
class LagEvent:
    """One observed divergence between our reprice and the market's."""

    match_id: str
    kind: Divergence
    model_delta: float
    market_delta: float
    model_p: float
    market_p: float
    ts_ms: int
    # Filled in when the market later moves the same way — the headline number.
    reaction_ms: Optional[int] = None

    @property
    def gap(self) -> float:
        """How much of our move the market has not yet made."""
        return self.model_delta - self.market_delta

    def as_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "kind": self.kind.value,
            "model_delta": round(self.model_delta, 4),
            "market_delta": round(self.market_delta, 4),
            "gap": round(self.gap, 4),
            "model_p": round(self.model_p, 4),
            "market_p": round(self.market_p, 4),
            "ts_ms": self.ts_ms,
            "reaction_ms": self.reaction_ms,
        }


@dataclass
class _Track:
    model_p: Optional[float] = None
    market_p: Optional[float] = None
    ts_ms: int = 0
    open_lag: Optional[LagEvent] = None
    reactions: Deque = field(default_factory=lambda: deque(maxlen=HISTORY))


class MarketLagDetector:
    """Per-match comparison of our reprice against the market's.

    Deliberately stateful and cheap: two floats and a timestamp per match. The
    expensive thing here would be storing every tick, and the only statistic
    worth keeping is the reaction time distribution.
    """

    def __init__(self, *, min_model_move: float = MIN_MODEL_MOVE,
                 market_flat: float = MARKET_FLAT, steam_move: float = STEAM_MOVE,
                 lag_ttl_ms: int = LAG_TTL_MS, now_ms=None):
        self.min_model_move = min_model_move
        self.market_flat = market_flat
        self.steam_move = steam_move
        self.lag_ttl_ms = lag_ttl_ms
        self._now = now_ms or (lambda: int(time.time() * 1000))
        self._tracks: dict[str, _Track] = {}

    def observe(self, match_id: str, *, model_p: Optional[float],
                market_p: Optional[float]) -> Optional[LagEvent]:
        """Record the current pair and classify any divergence.

        Returns an event only on a TRANSITION worth reporting. Called on every
        tick, so returning something every time would make the caller do the
        de-duplication that belongs here.
        """
        if model_p is None or market_p is None:
            return None

        now = self._now()
        tr = self._tracks.setdefault(match_id, _Track())

        # First observation establishes a baseline; a delta needs two points.
        if tr.model_p is None or tr.market_p is None:
            tr.model_p, tr.market_p, tr.ts_ms = model_p, market_p, now
            return None

        d_model = model_p - tr.model_p
        d_market = market_p - tr.market_p

        # Close an open lag as soon as the market moves our way — that closing
        # time IS the measurement this module exists to produce.
        event: Optional[LagEvent] = None
        if tr.open_lag is not None:
            same_way = (d_market > self.market_flat and tr.open_lag.model_delta > 0) or \
                       (d_market < -self.market_flat and tr.open_lag.model_delta < 0)
            expired = now - tr.open_lag.ts_ms > self.lag_ttl_ms
            if same_way:
                tr.open_lag.reaction_ms = now - tr.open_lag.ts_ms
                tr.reactions.append(tr.open_lag.reaction_ms)
                event = LagEvent(match_id=match_id, kind=Divergence.AGREE,
                                 model_delta=d_model, market_delta=d_market,
                                 model_p=model_p, market_p=market_p, ts_ms=now,
                                 reaction_ms=tr.open_lag.reaction_ms)
                tr.open_lag = None
            elif expired:
                # The market had its chance and declined. Not a reaction —
                # counting it as one would inflate every latency statistic.
                tr.open_lag = None

        if event is None:
            moved_model = abs(d_model) >= self.min_model_move
            moved_market = abs(d_market)

            if moved_model and moved_market <= self.market_flat:
                event = LagEvent(match_id=match_id, kind=Divergence.LAG,
                                 model_delta=d_model, market_delta=d_market,
                                 model_p=model_p, market_p=market_p, ts_ms=now)
                tr.open_lag = event
            elif moved_market >= self.steam_move and not moved_model:
                # The market knows something we do not. Report it as a reason
                # to stand aside, never as an edge.
                event = LagEvent(match_id=match_id, kind=Divergence.STEAM,
                                 model_delta=d_model, market_delta=d_market,
                                 model_p=model_p, market_p=market_p, ts_ms=now)

        tr.model_p, tr.market_p, tr.ts_ms = model_p, market_p, now
        return event

    def open_lag(self, match_id: str) -> Optional[LagEvent]:
        """The unresolved lag on this match, if any and still fresh."""
        tr = self._tracks.get(match_id)
        if tr is None or tr.open_lag is None:
            return None
        if self._now() - tr.open_lag.ts_ms > self.lag_ttl_ms:
            return None
        return tr.open_lag

    def reaction_stats(self, match_id: Optional[str] = None) -> dict:
        """Observed market reaction times.

        Median, not mean: reaction times are long-tailed — most are quick and a
        few sit near the TTL — and a mean would be dragged by the tail into a
        number that describes no actual reaction.
        """
        if match_id is not None:
            tr = self._tracks.get(match_id)
            samples = list(tr.reactions) if tr else []
        else:
            samples = [r for t in self._tracks.values() for r in t.reactions]
        if not samples:
            return {"n": 0, "median_ms": None, "fastest_ms": None, "slowest_ms": None}
        s = sorted(samples)
        return {
            "n": len(s),
            "median_ms": s[len(s) // 2],
            "fastest_ms": s[0],
            "slowest_ms": s[-1],
        }

    def forget(self, match_id: str) -> None:
        self._tracks.pop(match_id, None)
