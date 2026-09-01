"""The orchestrator — the loop that ties every piece together.

    provider -> normalize -> sequence check -> state -> model -> market
             -> signal gate -> room broadcast

Everything it calls already exists and is tested on its own; this file owns the
ORDER, and the order is the design:

  1. Integrity BEFORE state. An event the tracker rejects must not reach the
     scoreboard.
  2. State BEFORE model. The model prices what the feed said, never a guess.
  3. Model AND market BEFORE the signal gate. A signal needs both, and a
     missing one is silence rather than a default.
  4. The gate BEFORE the broadcast. Nothing reaches a user that the gate has
     not cleared — §24 is enforced in exactly one place, here.

Latency (§21) is stamped as the event moves so the breakdown reflects real work
rather than an estimate.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from execution.live.calibration import CalibrationRecorder
from execution.live.edge_publisher import EdgePublisher
from execution.live.engine import ModelBridge, game_ladder
from execution.live.events import LiveEvent
from execution.live.feed import Health, LatencyBreakdown, worst
from execution.live.gateway import RoomRegistry, build_payload
from execution.live.odds import MarketView
from execution.live.providers.failover import FailoverManager
from execution.live.signals import SignalEngine
from execution.live.state import InMemoryStateStore, MatchStateMachine, StateStore


@dataclass
class MatchContext:
    """Per-match working set: the state machine, the market, the metadata."""

    machine: MatchStateMachine
    market: dict = field(default_factory=dict)      # market name -> MarketView
    player1: str = ""
    player2: str = ""
    surface: str = "Hard"
    # Last health we told viewers about. A transition is news in its own right.
    last_health: Optional[Health] = None

    def view(self, match_id: str, market: str) -> MarketView:
        mv = self.market.get(market)
        if mv is None:
            mv = MarketView(match_id=match_id, market=market)
            self.market[market] = mv
        return mv


class LiveRuntime:
    """Owns the pipeline for every watched match."""

    def __init__(self, *, provider=None, store: Optional[StateStore] = None,
                 registry: Optional[RoomRegistry] = None,
                 model: Optional[ModelBridge] = None,
                 signals: Optional[SignalEngine] = None,
                 failover: Optional[FailoverManager] = None,
                 publisher: Optional[EdgePublisher] = None,
                 recorder: Optional[CalibrationRecorder] = None,
                 now_ms=None):
        self.provider = provider
        self.store: StateStore = store or InMemoryStateStore()
        self.model = model or ModelBridge()
        self.signals = signals or SignalEngine()
        self.failover = failover
        # Optional edge fan-out. Local rooms work without it; this is what
        # makes viewer number 1,000 free.
        self.publisher = publisher
        # Optional calibration recording. Every published probability is an
        # observation, and the reason no calibration exists today is that
        # nothing ever wrote them down with an unambiguous orientation.
        self.recorder = recorder
        self._recorded: set = set()
        self._now = now_ms or (lambda: int(time.time() * 1000))
        self.contexts: dict[str, MatchContext] = {}
        self.registry = registry or RoomRegistry(
            on_first_viewer=self._on_first_viewer,
            on_last_viewer=self._on_last_viewer,
        )
        self.processed = 0
        self.published = 0
        self.rejected = 0
        self._last_latency = LatencyBreakdown()
        self._running = False

    # ── room lifecycle -> upstream subscription (§20) ────────────────────────

    async def _on_first_viewer(self, match_id: str) -> None:
        if self.provider is not None:
            await self.provider.subscribe(match_id)

    async def _on_last_viewer(self, match_id: str) -> None:
        """Unsubscribing discards the point tape (see provider.unsubscribe).

        The context is kept rather than deleted so that a viewer returning
        within the same match does not restart momentum from nothing. It is
        dropped when the match finishes.
        """
        if self.provider is not None:
            await self.provider.unsubscribe(match_id)

    def register_match(self, match_id: str, *, player1: str, player2: str,
                       surface: str = "Hard", best_of: int = 3) -> MatchContext:
        """Attach player identity to a match id.

        The feed gives ids; the model needs names, because `inplay.py` looks up
        serve strength by player. Without this the model has no opinion and the
        gate correctly stays silent — so a missing registration is a silent
        product, which is why it is explicit rather than inferred.
        """
        ctx = MatchContext(
            machine=MatchStateMachine(match_id, store=self.store, player1=player1,
                                      player2=player2, surface=surface, best_of=best_of),
            player1=player1, player2=player2, surface=surface,
        )
        self.contexts[match_id] = ctx
        return ctx

    def _context(self, match_id: str) -> MatchContext:
        ctx = self.contexts.get(match_id)
        if ctx is None:
            ctx = MatchContext(machine=MatchStateMachine(match_id, store=self.store))
            self.contexts[match_id] = ctx
        return ctx

    # ── the pipeline ─────────────────────────────────────────────────────────

    async def handle_event(self, event: LiveEvent) -> Optional[dict]:
        """One event, end to end. Returns the payload broadcast, or None."""
        t_received = event.received_ts or self._now()
        ctx = self._context(event.match_id)

        # 1. integrity + state
        state, status, transitions = ctx.machine.apply(event)
        t_state = time.perf_counter()

        if self.failover is not None and self.provider is not None:
            self.failover.record_event(getattr(self.provider, "name", "unknown"),
                                       event.provider_latency_ms)

        self.processed += 1

        # 2. model
        t0 = time.perf_counter()
        fair = self.model.price(state, transitions)
        model_us = int((time.perf_counter() - t0) * 1_000_000)

        # 3. market
        mv = ctx.view(event.match_id, "match_winner")
        fair_market = mv.fair()
        market_p1 = fair_market[0] if fair_market else None
        price_source = fair_market[2] if fair_market else ""
        # An absent market must not drag the reported health down: the gate
        # already refuses to signal without a price (p_market is None), so the
        # only thing folding OFFLINE in here achieves is telling the user their
        # working scoreboard is broken.
        odds_health = mv.health(now_ms=self._now()) if mv.has_price else status.health

        # 4. the gate
        signal = self.signals.evaluate(
            match_id=event.match_id, market="match_winner", selection="p1",
            p_model=fair.p1 if fair else None,
            p_market=market_p1,
            feed_health=status.health, odds_health=odds_health,
            estimates=(fair.components if fair else {}),
            is_live=True, price_source=price_source,
        )
        if signal is None and not status.tradeable:
            self.rejected += 1

        latency = LatencyBreakdown(
            provider_ms=event.provider_latency_ms,
            processing_ms=int((t_state - t0) * 1000) if t_state > t0 else 0,
            model_ms=model_us // 1000,
        )
        self._last_latency = latency

        # A signal or a score change is always news; a model nudge may not be.
        #
        # A HEALTH TRANSITION is also always news, and this is easy to get
        # wrong: suppression only inspects the model probability, so a feed
        # going DEGRADED while the probability sits still was silently dropped
        # and the client kept rendering the last good state as if it were live.
        # That is the exact failure §22 exists to prevent, arriving through the
        # broadcast layer instead of the data layer.
        combined = worst(status.health, odds_health)   # odds_health == feed health when no price
        health_changed = ctx.last_health is not None and combined is not ctx.last_health
        ctx.last_health = combined

        payload = build_payload(
            match_id=event.match_id, state=state, fair=fair,
            ladder=game_ladder(state, fair),
            market=({"p1": round(market_p1, 4), "source": price_source}
                    if market_p1 is not None else None),
            signal=signal,
            health=combined,
            latency=latency,
            sequence=status.last_sequence,
        )

        if signal is not None:
            kind = "signal"
        elif health_changed:
            kind = "health"
        elif transitions:
            kind = "score"
        else:
            kind = "model"
        sent = await self.registry.broadcast(event.match_id, payload, kind=kind)
        if self.publisher is not None:
            # Independent of local delivery: the edge has its own viewers, and
            # suppressing a push because nobody is attached HERE would leave
            # them stale.
            await self.publisher.publish(event.match_id, payload)
        if sent:
            self.published += 1

        self._record_for_calibration(event.match_id, ctx, fair)
        return payload if sent else None

    def _record_for_calibration(self, match_id: str, ctx, fair) -> None:
        """Write one observation per match per model tier change.

        Recording every point would store thousands of near-identical rows for
        one match and let a single long match dominate the fit. One prediction
        per match per significant re-price keeps the dataset weighted by
        MATCHES, which is the unit the model is actually wrong about.
        """
        if self.recorder is None or fair is None or not ctx.player1:
            return
        key = (match_id, fair.tier)
        if key in self._recorded:
            return
        self._recorded.add(key)
        try:
            self.recorder.predict(
                match_id=match_id, market="match", selection=ctx.player1,
                p_model=fair.p1, source=fair.source)
        except Exception:
            # Calibration bookkeeping must never break the live path.
            pass

    def settle_match(self, match_id: str, winner: str) -> None:
        """Close the loop: tell the recorder who actually won.

        Without this the recorder accumulates predictions that never become
        data, which is indistinguishable from having no recorder at all.
        """
        if self.recorder is None:
            return
        try:
            self.recorder.settle(match_id=match_id, market="match", winner=winner)
        except Exception:
            pass

    async def run(self) -> None:
        """Consume the provider until stopped."""
        if self.provider is None:
            raise RuntimeError("LiveRuntime has no provider")
        self._running = True
        await self.provider.connect()
        try:
            async for event in self.provider.events():
                if not self._running:
                    break
                await self.handle_event(event)
        finally:
            await self.provider.close()

    async def stop(self) -> None:
        self._running = False

    def health(self) -> dict:
        out = {
            "processed": self.processed,
            "published": self.published,
            "rejected": self.rejected,
            "matches": len(self.contexts),
            "model_available": self.model.available,
            "latency": self._last_latency.as_dict(),
        }
        if self.model.unavailable_reason:
            out["model_error"] = self.model.unavailable_reason
        if self.failover is not None:
            out["feeds"] = self.failover.as_dict()
        if self.publisher is not None:
            out["edge"] = self.publisher.health()
        if self.recorder is not None:
            total, settled = self.recorder.count()
            out["calibration"] = {"observations": total, "settled": settled}
        return out
