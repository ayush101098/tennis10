"""Signal engine (PRD §15, §16, §24) — when to speak, and when to shut up.

TWO THINGS THIS FILE IS RESPONSIBLE FOR

  1. HYSTERESIS. An edge that oscillates around a single threshold produces
     ENTRY/EXIT/ENTRY/EXIT every few seconds. That is not a signal, it is a
     strobe, and a user watching it learns to ignore the panel. So entry and
     exit use different thresholds and a signal has to EARN its way out of the
     state it is in.

  2. THE PUBLISH GATE. Nothing reaches a user unless the data behind it is
     vouched for: the feed is not degraded, the price is not stale, the model
     actually has an opinion, and `edgescore` grades the edge above noise.

WHY THE GATE LEANS ON edgescore.py RATHER THAN A NEW THRESHOLD
    The PRD proposes ENTRY at edge > 6%. Against this model that fires on
    almost everything: the measured model-vs-market gap is ~13 points per leg
    and ~14 points high on favourites, and `execution/calibrate.py` records the
    same finding independently ("predicts 80%+, wins ~60%"). A fixed edge
    threshold cannot tell a real 6% from the 13% of bias sitting under it.

    `edgescore.score_edge` already solves this: it divides the edge by an
    uncertainty that INCLUDES the measured calibration error, so an edge is
    only green when it is large relative to how wrong we know we are. That is
    the gate this engine uses. A raw-edge floor is kept as a second, blunter
    condition — both must pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from execution.live.feed import Health, worst


class SignalStatus(str, Enum):
    WATCH = "WATCH"         # edge forming, not actionable
    ENTRY = "ENTRY"         # act now
    HOLD = "HOLD"           # position on, edge still there
    EXIT = "EXIT"           # edge gone, close
    STOP = "STOP"           # invalidated — data or model failed, not a price move
    EXPIRED = "EXPIRED"     # market or match ended


# §16. Deliberately asymmetric: it is harder to get in than to stay in, and
# harder to stay in than to be warned about. Equal thresholds are what produce
# the strobe.
ENTRY_EDGE = 0.06
HOLD_EDGE = 0.04
EXIT_EDGE = 0.02
WATCH_EDGE = 0.03

# A signal must survive this long before it is published at all. A one-tick
# spike is usually a bad price about to be corrected, not an opportunity.
MIN_DWELL_MS = 1_500

# Beyond this the edge is a data fault, not money. Matches the 20% quarantine
# the rest of the product already applies.
SUSPECT_EDGE = 0.20


@dataclass
class Signal:
    """One actionable opinion on one selection of one market (§15)."""

    match_id: str
    market: str
    selection: str                     # "p1" | "p2"
    model_probability: float
    market_probability: float
    edge: float
    confidence: float                  # 0-1, from edgescore's grade + EdgeScore
    status: SignalStatus
    timestamp: int
    reasons: list = field(default_factory=list)
    price_source: str = ""
    health: Health = Health.LIVE

    def as_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "market": self.market,
            "selection": self.selection,
            "model_probability": round(self.model_probability, 4),
            "market_probability": round(self.market_probability, 4),
            "edge": round(self.edge, 4),
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "timestamp": self.timestamp,
            "reasons": self.reasons,
            "price_source": self.price_source,
            "health": self.health.value,
        }


@dataclass
class _Track:
    """Per (match, market, selection) memory — what hysteresis needs."""

    status: SignalStatus = SignalStatus.EXPIRED
    since_ms: int = 0
    first_seen_ms: int = 0
    last_published: Optional[Signal] = None


class SignalEngine:
    """Turns (model probability, market price, health) into stable signals."""

    def __init__(self, *, entry=ENTRY_EDGE, hold=HOLD_EDGE, exit_=EXIT_EDGE,
                 watch=WATCH_EDGE, dwell_ms=MIN_DWELL_MS, now_ms=None, scorer=None):
        self.entry, self.hold, self.exit_, self.watch = entry, hold, exit_, watch
        self.dwell_ms = dwell_ms
        self._now = now_ms or (lambda: int(time.time() * 1000))
        # `scorer(p_model, p_market, estimates, is_live, source, liquidity_ok)`
        # -> anything with .tradeable / .reasons / .edge_score. Defaults to
        # edgescore.score_edge; swappable so the ensemble can change what
        # "confident" means without touching the state machine.
        self._scorer = scorer
        self._tracks: dict[tuple, _Track] = {}

    def evaluate(self, *, match_id: str, market: str, selection: str,
                 p_model: Optional[float], p_market: Optional[float],
                 feed_health: Health, odds_health: Health,
                 estimates: Optional[dict] = None,
                 is_live: bool = True, price_source: str = "",
                 liquidity_ok: bool = True) -> Optional[Signal]:
        """Evaluate one selection. Returns a Signal to publish, or None.

        None means "say nothing", which is different from EXIT. EXIT is a
        statement about a position; None is the absence of a statement, and
        conflating them would have the UI announce the end of a signal it never
        announced the start of.
        """
        key = (match_id, market, selection)
        tr = self._tracks.setdefault(key, _Track())
        now = self._now()
        health = worst(feed_health, odds_health)

        # ── §24: the publish gate ──
        # Any of these is a hard stop. Ordered so the most fundamental failure
        # is reported rather than a downstream symptom of it.
        if p_model is None:
            return self._stop(tr, key, "model has no opinion", match_id, market,
                              selection, now, health, price_source)
        if p_market is None:
            return self._stop(tr, key, "no market price", match_id, market,
                              selection, now, health, price_source)
        if health in (Health.DEGRADED, Health.STALE, Health.OFFLINE):
            return self._stop(tr, key, f"data {health.value.lower()}", match_id,
                              market, selection, now, health, price_source)

        edge = p_model - p_market

        if edge > SUSPECT_EDGE:
            return self._stop(tr, key,
                              f"edge {edge:+.1%} exceeds the {SUSPECT_EDGE:.0%} sanity bound "
                              "— treat as bad data, not an opportunity",
                              match_id, market, selection, now, health, price_source)

        scored = self._score(p_model, p_market, estimates or {}, is_live,
                             price_source, liquidity_ok)

        # ── hysteresis ──
        prev = tr.status
        nxt = self._next_status(prev, edge, scored)

        if nxt != prev:
            tr.status = nxt
            tr.since_ms = now
            if prev in (SignalStatus.EXPIRED, SignalStatus.STOP):
                tr.first_seen_ms = now

        # A signal must persist before it is worth anyone's attention.
        if nxt in (SignalStatus.ENTRY, SignalStatus.WATCH) and now - tr.since_ms < self.dwell_ms:
            return None
        if nxt is SignalStatus.EXPIRED:
            return None

        sig = Signal(
            match_id=match_id, market=market, selection=selection,
            model_probability=p_model, market_probability=p_market, edge=edge,
            confidence=_confidence(scored), status=nxt, timestamp=now,
            reasons=list(scored.reasons) if scored else [],
            price_source=price_source, health=health,
        )
        tr.last_published = sig
        return sig

    # ── internals ────────────────────────────────────────────────────────────

    def _score(self, p_model, p_market, estimates, is_live, price_source, liquidity_ok):
        """Grade the edge with the existing uncertainty engine.

        Imported lazily so this module is usable (and testable) in an
        environment where the model DB is absent; without it we fall back to a
        conservative stub that grades on raw edge alone and says so.
        """
        if self._scorer is not None:
            return self._scorer(p_model, p_market, estimates, is_live,
                                price_source, liquidity_ok)
        try:
            from execution.edgescore import score_edge
        except Exception:                                  # pragma: no cover
            return _StubScore(p_model - p_market)
        try:
            est = dict(estimates)
            est.setdefault("inplay", p_model)
            est.setdefault("market", p_market)
            return score_edge(p_model, p_market, est, is_live,
                              source="inplay", liquidity_ok=liquidity_ok, stale=False)
        except Exception:                                  # pragma: no cover
            return _StubScore(p_model - p_market)

    def _next_status(self, prev: SignalStatus, edge: float, scored) -> SignalStatus:
        """The hysteresis table.

        Entry additionally requires `scored.tradeable` (edgescore's green
        grade). Exit does NOT require it: a signal must be allowed to end
        because the edge went away, even when the confidence machinery is
        unavailable — otherwise a failure in grading silently pins positions
        open, which is the expensive direction to fail in.
        """
        green = bool(getattr(scored, "tradeable", False))

        if prev in (SignalStatus.ENTRY, SignalStatus.HOLD):
            if edge < self.exit_:
                return SignalStatus.EXIT
            return SignalStatus.HOLD

        if prev is SignalStatus.EXIT:
            # Re-entry has to clear the full entry bar again, not the exit one.
            if edge >= self.entry and green:
                return SignalStatus.ENTRY
            if edge < self.watch:
                return SignalStatus.EXPIRED
            return SignalStatus.EXIT

        # WATCH / STOP / EXPIRED
        if edge >= self.entry and green:
            return SignalStatus.ENTRY
        if edge >= self.watch:
            return SignalStatus.WATCH
        return SignalStatus.EXPIRED

    def _stop(self, tr, key, reason, match_id, market, selection, now, health, price_source):
        """Emit STOP once, then stay quiet.

        Repeating STOP every tick while a feed is down would bury the panel in
        identical rows. The transition is the news; the ongoing condition is
        state, and state belongs in the health indicator.
        """
        if tr.status is SignalStatus.STOP:
            return None
        was_active = tr.status in (SignalStatus.ENTRY, SignalStatus.HOLD,
                                   SignalStatus.WATCH, SignalStatus.EXIT)
        tr.status = SignalStatus.STOP
        tr.since_ms = now
        if not was_active:
            # Never had a signal here, so there is nothing to stop.
            return None
        sig = Signal(
            match_id=match_id, market=market, selection=selection,
            model_probability=0.0, market_probability=0.0, edge=0.0,
            confidence=0.0, status=SignalStatus.STOP, timestamp=now,
            reasons=[reason], price_source=price_source, health=health,
        )
        tr.last_published = sig
        return sig


@dataclass
class _StubScore:
    """Fallback when edgescore is unavailable. Deliberately pessimistic: it
    never grades anything tradeable, so a missing confidence engine degrades to
    WATCH-only rather than to unguarded ENTRY."""

    edge: float

    @property
    def tradeable(self) -> bool:
        return False

    @property
    def reasons(self) -> list:
        return ["confidence engine unavailable — signals limited to WATCH"]

    edge_score: float = 0.0


def _confidence(scored) -> float:
    """Map the grade + EdgeScore onto 0-1 for display.

    Bounded at 0.95: the model's own calibration report says it is
    over-confident, so a UI that can print 100% confidence would be repeating
    the exact error the number is meant to warn about.
    """
    grade = getattr(scored, "grade", "red")
    escore = abs(float(getattr(scored, "edge_score", 0.0)))
    base = {"green": 0.62, "amber": 0.38, "red": 0.12}.get(grade, 0.12)
    return round(min(0.95, base + min(0.33, escore / 12.0)), 3)
