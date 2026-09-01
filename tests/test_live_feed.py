"""Tests for the live ingestion layer.

These assert the cases a live feed cannot be asked to produce on demand: a
dropped event, a reordered one, a replayed one, and silence. Those are exactly
the conditions under which the model must refuse to publish a signal, so they
are the ones worth pinning down.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.live.events import (  # noqa: E402
    EventType, LiveEvent, P1, P2, Score, derive_transitions,
)
from execution.live.feed import (  # noqa: E402
    Health, SequenceTracker, health_for_age, worst,
)
from execution.live.provider import ReplayProvider, ScriptedEvent  # noqa: E402


T0 = 1_788_000_000_000


def ev(seq, *, ts=T0, received=None, score=None, server=None,
       etype=EventType.POINT) -> LiveEvent:
    return LiveEvent(
        match_id="m1", sequence=seq, event_type=etype,
        provider_ts=ts, received_ts=received if received is not None else ts,
        score=score or Score(), server=server,
    )


class Clock:
    """Virtual clock, so a test for 'offline after 15 seconds' runs instantly."""

    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms
        return self.t


# ── Sequence integrity ───────────────────────────────────────────────────────

def test_contiguous_stream_stays_live():
    clock = Clock()
    tr = SequenceTracker(now_ms=clock)
    for s in range(100, 105):
        st = tr.observe(ev(s, received=clock.t))
    assert st.health is Health.LIVE
    assert st.missing == ()
    assert st.tradeable


def test_gap_is_not_declared_inside_the_reorder_window():
    # 100, 101, 103 — 102 may still be in flight. Declaring a gap immediately
    # would resync on ordinary network jitter, and constant resyncing is its
    # own outage.
    clock = Clock()
    tr = SequenceTracker(now_ms=clock, grace_ms=750)
    tr.observe(ev(100, received=clock.t))
    tr.observe(ev(101, received=clock.t))
    st = tr.observe(ev(103, received=clock.advance(50)))
    assert st.health is Health.LIVE
    assert st.missing == ()


def test_late_event_closes_the_hole_and_counts_as_reorder():
    clock = Clock()
    tr = SequenceTracker(now_ms=clock, grace_ms=750)
    for s in (100, 101, 103):
        tr.observe(ev(s, received=clock.t))
    st = tr.observe(ev(102, received=clock.advance(80)))
    assert st.missing == ()
    assert st.reordered == 1
    assert st.health is Health.LIVE


def test_hole_that_outlives_the_window_degrades_the_feed():
    clock = Clock()
    tr = SequenceTracker(now_ms=clock, grace_ms=750)
    tr.observe(ev(100, received=clock.t))
    tr.observe(ev(103, received=clock.t))
    clock.advance(800)                     # window closes with 101, 102 absent
    st = tr.status()
    assert st.health is Health.DEGRADED
    assert st.missing == (101, 102)
    # The whole point: a degraded feed may not produce a signal, however fresh.
    assert not st.tradeable


def test_degraded_outranks_freshness():
    # A hole matters even when events are arriving briskly — the state we hold
    # is wrong, and being wrong quickly is not better.
    clock = Clock()
    tr = SequenceTracker(now_ms=clock, grace_ms=100)
    tr.observe(ev(1, received=clock.t))
    tr.observe(ev(5, received=clock.t))
    clock.advance(150)
    st = tr.observe(ev(6, received=clock.t))
    assert st.health is Health.DEGRADED
    assert not st.tradeable


def test_duplicate_replay_is_counted_not_priced_twice():
    # Providers replay on reconnect. Processing a point twice would double-count
    # it in the momentum engine.
    clock = Clock()
    tr = SequenceTracker(now_ms=clock)
    tr.observe(ev(10, received=clock.t))
    tr.observe(ev(11, received=clock.t))
    st = tr.observe(ev(11, received=clock.t))
    assert st.duplicates == 1
    assert st.last_sequence == 11


def test_resync_clears_the_gap():
    # Without this a single early loss would pin a match to DEGRADED for its
    # whole duration, which in practice means nobody trusts the flag.
    clock = Clock()
    tr = SequenceTracker(now_ms=clock, grace_ms=10)
    tr.observe(ev(1, received=clock.t))
    tr.observe(ev(4, received=clock.t))
    clock.advance(50)
    assert tr.status().health is Health.DEGRADED
    tr.resynced(up_to_sequence=4)
    assert tr.status().health is not Health.DEGRADED


# ── Freshness ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("age_ms,expected", [
    (0, Health.LIVE), (1_999, Health.LIVE),
    (2_000, Health.DELAYED), (4_999, Health.DELAYED),
    (5_000, Health.STALE), (14_999, Health.STALE),
    (15_000, Health.OFFLINE), (60_000, Health.OFFLINE),
])
def test_freshness_bands(age_ms, expected):
    assert health_for_age(age_ms) is expected


def test_silence_ages_the_feed_out():
    clock = Clock()
    tr = SequenceTracker(now_ms=clock)
    tr.observe(ev(1, received=clock.t))
    assert tr.status().health is Health.LIVE
    clock.advance(3_000)
    assert tr.status().health is Health.DELAYED
    clock.advance(10_000)
    assert tr.status().health is Health.STALE
    clock.advance(10_000)
    assert tr.status().health is Health.OFFLINE
    assert not tr.status().tradeable


def test_delayed_is_still_tradeable_but_stale_is_not():
    # A two-second-old score is still a score. A five-second-old one, in a
    # market that reprices on every point, is not.
    clock = Clock()
    tr = SequenceTracker(now_ms=clock)
    tr.observe(ev(1, received=clock.t))
    clock.advance(3_000)
    assert tr.status().tradeable
    clock.advance(3_000)
    assert not tr.status().tradeable


def test_worst_combines_two_feeds():
    # A live score against a stale price is not a live signal.
    assert worst(Health.LIVE, Health.STALE) is Health.STALE
    assert worst(Health.LIVE, Health.LIVE) is Health.LIVE
    assert worst(Health.DELAYED, Health.DEGRADED) is Health.DEGRADED


# ── Latency ──────────────────────────────────────────────────────────────────

def test_provider_latency_never_negative():
    # A provider clock running ahead of ours must not report negative latency,
    # which would drag any average built on it below the truth.
    assert ev(1, ts=T0 + 500, received=T0).provider_latency_ms == 0
    assert ev(1, ts=T0, received=T0 + 180).provider_latency_ms == 180


# ── Derived transitions ──────────────────────────────────────────────────────

def test_game_end_and_break_are_derived_from_the_scoreboard():
    prev = ev(1, score=Score(games=(3, 3)), server=P1)
    curr = ev(2, score=Score(games=(3, 4)), server=P2)
    out = derive_transitions(prev, curr)
    # P2 won a game P1 was serving — that is a break.
    assert EventType.GAME_END in out
    assert EventType.BREAK in out


def test_hold_is_not_reported_as_a_break():
    prev = ev(1, score=Score(games=(3, 3)), server=P1)
    curr = ev(2, score=Score(games=(4, 3)), server=P1)
    out = derive_transitions(prev, curr)
    assert EventType.GAME_END in out
    assert EventType.BREAK not in out


def test_break_is_never_invented_without_a_known_server():
    # Feeds that omit the server are common. A fabricated break is worse than
    # a missing one: momentum and the signal engine both weight breaks heavily.
    prev = ev(1, score=Score(games=(3, 3)), server=None)
    curr = ev(2, score=Score(games=(3, 4)), server=None)
    out = derive_transitions(prev, curr)
    assert EventType.GAME_END in out
    assert EventType.BREAK not in out


def test_missed_game_does_not_produce_a_phantom_winner():
    # A jump of two games means we lost one. Attributing it to somebody would
    # feed the momentum engine a result that never happened.
    prev = ev(1, score=Score(games=(3, 3)), server=P1)
    curr = ev(2, score=Score(games=(5, 3)), server=P1)
    assert EventType.BREAK not in derive_transitions(prev, curr)


def test_set_change_emits_set_boundaries():
    prev = ev(1, score=Score(sets=(0, 0), games=(5, 4)), server=P1)
    curr = ev(2, score=Score(sets=(1, 0), games=(0, 0)), server=P2)
    out = derive_transitions(prev, curr)
    assert EventType.SET_END in out
    assert EventType.SET_START in out


# ── Provider contract ────────────────────────────────────────────────────────

def test_replay_provider_emits_the_script_in_arrival_order():
    # Arrival order, not sequence order — hiding the reorder here would defeat
    # the tracker that exists to distinguish a reorder from a loss.
    script = [ScriptedEvent(sequence=s) for s in (1, 2, 4, 3)]
    prov = ReplayProvider(script)

    async def run():
        await prov.connect()
        await prov.subscribe("m1")
        return [e.sequence async for e in prov.events()]

    assert asyncio.run(run()) == [1, 2, 4, 3]
    assert prov.connected
    assert prov.subscriptions == {"m1"}


def test_replay_clock_advances_without_real_sleeping():
    script = [ScriptedEvent(sequence=1), ScriptedEvent(sequence=2, delay_ms=20_000)]
    prov = ReplayProvider(script)

    async def run():
        return [e.received_ts async for e in prov.events()]

    ts = asyncio.run(run())
    assert ts[1] - ts[0] == 20_000


def test_subscribe_and_unsubscribe_track_watched_matches():
    # §20: upstream cost should follow the number of matches watched, not the
    # number of viewers.
    prov = ReplayProvider([])

    async def run():
        await prov.subscribe("a")
        await prov.subscribe("b")
        await prov.unsubscribe("a")
        await prov.unsubscribe("missing")   # must not raise
        return prov.subscriptions

    assert asyncio.run(run()) == {"b"}


def test_payload_is_compact():
    # §17: the browser gets what changed, not the event's provenance.
    payload = ev(7, score=Score(sets=(1, 0), games=(4, 3), points=("30", "15")),
                 server=P1).as_payload()
    assert set(payload) == {"match_id", "sequence", "type", "score", "server", "ts"}
    assert payload["score"]["points"] == ["30", "15"]
