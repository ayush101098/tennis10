"""Livesport provider — the second failover leg.

Driven through an injected client, because the cases that matter (a score jump,
an unchanged poll, a match ending, a client that throws) cannot be requested
from a live source.
"""

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.live.events import EventType, P1, P2  # noqa: E402
from execution.live.events import derive_transitions  # noqa: E402
from execution.live.providers.failover import FailoverManager, ProviderHealth  # noqa: E402
from execution.live.providers.livesport import (  # noqa: E402
    LivesportProvider, match_to_event,
)


@dataclass
class FakeMatch:
    """Mirrors execution.flashscore.Match in the fields this provider reads."""

    fs_id: str = "abc123"
    tour: str = "ATP"
    tournament: str = "US Open"
    surface: str = "Hard"
    home: str = "Alice"
    away: str = "Bob"
    status: str = "live"
    set_index: int = 1
    home_sets: int = 0
    away_sets: int = 0
    home_games: list = field(default_factory=lambda: [3])
    away_games: list = field(default_factory=lambda: [2])
    home_tb: dict = field(default_factory=dict)
    away_tb: dict = field(default_factory=dict)
    point_home: str = "30"
    point_away: str = "15"


class FakeClient:
    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def live_matches(self, with_points=True):
        self.calls += 1
        if not self.frames:
            return []
        return self.frames.pop(0)


# ── conversion ───────────────────────────────────────────────────────────────

def test_match_becomes_a_canonical_event():
    ev = match_to_event(FakeMatch(), sequence=1, received_ms=1000)
    assert ev.match_id == "abc123"
    assert ev.score.games == (3, 2)
    assert ev.score.points == ("30", "15")
    assert ev.score.sets == (0, 0)
    assert ev.event_type is EventType.POINT


def test_the_current_set_is_used_not_the_first():
    # home_games is per-set; taking [0] would freeze the board on set one, which
    # looks exactly like a stale feed.
    m = FakeMatch(set_index=2, home_games=[6, 1], away_games=[4, 3],
                  home_sets=1, away_sets=0)
    ev = match_to_event(m, sequence=1)
    assert ev.score.games == (1, 3)
    assert ev.score.sets == (1, 0)


def test_advantage_is_normalised():
    assert match_to_event(FakeMatch(point_home="AD", point_away="40"),
                          sequence=1).score.points == ("A", "40")


def test_missing_point_score_is_love_not_a_crash():
    ev = match_to_event(FakeMatch(point_home=None, point_away=None), sequence=1)
    assert ev.score.points == ("0", "0")


def test_tiebreak_is_flagged_from_the_tiebreak_map():
    m = FakeMatch(set_index=1, home_tb={1: 5}, away_tb={1: 4})
    assert match_to_event(m, sequence=1).score.tiebreak is True


def test_six_all_is_a_tiebreak_even_before_the_map_populates():
    # Seen live: a set at 6-6 reported tiebreak=False because the tiebreak map
    # only fills once points are played. Both setengine and the model's
    # significance tiering key off this flag, so the most important game of the
    # set was being priced as an ordinary one.
    m = FakeMatch(home_games=[6], away_games=[6], home_tb={}, away_tb={})
    assert match_to_event(m, sequence=1).score.tiebreak is True


def test_five_all_is_not_a_tiebreak():
    m = FakeMatch(home_games=[5], away_games=[5])
    assert match_to_event(m, sequence=1).score.tiebreak is False


def test_the_server_is_never_guessed():
    # Flashscore renders it as an icon. None means unknown, and downstream
    # refuses to infer a break without it.
    assert match_to_event(FakeMatch(), sequence=1).server is None


def test_no_break_is_fabricated_without_a_server():
    prev = match_to_event(FakeMatch(home_games=[3], away_games=[3]), sequence=1)
    curr = match_to_event(FakeMatch(home_games=[3], away_games=[4]), sequence=2)
    out = derive_transitions(prev, curr)
    assert EventType.GAME_END in out
    assert EventType.BREAK not in out, "a fabricated break is worse than a missing one"


# ── polling behaviour ────────────────────────────────────────────────────────

def test_only_real_changes_are_emitted():
    # Re-emitting an identical scoreboard every 8s would make the model burn a
    # reprice on an event that moved nothing.
    same = FakeMatch()
    p = LivesportProvider(client=FakeClient([[same], [FakeMatch()], [FakeMatch(point_home="40")]]))
    assert len(p.poll_events()) == 1        # first sighting
    assert p.poll_events() == []            # unchanged
    assert len(p.poll_events()) == 1        # point moved


def test_sequence_increments_per_match():
    p = LivesportProvider(client=FakeClient([
        [FakeMatch(point_home="30")],
        [FakeMatch(point_home="40")],
        [FakeMatch(point_home="A")],
    ]))
    seqs = [p.poll_events()[0].sequence for _ in range(3)]
    assert seqs == [1, 2, 3]


def test_two_matches_have_independent_sequences():
    a = FakeMatch(fs_id="a", point_home="15")
    b = FakeMatch(fs_id="b", point_home="30")
    p = LivesportProvider(client=FakeClient([[a, b]]))
    evs = {e.match_id: e.sequence for e in p.poll_events()}
    assert evs == {"a": 1, "b": 1}


def test_subscriptions_filter_the_poll():
    # Upstream cost should follow matches watched, not matches available.
    p = LivesportProvider(client=FakeClient([[FakeMatch(fs_id="a"), FakeMatch(fs_id="b")]]))
    asyncio.run(p.subscribe("a"))
    evs = p.poll_events()
    assert [e.match_id for e in evs] == ["a"]


def test_no_subscriptions_means_everything():
    p = LivesportProvider(client=FakeClient([[FakeMatch(fs_id="a"), FakeMatch(fs_id="b")]]))
    assert len(p.poll_events()) == 2


def test_a_throwing_client_is_recorded_not_raised():
    # An exception here would kill the consumer for every match, not just this
    # feed. The failover manager watches last_error instead.
    class Boom:
        def live_matches(self, with_points=True):
            raise ConnectionError("livesport unreachable")

    p = LivesportProvider(client=Boom())
    assert p.poll_events() == []
    assert "ConnectionError" in (p.last_error or "")


def test_resync_forces_the_next_poll_to_re_emit():
    m = FakeMatch()
    p = LivesportProvider(client=FakeClient([[m], [FakeMatch()], [FakeMatch()]]))
    p.poll_events()
    assert p.poll_events() == []             # unchanged, so silent
    asyncio.run(p.resync("abc123"))
    assert len(p.poll_events()) == 1         # state re-sent after a resync


def test_provider_declares_it_cannot_detect_gaps():
    # A polling source has no sequence of record. Saying so is what stops a
    # consumer reading "no gaps" as evidence of completeness.
    assert LivesportProvider.has_sequence_guarantee is False


def test_connect_is_idempotent_and_close_is_safe():
    p = LivesportProvider(client=FakeClient([]))

    async def run():
        await p.connect()
        await p.connect()
        assert p.connected
        await p.close()
        await p.close()
        assert not p.connected

    asyncio.run(run())


def test_events_stream_yields_and_stops_on_close():
    p = LivesportProvider(client=FakeClient([[FakeMatch()]]), poll_s=0.01)

    async def run():
        await p.connect()
        got = []
        async for ev in p.events():
            got.append(ev)
            await p.close()          # stop after the first batch
        return got

    assert len(asyncio.run(run())) == 1


# ── failover, now with two real legs ─────────────────────────────────────────

def test_failover_moves_to_livesport_when_sofascore_goes_silent():
    """The reason this provider exists.

    Until now FailoverManager managed one provider, which makes it a health
    tracker rather than a failover.
    """
    clock = _Clock()
    fm = FailoverManager(["sofascore", "livesport"], now_ms=clock,
                         silence_ms=20_000, debounce_ms=10_000)
    fm.record_event("sofascore", 200)
    fm.record_event("livesport", 400)
    assert fm.decide() == "sofascore"

    # Silence DETECTS, debounce CONFIRMS — the two compose, so failover takes
    # silence_ms + debounce_ms. A tennis point can be 30s apart, which is why
    # the bar is a sustained outage rather than a gap between points.
    clock.advance(25_000)                     # sofascore challenged the IP
    fm.record_event("livesport", 400)
    assert fm.decide() == "sofascore"         # detected silent, not yet confirmed
    assert fm.stats["sofascore"].health is ProviderHealth.SILENT

    clock.advance(11_000)                     # still nothing from sofascore
    fm.record_event("livesport", 400)
    assert fm.decide() == "livesport"


class _Clock:
    def __init__(self, t=1_788_000_000_000):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms
        return self.t
