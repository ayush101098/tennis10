"""Reconstructing the server and the point tape.

Both exist because SofaScore is challenging this IP: the feed falls back to
Flashscore, which carries neither. These are the cases that decide whether the
reconstruction is trustworthy — a wrong server is worse than no server.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.live.events import P1, P2  # noqa: E402
from execution.live.pointtape import (  # noqa: E402
    PointTape, ServeTracker, served_counts,
)


def stats(home_served, away_served):
    """A statistics payload carrying 'Service Points Won (won/served)'."""
    return {"statistics": [{"period": "ALL", "groups": [{
        "groupName": "Points",
        "statisticsItems": [
            {"name": "Service Points Won",
             "home": f"70% (7/{home_served})", "away": f"60% (6/{away_served})"},
        ]}]}]}


# ── reading the counters ─────────────────────────────────────────────────────

def test_served_counts_reads_the_denominator():
    assert served_counts(stats(11, 22)) == (11, 22)


def test_unrecognised_payload_gives_no_anchor_rather_than_a_wrong_one():
    assert served_counts({}) == (None, None)
    assert served_counts({"statistics": [{"groups": [{"statisticsItems": [
        {"name": "Aces", "home": "3", "away": "1"}]}]}]}) == (None, None)


# ── anchoring the server ─────────────────────────────────────────────────────

def test_one_sample_is_not_enough():
    # The trick is the DELTA; a single reading says nothing about who is on
    # serve now.
    t = ServeTracker()
    assert t.observe_statistics("m1", stats(10, 10)) is None
    assert not t.state("m1").anchored


def test_the_player_whose_served_count_grew_is_serving():
    t = ServeTracker()
    t.observe_statistics("m1", stats(10, 10))
    assert t.observe_statistics("m1", stats(10, 12)) == P2
    assert t.state("m1").source == "statistics"

    t2 = ServeTracker()
    t2.observe_statistics("m2", stats(4, 5))
    assert t2.observe_statistics("m2", stats(6, 5)) == P1


def test_no_points_played_teaches_nothing():
    # Measured live: a match can sit unchanged across a poll. Guessing from
    # that is how a wrong server gets in.
    t = ServeTracker()
    t.observe_statistics("m1", stats(11, 22))
    assert t.observe_statistics("m1", stats(11, 22)) is None
    assert not t.state("m1").anchored


def test_serve_alternates_when_a_game_completes():
    t = ServeTracker()
    t.observe_statistics("m1", stats(0, 0))
    t.observe_statistics("m1", stats(4, 0))          # anchored to P1
    assert t.server("m1") == P1
    assert t.observe_game_completed("m1") == P2
    assert t.observe_game_completed("m1") == P1
    assert t.state("m1").source == "alternation"


def test_alternation_does_nothing_before_an_anchor():
    # Without an anchor there is no server to flip, and inventing one here
    # would defeat the whole point.
    t = ServeTracker()
    assert t.observe_game_completed("m1") is None
    assert t.server("m1") is None


def test_reanchor_is_requested_after_enough_games():
    t = ServeTracker(reanchor_every=3)
    t.observe_statistics("m1", stats(0, 0))
    t.observe_statistics("m1", stats(4, 0))
    assert not t.needs_anchor("m1")
    for _ in range(3):
        t.observe_game_completed("m1")
    assert t.needs_anchor("m1"), "drift must be correctable"


def test_an_unanchored_match_always_wants_an_anchor():
    assert ServeTracker().needs_anchor("never-seen")


# ── the point tape ───────────────────────────────────────────────────────────

def test_first_reading_is_only_a_baseline():
    assert PointTape().observe("m1", ("0", "0"), (0, 0)) is None


def test_a_point_is_attributed_to_whoever_advanced():
    tape = PointTape()
    tape.observe("m1", ("0", "0"), (0, 0))
    pt = tape.observe("m1", ("15", "0"), (0, 0), server=P1)
    assert pt is not None and pt.winner == P1
    assert pt.server == P1

    pt = tape.observe("m1", ("15", "15"), (0, 0), server=P1)
    assert pt.winner == P2


def test_advantage_and_back_to_deuce():
    tape = PointTape()
    tape.observe("m1", ("40", "40"), (3, 3))
    assert tape.observe("m1", ("A", "40"), (3, 3)).winner == P1
    # Losing advantage is the OPPONENT winning a point, not a score correction.
    assert tape.observe("m1", ("40", "40"), (3, 3)).winner == P2


def test_a_game_boundary_is_not_a_point():
    # The point score resets to 0-0 when a game ends; reading that as a point
    # would put a phantom entry in the tape every game.
    tape = PointTape()
    tape.observe("m1", ("40", "30"), (3, 3))
    assert tape.observe("m1", ("0", "0"), (4, 3)) is None


def test_two_points_inside_one_poll_are_counted_as_a_gap_not_invented():
    tape = PointTape()
    tape.observe("m1", ("0", "0"), (0, 0))
    assert tape.observe("m1", ("30", "15"), (0, 0)) is None
    assert tape.gaps == 1, "a sampled tape must admit what it missed"


def test_an_unchanged_scoreboard_produces_nothing():
    tape = PointTape()
    tape.observe("m1", ("15", "0"), (0, 0))
    assert tape.observe("m1", ("15", "0"), (0, 0)) is None
    assert tape.gaps == 0


def test_tiebreak_points_are_counted_numerically():
    # Tiebreaks score 1,2,3… not 15/30/40, so the game-scoring ladder does not
    # apply and would otherwise register every tiebreak point as a gap.
    tape = PointTape()
    tape.observe("m1", ("3", "2"), (6, 6))
    assert tape.observe("m1", ("3", "3"), (6, 6)).winner == P2
    assert tape.observe("m1", ("4", "3"), (6, 6)).winner == P1
    assert tape.gaps == 0


def test_the_tape_accumulates_per_match():
    tape = PointTape()
    for pts in [("0", "0"), ("15", "0"), ("15", "15"), ("30", "15")]:
        tape.observe("m1", pts, (0, 0), server=P1)
    tape.observe("m2", ("0", "0"), (0, 0))
    tape.observe("m2", ("0", "15"), (0, 0))
    assert [p.winner for p in tape.points("m1")] == [P1, P2, P1]
    assert [p.winner for p in tape.points("m2")] == [P2]


def test_forget_releases_a_finished_match():
    tape = PointTape()
    tape.observe("m1", ("0", "0"), (0, 0))
    tape.observe("m1", ("15", "0"), (0, 0))
    tape.forget("m1")
    assert tape.points("m1") == []
