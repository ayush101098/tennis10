"""Local sofa_proxy provider — the personal-use feed.

Driven through an injected fetch, so the cases that matter (a tiebreak, a
doubles match, a proxy that is down, an unchanged poll) are ordinary tests
rather than things you wait for.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.live.events import P1, P2  # noqa: E402
from execution.live.providers.sofaproxy import (  # noqa: E402
    SofaProxyProvider, derive_server, event_from_sofa,
)


def sofa_event(*, eid=1, home="Alice", away="Bob", cat="atp",
               h_sets=0, a_sets=0, h_games=(3,), a_games=(2,),
               h_point="30", a_point="15", first_to_serve=1, tb=None):
    hs = {"current": h_sets, "point": h_point}
    as_ = {"current": a_sets, "point": a_point}
    for i, v in enumerate(h_games, 1):
        hs[f"period{i}"] = v
    for i, v in enumerate(a_games, 1):
        as_[f"period{i}"] = v
    if tb:
        hs[f"period{len(h_games)}TieBreak"] = tb[0]
        as_[f"period{len(a_games)}TieBreak"] = tb[1]
    return {
        "id": eid,
        "homeTeam": {"name": home}, "awayTeam": {"name": away},
        "homeScore": hs, "awayScore": as_,
        "firstToServe": first_to_serve,
        "groundType": "Hardcourt outdoor",
        "status": {"description": "2nd set"},
        "tournament": {"name": "Test Open", "category": {"slug": cat}},
    }


def feed(*events):
    return lambda: {"events": list(events)}


# ── server derivation: the thing Livesport cannot do ─────────────────────────

def test_server_alternates_with_games_played():
    assert derive_server(1, [0], [0]) == P1        # nobody has served yet
    assert derive_server(1, [1], [0]) == P2        # one game gone
    assert derive_server(1, [1], [1]) == P1        # two games gone
    assert derive_server(2, [0], [0]) == P2
    assert derive_server(2, [2], [1]) == P1        # three games gone


def test_parity_carries_across_sets():
    # Serve alternates continuously through the match, not per set.
    assert derive_server(1, [6, 0], [4, 0]) == derive_server(1, [10], [0])


def test_unknown_first_server_is_not_guessed():
    # A wrong server inverts the game market and misprices the set.
    assert derive_server(None, [1], [1]) is None
    assert derive_server(0, [1], [1]) is None


# ── event conversion ─────────────────────────────────────────────────────────

def test_basic_conversion():
    ev = event_from_sofa(sofa_event(), sequence=1)
    assert ev.match_id == "1"
    assert ev.score.games == (3, 2)
    assert ev.score.points == ("30", "15")
    assert ev.server == P2                      # 5 games played, home served first


def test_the_current_set_is_used():
    ev = event_from_sofa(sofa_event(h_games=(6, 2), a_games=(4, 3),
                                    h_sets=1, a_sets=0), sequence=1)
    assert ev.score.games == (2, 3)
    assert ev.score.sets == (1, 0)


def test_tiebreak_points_are_not_mapped_onto_game_scoring():
    # Tiebreak points are plain integers; running them through the 15/30/40
    # map would turn "5" into "0" and silently misreport the tiebreak.
    ev = event_from_sofa(sofa_event(h_games=(6,), a_games=(6,),
                                    h_point="5", a_point="3", tb=(5, 3)), sequence=1)
    assert ev.score.tiebreak is True
    assert ev.score.points == ("5", "3")


def test_six_all_is_a_tiebreak_even_without_the_tiebreak_field():
    ev = event_from_sofa(sofa_event(h_games=(6,), a_games=(6,),
                                    h_point="0", a_point="0"), sequence=1)
    assert ev.score.tiebreak is True


def test_advantage_is_normalised():
    ev = event_from_sofa(sofa_event(h_point="AD", a_point="40"), sequence=1)
    assert ev.score.points == ("A", "40")


def test_a_match_with_no_games_yet_is_skipped():
    e = sofa_event()
    e["homeScore"] = {"current": 0}
    e["awayScore"] = {"current": 0}
    assert event_from_sofa(e, sequence=1) is None


# ── polling ──────────────────────────────────────────────────────────────────

def test_only_real_changes_are_emitted():
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        pt = "40" if calls["n"] == 3 else "30"
        return {"events": [sofa_event(h_point=pt)]}

    p = SofaProxyProvider(fetch=fetch)
    assert len(p.poll_events()) == 1     # first sighting
    assert p.poll_events() == []         # unchanged
    assert len(p.poll_events()) == 1     # point moved


def test_doubles_are_excluded():
    p = SofaProxyProvider(fetch=feed(sofa_event(home="Alice/Carol", away="Bob/Dave")))
    assert p.poll_events() == []


def test_categories_are_filtered():
    # Deep ITF qualifying is mostly players with no ranking, so the model has
    # no prior and the rows would be noise on a personal board.
    p = SofaProxyProvider(fetch=feed(sofa_event(eid=1, cat="atp"),
                                     sofa_event(eid=2, cat="itf-men")))
    assert [e.match_id for e in p.poll_events()] == ["1"]


def test_categories_can_be_widened():
    p = SofaProxyProvider(fetch=feed(sofa_event(eid=2, cat="itf-men")),
                          categories=frozenset({"itf-men"}))
    assert [e.match_id for e in p.poll_events()] == ["2"]


def test_subscriptions_filter_the_poll():
    p = SofaProxyProvider(fetch=feed(sofa_event(eid=1), sofa_event(eid=2)))
    asyncio.run(p.subscribe("2"))
    assert [e.match_id for e in p.poll_events()] == ["2"]


def test_sequence_increments_per_match():
    seq = {"n": 0}

    def fetch():
        seq["n"] += 1
        return {"events": [sofa_event(h_point=str(15 * seq["n"]))]}

    p = SofaProxyProvider(fetch=fetch)
    assert [p.poll_events()[0].sequence for _ in range(2)] == [1, 2]


def test_a_failing_fetch_is_recorded_not_raised():
    def boom():
        raise ConnectionError("proxy down")

    p = SofaProxyProvider(fetch=boom)
    assert p.poll_events() == []
    assert "ConnectionError" in (p.last_error or "")


def test_connect_fails_loudly_when_the_proxy_is_down():
    # An empty board with no explanation is the most likely personal-use
    # failure, and the least diagnosable. Say which command to run.
    def boom():
        raise ConnectionError("refused")

    p = SofaProxyProvider(fetch=boom)
    with pytest.raises(RuntimeError, match="sofa_proxy.py"):
        asyncio.run(p.connect())


def test_resync_forces_a_re_emit():
    p = SofaProxyProvider(fetch=feed(sofa_event()))
    p.poll_events()
    assert p.poll_events() == []
    asyncio.run(p.resync("1"))
    assert len(p.poll_events()) == 1


def test_provider_admits_it_cannot_detect_gaps():
    assert SofaProxyProvider.has_sequence_guarantee is False
