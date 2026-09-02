"""Set engine, market-lag detector and opportunity scanner.

These three fill the gaps the strategy framework identified that were NOT
already in the codebase. The Markov re-pricer, EWMA momentum, fatigue, surface
weighting, fractional Kelly and the EdgeScore ranking quantity all predate it
and are tested elsewhere.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.live.feed import Health  # noqa: E402
from execution.live.marketlag import (  # noqa: E402
    Divergence, MarketLagDetector,
)
from execution.live.scanner import Opportunity, ScanFilter, Scanner  # noqa: E402
from execution.live.setengine import set_win_prob, tiebreak_prob  # noqa: E402

T0 = 1_788_000_000_000
H1, H2 = 0.83, 0.78          # hold probabilities used across the set tests


def sw(a, b, server=1, hold_p1=H1, hold_p2=H2, **kw):
    return set_win_prob(a, b, server=server, hold_p1=hold_p1, hold_p2=hold_p2, **kw)


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms
        return self.t


# ── Set engine ───────────────────────────────────────────────────────────────

def test_a_won_set_is_certain():
    assert sw(6, 0) == 1.0
    assert sw(6, 4) == 1.0
    assert sw(7, 5) == 1.0
    assert sw(0, 6) == 0.0
    assert sw(4, 6) == 0.0


def test_seven_six_resolves_without_replaying_the_tiebreak():
    assert sw(7, 6) == 1.0
    assert sw(6, 7) == 0.0


def test_six_all_is_the_tiebreak():
    # At 6-6 the games are gone; only relative serve strength remains.
    assert sw(6, 6, sp1=0.62, sp2=0.62) == pytest.approx(0.5, abs=1e-9)
    assert sw(6, 6, sp1=0.70, sp2=0.60) > 0.5


def test_probability_rises_monotonically_with_games_banked():
    # The property that makes the set market tradeable: banked games dominate.
    probs = [sw(g, 0) for g in range(6)]
    assert all(b > a for a, b in zip(probs, probs[1:])), probs


def test_holding_serve_is_worth_more_than_the_match_number_implies():
    # 5-2 up serving is nearly won regardless of who the players are — this is
    # exactly what converting from a match probability throws away.
    assert sw(5, 2, server=1) > 0.95
    assert sw(2, 5, server=1) < 0.15


def test_serving_next_matters_only_when_the_service_count_is_odd():
    """A non-obvious property, and the reason this walk beats a heuristic.

    Before 6-6 there are 12 - (a + b) games left. When that number is EVEN
    (0-0, 4-4, 5-5) both players serve it equally often, so who serves next is
    worth exactly nothing. When it is ODD (5-4, 3-4) one player banks an extra
    service game, and it is worth a great deal — 26 points of probability at
    5-4.

    A model that applied a flat "server bonus" would be wrong in both
    directions at once.
    """
    for a, b in [(0, 0), (4, 4), (5, 5)]:          # even games remaining
        assert sw(a, b, server=1) == pytest.approx(sw(a, b, server=2), abs=1e-9)

    for a, b in [(5, 4), (4, 5), (3, 4)]:          # odd games remaining
        assert sw(a, b, server=1) - sw(a, b, server=2) > 0.20


def test_the_in_progress_game_is_priced_from_the_point_score():
    # 0-40 down on serve is materially worse than a fresh game, and a live model
    # that cannot see that is not live.
    fresh = sw(4, 4, server=1)
    losing = sw(4, 4, server=1, current_game_p1=0.10)
    winning = sw(4, 4, server=1, current_game_p1=0.95)
    assert losing < fresh < winning


def test_stronger_server_wins_more_sets():
    assert sw(0, 0, hold_p1=0.90, hold_p2=0.60) > sw(0, 0, hold_p1=0.60, hold_p2=0.90)


def test_impossible_inputs_return_none_not_a_number():
    # A feed error must surface as "no opinion", never as confident fiction.
    assert sw(0, 0, hold_p1=0.0) is None
    assert sw(0, 0, hold_p1=1.0) is None
    assert sw(-1, 0) is None
    assert sw(0, 0, server=3) is None
    assert sw(99, 0) is None


def test_tiebreak_is_symmetric():
    assert tiebreak_prob(0.7, 0.6) == pytest.approx(1 - tiebreak_prob(0.6, 0.7), abs=1e-9)


# ── Market-lag detector ──────────────────────────────────────────────────────

def test_first_observation_establishes_a_baseline_only():
    d = MarketLagDetector(now_ms=Clock())
    assert d.observe("m1", model_p=0.60, market_p=0.58) is None


def test_model_moves_and_market_does_not_is_a_lag():
    clock = Clock()
    d = MarketLagDetector(now_ms=clock)
    d.observe("m1", model_p=0.60, market_p=0.58)
    clock.advance(1_000)
    ev = d.observe("m1", model_p=0.68, market_p=0.581)
    assert ev is not None and ev.kind is Divergence.LAG
    assert ev.gap == pytest.approx(0.079, abs=0.002)
    assert d.open_lag("m1") is not None


def test_market_catching_up_closes_the_lag_and_times_it():
    # The reaction time is the measurement this module exists to produce.
    clock = Clock()
    d = MarketLagDetector(now_ms=clock)
    d.observe("m1", model_p=0.60, market_p=0.58)
    clock.advance(500)
    d.observe("m1", model_p=0.70, market_p=0.58)          # we move, market flat
    clock.advance(900)
    ev = d.observe("m1", model_p=0.70, market_p=0.66)     # market catches up
    assert ev is not None and ev.kind is Divergence.AGREE
    assert ev.reaction_ms == 900
    assert d.open_lag("m1") is None
    assert d.reaction_stats("m1")["median_ms"] == 900


def test_a_market_that_never_moves_is_not_counted_as_a_reaction():
    # Otherwise every latency statistic is inflated by markets that simply
    # disagreed with us.
    clock = Clock()
    d = MarketLagDetector(now_ms=clock, lag_ttl_ms=5_000)
    d.observe("m1", model_p=0.60, market_p=0.58)
    clock.advance(100)
    d.observe("m1", model_p=0.70, market_p=0.58)
    clock.advance(6_000)                                   # past the TTL
    d.observe("m1", model_p=0.70, market_p=0.58)
    assert d.open_lag("m1") is None
    assert d.reaction_stats("m1")["n"] == 0


def test_market_moving_alone_is_steam_not_an_edge():
    # Information arriving somewhere else first — a reason to stand aside.
    clock = Clock()
    d = MarketLagDetector(now_ms=clock)
    d.observe("m1", model_p=0.60, market_p=0.60)
    clock.advance(1_000)
    ev = d.observe("m1", model_p=0.601, market_p=0.68)
    assert ev is not None and ev.kind is Divergence.STEAM


def test_small_moves_are_noise_not_signal():
    clock = Clock()
    d = MarketLagDetector(now_ms=clock)
    d.observe("m1", model_p=0.60, market_p=0.60)
    clock.advance(1_000)
    assert d.observe("m1", model_p=0.605, market_p=0.601) is None


def test_a_lag_downward_is_detected_too():
    # Direction-agnostic: the market being slow to mark a player DOWN is the
    # same opportunity from the other side.
    clock = Clock()
    d = MarketLagDetector(now_ms=clock)
    d.observe("m1", model_p=0.60, market_p=0.60)
    clock.advance(500)
    ev = d.observe("m1", model_p=0.50, market_p=0.599)
    assert ev.kind is Divergence.LAG and ev.model_delta < 0


def test_missing_inputs_produce_nothing():
    d = MarketLagDetector(now_ms=Clock())
    assert d.observe("m1", model_p=None, market_p=0.5) is None
    assert d.observe("m1", model_p=0.5, market_p=None) is None


def test_reaction_stats_use_the_median():
    clock = Clock()
    d = MarketLagDetector(now_ms=clock)
    for i, delay in enumerate((200, 400, 20_000)):
        mid = f"m{i}"
        d.observe(mid, model_p=0.50, market_p=0.50)
        clock.advance(10)
        d.observe(mid, model_p=0.62, market_p=0.50)
        clock.advance(delay)
        d.observe(mid, model_p=0.62, market_p=0.60)
    stats = d.reaction_stats()
    assert stats["n"] == 3
    assert stats["median_ms"] == 400            # not dragged by the 20s tail
    assert stats["slowest_ms"] == 20_000


# ── Scanner ──────────────────────────────────────────────────────────────────

class _Scored:
    def __init__(self, grade, escore, sigma=0.05):
        self.grade, self.edge_score, self.sigma = grade, escore, sigma
        self.reasons = []


def _op(**kw):
    defaults = dict(match_id="m1", label="A vs B", market="match", selection="A",
                    model_p=0.60, market_p=0.54, edge=0.06, edge_score=2.0,
                    sigma=0.03, confidence=0.8, health=Health.LIVE)
    defaults.update(kw)
    return Opportunity(**defaults)


def test_ranking_is_by_edge_score_not_raw_edge():
    # The whole point: a 4% edge on a well-sourced match outranks an 11% edge
    # on a thin one, because sigma is what makes them comparable.
    thin = _op(match_id="thin", edge=0.11, edge_score=0.9)
    solid = _op(match_id="solid", edge=0.04, edge_score=3.1)
    ranked = Scanner.rank([thin, solid])
    assert [o.match_id for o in ranked] == ["solid", "thin"]


def test_degraded_and_stale_rows_never_reach_the_scanner():
    # A scanner listing opportunities the publish gate would refuse is a list
    # of disappointments.
    rows = [_op(match_id="ok"), _op(match_id="bad", health=Health.DEGRADED),
            _op(match_id="stale", health=Health.STALE)]
    assert [o.match_id for o in Scanner.rank(rows)] == ["ok"]


def test_filters_compose():
    rows = [
        _op(match_id="a", edge=0.08, edge_score=3.0, model_p=0.60, market="match"),
        _op(match_id="b", edge=0.02, edge_score=3.0, model_p=0.60, market="match"),
        _op(match_id="c", edge=0.08, edge_score=0.5, model_p=0.60, market="match"),
        _op(match_id="d", edge=0.08, edge_score=3.0, model_p=0.95, market="match"),
        _op(match_id="e", edge=0.08, edge_score=3.0, model_p=0.60, market="game"),
    ]
    f = ScanFilter(min_edge=0.05, min_edge_score=1.0, prob_range=(0.55, 0.85),
                   markets=frozenset({"match"}))
    assert [o.match_id for o in Scanner.rank(rows, filt=f)] == ["a"]


def test_unknown_liquidity_fails_a_liquidity_filter():
    # A size requirement that silently admits unmeasured markets is not a size
    # requirement.
    rows = [_op(match_id="measured", liquidity=5_000),
            _op(match_id="unknown", liquidity=None)]
    f = ScanFilter(min_liquidity=1_000)
    assert [o.match_id for o in Scanner.rank(rows, filt=f)] == ["measured"]


def test_lag_only_filter():
    rows = [_op(match_id="lagging", lag=Divergence.LAG.value),
            _op(match_id="quiet", lag=None)]
    f = ScanFilter(lag_only=True)
    assert [o.match_id for o in Scanner.rank(rows, filt=f)] == ["lagging"]


def test_limit_is_respected():
    rows = [_op(match_id=str(i), edge_score=float(i)) for i in range(10)]
    assert len(Scanner.rank(rows, limit=3)) == 3


def test_evaluate_builds_a_row_from_model_and_market():
    sc = Scanner(scorer=lambda *a, **k: _Scored("green", 2.5))
    o = sc.evaluate(match_id="m1", label="A vs B", market="match", selection="A",
                    model_p=0.66, market_p=0.58,
                    feed_health=Health.LIVE, odds_health=Health.LIVE)
    assert o is not None
    assert o.edge == pytest.approx(0.08)
    assert o.edge_score == 2.5
    assert o.health is Health.LIVE


def test_evaluate_returns_nothing_without_both_numbers():
    sc = Scanner(scorer=lambda *a, **k: _Scored("green", 2.5))
    assert sc.evaluate(match_id="m", label="l", market="match", selection="A",
                       model_p=None, market_p=0.5,
                       feed_health=Health.LIVE, odds_health=Health.LIVE) is None


def test_health_is_the_worse_of_feed_and_market():
    sc = Scanner(scorer=lambda *a, **k: _Scored("green", 2.5))
    o = sc.evaluate(match_id="m", label="l", market="match", selection="A",
                    model_p=0.6, market_p=0.5,
                    feed_health=Health.LIVE, odds_health=Health.DELAYED)
    assert o.health is Health.DELAYED


def test_summary_reports_the_headline_numbers():
    rows = [_op(edge=0.04, edge_score=1.0),
            _op(edge=0.09, edge_score=2.4, lag=Divergence.LAG.value)]
    s = Scanner.summary(rows)
    assert s == {"count": 2, "best_edge": 0.09, "best_score": 2.4, "lagging": 1}


def test_summary_of_nothing_is_not_a_crash():
    assert Scanner.summary([])["count"] == 0
