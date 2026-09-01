"""Tests for state, odds, signals, rooms and the runtime pipeline.

The emphasis is on the cases that cost money if they are wrong: a de-vig that
drifts, a thin exchange quote treated as a real price, a signal that fires on
degraded data, and a room that keeps an upstream subscription alive after
everyone has left.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.live.events import EventType, LiveEvent, P1, P2, Score  # noqa: E402
from execution.live.feed import Health  # noqa: E402
from execution.live.gateway import Room, RoomRegistry, Viewer, build_payload  # noqa: E402
from execution.live.odds import (  # noqa: E402
    BookmakerQuote, ExchangeQuote, MarketView, implied, to_decimal,
)
from execution.live.providers.failover import FailoverManager, ProviderHealth  # noqa: E402
from execution.live.signals import SignalEngine, SignalStatus  # noqa: E402
from execution.live.state import MatchStateMachine, is_significant  # noqa: E402

T0 = 1_788_000_000_000


def ev(seq, *, games=(0, 0), sets=(0, 0), points=("0", "0"), server=None,
       winner=None, etype=EventType.POINT, ts=T0, match="m1"):
    return LiveEvent(match_id=match, sequence=seq, event_type=etype,
                     provider_ts=ts, received_ts=ts,
                     score=Score(sets=sets, games=games, points=points),
                     server=server, point_winner=winner)


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, ms):
        self.t += ms
        return self.t


# ── Odds: de-vig and the bookmaker/exchange separation ───────────────────────

def test_devig_matches_the_worked_example():
    # 1.80 / 2.10 -> 53.85% / 46.15%. Pinned because every edge in the product
    # is measured against this number.
    p1, p2 = BookmakerQuote("b", "match_winner", 1.80, 2.10).devigged()
    assert round(p1, 4) == 0.5385
    assert round(p2, 4) == 0.4615
    assert abs(p1 + p2 - 1.0) < 1e-9


def test_devig_removes_the_margin_entirely():
    q = BookmakerQuote("b", "match_winner", 1.50, 2.50)
    raw = implied(1.50) + implied(2.50)
    assert raw > 1.0                       # the book has margin
    p1, p2 = q.devigged()
    assert abs(p1 + p2 - 1.0) < 1e-12      # we removed all of it


def test_non_prices_are_rejected_not_parsed():
    assert BookmakerQuote("b", "m", 1.0, 2.0).devigged() is None
    assert BookmakerQuote("b", "m", 0.5, 2.0).devigged() is None
    assert implied(1.0) is None
    assert to_decimal(0.0) is None


def test_thin_exchange_quote_is_not_tradeable():
    # The dangerous case: a mid that looks like a price and has nobody behind
    # it. An edge against this is an edge against nobody.
    thin = ExchangeQuote("pm", "match_winner", "p1", bid=0.60, ask=0.62,
                         bid_size=5, ask_size=5)
    assert thin.mid == pytest.approx(0.61)
    assert not thin.tradeable

    deep = ExchangeQuote("pm", "match_winner", "p1", bid=0.60, ask=0.62,
                         bid_size=500, ask_size=500)
    assert deep.tradeable


def test_wide_spread_is_not_tradeable():
    wide = ExchangeQuote("pm", "m", "p1", bid=0.40, ask=0.70,
                         bid_size=1000, ask_size=1000)
    assert not wide.tradeable


def test_crossed_book_has_no_mid():
    crossed = ExchangeQuote("pm", "m", "p1", bid=0.70, ask=0.60,
                            bid_size=100, ask_size=100)
    assert crossed.mid is None
    assert not crossed.tradeable


def test_book_consensus_uses_median_not_mean():
    # One stale book must not drag the consensus.
    mv = MarketView(match_id="m1", market="match_winner")
    mv.add_book(BookmakerQuote("a", "match_winner", 2.00, 2.00))   # 50%
    mv.add_book(BookmakerQuote("b", "match_winner", 2.00, 2.00))   # 50%
    mv.add_book(BookmakerQuote("c", "match_winner", 1.10, 9.00))   # ~89% outlier
    p1, _ = mv.book_consensus()
    assert p1 == pytest.approx(0.5, abs=0.02)


def test_a_book_replaces_its_own_previous_quote():
    mv = MarketView(match_id="m1", market="match_winner")
    mv.add_book(BookmakerQuote("a", "match_winner", 2.00, 2.00))
    mv.add_book(BookmakerQuote("a", "match_winner", 1.50, 3.00))
    assert len(mv.books) == 1
    assert mv.book_consensus()[0] == pytest.approx(2 / 3, abs=0.01)


def test_deep_exchange_is_preferred_over_books():
    # Money actually risked outranks a posted line.
    mv = MarketView(match_id="m1", market="match_winner")
    mv.add_book(BookmakerQuote("a", "match_winner", 2.00, 2.00))
    mv.add_exchange(ExchangeQuote("pm", "match_winner", "p1", bid=0.64, ask=0.66,
                                  bid_size=900, ask_size=900))
    mv.add_exchange(ExchangeQuote("pm", "match_winner", "p2", bid=0.34, ask=0.36,
                                  bid_size=900, ask_size=900))
    p1, _, source = mv.fair()
    assert source == "exchange"
    assert p1 == pytest.approx(0.65, abs=0.01)


def test_thin_exchange_falls_back_to_books():
    mv = MarketView(match_id="m1", market="match_winner")
    mv.add_book(BookmakerQuote("a", "match_winner", 1.80, 2.10))
    mv.add_exchange(ExchangeQuote("pm", "match_winner", "p1", bid=0.60, ask=0.62,
                                  bid_size=1, ask_size=1))
    p1, _, source = mv.fair()
    assert source == "bookmaker"
    assert p1 == pytest.approx(0.5385, abs=0.001)


def test_thin_exchange_alone_is_labelled_thin():
    # Used, because it is all we have — but labelled so the caller can widen
    # its uncertainty rather than trusting it like a deep book.
    mv = MarketView(match_id="m1", market="match_winner")
    mv.add_exchange(ExchangeQuote("pm", "match_winner", "p1", bid=0.60, ask=0.62,
                                  bid_size=1, ask_size=1))
    assert mv.fair()[2] == "exchange_thin"


def test_one_sided_exchange_infers_the_complement():
    mv = MarketView(match_id="m1", market="match_winner")
    mv.add_exchange(ExchangeQuote("pm", "match_winner", "p1", bid=0.70, ask=0.72,
                                  bid_size=900, ask_size=900))
    p1, p2, _ = mv.fair()
    assert p1 + p2 == pytest.approx(1.0)
    assert p1 == pytest.approx(0.71, abs=0.01)


def test_absent_market_is_not_the_same_as_a_stale_one():
    # Found by running the pipeline: a scoreboard with no odds attached was
    # reporting OFFLINE, i.e. a working feed diagnosed as broken.
    mv = MarketView(match_id="m1", market="match_winner")
    assert not mv.has_price
    mv.add_exchange(ExchangeQuote("pm", "match_winner", "p1", bid=0.5, ask=0.52,
                                  bid_size=900, ask_size=900))
    assert mv.has_price
    assert mv.health() is Health.LIVE


# ── State machine ────────────────────────────────────────────────────────────

def test_state_folds_score_and_records_the_game_tape():
    m = MatchStateMachine("m1", player1="A", player2="B")
    m.apply(ev(1, games=(3, 3), server=P1))
    st, _, transitions = m.apply(ev(2, games=(3, 4), server=P2))
    assert EventType.BREAK in transitions
    assert st.score.games == (3, 4)
    assert st.games[-1]["winner"] == P2
    assert st.games[-1]["break"] is True


def test_serve_tallies_track_points_on_serve():
    m = MatchStateMachine("m1", player1="A", player2="B")
    m.apply(ev(1, server=P1, winner=P1, points=("15", "0")))
    m.apply(ev(2, server=P1, winner=P2, points=("15", "15")))
    st, _, _ = m.apply(ev(3, server=P1, winner=P1, points=("30", "15")))
    assert st.serve_points[P1] == 3
    assert st.serve_points_won[P1] == 2


def test_duplicate_event_is_not_folded_in_twice():
    # A replayed point must not double-count in the serve tallies — that would
    # quietly bias the serve percentage the model prices from.
    m = MatchStateMachine("m1", player1="A", player2="B")
    m.apply(ev(1, server=P1, winner=P1))
    m.apply(ev(2, server=P1, winner=P1))
    m.apply(ev(2, server=P1, winner=P1))
    st = m.store.get("m1")
    assert st.serve_points[P1] == 2


def test_significance_tiering():
    m = MatchStateMachine("m1")
    st, _, _ = m.apply(ev(1, points=("15", "0")))
    assert not is_significant([], st)                       # ordinary point
    st, _, _ = m.apply(ev(2, points=("40", "30")))
    assert is_significant([], st)                           # game point
    assert is_significant([EventType.BREAK], st)


# ── Signal engine: hysteresis and the publish gate ───────────────────────────

def _sig(engine, p_model, p_market, *, feed=Health.LIVE, odds=Health.LIVE):
    return engine.evaluate(match_id="m1", market="match_winner", selection="p1",
                           p_model=p_model, p_market=p_market,
                           feed_health=feed, odds_health=odds,
                           price_source="exchange")


def test_no_signal_without_a_model_opinion():
    # None is "no opinion", never 50/50 — a coin flip presented as a model
    # output is the easiest way to manufacture a fake edge.
    clock = Clock()
    eng = SignalEngine(now_ms=clock)
    assert _sig(eng, None, 0.50) is None


def test_no_signal_without_a_price():
    clock = Clock()
    eng = SignalEngine(now_ms=clock)
    assert _sig(eng, 0.70, None) is None


def test_degraded_feed_blocks_signals_entirely():
    clock = Clock()
    eng = SignalEngine(now_ms=clock)
    # Establish a live signal first so there is something to stop.
    _sig(eng, 0.70, 0.50)
    clock.advance(2_000)
    _sig(eng, 0.70, 0.50)
    out = _sig(eng, 0.70, 0.50, feed=Health.DEGRADED)
    assert out is None or out.status is SignalStatus.STOP


def test_stale_price_blocks_signals_even_with_a_live_feed():
    # A live score against a stale price is not a live signal.
    clock = Clock()
    eng = SignalEngine(now_ms=clock)
    _sig(eng, 0.70, 0.50)
    clock.advance(2_000)
    out = _sig(eng, 0.70, 0.50, odds=Health.STALE)
    assert out is None or out.status is SignalStatus.STOP


def test_absurd_edge_is_quarantined_not_published_as_entry():
    clock = Clock()
    eng = SignalEngine(now_ms=clock)
    _sig(eng, 0.60, 0.55)                 # get a live track going
    clock.advance(2_000)
    _sig(eng, 0.60, 0.55)
    out = _sig(eng, 0.95, 0.30)           # +65 points: a data fault
    assert out is None or out.status is SignalStatus.STOP


def test_dwell_prevents_a_one_tick_spike_from_publishing():
    clock = Clock()
    eng = SignalEngine(now_ms=clock, dwell_ms=1_500)
    assert _sig(eng, 0.70, 0.50) is None          # too new to trust
    clock.advance(1_600)
    assert _sig(eng, 0.70, 0.50) is not None


def test_hysteresis_does_not_strobe_around_one_threshold():
    # The behaviour the asymmetric thresholds exist to prevent: an edge
    # hovering near entry must not produce ENTRY/EXIT/ENTRY/EXIT.
    clock = Clock()
    eng = SignalEngine(now_ms=clock, dwell_ms=0)
    statuses = []
    for p_market in (0.44, 0.45, 0.44, 0.45, 0.44):   # edge oscillates ~6%
        s = _sig(eng, 0.50, p_market)
        statuses.append(s.status if s else None)
        clock.advance(500)
    exits = [s for s in statuses if s is SignalStatus.EXIT]
    assert not exits, f"signal strobed: {statuses}"


class _Green:
    """A scorer that always grades green, so the state machine can be tested
    independently of the confidence engine's thresholds."""
    tradeable = True
    reasons: list = []
    edge_score = 3.0


def _green_scorer(*_a, **_k):
    return _Green()


def test_never_entered_means_nothing_to_exit():
    # A collapsing edge that only ever reached WATCH goes quiet — it does not
    # announce an EXIT from a position the user never held.
    clock = Clock()
    eng = SignalEngine(now_ms=clock, dwell_ms=0)
    _sig(eng, 0.60, 0.50)
    clock.advance(1_000)
    assert _sig(eng, 0.60, 0.59) is None


def test_exit_requires_the_edge_to_actually_collapse():
    clock = Clock()
    eng = SignalEngine(now_ms=clock, dwell_ms=0, scorer=_green_scorer)
    s = _sig(eng, 0.60, 0.50)             # +10% and confident -> ENTRY
    assert s.status is SignalStatus.ENTRY
    clock.advance(1_000)
    s = _sig(eng, 0.60, 0.55)             # +5% -> above the exit floor, hold
    assert s.status is SignalStatus.HOLD
    clock.advance(1_000)
    s = _sig(eng, 0.60, 0.59)             # +1% -> gone
    assert s.status is SignalStatus.EXIT


def test_stop_is_emitted_once_not_every_tick():
    # A feed that is down for a minute must not fill the panel with identical
    # STOP rows; the transition is the news.
    clock = Clock()
    eng = SignalEngine(now_ms=clock, dwell_ms=0)
    _sig(eng, 0.60, 0.50)
    clock.advance(1_000)
    first = _sig(eng, 0.60, 0.50, feed=Health.OFFLINE)
    second = _sig(eng, 0.60, 0.50, feed=Health.OFFLINE)
    third = _sig(eng, 0.60, 0.50, feed=Health.OFFLINE)
    assert first is not None and first.status is SignalStatus.STOP
    assert second is None and third is None


def test_confidence_never_reaches_certainty():
    # The model's own calibration report says it is over-confident; a UI that
    # can print 100% would repeat exactly the error this number warns about.
    clock = Clock()
    eng = SignalEngine(now_ms=clock, dwell_ms=0)
    s = _sig(eng, 0.95, 0.80)
    if s is not None:
        assert s.confidence <= 0.95


# ── Rooms: the fan-out economics ─────────────────────────────────────────────

def test_upstream_subscription_happens_once_per_match_not_per_viewer():
    # The entire cost argument, asserted: 50 viewers, one subscription.
    subs, unsubs = [], []
    reg = RoomRegistry(on_first_viewer=lambda m: _async(subs.append(m)),
                       on_last_viewer=lambda m: _async(unsubs.append(m)))

    async def run():
        for i in range(50):
            await reg.join("m1", Viewer(id=f"v{i}", send=_noop))
        assert reg.rooms["m1"].viewer_count == 50
        for i in range(49):
            await reg.leave("m1", f"v{i}")
        assert subs == ["m1"] and unsubs == []      # still one viewer left
        await reg.leave("m1", "v49")
        assert unsubs == ["m1"]                     # now the room is gone
        assert "m1" not in reg.rooms

    asyncio.run(run())


def test_a_joiner_gets_the_last_state_immediately():
    # Otherwise a mid-match joiner stares at an empty panel until the next
    # point, which in tennis can be 30+ seconds.
    reg = RoomRegistry()
    got = []

    async def run():
        await reg.join("m1", Viewer(id="a", send=_noop))
        await reg.broadcast("m1", {"probability": {"p1": 0.6}}, kind="score")
        await reg.join("m1", Viewer(id="b", send=lambda p: _collect(got, p)))

    asyncio.run(run())
    assert got and got[0]["probability"]["p1"] == 0.6


def test_tiny_model_moves_are_suppressed_but_prices_never_are():
    reg = RoomRegistry()
    sent = []

    async def run():
        await reg.join("m1", Viewer(id="a", send=lambda p: _collect(sent, p)))
        await reg.broadcast("m1", {"probability": {"p1": 0.600}}, kind="model")
        await reg.broadcast("m1", {"probability": {"p1": 0.6005}}, kind="model")   # noise
        await reg.broadcast("m1", {"probability": {"p1": 0.6006}}, kind="price")   # a PRICE
    asyncio.run(run())

    assert len(sent) == 2, "model noise should be suppressed, price must not be"
    assert reg.rooms["m1"].suppressed == 1


def test_a_dead_socket_removes_itself():
    # A disconnected viewer that lingers keeps the upstream subscription alive
    # — a slow leak that shows up as an unexplained provider bill.
    reg = RoomRegistry()

    async def boom(_):
        raise ConnectionResetError("gone")

    async def run():
        await reg.join("m1", Viewer(id="dead", send=boom))
        await reg.join("m1", Viewer(id="ok", send=_noop))
        await reg.broadcast("m1", {"probability": {"p1": 0.5}}, kind="score")
        return reg.rooms["m1"].viewer_count

    assert asyncio.run(run()) == 1


def test_payload_is_compact_and_always_states_freshness():
    class S:
        score = Score(sets=(1, 0), games=(4, 3), points=("30", "15"))
        server = P1
    p = build_payload(match_id="m1", state=S(), health=Health.DELAYED, sequence=42)
    assert p["health"] == "DELAYED"
    assert p["state"]["points"] == ["30", "15"]
    assert "probability" not in p          # nothing invented when there is no model


# ── Failover ─────────────────────────────────────────────────────────────────

def test_silent_provider_triggers_failover_after_silence_plus_debounce():
    clock = Clock()
    fm = FailoverManager(["a", "b"], now_ms=clock, silence_ms=20_000, debounce_ms=10_000)
    fm.record_event("a", 100)
    fm.record_event("b", 100)
    assert fm.decide() == "a"

    clock.advance(25_000)                  # a quiet 25s: detected, not yet confirmed
    fm.record_event("b", 100)
    assert fm.decide() == "a"

    clock.advance(11_000)                  # past silence + debounce
    fm.record_event("b", 100)
    assert fm.decide() == "b"


def test_failover_does_not_flap_on_a_brief_blip():
    clock = Clock()
    fm = FailoverManager(["a", "b"], now_ms=clock, debounce_ms=15_000)
    fm.record_event("a", 100)
    fm.record_event("b", 100)
    assert fm.decide() == "a"              # both healthy to begin with
    clock.advance(21_000)                  # a is silent, but only just
    fm.record_event("b", 100)
    # Detected silent, yet inside the debounce window -> keep the tape.
    assert fm.decide() == "a"
    assert fm.stats["a"].health is ProviderHealth.SILENT


def test_failback_requires_a_longer_healthy_run_than_failover():
    clock = Clock()
    fm = FailoverManager(["a", "b"], now_ms=clock, debounce_ms=5_000, recovery_ms=60_000)
    fm.record_event("b", 100)
    clock.advance(30_000)
    fm.record_event("b", 100)
    assert fm.decide() == "b"              # a never spoke

    clock.advance(1_000)                   # a comes back
    fm.record_event("a", 100)
    fm.record_event("b", 100)
    assert fm.decide() == "b", "must not fail back immediately"

    for _ in range(7):                     # a stays healthy for a full minute
        clock.advance(10_000)
        fm.record_event("a", 100)
        fm.record_event("b", 100)
    assert fm.decide() == "a"


def test_lagging_beats_silent():
    clock = Clock()
    fm = FailoverManager(["a", "b"], now_ms=clock, silence_ms=20_000,
                         debounce_ms=1_000, lag_ms=1_000)
    fm.record_event("a", 100)
    fm.record_event("b", 5_000)            # b is slow but alive
    assert fm.decide() == "a"
    clock.advance(25_000)                  # a goes silent — detected
    fm.record_event("b", 5_000)
    assert fm.decide() == "a"              # not yet confirmed
    clock.advance(2_000)                   # past the debounce
    fm.record_event("b", 5_000)
    # b is LAGGING, not healthy — but a lagging feed beats a silent one.
    assert fm.stats["b"].health is ProviderHealth.LAGGING
    assert fm.decide() == "b"


# ── helpers ──────────────────────────────────────────────────────────────────

async def _noop(_payload):
    return None


def _collect(bucket, payload):
    bucket.append(payload)

    async def done():
        return None
    return done()


def _async(value):
    async def run():
        return value
    return run()
