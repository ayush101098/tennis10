"""End-to-end pipeline test.

The unit tests prove each stage in isolation. This proves the ORDER holds when
they are wired together — integrity before state, state before model, model and
market before the gate, gate before the broadcast — because the order is the
design and it is the thing a refactor silently breaks.

The model bridge is stubbed. `inplay.py` needs the model DB and the
hierarchical Markov model, which are not present in every environment; what is
under test here is the plumbing, and a test that only runs on a machine with a
6MB .db file is a test that stops running.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.live.engine import Fair  # noqa: E402
from execution.live.events import EventType, LiveEvent, P1, P2, Score  # noqa: E402
from execution.live.gateway import RoomRegistry, Viewer  # noqa: E402
from execution.live.odds import ExchangeQuote  # noqa: E402
from execution.live.runtime import LiveRuntime  # noqa: E402
from execution.live.signals import SignalStatus  # noqa: E402
from execution.live.state import InMemoryStateStore  # noqa: E402

T0 = 1_788_000_000_000


class StubModel:
    """A model bridge with a dial, so the pipeline can be driven to a known
    probability without loading the real engines."""

    def __init__(self, p1=0.70):
        self.p1 = p1
        self.available = True
        self.unavailable_reason = None
        self.calls = 0

    def price(self, state, transitions, force_full=False):
        self.calls += 1
        if self.p1 is None:
            return None
        return Fair(p1=self.p1, source="stub", tier="cheap", computed_ms=T0,
                    components={"stub": self.p1})


def ev(seq, *, games=(0, 0), points=("0", "0"), server=P1, ts=None, match="m1"):
    """Timestamped NOW by default.

    Freshness is measured against wall time, so a fixed past constant makes
    every event read as OFFLINE — which is the tracker working, not a bug, but
    it hides whatever the test was actually trying to assert.
    """
    now = ts if ts is not None else int(time.time() * 1000)
    return LiveEvent(match_id=match, sequence=seq, event_type=EventType.POINT,
                     provider_ts=now, received_ts=now,
                     score=Score(games=games, points=points), server=server)


def _rt(model=None, **kw):
    rt = LiveRuntime(store=InMemoryStateStore(), model=model or StubModel(), **kw)
    rt.register_match("m1", player1="A. Player", player2="B. Player", surface="Hard")
    return rt


def _price(rt, p1=0.55, size=900.0):
    """Put a deep two-sided exchange price on the match-winner market."""
    ctx = rt.contexts["m1"]
    mv = ctx.view("m1", "match_winner")
    mv.add_exchange(ExchangeQuote("pm", "match_winner", "p1", bid=p1 - 0.01,
                                  ask=p1 + 0.01, bid_size=size, ask_size=size))
    mv.add_exchange(ExchangeQuote("pm", "match_winner", "p2", bid=1 - p1 - 0.01,
                                  ask=1 - p1 + 0.01, bid_size=size, ask_size=size))
    return mv


def test_event_flows_through_to_a_broadcast():
    rt = _rt()
    got = []

    async def run():
        await rt.registry.join("m1", Viewer(id="v1", send=lambda p: _collect(got, p)))
        _price(rt)
        await rt.handle_event(ev(1, points=("15", "0")))

    asyncio.run(run())
    assert got, "nothing reached the viewer"
    p = got[-1]
    assert p["match_id"] == "m1"
    assert p["probability"]["p1"] == 0.70
    assert p["market"]["p1"] == 0.55
    assert p["market"]["source"] == "exchange"
    assert p["state"]["points"] == ["15", "0"]
    assert "latency" in p


def test_no_price_means_no_signal_but_still_a_scoreboard():
    # The board is useful without a market. Losing the price must not take the
    # score down with it.
    rt = _rt()
    got = []

    async def run():
        await rt.registry.join("m1", Viewer(id="v1", send=lambda p: _collect(got, p)))
        await rt.handle_event(ev(1, points=("15", "0")))

    asyncio.run(run())
    assert got
    assert "market" not in got[-1] or got[-1].get("market") is None
    assert got[-1].get("signal") is None
    assert got[-1]["state"]["points"] == ["15", "0"]


def test_model_silence_does_not_fabricate_a_probability():
    rt = _rt(model=StubModel(p1=None))
    got = []

    async def run():
        await rt.registry.join("m1", Viewer(id="v1", send=lambda p: _collect(got, p)))
        _price(rt)
        await rt.handle_event(ev(1, points=("15", "0")))

    asyncio.run(run())
    # No probability key at all — not 0.5. A coin flip presented as an opinion
    # is the easiest way to manufacture a fake edge.
    assert "probability" not in got[-1]
    assert got[-1].get("signal") is None


def test_sequence_gap_degrades_health_and_blocks_the_signal():
    rt = _rt(model=StubModel(p1=0.90))
    got = []

    async def run():
        await rt.registry.join("m1", Viewer(id="v1", send=lambda p: _collect(got, p)))
        _price(rt, p1=0.55)
        await rt.handle_event(ev(1))
        await rt.handle_event(ev(9))              # 2..8 missing
        # Age the hole past the reorder window. The test runs inside a
        # millisecond, so elapsed time has to be simulated rather than waited
        # for — backdating when the hole was first noticed is exactly what
        # "750ms went by and 2..8 never arrived" means.
        tr = rt.contexts["m1"].machine.tracker
        tr._pending = {seq: first - 5_000 for seq, first in tr._pending.items()}
        await rt.handle_event(ev(10))

    asyncio.run(run())
    last = got[-1]
    assert last["health"] == "DEGRADED"
    # A 35-point edge on a degraded feed must never publish as actionable.
    sig = last.get("signal")
    assert sig is None or sig["status"] == SignalStatus.STOP.value


def test_missing_market_does_not_report_the_feed_as_broken():
    rt = _rt()
    got = []

    async def run():
        await rt.registry.join("m1", Viewer(id="v1", send=lambda p: _collect(got, p)))
        await rt.handle_event(ev(1, points=("15", "0")))     # no price attached

    asyncio.run(run())
    assert got[-1]["health"] == "LIVE", "a scoreboard with no odds is not an outage"
    assert got[-1].get("signal") is None, "and it still must not signal"


def test_one_subscription_regardless_of_viewer_count():
    subs = []

    class P:
        name = "stub"

        async def subscribe(self, m):
            subs.append(m)

        async def unsubscribe(self, m):
            subs.remove(m)

    rt = LiveRuntime(provider=P(), store=InMemoryStateStore(), model=StubModel())
    rt.register_match("m1", player1="A", player2="B")

    async def run():
        for i in range(25):
            await rt.registry.join("m1", Viewer(id=f"v{i}", send=_noop))
        assert subs == ["m1"]
        for i in range(25):
            await rt.registry.leave("m1", f"v{i}")
        assert subs == []

    asyncio.run(run())


def test_the_game_tape_survives_across_events():
    # momentum.py is a function of recent history, so the tape has to persist
    # between events rather than be recomputed from the current scoreboard.
    rt = _rt()

    async def run():
        await rt.registry.join("m1", Viewer(id="v1", send=_noop))
        await rt.handle_event(ev(1, games=(3, 3), server=P1))
        await rt.handle_event(ev(2, games=(3, 4), server=P2))   # break
        await rt.handle_event(ev(3, games=(4, 4), server=P2))   # break back

    asyncio.run(run())
    st = rt.store.get("m1")
    assert len(st.games) == 2
    assert [g["break"] for g in st.games] == [True, True]


def test_health_endpoint_reports_the_pipeline():
    rt = _rt()

    async def run():
        await rt.registry.join("m1", Viewer(id="v1", send=_noop))
        _price(rt)
        await rt.handle_event(ev(1))

    asyncio.run(run())
    h = rt.health()
    assert h["processed"] == 1
    assert h["matches"] == 1
    assert "latency" in h and "total" in h["latency"]


def test_runtime_consumes_a_replay_provider_end_to_end():
    from execution.live.provider import ReplayProvider, ScriptedEvent

    script = [
        ScriptedEvent(sequence=1, score=Score(games=(0, 0), points=("15", "0")), server=P1),
        ScriptedEvent(sequence=2, score=Score(games=(0, 0), points=("30", "0")), server=P1),
        ScriptedEvent(sequence=3, score=Score(games=(1, 0), points=("0", "0")), server=P2),
    ]
    prov = ReplayProvider(script, match_id="m1",
                          clock_ms=int(time.time() * 1000))
    rt = LiveRuntime(provider=prov, store=InMemoryStateStore(), model=StubModel())
    rt.register_match("m1", player1="A", player2="B")
    got = []

    async def run():
        await rt.registry.join("m1", Viewer(id="v1", send=lambda p: _collect(got, p)))
        _price(rt)
        await rt.run()

    asyncio.run(run())
    assert rt.processed == 3
    assert got, "the pipeline produced no broadcasts"
    assert got[-1]["state"]["games"] == [1, 0]
    assert rt.store.get("m1").games, "the completed game was not recorded"


# ── helpers ──────────────────────────────────────────────────────────────────

async def _noop(_payload):
    return None


def _collect(bucket, payload):
    bucket.append(payload)

    async def done():
        return None
    return done()
