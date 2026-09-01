"""Edge publisher, and the runtime's calibration recording loop."""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.live.calibration import CalibrationRecorder  # noqa: E402
from execution.live.edge_publisher import EdgePublisher  # noqa: E402
from execution.live.gateway import Viewer  # noqa: E402
from execution.live.odds import ExchangeQuote  # noqa: E402
from execution.live.runtime import LiveRuntime  # noqa: E402
from execution.live.state import InMemoryStateStore  # noqa: E402
from test_live_runtime import StubModel, ev, _noop  # noqa: E402


def test_publisher_is_inert_until_configured():
    # An unconfigured publisher must not raise or count failures — running
    # without Cloudflare is a supported deployment, not an error state.
    p = EdgePublisher(base_url="", token="")
    assert not p.configured
    assert asyncio.run(p.publish("m1", {"x": 1})) is False
    assert p.failed == 0


def test_publisher_sends_through_the_transport():
    sent = []
    p = EdgePublisher(transport=lambda m, payload: sent.append((m, payload)))
    assert asyncio.run(p.publish("m1", {"sequence": 4})) is True
    assert sent == [("m1", {"sequence": 4})]
    assert p.pushed == 1 and p.failed == 0


def test_publisher_failure_is_counted_not_raised():
    # A push failure must never stop the pipeline; the engine's job is to keep
    # pricing, delivery is best effort.
    def boom(_m, _p):
        raise ConnectionError("edge unreachable")

    p = EdgePublisher(transport=boom)
    assert asyncio.run(p.publish("m1", {})) is False
    assert p.failed == 1
    assert "ConnectionError" in p.last_error
    assert p.health()["failed"] == 1


def test_runtime_pushes_to_the_edge_even_with_no_local_viewers():
    # The edge has its own viewers. Suppressing a push because nobody is
    # attached to THIS process would leave them stale.
    sent = []
    rt = LiveRuntime(store=InMemoryStateStore(), model=StubModel(),
                     publisher=EdgePublisher(transport=lambda m, p: sent.append(m)))
    rt.register_match("m1", player1="A", player2="B")
    asyncio.run(rt.handle_event(ev(1)))
    assert sent == ["m1"]


def _recorder():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return CalibrationRecorder(Path(tmp.name))


def test_runtime_records_a_prediction_for_calibration():
    rec = _recorder()
    rt = LiveRuntime(store=InMemoryStateStore(), model=StubModel(0.72), recorder=rec)
    rt.register_match("m1", player1="Alice", player2="Bob")

    async def run():
        await rt.registry.join("m1", Viewer(id="v", send=_noop))
        await rt.handle_event(ev(1))

    asyncio.run(run())
    total, settled = rec.count()
    assert total == 1 and settled == 0

    rt.settle_match("m1", winner="Alice")
    obs = rec.settled()
    assert len(obs) == 1
    assert obs[0].selection == "Alice"
    assert obs[0].p_model == 0.72
    assert obs[0].won is True


def test_one_match_does_not_flood_the_calibration_set():
    # Recording every point would let a single long match dominate the fit;
    # the model is wrong per MATCH, so that is the unit to weight by.
    rec = _recorder()
    rt = LiveRuntime(store=InMemoryStateStore(), model=StubModel(0.6), recorder=rec)
    rt.register_match("m1", player1="Alice", player2="Bob")

    async def run():
        for i in range(1, 40):
            await rt.handle_event(ev(i))

    asyncio.run(run())
    total, _ = rec.count()
    assert total <= 2, f"one match produced {total} observations"


def test_calibration_recording_never_breaks_the_live_path():
    class Broken:
        def predict(self, **_kw):
            raise RuntimeError("disk full")

        def count(self):
            return (0, 0)

    rt = LiveRuntime(store=InMemoryStateStore(), model=StubModel(), recorder=Broken())
    rt.register_match("m1", player1="A", player2="B")
    # Must not raise.
    asyncio.run(rt.handle_event(ev(1)))
    assert rt.processed == 1


def test_health_surfaces_edge_and_calibration_state():
    rec = _recorder()
    rt = LiveRuntime(store=InMemoryStateStore(), model=StubModel(), recorder=rec,
                     publisher=EdgePublisher(transport=lambda m, p: None))
    rt.register_match("m1", player1="A", player2="B")
    asyncio.run(rt.handle_event(ev(1)))
    h = rt.health()
    assert h["edge"]["pushed"] == 1
    assert h["calibration"]["observations"] == 1
