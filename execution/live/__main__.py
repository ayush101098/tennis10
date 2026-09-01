"""Run the live market engine.

    python -m execution.live serve            # gateway on :8080
    python -m execution.live replay <file>    # drive it from a recorded tape
    python -m execution.live doctor           # what is configured, what is not

WHY `doctor` EXISTS
    This system has several independent reasons to be silent — no provider key,
    no model DB, no market prices — and from the outside they all look the
    same: a board with nothing on it. `doctor` says which one it is, so
    "nothing is happening" is diagnosable without reading logs.

WHY `replay` EXISTS
    The Ultra tier is what makes sub-second updates possible, and until it is
    paid for there is no live point feed to develop against. A recorded tape
    makes the whole pipeline runnable — and demonstrable — on a laptop with no
    credentials at all.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.live.events import EventType, Score  # noqa: E402
from execution.live.gateway import RoomRegistry, create_app  # noqa: E402
from execution.live.provider import ReplayProvider, ScriptedEvent  # noqa: E402
from execution.live.runtime import LiveRuntime  # noqa: E402


def _build_provider():
    """The configured tennis feed, or None.

    None is a supported state: the gateway still serves, `doctor` still
    reports, and the failure is one line in /health rather than a crash at
    import time.
    """
    key = os.getenv("LIVETENNIS_API_KEY")
    if not key:
        return None, "LIVETENNIS_API_KEY is not set"
    try:
        from execution.live.providers.livetennis import LiveTennisProvider
        return LiveTennisProvider(api_key=key), None
    except Exception as e:                                   # pragma: no cover
        return None, f"{type(e).__name__}: {e}"


def cmd_doctor() -> int:
    provider, perr = _build_provider()
    runtime = LiveRuntime(provider=provider)

    rows = [
        ("tennis feed", provider.name if provider else "NOT CONFIGURED", perr or "ok"),
        ("model engine", "available" if runtime.model.available else "UNAVAILABLE",
         runtime.model.unavailable_reason or "inplay.py + momentum.py loaded"),
    ]
    try:
        import fastapi  # noqa: F401
        rows.append(("gateway", "available", "fastapi installed"))
    except Exception:
        rows.append(("gateway", "UNAVAILABLE", "pip install fastapi uvicorn"))

    width = max(len(r[0]) for r in rows)
    print("\nTennisAlpha live engine — configuration\n")
    for name, status, detail in rows:
        print(f"  {name.ljust(width)}  {status:<16} {detail}")

    print("\nnotes")
    print("  • Sub-second updates need the provider's WebSocket tier. Polling")
    print("    tiers cannot beat their own poll interval, whatever this does.")
    print("  • With no market price the engine still serves a scoreboard and")
    print("    stays silent on signals — by design, not by failure.")
    print()
    return 0 if provider else 1


def _load_tape(path: str) -> list:
    """Read a recorded tape: one JSON event per line (JSONL).

    Accepts the canonical shape so a tape can be captured straight off the
    normalizer — recording provider-native frames would tie every tape to the
    provider that produced it.
    """
    events = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        sc = d.get("score") or {}
        events.append(ScriptedEvent(
            sequence=int(d.get("sequence", len(events) + 1)),
            event_type=EventType(d.get("type", "POINT")),
            score=Score(
                sets=tuple(sc.get("sets", (0, 0))),
                games=tuple(sc.get("games", (0, 0))),
                points=tuple(sc.get("points", ("0", "0"))),
            ),
            server=d.get("server"),
            point_winner=(d.get("point") or {}).get("winner"),
            delay_ms=int(d.get("delay_ms", 0)),
        ))
    return events


async def _run_replay(path: str, match_id: str, p1: str, p2: str) -> None:
    import time
    provider = ReplayProvider(_load_tape(path), match_id=match_id,
                              clock_ms=int(time.time() * 1000), real_time=True)
    runtime = LiveRuntime(provider=provider)
    runtime.register_match(match_id, player1=p1, player2=p2)

    async def echo(payload):
        st = payload["state"]
        prob = payload.get("probability")
        line = (f"{st['sets']} {st['games']} {st['points']}"
                f"  server={st['server']}  health={payload['health']}")
        if prob:
            line += f"  p1={prob['p1']:.3f}"
        if payload.get("signal"):
            line += f"  SIGNAL={payload['signal']['status']}"
        print(line)

    from execution.live.gateway import Viewer
    await runtime.registry.join(match_id, Viewer(id="cli", send=echo))
    await runtime.run()
    print(json.dumps(runtime.health(), indent=2))


def cmd_serve(port: int) -> int:
    try:
        import uvicorn
    except Exception:
        print("uvicorn is not installed:  pip install uvicorn fastapi", file=sys.stderr)
        return 1

    provider, perr = _build_provider()
    if perr:
        # Serve anyway. A gateway that refuses to start without a feed takes
        # the scoreboard down with the feed, and the scoreboard is useful alone.
        print(f"warning: no tennis feed ({perr}) — serving without live data",
              file=sys.stderr)

    registry = RoomRegistry()
    runtime = LiveRuntime(provider=provider, registry=registry)
    registry._on_first = runtime._on_first_viewer
    registry._on_last = runtime._on_last_viewer
    app = create_app(registry, runtime=runtime)

    if provider is not None:
        @app.on_event("startup")
        async def _start():
            asyncio.create_task(runtime.run())

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    return 0


def main(argv: list) -> int:
    cmd = argv[1] if len(argv) > 1 else "doctor"
    if cmd == "doctor":
        return cmd_doctor()
    if cmd == "serve":
        return cmd_serve(int(os.getenv("PORT", "8080")))
    if cmd == "replay":
        if len(argv) < 3:
            print("usage: python -m execution.live replay <tape.jsonl> "
                  "[match_id] [player1] [player2]", file=sys.stderr)
            return 2
        asyncio.run(_run_replay(
            argv[2],
            argv[3] if len(argv) > 3 else "replay-match",
            argv[4] if len(argv) > 4 else "Player A",
            argv[5] if len(argv) > 5 else "Player B",
        ))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
