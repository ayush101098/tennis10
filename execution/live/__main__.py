"""Run the live market engine.

    python -m execution.live serve            # gateway on :8080
    python -m execution.live replay <file>    # drive it from a recorded tape
    python -m execution.live doctor           # what is configured, what is not
    python -m execution.live calibrate        # reliability diagram + fit
    python -m execution.live smoke            # probe the real provider endpoint
    python -m execution.live board            # PERSONAL: live board off the local proxy

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
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.live.events import EventType, Score  # noqa: E402
from execution.live.gateway import RoomRegistry, create_app  # noqa: E402
from execution.live.provider import ReplayProvider, ScriptedEvent  # noqa: E402
from execution.live.runtime import LiveRuntime  # noqa: E402


def _build_provider():
    """The tennis feed to use, preferring the licensed one.

    Order matters and is not arbitrary. Live Tennis API is preferred when a key
    exists: it is licensed, carries the server and true point events, and can
    stamp provider time. Livesport is the fallback that needs no credentials —
    it answers from an address SofaScore has burned, which is exactly when a
    fallback earns its keep, but it cannot see the server and has no sequence
    of record (see providers/livesport.py).

    Returning None is still supported: the gateway serves, `doctor` reports,
    and the failure is one line in /health rather than a crash at import.
    """
    key = os.getenv("LIVETENNIS_API_KEY")
    if key:
        try:
            from execution.live.providers.livetennis import LiveTennisProvider
            return LiveTennisProvider(api_key=key), None
        except Exception as e:                               # pragma: no cover
            return None, f"{type(e).__name__}: {e}"
    try:
        from execution.live.providers.livesport import LivesportProvider
        return LivesportProvider(), "LIVETENNIS_API_KEY unset — using livesport (no server, polled)"
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


def cmd_calibrate() -> int:
    """Reliability diagram and a fit, from recorded observations.

    Prints the diagram FIRST and the fitted parameters second, deliberately.
    A single fitted number hides which probability band is wrong, and the band
    is the thing you can act on.
    """
    from execution.live.calibration import (
        CalibrationRecorder, MIN_OBSERVATIONS, fit_and_evaluate, format_reliability,
    )

    rec = CalibrationRecorder()
    total, settled = rec.count()
    obs = rec.settled()

    print(f"\ncalibration observations: {total} recorded, {settled} settled\n")
    if not obs:
        print("Nothing settled yet, so there is nothing to calibrate.")
        print()
        print("This is the honest state of the project, not a missing feature.")
        print("The 8,218 settled rows in tennis_betting.db:trade_log cannot be")
        print("used: corr(true_p, market_price) is +0.15 where it should be ~+0.8,")
        print("predictions of 0.05 won 88% of the time, and no orientation")
        print("convention makes match/set1/set2 consistent — the log mixes")
        print("conventions across market types and code versions.")
        print()
        print("The recorder writes one unambiguous orientation. Run the engine,")
        print("call settle_match() as matches finish, and this becomes real.")
        return 1

    print(format_reliability(obs))
    rep = fit_and_evaluate(obs)
    print(f"\nfit: {rep.model}")
    print(f"  Brier    {rep.brier_raw:.4f} -> {rep.brier_cal:.4f}")
    print(f"  log-loss {rep.logloss_raw:.4f} -> {rep.logloss_cal:.4f}")
    print(f"  ECE      {rep.ece_raw:.1%} -> {rep.ece_cal:.1%}")
    if not rep.honest:
        print(f"\n  ⚠ {rep.note}")
    return 0


def cmd_smoke() -> int:
    """Probe the real provider and show what normalization does to its frames.

    The adapter is written against documented shapes and has never spoken to
    the live endpoint, because that needs a paid key. This is the tool for the
    first five minutes after a key exists: it shows the raw frame beside the
    normalized event, so any field-name mismatch is visible immediately rather
    than as a silently empty board.
    """
    key = os.getenv("LIVETENNIS_API_KEY")
    if not key:
        print("LIVETENNIS_API_KEY is not set — nothing to probe.", file=sys.stderr)
        print("\nThe adapter has never been run against the live endpoint.",
              file=sys.stderr)
        print("Expect to adjust normalize() on first contact; this command is",
              file=sys.stderr)
        print("what makes that a five-minute job instead of a debugging session.",
              file=sys.stderr)
        return 1

    import json as _json
    import urllib.request
    from execution.live.providers.livetennis import BASE_URL, normalize

    url = f"{BASE_URL}/matches/live"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}",
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = _json.loads(r.read())
    except Exception as e:
        print(f"request failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    frames = data if isinstance(data, list) else (data.get("matches") or data.get("data") or [])
    print(f"\n{len(frames)} live frames from {url}\n")
    for frame in frames[:3]:
        print("  RAW:", _json.dumps(frame)[:220])
        ev_ = normalize(frame)
        if ev_ is None:
            print("  ->  normalize() returned None — the frame shape is not recognised")
        else:
            print(f"  ->  match={ev_.match_id} seq={ev_.sequence} "
                  f"score={ev_.score.as_dict()} server={ev_.server}")
        print()
    unparsed = sum(1 for f in frames if normalize(f) is None)
    if unparsed:
        print(f"⚠ {unparsed}/{len(frames)} frames did not normalize — adjust "
              "normalize() field names before trusting the board.")
    return 0


async def _run_board(poll_s: float, price_s: float, min_edge: float) -> None:
    """Personal live board: local proxy in, ranked edges out.

    Deliberately not the gateway. There is no fan-out, no cloud hop and no
    quota here — one machine reading a proxy that is already running, which is
    why the lag is just the poll interval instead of push + CDN + client poll.

    TWO CADENCES, because the two things move at different speeds. The
    scoreboard changes on every point; market prices do not, and the CLOB book
    costs a request per token. Polling prices at the scoreboard rate would
    multiply upstream calls for numbers that had not moved.
    """
    from execution.live.providers.sofaproxy import SofaProxyProvider
    from execution.live.providers.polymarket_odds import PolymarketOddsSource
    from execution.live.runtime import LiveRuntime
    from execution.live.scanner import ScanFilter

    prov = SofaProxyProvider(poll_s=poll_s)
    rt = LiveRuntime(provider=prov)
    odds = PolymarketOddsSource()

    await prov.connect()
    print(f"live board — scoreboard {poll_s:.0f}s · prices {price_s:.0f}s · "
          f"min edge {min_edge:.0%}   (ctrl-c to stop)\n")

    tokens: dict = {}          # match_id -> (token_p1, token_p2)
    last_price = 0.0
    loop = asyncio.get_running_loop()

    while True:
        t0 = time.time()
        events = await loop.run_in_executor(None, prov.poll_events)

        for ev in events:
            mid, r = ev.match_id, ev.raw
            if mid not in rt.contexts:
                rt.register_match(mid, player1=r["home"], player2=r["away"],
                                  surface=_surface(r.get("surface")))
            await rt.handle_event(ev)

        # Prices on their own beat.
        if time.time() - last_price >= price_s:
            last_price = time.time()
            for mid, ctx in rt.contexts.items():
                if mid not in tokens:
                    # Resolved once per match, not per cycle — the gamma lookup
                    # is the expensive half and the market does not change id.
                    found = await loop.run_in_executor(
                        None, odds.find_tokens, ctx.player1, ctx.player2)
                    tokens[mid] = found[:2] if found else None
                tk = tokens.get(mid)
                if not tk:
                    continue
                await loop.run_in_executor(
                    None, lambda c=ctx, m=mid, t=tk: odds.update(
                        c.view(m, "match_winner"), token_p1=t[0], token_p2=t[1]))

        _print_board(rt, min_edge)
        await asyncio.sleep(max(0.5, poll_s - (time.time() - t0)))


def _surface(ground: str) -> str:
    g = (ground or "").lower()
    if "clay" in g:
        return "Clay"
    if "grass" in g:
        return "Grass"
    return "Hard"


def _print_board(rt, min_edge: float) -> None:
    from execution.live.scanner import ScanFilter
    rows = rt.opportunities(filt=ScanFilter(min_edge=min_edge), limit=20)
    summary = rt.scanner_summary()

    print("\033[2J\033[H", end="")          # clear, so the board updates in place
    print(f"{time.strftime('%H:%M:%S')}   matches {len(rt.contexts)}   "
          f"priced {summary['count']}   events {rt.processed}")
    lag = rt.lag.reaction_stats()
    if lag["n"]:
        print(f"market reaction: median {lag['median_ms']}ms over {lag['n']} moves")
    print()
    print(f"{'MATCH':<38} {'SCORE':<16} {'SRV':<4} {'MODEL':>7} {'MARKET':>7} {'EDGE':>7} {'SCORE':>6}")
    print("-" * 92)

    if not rows:
        print("  no edge above the floor — that is the system working, not an empty feed")
    for o in rows:
        st = rt.store.get(o.match_id)
        if st is None:
            continue
        sc = f"{'-'.join(map(str, st.score.sets))}  {'-'.join(map(str, st.score.games))} " \
             f"{'/'.join(st.score.points)}"
        srv = "p1" if st.server == "p1" else ("p2" if st.server == "p2" else "?")
        print(f"{o.label[:37]:<38} {sc:<16} {srv:<4} {o.model_p:>6.1%} {o.market_p:>7.1%} "
              f"{o.edge:>+7.1%} {o.edge_score:>6.2f}")

    # Everything the model priced but the market has not, so a quiet board is
    # distinguishable from a broken one.
    unpriced = [m for m in rt.contexts if m not in {o.match_id for o in rows}]
    if unpriced:
        print(f"\n  {len(unpriced)} live match(es) without a tradeable edge or price")


def cmd_board(poll_s: float, price_s: float, min_edge: float) -> int:
    try:
        asyncio.run(_run_board(poll_s, price_s, min_edge))
    except KeyboardInterrupt:
        print("\nstopped")
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    return 0


def main(argv: list) -> int:
    cmd = argv[1] if len(argv) > 1 else "doctor"
    if cmd == "doctor":
        return cmd_doctor()
    if cmd == "serve":
        return cmd_serve(int(os.getenv("PORT", "8080")))
    if cmd == "calibrate":
        return cmd_calibrate()
    if cmd == "smoke":
        return cmd_smoke()
    if cmd == "board":
        return cmd_board(
            poll_s=float(os.getenv("BOARD_POLL_S", "3")),
            price_s=float(os.getenv("BOARD_PRICE_S", "10")),
            min_edge=float(os.getenv("BOARD_MIN_EDGE", "0.02")),
        )
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
