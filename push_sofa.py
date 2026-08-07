"""
Push SofaScore tennis data from this machine to the deployed site.

WHY
  SofaScore blocks servers by TLS fingerprint. sofa_proxy.py defeats that
  locally, but the deployed proxy gets challenged too, so production had no
  working tennis source at all:
      /api/sofa/...  -> 403 {"error":{"code":403,"reason":"challenge"}}
      ESPN           -> returns tournaments but ZERO individual matches
  The board therefore rendered empty while ATP/WTA matches were live.

  This uploads the exact endpoints the terminal requests to the site's `sofa`
  blob cache (netlify/functions/sofa-proxy.js), which serves them whenever
  upstream is challenged. The client URL contract is unchanged.

WHAT IT PUSHES  (everything the schedule needs for ATP + WTA + Challenger + ITF)
  sport/tennis/events/live                        every cycle — live scores
  sport/tennis/scheduled-events/<date>            today + tomorrow
  category/{3,6,72,785,213}/scheduled-events/<date>   ATP/WTA/Ch/ITF-M/ITF-W
  sport/tennis/odds/1/<date>                      bulk match-winner odds

SETUP
  Needs sofa_proxy.py running locally, and the same secret both sides:
    • Netlify env:  TT_PUSH_TOKEN=<secret>
    • local .env:   TT_PUSH_TOKEN=<secret>
                    TT_SITE_URL=https://tennispredictions.netlify.app

    python push_sofa.py --once      # one push, prints each path
    python push_sofa.py             # loop every 30s (leave running)
"""

import argparse
import datetime as _dt
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_PROXY = os.getenv("SOFA_PROXY", "http://127.0.0.1:3001")


def _ssl_ctx() -> ssl.SSLContext:
    """A context with a real CA bundle.

    A stock python.org build on macOS ships no root certificates, so every
    HTTPS push dies with CERTIFICATE_VERIFY_FAILED while the local (plain HTTP)
    proxy reads succeed — i.e. it looks like the site is rejecting us when
    nothing has actually left the machine. Prefer certifi when it's installed.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


SSL_CTX = _ssl_ctx()

# ATP=3 WTA=6 Challenger=72 ITF Men=785 ITF Women=213 (must match
# SOFA_CAT_URLS in trading-terminal/src/lib/scheduleService.ts)
CATEGORIES = [3, 6, 72, 785, 213]


def load_env() -> None:
    for name in (".env", ".env.local"):
        p = REPO_ROOT / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def paths_for(days: int = 2) -> list[str]:
    today = _dt.date.today()
    out = ["sport/tennis/events/live"]
    for i in range(days):
        d = (today + _dt.timedelta(days=i)).isoformat()
        # sport/tennis/scheduled-events/<date> was dropped upstream — it 404s
        # every time, so it is no longer requested. Coverage comes from the
        # per-category endpoints below.
        out += [f"category/{c}/scheduled-events/{d}" for c in CATEGORIES]
        out.append(f"sport/tennis/odds/1/{d}")
    return out


# Per-event detail the terminal asks for on every LIVE match. Without these
# pushed, the board renders matches and scores but every in-play request 503s:
# no live odds, so no live edge, and no point-by-point, so no momentum.
LIVE_EVENT_PATHS = (
    "event/{id}/odds/1/all",
    "event/{id}/point-by-point",
    "event/{id}/statistics",
)


def live_event_paths(limit: int = 40) -> list[str]:
    """Paths for the events currently in play, straight from the live feed."""
    payload, err = fetch_local("sport/tennis/events/live")
    if not payload:
        return []
    out = []
    for evt in (payload.get("events") or [])[:limit]:
        eid = evt.get("id")
        if not eid:
            continue
        out += [t.format(id=eid) for t in LIVE_EVENT_PATHS]
    return out


def fetch_local(path: str):
    """Pull one endpoint through the local (TLS-impersonating) sofa proxy."""
    try:
        req = urllib.request.Request(f"{LOCAL_PROXY}/{path}",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as r:
            if r.status != 200:
                return None, f"proxy HTTP {r.status}"
            return json.loads(r.read().decode()), None
    except urllib.error.HTTPError as e:
        return None, f"proxy HTTP {e.code}"
    except Exception as e:
        return None, str(e)[:120]


def push(path: str, payload, endpoint: str, token: str):
    body = json.dumps({"path": path, "payload": payload}).encode()
    req = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={"Content-Type": "application/json", "x-tt-token": token})
    try:
        with urllib.request.urlopen(req, timeout=90, context=SSL_CTX) as r:
            return r.status == 200, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.read()[:100].decode(errors='replace')}"
    except Exception as e:
        return False, str(e)[:120]


def _count(payload) -> str:
    if isinstance(payload, dict):
        for k in ("events", "odds"):
            v = payload.get(k)
            if isinstance(v, list):
                return f"{len(v)} events"
            if isinstance(v, dict):
                return f"{len(v)} odds"
    return ""


def cycle(endpoint: str, token: str, days: int, verbose: bool) -> tuple[int, int]:
    ok_n = fail_n = 0
    # Scheduled/odds first, then per-event detail for whatever is in play now.
    for path in paths_for(days) + live_event_paths():
        payload, err = fetch_local(path)
        if payload is None:
            # SofaScore legitimately 404s some category/date combos — not an error
            if verbose:
                print(f"    skip  {path}  ({err})", flush=True)
            continue
        ok, msg = push(path, payload, endpoint, token)
        ok_n += ok
        fail_n += (not ok)
        if verbose or not ok:
            print(f"    {'ok  ' if ok else 'FAIL'}  {path}  {_count(payload)} {'' if ok else msg}",
                  flush=True)
    return ok_n, fail_n


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description="Push SofaScore tennis data to the deployed site")
    ap.add_argument("--url", default=os.getenv("TT_SITE_URL", ""))
    ap.add_argument("--token", default=os.getenv("TT_PUSH_TOKEN", ""))
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--days", type=int, default=2, help="today + N-1 following days")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.url or not args.token:
        print("ERROR: set TT_SITE_URL and TT_PUSH_TOKEN (env or .env), or pass "
              "--url/--token. The same TT_PUSH_TOKEN must be set in the Netlify "
              "site env.", file=sys.stderr)
        sys.exit(2)

    # fail fast if the local proxy isn't up — otherwise every cycle is a no-op
    probe, err = fetch_local("sport/tennis/events/live")
    if probe is None:
        print(f"ERROR: local sofa proxy unreachable at {LOCAL_PROXY} ({err}).\n"
              f"       Start it first:  python sofa_proxy.py", file=sys.stderr)
        sys.exit(2)

    endpoint = args.url.rstrip("/") + "/api/sofa/_push"
    print("=" * 68)
    print(f"SOFA PUSH → {endpoint}   every {args.interval}s · {args.days}d")
    print("=" * 68, flush=True)
    try:
        while True:
            t0 = time.time()
            ok_n, fail_n = cycle(endpoint, args.token, args.days, not args.quiet)
            print(f"[{time.strftime('%H:%M:%S')}] pushed {ok_n} path(s)"
                  f"{f', {fail_n} failed' if fail_n else ''}", flush=True)
            if args.once:
                break
            time.sleep(max(10.0, args.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
