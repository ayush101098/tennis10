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
    python push_sofa.py             # loop every 45s (leave running)
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


# ─── Request budget ──────────────────────────────────────────────────────────
# Everything below exists because the first version of this loop fetched EVERY
# path every 30s: 1 live + 10 scheduled + 40 live events x 3 detail paths = 131
# requests per cycle, 262/min, ~377,000/day. SofaScore challenged the IP for it
# (403 {"reason":"challenge"} on every endpoint, including from a real browser),
# the feed froze, and the board served hours-old fixtures as if they were current.
#
# So each path now refreshes at the rate it actually changes. A scheduled draw
# does not move every 30 seconds; a live point does.
SCHEDULED_EVERY = 20   # cycles between scheduled-events refreshes (~15 min)
ODDS_EVERY = 10        # cycles between bulk daily-odds refreshes (~7.5 min)
STATS_EVERY = 4        # cycles between per-event statistics refreshes
LIVE_DETAIL_CAP = 12   # in-play matches carrying full detail, best tours first


def paths_for(days: int = 2, cycle_n: int = 0) -> list[str]:
    """Paths due this cycle. The live feed every time; the rest on their own beat."""
    today = _dt.date.today()
    out = ["sport/tennis/events/live"]
    want_sched = cycle_n % SCHEDULED_EVERY == 0
    want_odds = cycle_n % ODDS_EVERY == 0
    for i in range(days):
        d = (today + _dt.timedelta(days=i)).isoformat()
        # sport/tennis/scheduled-events/<date> was dropped upstream — it 404s
        # every time, so it is no longer requested. Coverage comes from the
        # per-category endpoints below.
        if want_sched:
            out += [f"category/{c}/scheduled-events/{d}" for c in CATEGORIES]
        if want_odds:
            out.append(f"sport/tennis/odds/1/{d}")
    return out


# Per-event detail the terminal asks for on every LIVE match. Without these
# pushed, the board renders matches and scores but every in-play request 503s:
# no live odds, so no live edge, and no point-by-point, so no momentum.
LIVE_EVENT_PATHS = (
    "event/{id}/odds/1/all",
    "event/{id}/point-by-point",
)
# Serve/return splits move far slower than the score does.
SLOW_EVENT_PATHS = ("event/{id}/statistics",)

# Detail budget goes to the tours people actually open, not to whichever ITF
# match happens to sort first.
_TIER = {3: 0, 6: 0, 72: 1, 785: 2, 213: 2}


def live_event_paths(limit: int = LIVE_DETAIL_CAP, cycle_n: int = 0) -> list[str]:
    """Paths for the events currently in play, straight from the live feed."""
    payload, err = fetch_local("sport/tennis/events/live")
    if not payload:
        return []

    def rank(evt) -> tuple:
        cat = ((evt.get("tournament") or {}).get("category") or {}).get("id")
        return (_TIER.get(cat, 3), evt.get("startTimestamp") or 0)

    inplay = [e for e in (payload.get("events") or [])
              if ((e.get("status") or {}).get("type") == "inprogress") and e.get("id")]
    inplay.sort(key=rank)

    templates = list(LIVE_EVENT_PATHS)
    if cycle_n % STATS_EVERY == 0:
        templates += SLOW_EVENT_PATHS

    out = []
    for evt in inplay[:limit]:
        out += [t.format(id=evt["id"]) for t in templates]
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
        if e.code == 403:
            return None, CHALLENGED
        return None, f"proxy HTTP {e.code}"
    except Exception as e:
        return None, str(e)[:120]


# Sentinel for "SofaScore is refusing us", as distinct from "this path 404s".
# The difference matters: a 404 is one dead path, a challenge means every
# further request digs the hole deeper, so the loop has to stand down.
CHALLENGED = "challenged"


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


def cycle(endpoint: str, token: str, days: int, verbose: bool,
          cycle_n: int = 0) -> tuple[int, int, bool]:
    ok_n = fail_n = 0
    challenged_n = 0
    # Scheduled/odds first, then per-event detail for whatever is in play now.
    for path in paths_for(days, cycle_n) + live_event_paths(cycle_n=cycle_n):
        payload, err = fetch_local(path)
        if payload is None:
            if err == CHALLENGED:
                # A challenge on ONE path is no longer a reason to abandon the
                # cycle. sofa_proxy now owns upstream backoff — it benches burned
                # egresses itself and serves what it can from Flashscore — so
                # these requests never reach SofaScore and cannot deepen a ban.
                # Some paths simply have no fallback (odds, point-by-point);
                # skipping those while pushing everything else is the difference
                # between a working board and an empty one.
                challenged_n += 1
                if verbose:
                    print(f"    skip  {path}  (no upstream, no fallback)", flush=True)
                continue
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
    # Only stand down when the cycle got NOTHING through. That is the real
    # "we are locked out everywhere" signal; anything less is partial coverage,
    # which is worth pushing.
    return ok_n, fail_n, (ok_n == 0 and challenged_n > 0)


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description="Push SofaScore tennis data to the deployed site")
    ap.add_argument("--url", default=os.getenv("TT_SITE_URL", ""))
    ap.add_argument("--token", default=os.getenv("TT_PUSH_TOKEN", ""))
    ap.add_argument("--interval", type=int, default=45)
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
    if probe is None and err != CHALLENGED:
        print(f"ERROR: local sofa proxy unreachable at {LOCAL_PROXY} ({err}).\n"
              f"       Start it first:  python sofa_proxy.py", file=sys.stderr)
        sys.exit(2)
    if err == CHALLENGED:
        print("WARNING: SofaScore is currently challenging this IP. Starting in "
              "backoff — the loop will keep probing at a widening interval and "
              "resume full pushes the moment it is let back in.", flush=True)

    endpoint = args.url.rstrip("/") + "/api/sofa/_push"
    print("=" * 68)
    print(f"SOFA PUSH → {endpoint}   every {args.interval}s · {args.days}d")
    print("=" * 68, flush=True)

    # While challenged the loop probes with ONE request at 5, 10, 20, 40, 60min
    # rather than continuing to poll. A ban that is still being hammered is a
    # ban that does not expire.
    BACKOFF = (300, 600, 1200, 2400, 3600)
    strikes = 0
    n = 0
    try:
        while True:
            t0 = time.time()
            if strikes:
                # Single cheap probe; only a success ends the standdown.
                payload, err = fetch_local("sport/tennis/events/live")
                if payload is None:
                    wait = BACKOFF[min(strikes - 1, len(BACKOFF) - 1)]
                    print(f"[{time.strftime('%H:%M:%S')}] still challenged "
                          f"(strike {strikes}) — next probe in {wait // 60}min",
                          flush=True)
                    strikes += 1
                    if args.once:
                        break
                    time.sleep(wait)
                    continue
                print(f"[{time.strftime('%H:%M:%S')}] upstream is back — resuming",
                      flush=True)
                strikes = 0

            ok_n, fail_n, challenged = cycle(endpoint, args.token, args.days,
                                             not args.quiet, n)
            if challenged:
                strikes = 1
                if args.once:
                    break
                print(f"[{time.strftime('%H:%M:%S')}] backing off "
                      f"{BACKOFF[0] // 60}min", flush=True)
                time.sleep(BACKOFF[0])
                continue

            n += 1
            print(f"[{time.strftime('%H:%M:%S')}] pushed {ok_n} path(s)"
                  f"{f', {fail_n} failed' if fail_n else ''}", flush=True)
            if args.once:
                break
            time.sleep(max(10.0, args.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
