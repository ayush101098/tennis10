#!/usr/bin/env python3
"""
Lightweight SofaScore proxy server — multi-egress edition.

SofaScore blocks programmatic HTTP two different ways, and they need two
different answers:

  1. TLS fingerprint.  A stock python/curl handshake is 403'd by the Varnish CDN
     ({"reason":"Forbidden"}).  `tls_client` impersonates a real Chrome handshake
     and gets past it.  This has always been here.

  2. IP reputation.  Once an address has pulled enough API traffic, SofaScore
     challenges that *address* on /api/v1/* specifically — every fingerprint
     403s with {"reason":"challenge"} while plain HTML pages still return 200.
     No handshake trick fixes this; the request has to leave from somewhere
     else.  That is what the egress pool below is for.

EGRESS POOL
  Sessions are grouped into "lanes".  Each lane owns one TLS session pinned to
  one egress (a residential/mobile proxy, or the direct connection).  A lane
  that gets challenged is benched with an escalating cooldown and traffic moves
  to the others, so one burned IP degrades throughput instead of killing the
  feed.  Configure with SOFA_EGRESS (see below); with nothing set it runs
  exactly as before, one direct lane.

Usage:
    python sofa_proxy.py          # runs on port 3001
    python sofa_proxy.py 8888     # custom port
    python sofa_proxy.py --check  # probe every egress, print status, exit

Config (env or .env):
    SOFA_EGRESS          comma/space/newline separated proxy URLs, e.g.
                         http://user:pass@gate.provider.com:7000
                         socks5://user:pass@host:1080
                         Rotating-gateway URLs are fine — each lane holds its
                         own session, so each gets its own sticky exit IP.
    SOFA_EGRESS_FILE     path to a file of the same, one per line
    SOFA_EGRESS_DIRECT   1 = also keep one direct (un-proxied) lane. Default 0
                         when proxies are configured, 1 when none are.
    SOFA_LANES           lanes per egress (default 2)

The Next.js API route at /api/sofa/[...path] calls http://localhost:3001/...
"""

import os
import re
import sys
import json
import time
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime, timedelta

import tls_client

SOFA_BASE = "https://www.sofascore.com/api/v1"
REPO_ROOT = Path(__file__).resolve().parent

# ─── Browser identity ────────────────────────────────────────────────────────
# NOTE: browser headers + random_tls_extension_order are required — without
# them SofaScore's CDN returns 403 for every request.
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}

# Lanes cycle through these so the pool doesn't present one identical
# fingerprint from every exit IP — that pattern is itself a signal.
_CLIENT_IDS = ("chrome_120", "chrome_124", "chrome_131", "chrome_133")


def load_env() -> None:
    """Read .env / .env.local without adding a dependency."""
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


def _mask(url: str | None) -> str:
    """Proxy URLs carry credentials — never log them raw."""
    if not url:
        return "direct"
    try:
        p = urlparse(url)
        host = p.hostname or "?"
        port = f":{p.port}" if p.port else ""
        user = f"{p.username[:3]}***@" if p.username else ""
        return f"{p.scheme}://{user}{host}{port}"
    except Exception:
        return "proxy"


def egress_list() -> list[str | None]:
    """Configured egresses, in order. `None` means the direct connection."""
    raw = os.getenv("SOFA_EGRESS", "").strip()
    if not raw:
        f = os.getenv("SOFA_EGRESS_FILE", "").strip()
        if f and Path(f).expanduser().exists():
            raw = Path(f).expanduser().read_text()
    urls = [u.strip() for u in re.split(r"[,\s]+", raw) if u.strip()
            and not u.strip().startswith("#")]

    default_direct = "0" if urls else "1"
    want_direct = os.getenv("SOFA_EGRESS_DIRECT", default_direct) not in ("0", "false", "no")

    out: list[str | None] = list(urls)
    if want_direct or not out:
        out.append(None)
    return out


# ─── Lanes ───────────────────────────────────────────────────────────────────
# A lane is one TLS session pinned to one egress, plus that egress's health.
# Cooldowns escalate so a burned IP is rested rather than hammered — a ban that
# keeps getting traffic is a ban that does not expire.
_COOLDOWN = (60, 300, 900, 1800, 3600)


class Lane:
    def __init__(self, idx: int, egress: str | None):
        self.idx = idx
        self.egress = egress
        self.lock = threading.Lock()
        self.strikes = 0
        self.blocked_until = 0.0
        self.ok_n = 0
        self.fail_n = 0
        self.exit_ip: str | None = None
        self.session = self._new_session()

    def _new_session(self) -> "tls_client.Session":
        s = tls_client.Session(
            client_identifier=_CLIENT_IDS[self.idx % len(_CLIENT_IDS)],
            random_tls_extension_order=True,
        )
        s.headers.update(_BROWSER_HEADERS)
        if self.egress:
            s.proxies = {"http": self.egress, "https": self.egress}
        return s

    @property
    def available(self) -> bool:
        return time.time() >= self.blocked_until

    def penalise(self) -> None:
        """Challenged — bench this lane and hand it a fresh session."""
        self.strikes += 1
        self.fail_n += 1
        self.blocked_until = time.time() + _COOLDOWN[min(self.strikes - 1, len(_COOLDOWN) - 1)]
        # A challenged session may be carrying a poisoned cookie/TLS ticket;
        # start clean so the cooldown is actually testing the IP, not the state.
        self.session = self._new_session()

    def reward(self) -> None:
        self.strikes = 0
        self.ok_n += 1

    def status(self) -> dict:
        return {
            "lane": self.idx,
            "egress": _mask(self.egress),
            "exit_ip": self.exit_ip,
            "available": self.available,
            "cooldown_s": max(0, round(self.blocked_until - time.time())),
            "strikes": self.strikes,
            "ok": self.ok_n,
            "fail": self.fail_n,
        }

    def check(self) -> dict:
        """Probe this lane: resolve its exit IP, then try the live endpoint."""
        try:
            with self.lock:
                r = self.session.get("https://api.ipify.org?format=json", timeout_seconds=20)
                self.exit_ip = (r.json() or {}).get("ip")
        except Exception as e:
            self.exit_ip = f"unreachable ({str(e)[:40]})"
        try:
            with self.lock:
                r = self.session.get(f"{SOFA_BASE}/sport/tennis/events/live", timeout_seconds=25)
            st = r.status_code
            n = len((r.json() or {}).get("events", [])) if st == 200 else 0
            return {**self.status(), "probe_status": st, "events": n}
        except Exception as e:
            return {**self.status(), "probe_status": "error", "error": str(e)[:60]}


_lanes: list[Lane] = []
_lane_idx = 0
_lane_idx_lock = threading.Lock()


def _build_lanes() -> list[Lane]:
    egresses = egress_list()
    per = max(1, int(os.getenv("SOFA_LANES", "2")))
    lanes = []
    i = 0
    for eg in egresses:
        for _ in range(per):
            lanes.append(Lane(i, eg))
            i += 1
    return lanes


def _get_lane() -> Lane:
    """Round-robin over healthy lanes; if all are benched, take the freest."""
    global _lane_idx
    with _lane_idx_lock:
        n = len(_lanes)
        for _ in range(n):
            lane = _lanes[_lane_idx % n]
            _lane_idx += 1
            if lane.available:
                return lane
        # Everything is cooling down — use whichever frees up soonest.
        return min(_lanes, key=lambda l: l.blocked_until)


# ─── Stale-while-revalidate cache ────────────────────────────────────────────
# { url: { "ts": float, "data": bytes, "refreshing": bool } }
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
CACHE_FRESH = 3       # seconds — serve instantly without refresh
CACHE_STALE_MAX = 30  # seconds — serve stale & refresh in background

_CHALLENGE_RE = re.compile(rb"challenge", re.I)


# ─── Flashscore fallback ─────────────────────────────────────────────────────
# When every egress is challenged, an empty board is the worst possible answer.
# Flashscore covers the same tours from separate infrastructure and answers from
# addresses SofaScore has burned, so we translate its feed into SofaScore's
# schema and serve that instead. The client URL contract is unchanged — the
# terminal, push_sofa and the execution pipeline cannot tell the difference
# beyond a "source":"flashscore" marker on each event.
#
# Not everything has an equivalent: Flashscore publishes no odds and no
# point-by-point (see execution/flashscore.py). Those paths return empty rather
# than fabricating numbers a trading model would then price off.
_FALLBACK_ON = os.getenv("SOFA_FALLBACK", "1") not in ("0", "false", "no")
_fs_client = None
_fs_lock = threading.Lock()

_RE_SCHEDULED = re.compile(r"^category/(\d+)/scheduled-events/(\d{4}-\d{2}-\d{2})$")
_RE_EVENT_STATS = re.compile(r"^event/(\d+)/statistics$")


def _flashscore():
    global _fs_client
    with _fs_lock:
        if _fs_client is None:
            from execution.flashscore import FlashscoreClient
            # Reuse the egress pool — if a proxy is configured it is just as
            # useful here, and a lane already proven to work is a good bet.
            eg = next((l.egress for l in _lanes if l.egress and l.available), None)
            _fs_client = FlashscoreClient(egress=eg)
        return _fs_client


def _fallback(path: str) -> tuple[int, bytes] | None:
    """Serve `path` from Flashscore, or None if it has no equivalent."""
    if not _FALLBACK_ON:
        return None
    try:
        from execution import flashscore as fsmod
        client = _flashscore()

        if path == "sport/tennis/events/live":
            payload = fsmod.sofa_live_payload(client)
        elif (m := _RE_SCHEDULED.match(path)):
            payload = fsmod.sofa_scheduled_payload(client, int(m.group(1)), m.group(2))
        elif (m := _RE_EVENT_STATS.match(path)):
            fs_id = fsmod.flashscore_id(int(m.group(1)))
            if not fs_id:
                return None
            payload = fsmod.sofa_statistics_payload(client, fs_id)
        else:
            return None

        body = json.dumps(payload).encode()
        print(f"[sofa-proxy] fallback→flashscore  {path}  "
              f"({len(payload.get('events', [])) if 'events' in payload else 'ok'})",
              flush=True)
        return 200, body
    except Exception as e:
        print(f"[sofa-proxy] fallback failed for {path}: {str(e)[:110]}", flush=True)
        return None


def _is_challenge(status: int, body: bytes) -> bool:
    """403 + 'challenge' means the IP is burned, not that the path is wrong."""
    return status == 403 and bool(_CHALLENGE_RE.search(body or b""))


def _raw_get(url: str) -> tuple[int, bytes, Lane]:
    """One attempt down one lane, updating that lane's health."""
    lane = _get_lane()
    try:
        with lane.lock:
            r = lane.session.get(url, timeout_seconds=30)
        status, body = r.status_code, r.content
    except Exception as e:
        lane.fail_n += 1
        return 0, json.dumps({"error": str(e)[:120]}).encode(), lane

    if _is_challenge(status, body):
        lane.penalise()
    elif status == 200:
        lane.reward()
    return status, body, lane


def _fetch_upstream(url: str) -> tuple[int, bytes]:
    """Try each lane once before giving up — a challenge on one IP is not a
    challenge on all of them, which is the entire point of the pool."""
    attempts = max(1, len(_lanes))
    last: tuple[int, bytes] = (502, b'{"error":"no lanes"}')
    for _ in range(attempts):
        status, body, lane = _raw_get(url)
        if status == 200:
            return 200, body
        last = (status or 502, body)
        if not _is_challenge(status, body) and status != 0:
            return last          # a real 404/500 — trying another IP won't help
    return last


def _bg_refresh(url: str, path: str):
    """Background thread: re-fetch one URL and update the cache."""
    try:
        status, body = _fetch_upstream(url)
        with _cache_lock:
            if status == 200:
                _cache[url] = {"ts": time.time(), "data": body, "refreshing": False}
            elif url in _cache:
                _cache[url]["refreshing"] = False
    except Exception:
        with _cache_lock:
            if url in _cache:
                _cache[url]["refreshing"] = False


def _fetch_sofa(path: str) -> tuple[int, bytes]:
    """Fetch from SofaScore with TLS impersonation. Returns (status, body_bytes)."""
    url = f"{SOFA_BASE}/{path}"

    # Check cache
    with _cache_lock:
        entry = _cache.get(url)
        if entry:
            age = time.time() - entry["ts"]
            if age < CACHE_FRESH:
                return 200, entry["data"]                    # fresh — serve directly
            if age < CACHE_STALE_MAX:
                # Stale but usable — serve immediately, refresh in background
                if not entry.get("refreshing"):
                    entry["refreshing"] = True
                    threading.Thread(target=_bg_refresh, args=(url, path), daemon=True).start()
                return 200, entry["data"]
            # Expired beyond stale limit — fall through to synchronous fetch

    status, body = _fetch_upstream(url)
    if status == 200:
        with _cache_lock:
            _cache[url] = {"ts": time.time(), "data": body, "refreshing": False}
        return 200, body

    # Upstream refused every lane. Try Flashscore before falling back to stale
    # data: a live score from another provider beats an hours-old one from this
    # one. Cache the result so the next caller isn't charged the same round trip.
    fb = _fallback(path)
    if fb:
        with _cache_lock:
            _cache[url] = {"ts": time.time(), "data": fb[1], "refreshing": False}
        return fb

    # Nothing live anywhere. Stale data still beats an empty board.
    with _cache_lock:
        if url in _cache:
            return 200, _cache[url]["data"]
    return status, body


class SofaHandler(BaseHTTPRequestHandler):
    # Increase timeout so connections don't die under load
    timeout = 30

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        sofa_path = parsed.path.lstrip("/")

        if not sofa_path:
            self._json(200, {"status": "ok", "service": "sofa-proxy",
                             "lanes": len(_lanes)})
            return

        # Operational view: which egresses are alive, which are benched.
        if sofa_path in ("_health", "_lanes"):
            healthy = sum(1 for l in _lanes if l.available)
            self._json(200, {
                "service": "sofa-proxy",
                "lanes_total": len(_lanes),
                "lanes_available": healthy,
                "degraded": healthy == 0,
                "cached_endpoints": len(_cache),
                "detail": [l.status() for l in _lanes],
            })
            return

        status, body = _fetch_sofa(sofa_path)

        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "public, max-age=3")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected before we finished writing

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    # Suppress request logging noise
    def log_message(self, fmt, *args):
        pass  # silent — remove noise from terminal


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread for concurrent fast polling."""
    allow_reuse_address = True
    allow_reuse_port = True
    daemon_threads = True
    request_queue_size = 64  # allow many queued connections (default is 5)

    def server_bind(self):
        """Set SO_REUSEADDR and increase listen backlog to prevent ECONNRESET."""
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass  # SO_REUSEPORT not available on all platforms
        super().server_bind()


def _warm_cache():
    """Pre-fetch popular schedule endpoints so the first UI load is instant."""
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    # NOTE: the sport-level generic endpoint currently 404s upstream, so the
    # per-category endpoints below are what actually carry each tour. ATP=3,
    # WTA=6, Challenger=72, ITF Men=785, ITF Women=213.
    paths = [
        f"category/{cat}/scheduled-events/{d}"
        for d in (today, tomorrow)
        for cat in (3, 6, 72, 785, 213)  # ATP, WTA, Challenger, ITF M, ITF W
    ] + [
        f"sport/tennis/odds/1/{today}",      # daily bulk odds — feeds the Value Board
        f"sport/tennis/odds/1/{tomorrow}",
        "sport/tennis/events/live",
    ]
    def fetch_one(p):
        try:
            _fetch_sofa(p)
        except Exception:
            pass
    threads = [threading.Thread(target=fetch_one, args=(p,), daemon=True) for p in paths]
    for t in threads:
        t.start()
    # Wait up to 5s for warm-up to complete
    for t in threads:
        t.join(timeout=5)
    print(f"[sofa-proxy] cache warmed: {len(_cache)} endpoints ready")


def _print_check() -> int:
    """--check: probe every lane and report. Exit 0 if any lane can reach the API."""
    print(f"[sofa-proxy] probing {len(_lanes)} lane(s)…\n")
    good = 0
    for lane in _lanes:
        r = lane.check()
        ok = r.get("probe_status") == 200
        good += ok
        mark = "OK  " if ok else "FAIL"
        extra = f"{r.get('events', 0)} events" if ok else f"status={r.get('probe_status')}"
        print(f"  {mark}  lane {r['lane']:<2} {r['egress']:<44} exit={r['exit_ip']}  {extra}")
    print(f"\n[sofa-proxy] {good}/{len(_lanes)} lane(s) can reach SofaScore.")
    if not good:
        print("\n  Every egress is challenged. Add residential/mobile proxies:\n"
              "    SOFA_EGRESS=http://user:pass@gateway:port,http://user:pass@gateway2:port\n"
              "  in .env, then re-run with --check.")
    return 0 if good else 1


def main():
    load_env()
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    global _lanes
    _lanes = _build_lanes()

    if "--check" in sys.argv:
        sys.exit(_print_check())

    port = int(os.environ.get("PORT", argv[0] if argv else 3001))
    host = os.environ.get("HOST", "0.0.0.0")
    server = ThreadedHTTPServer((host, port), SofaHandler)
    print(f"[sofa-proxy] listening on http://{host}:{port}")
    print(f"[sofa-proxy] lanes: {len(_lanes)} | fresh: {CACHE_FRESH}s | "
          f"stale: {CACHE_STALE_MAX}s | backlog: {server.request_queue_size}")
    for eg in dict.fromkeys(_mask(l.egress) for l in _lanes):
        print(f"[sofa-proxy]   egress: {eg}")
    print(f"[sofa-proxy] health: http://127.0.0.1:{port}/_health")

    # Pre-warm cache in background before serving
    threading.Thread(target=_warm_cache, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[sofa-proxy] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
