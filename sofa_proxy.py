#!/usr/bin/env python3
"""
Lightweight SofaScore proxy server — high-throughput edition.

SofaScore blocks all programmatic HTTP requests via TLS fingerprinting
(Varnish CDN returns 403).  This micro-server uses `tls_client` to
impersonate a real Chrome browser's TLS handshake, then exposes the
data on a local HTTP endpoint that Next.js can call.

Features:
  • Session pool (4 Chrome TLS sessions) — parallel requests never block
  • Stale-while-revalidate cache — stale data returns instantly while
    a background thread refreshes the entry
  • Auto-warm popular schedule endpoints on startup

Usage:
    python sofa_proxy.py          # runs on port 3001
    python sofa_proxy.py 8888     # custom port

The Next.js API route at /api/sofa/[...path] calls http://localhost:3001/...
"""

import os
import sys
import json
import time
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse
from datetime import datetime, timedelta

import tls_client

SOFA_BASE = "https://www.sofascore.com/api/v1"

# ─── Session pool (round-robin) ──────────────────────────────────────────────
POOL_SIZE = 4
_sessions = [tls_client.Session(client_identifier="chrome_120") for _ in range(POOL_SIZE)]
_session_locks = [threading.Lock() for _ in range(POOL_SIZE)]
_pool_idx = 0
_pool_idx_lock = threading.Lock()


def _get_session() -> tuple["tls_client.Session", "threading.Lock"]:
    """Round-robin a session from the pool — spreads load across 4 sessions."""
    global _pool_idx
    with _pool_idx_lock:
        idx = _pool_idx % POOL_SIZE
        _pool_idx += 1
    return _sessions[idx], _session_locks[idx]


# ─── Stale-while-revalidate cache ────────────────────────────────────────────
# { url: { "ts": float, "data": bytes, "refreshing": bool } }
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()
CACHE_FRESH = 3      # seconds — serve instantly without refresh
CACHE_STALE_MAX = 30  # seconds — serve stale & refresh in background


def _bg_refresh(url: str, path: str):
    """Background thread: re-fetch one URL and update the cache."""
    try:
        sess, lock = _get_session()
        with lock:
            r = sess.get(url)
        if r.status_code == 200:
            with _cache_lock:
                _cache[url] = {"ts": time.time(), "data": r.content, "refreshing": False}
        else:
            with _cache_lock:
                if url in _cache:
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

    # No cache or fully expired — synchronous fetch
    try:
        sess, lock = _get_session()
        with lock:
            r = sess.get(url)
        if r.status_code == 200:
            body = r.content
            with _cache_lock:
                _cache[url] = {"ts": time.time(), "data": body, "refreshing": False}
            return 200, body
        return r.status_code, r.content
    except Exception as e:
        # If we have any stale data, prefer it over an error
        with _cache_lock:
            if url in _cache:
                return 200, _cache[url]["data"]
        err = json.dumps({"error": str(e)}).encode()
        return 502, err


class SofaHandler(BaseHTTPRequestHandler):
    # Increase timeout so connections don't die under load
    timeout = 30

    def do_GET(self):
        parsed = urlparse(self.path)
        sofa_path = parsed.path.lstrip("/")

        if not sofa_path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","service":"sofa-proxy"}')
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
    paths = [
        f"sport/tennis/scheduled-events/{today}",
        f"sport/tennis/scheduled-events/{tomorrow}",
        f"category/785/scheduled-events/{today}",   # ITF men
        f"category/213/scheduled-events/{today}",   # ITF women
        f"category/785/scheduled-events/{tomorrow}",
        f"category/213/scheduled-events/{tomorrow}",
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


def main():
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 3001))
    host = os.environ.get("HOST", "0.0.0.0")
    server = ThreadedHTTPServer((host, port), SofaHandler)
    print(f"[sofa-proxy] listening on http://{host}:{port}")
    print(f"[sofa-proxy] pool: {POOL_SIZE} sessions | fresh: {CACHE_FRESH}s | stale: {CACHE_STALE_MAX}s | backlog: {server.request_queue_size}")

    # Pre-warm cache in background before serving
    threading.Thread(target=_warm_cache, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[sofa-proxy] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
