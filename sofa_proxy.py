#!/usr/bin/env python3
"""
Lightweight SofaScore proxy server.

SofaScore blocks all programmatic HTTP requests via TLS fingerprinting
(Varnish CDN returns 403).  This micro-server uses `tls_client` to
impersonate a real Chrome browser's TLS handshake, then exposes the
data on a local HTTP endpoint that Next.js can call.

Usage:
    python sofa_proxy.py          # runs on port 3001
    python sofa_proxy.py 8888     # custom port

The Next.js API route at /api/sofa/[...path] calls http://localhost:3001/...
"""

import os
import sys
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import tls_client

SOFA_BASE = "https://www.sofascore.com/api/v1"

# Reuse a single TLS session (Chrome fingerprint) across requests
_session = tls_client.Session(client_identifier="chrome_120")

# Simple in-memory cache: { url: (timestamp, json_bytes) }
_cache: dict[str, tuple[float, bytes]] = {}
_cache_lock = threading.Lock()
CACHE_TTL = 30  # seconds


def _fetch_sofa(path: str) -> tuple[int, bytes]:
    """Fetch from SofaScore with TLS impersonation. Returns (status, body_bytes)."""
    url = f"{SOFA_BASE}/{path}"

    # Check cache first
    with _cache_lock:
        if url in _cache:
            ts, data = _cache[url]
            if time.time() - ts < CACHE_TTL:
                return 200, data

    try:
        r = _session.get(url)
        if r.status_code == 200:
            body = r.content
            with _cache_lock:
                _cache[url] = (time.time(), body)
            return 200, body
        return r.status_code, r.content
    except Exception as e:
        err = json.dumps({"error": str(e)}).encode()
        return 502, err


class SofaHandler(BaseHTTPRequestHandler):
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
            self.send_header("Cache-Control", "public, max-age=30")
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass  # client disconnected before we finished writing

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    # Suppress request logging noise
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[sofa-proxy] {fmt % args}\n")


def main():
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 3001))
    host = os.environ.get("HOST", "0.0.0.0")
    server = HTTPServer((host, port), SofaHandler)
    print(f"[sofa-proxy] listening on http://{host}:{port}")
    print(f"[sofa-proxy] TLS impersonation: chrome_120")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[sofa-proxy] shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
