"""
SofaScore TLS-impersonation proxy — Vercel Python serverless function.

SofaScore's CDN 403s any request whose TLS ClientHello doesn't look like a
real browser (Node's fetch/undici and Python's plain requests/urllib both
get blocked). `tls_client` impersonates Chrome's TLS fingerprint; combined
with browser-shaped headers and randomized extension order, that's enough
to pass. This mirrors sofa_proxy.py (the original standalone server) but as
a stateless serverless function — every path under this deployment's domain
is forwarded to https://www.sofascore.com/api/v1/<path>.
"""

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

import tls_client

SOFA_BASE = "https://www.sofascore.com/api/v1"

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}


def _fetch(path: str) -> tuple[int, bytes]:
    session = tls_client.Session(client_identifier="chrome_120", random_tls_extension_order=True)
    session.headers.update(_BROWSER_HEADERS)
    try:
        r = session.get(f"{SOFA_BASE}/{path}")
        return r.status_code, r.content
    except Exception as e:
        return 502, f'{{"error": "{str(e)}"}}'.encode()


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        sofa_path = urlparse(self.path).path.lstrip("/")

        if not sofa_path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","service":"sofa-proxy-vercel"}')
            return

        status, body = _fetch(sofa_path)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=3")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def log_message(self, fmt, *args):
        pass
