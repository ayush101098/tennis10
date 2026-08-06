"""
Push local TT artifacts to the deployed site so table tennis works in production.

WHY
  The TT model can only run locally — ingest/predict/live all go through
  sofa_proxy.py. Netlify can't run that, and the static export deletes
  src/app/api at build time, so /api/tt did not exist in production at all and
  the TT tab reported "feed unreachable". This uploads the pipeline's JSON to
  the site's `tt` blob store (netlify/functions/tt.js), which then serves it.

WHAT IT PUSHES
  live        live_predictions.json   every cycle (small, changes every 8s)
  predictions predictions.json        only when generated_ts changes (~275KB)
  metrics     metrics.json            only when it changes

SETUP
  Set the same secret in both places:
    • Netlify env:  TT_PUSH_TOKEN=<secret>
    • local .env:   TT_PUSH_TOKEN=<secret>
                    TT_SITE_URL=https://your-site.netlify.app

    python -m tabletennis.push --once      # one push, prints what happened
    python -m tabletennis.push             # loop every 15s (run alongside live.py)

DEPLOY SNAPSHOT (no token needed)
  Every cycle also copies the pre-match artifacts into
  trading-terminal/public/tt/, which the static export ships. The terminal
  reads those when /api/tt comes back empty — so table tennis still has a
  board when the blob store is unavailable. Refreshes on redeploy, not on
  push, and carries no in-play True P.

    python -m tabletennis.push --snapshot-only   # refresh, then commit + deploy
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE = Path(__file__).resolve().parent / "site"


def _ssl_ctx() -> ssl.SSLContext:
    """A context with a real CA bundle — a stock python.org build on macOS ships
    no root certificates, so every HTTPS push would die with
    CERTIFICATE_VERIFY_FAILED."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


SSL_CTX = _ssl_ctx()

FILES = {
    "live": "live_predictions.json",
    "predictions": "predictions.json",
    "metrics": "metrics.json",
}


def load_env() -> None:
    """Read .env without a hard dependency on python-dotenv."""
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


def _read(name: str):
    try:
        return json.loads((SITE / name).read_text())
    except Exception:
        return None


# Files copied into the static export so table tennis survives a dead push path.
# live_predictions.json is deliberately excluded: it changes every 8s and would
# only ever be stale by the time a deploy finished.
SNAPSHOT_DEST = REPO_ROOT / "trading-terminal" / "public" / "tt"
SNAPSHOT_FILES = ("predictions.json", "metrics.json")


def snapshot() -> list[str]:
    """Copy the pre-match artifacts into the terminal's public/ directory.

    The deployed site reads these (`/tt/predictions.json`) when /api/tt comes
    back empty — which is what happens whenever Netlify Blobs is unavailable.
    They ship with the build, so they refresh on redeploy, not on push.
    """
    SNAPSHOT_DEST.mkdir(parents=True, exist_ok=True)
    written = []
    for fname in SNAPSHOT_FILES:
        src = SITE / fname
        if not src.exists():
            continue
        dst = SNAPSHOT_DEST / fname
        body = src.read_text()
        if not dst.exists() or dst.read_text() != body:
            dst.write_text(body)
            written.append(fname)
    return written


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects on a push.

    urllib follows a 301 and downgrades the POST to a GET. Pointing TT_SITE_URL
    at a host that redirects — which is exactly what the old *.netlify.app
    address does once a custom domain is made primary — would therefore turn
    every push into a GET of the feed, return HTTP 200, and report success while
    uploading nothing at all. Fail loudly instead.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code,
            f"redirected to {newurl} — point TT_SITE_URL at the final domain",
            headers, fp,
        )


_OPENER = urllib.request.build_opener(
    _NoRedirect,
    urllib.request.HTTPSHandler(context=SSL_CTX),
)


def push(kind: str, payload, url: str, token: str) -> tuple[bool, str]:
    body = json.dumps({"kind": kind, "payload": payload}).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "x-tt-token": token},
    )
    try:
        with _OPENER.open(req, timeout=60) as r:
            return r.status == 200, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.read()[:120].decode(errors='replace')}"
    except Exception as e:
        return False, str(e)[:160]


def cycle(url: str, token: str, sent: dict, force: bool = False,
          push_remote: bool = True) -> None:
    stamp = time.strftime("%H:%M:%S")

    written = snapshot()
    if written:
        print(f"[{stamp}] snapshot    ok   {', '.join(written)} → public/tt/ "
              f"(commit + redeploy to publish)", flush=True)
    if not push_remote:
        return

    for kind, fname in FILES.items():
        payload = _read(fname)
        if payload is None:
            print(f"[{stamp}] {kind:11s} skip — {fname} missing/unreadable", flush=True)
            continue
        # only re-upload the big pre-match/metrics files when they actually change
        stamp_key = payload.get("generated_ts") if isinstance(payload, dict) else None
        fp = stamp_key if stamp_key is not None else hash(json.dumps(payload, sort_keys=True))
        if kind != "live" and not force and sent.get(kind) == fp:
            continue
        ok, msg = push(kind, payload, url, token)
        sent[kind] = fp if ok else sent.get(kind)
        n = payload.get("n") if isinstance(payload, dict) else ""
        print(f"[{stamp}] {kind:11s} {'ok  ' if ok else 'FAIL'} {msg}"
              f"{f' · n={n}' if n else ''}", flush=True)


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description="Push TT artifacts to the deployed site")
    ap.add_argument("--url", default=os.getenv("TT_SITE_URL", ""),
                    help="site base URL, e.g. https://your-site.netlify.app")
    ap.add_argument("--token", default=os.getenv("TT_PUSH_TOKEN", ""))
    ap.add_argument("--interval", type=int, default=15, help="seconds between pushes")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--snapshot-only", action="store_true",
                    help="only refresh trading-terminal/public/tt/ (no token needed); "
                         "commit + redeploy to publish the pre-match board")
    args = ap.parse_args()

    if args.snapshot_only:
        written = snapshot()
        print(f"snapshot → {SNAPSHOT_DEST}")
        print("  " + (", ".join(written) + " updated" if written else "already up to date"))
        print("  commit these and redeploy — the terminal reads them when /api/tt is empty")
        return

    if not args.url or not args.token:
        print("ERROR: set TT_SITE_URL and TT_PUSH_TOKEN (env or .env), or pass "
              "--url/--token.\n       The same TT_PUSH_TOKEN must be set in the "
              "Netlify site env.\n       (Use --snapshot-only to refresh the deploy "
              "snapshot without pushing.)", file=sys.stderr)
        sys.exit(2)

    endpoint = args.url.rstrip("/") + "/api/tt"
    print("=" * 64)
    print(f"TT PUSH → {endpoint}   every {args.interval}s")
    print("=" * 64, flush=True)

    sent: dict = {}
    try:
        while True:
            t0 = time.time()
            cycle(endpoint, args.token, sent, force=not sent)
            if args.once:
                break
            time.sleep(max(5.0, args.interval - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
