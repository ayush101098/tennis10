"""Push updates to the Cloudflare edge (PRD §25).

WHERE THE FAN-OUT ACTUALLY HAPPENS
    `RoomRegistry` fans out to viewers connected to THIS process. That is
    correct for a single instance and is what the tests exercise, but it means
    every viewer holds a socket to one Python process.

    The edge moves that: the runtime pushes one HTTP request per update to a
    Durable Object, and the DO broadcasts to everyone in the room. Viewer
    number 1,000 then costs Cloudflare a socket write and costs this process
    nothing — which is the whole architecture in one sentence.

    Both paths are supported on purpose. Local rooms need no Cloudflare account
    and make the system runnable and testable on a laptop; the edge is what
    scales. The runtime does not care which is attached.

FAILING QUIET, ON PURPOSE
    A push failure must not stop the pipeline. The engine's job is to keep
    pricing; delivery is best-effort, and an exception here would take the
    scoreboard down for a CDN hiccup. Failures are counted and surfaced in
    /health rather than raised.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest


class EdgePublisher:
    """Posts payloads to `/match/{id}/push` on the edge worker."""

    def __init__(self, *, base_url: Optional[str] = None, token: Optional[str] = None,
                 timeout_s: float = 3.0, transport=None):
        self.base_url = (base_url or os.getenv("EDGE_BASE_URL", "")).rstrip("/")
        self.token = token or os.getenv("EDGE_PUSH_TOKEN", "")
        self.timeout_s = timeout_s
        # Injected for tests; the default does a real POST.
        self._transport = transport
        self.pushed = 0
        self.failed = 0
        self.last_error: Optional[str] = None
        self._last_sent_ms = 0

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    async def publish(self, match_id: str, payload: dict) -> bool:
        if not self.configured and self._transport is None:
            return False
        try:
            if self._transport is not None:
                await _maybe_await(self._transport(match_id, payload))
            else:
                # Blocking urllib on a worker thread: adding an async HTTP
                # client as a dependency for one POST per update is not worth
                # it at this volume (a busy slate is a few events a second).
                await asyncio.get_running_loop().run_in_executor(
                    None, self._post, match_id, payload)
            self.pushed += 1
            self._last_sent_ms = int(time.time() * 1000)
            return True
        except Exception as e:
            self.failed += 1
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            return False

    def _post(self, match_id: str, payload: dict) -> None:
        req = urlrequest.Request(
            f"{self.base_url}/match/{match_id}/push",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-push-token": self.token},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_s) as r:
                r.read()
        except urlerror.HTTPError as e:
            # 401 is a configuration error and will not fix itself; make it
            # loud in the message rather than blending into transport noise.
            raise RuntimeError(f"edge push {e.code}"
                               f"{' — check EDGE_PUSH_TOKEN' if e.code == 401 else ''}") from e

    def health(self) -> dict:
        return {
            "configured": self.configured,
            "pushed": self.pushed,
            "failed": self.failed,
            "last_error": self.last_error,
            "last_sent_ms": self._last_sent_ms,
        }


async def _maybe_await(v):
    if asyncio.iscoroutine(v):
        return await v
    return v
