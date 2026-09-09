"""Local sofa_proxy — the fastest source available, for personal use.

WHY THIS IS THE ONE TO RUN LOCALLY
    The deployed path is push_sofa -> Netlify blob -> CDN -> browser poll, and
    every hop adds staleness: ~3s push + up to 8s CDN + up to 10s client poll.
    That chain exists to serve many people cheaply.

    Running for yourself, none of it applies. sofa_proxy is already on this
    machine answering in milliseconds, so reading it directly collapses the lag
    to the poll interval and nothing else. No push, no blob, no CDN, no quota.

THE SERVER, WHICH MATTERS
    SofaScore's `firstToServe` is misleadingly named: on live events it updates
    every game and carries the CURRENT server. Taken at face value it gives us
    something the Livesport leg cannot — the server — which is what the set and
    game rungs of the ladder (`setengine.py`) need.

    It is absent whenever the proxy has failed over to Flashscore, and absent
    means unknown: the ladder shows nothing rather than guessing, because a
    wrong server inverts the game market.

REQUEST BUDGET — the real limit for personal use is not a bill
    There is no usage quota here; the constraint is SofaScore challenging the
    IP, which has happened before. That ban came from ~262 requests/minute
    across 131 paths per cycle.

    This polls ONE bulk endpoint. At the 3s default that is 20 requests/minute
    to the local proxy — an order of magnitude under what caused trouble, and
    the proxy is the only thing talking to SofaScore. Going below ~2s buys
    nothing anyway: SofaScore's own feed does not update faster than that.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from typing import AsyncIterator, Optional

from execution.live.events import EventType, LiveEvent, P1, P2, Score
from execution.live.provider import TennisDataProvider

DEFAULT_BASE = os.getenv("SOFA_PROXY", "http://127.0.0.1:3001")
LIVE_PATH = "/sport/tennis/events/live"

# Bookmakers reprice in roughly 1-3s. Matching that is the point; going faster
# only re-reads a feed that has not changed.
DEFAULT_POLL_S = 3.0

_POINT = {"0": "0", "15": "15", "30": "30", "40": "40", "A": "A", "AD": "A"}

# Tour categories worth pricing. Deep ITF qualifying is mostly players the
# rankings file does not cover, so the model has no prior and the gate stays
# silent — the rows would be noise on a personal board.
DEFAULT_CATEGORIES = frozenset({"atp", "wta", "challenger", "wta-125"})


def _games(score: dict) -> list:
    """Per-set games from SofaScore's period fields."""
    out = []
    for i in range(1, 6):
        v = score.get(f"period{i}")
        if v is None:
            break
        out.append(int(v))
    return out


def derive_server(first_to_serve: Optional[int], home_games: list = None,
                  away_games: list = None, in_tiebreak: bool = False) -> Optional[str]:
    """Who is serving now.

    `firstToServe` is a MISLEADING NAME: on live events SofaScore updates it
    every game, so it carries the CURRENT server, not the match opener. That
    was verified live against ESPN possession by the web client
    (`scheduleService.ts`, which has used it directly in production), and it is
    the reading this now follows.

    It previously flipped the value by the parity of games played, on the
    assumption the field meant "who served first". That assumption was never
    verified, and it is wrong in the worst possible way: it agrees with reality
    on an even game count and inverts it on an odd one, so the server displayed
    was correct about half the time. Reported from a live match, 2026-09-09.

    `home_games` / `away_games` are kept in the signature so callers do not
    change, and are deliberately unused — the parity they were for is the bug.

    Returns None when the field is absent, which is common: the Flashscore
    fallback source does not carry a server at all, and a guess here inverts
    the game market.
    """
    if first_to_serve == 1:
        return P1
    if first_to_serve == 2:
        return P2
    return None


def event_from_sofa(e: dict, *, sequence: int, received_ms: Optional[int] = None
                    ) -> Optional[LiveEvent]:
    """One SofaScore live event to one canonical `LiveEvent`."""
    hs = e.get("homeScore") or {}
    as_ = e.get("awayScore") or {}
    hg, ag = _games(hs), _games(as_)
    if not hg or not ag:
        return None

    idx = min(len(hg), len(ag)) - 1
    set_no = idx + 1
    tb = (hs.get(f"period{set_no}TieBreak") is not None
          or as_.get(f"period{set_no}TieBreak") is not None
          or (hg[idx] >= 6 and hg[idx] == ag[idx]))

    # In a tiebreak the "point" field carries tiebreak points, which are plain
    # integers rather than 15/30/40 — pass them through as-is instead of
    # mapping them onto game scoring they do not belong to.
    raw_hp = str(hs.get("point", "0")).upper()
    raw_ap = str(as_.get("point", "0")).upper()
    if tb:
        points = (raw_hp or "0", raw_ap or "0")
    else:
        points = (_POINT.get(raw_hp, "0"), _POINT.get(raw_ap, "0"))

    server = derive_server(e.get("firstToServe"))

    now = received_ms if received_ms is not None else int(time.time() * 1000)
    return LiveEvent(
        match_id=str(e.get("id")),
        sequence=sequence,
        event_type=EventType.POINT,
        provider_ts=now,
        received_ts=now,
        score=Score(
            sets=(int(hs.get("current") or 0), int(as_.get("current") or 0)),
            games=(hg[idx], ag[idx]),
            points=points,
            tiebreak=bool(tb),
        ),
        server=server,
        raw={
            "home": (e.get("homeTeam") or {}).get("name", ""),
            "away": (e.get("awayTeam") or {}).get("name", ""),
            "tournament": (e.get("tournament") or {}).get("name", ""),
            "category": (((e.get("tournament") or {}).get("category")) or {}).get("slug", ""),
            "surface": e.get("groundType", ""),
            "status": (e.get("status") or {}).get("description", ""),
        },
    )


class SofaProxyProvider(TennisDataProvider):
    """Polls the LOCAL sofa_proxy. No credentials, no cloud, no quota."""

    name = "sofaproxy"

    # One bulk endpoint per poll and no provider sequence field, so gaps are
    # undetectable by construction — same honest caveat as the Livesport leg.
    has_sequence_guarantee = False

    def __init__(self, *, base_url: str = DEFAULT_BASE, poll_s: float = DEFAULT_POLL_S,
                 categories: frozenset = DEFAULT_CATEGORIES, timeout_s: float = 8.0,
                 fetch=None):
        self.base_url = base_url.rstrip("/")
        self.poll_s = poll_s
        self.categories = categories
        self.timeout_s = timeout_s
        self._fetch = fetch          # injected in tests
        self.subscriptions: set[str] = set()
        self.connected = False
        self.polls = 0
        self.last_error: Optional[str] = None
        self._seq: dict[str, int] = {}
        self._last: dict[str, tuple] = {}

    async def connect(self) -> None:
        if self.connected:
            return
        # Fail loudly here rather than silently returning nothing forever: a
        # proxy that is not running is the single most likely reason a personal
        # run shows an empty board.
        try:
            self._raw()
        except Exception as e:
            raise RuntimeError(
                f"sofa_proxy is not answering at {self.base_url} ({type(e).__name__}). "
                "Start it with:  python sofa_proxy.py") from e
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def subscribe(self, match_id: str) -> None:
        self.subscriptions.add(str(match_id))

    async def unsubscribe(self, match_id: str) -> None:
        self.subscriptions.discard(str(match_id))

    def _raw(self) -> dict:
        if self._fetch is not None:
            return self._fetch()
        req = urllib.request.Request(self.base_url + LIVE_PATH,
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())

    def _fingerprint(self, ev: LiveEvent) -> tuple:
        return (ev.score.sets, ev.score.games, ev.score.points, ev.server)

    def poll_events(self) -> list:
        """One poll into the events that represent real changes."""
        self.polls += 1
        try:
            data = self._raw()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            return []

        out = []
        for e in (data.get("events") or []):
            cat = (((e.get("tournament") or {}).get("category")) or {}).get("slug", "")
            if self.categories and cat not in self.categories:
                continue
            # Singles only — doubles names carry a slash and the model prices
            # one player against one player.
            home = (e.get("homeTeam") or {}).get("name", "")
            if "/" in home:
                continue

            mid = str(e.get("id"))
            if self.subscriptions and mid not in self.subscriptions:
                continue

            ev = event_from_sofa(e, sequence=self._seq.get(mid, 0) + 1)
            if ev is None:
                continue
            fp = self._fingerprint(ev)
            if self._last.get(mid) == fp:
                continue                       # nothing moved
            self._last[mid] = fp
            self._seq[mid] = self._seq.get(mid, 0) + 1
            out.append(ev)
        return out

    async def events(self) -> AsyncIterator[LiveEvent]:  # type: ignore[override]
        loop = asyncio.get_running_loop()
        while self.connected:
            for ev in await loop.run_in_executor(None, self.poll_events):
                yield ev
            await asyncio.sleep(self.poll_s)

    async def resync(self, match_id: str) -> Optional[LiveEvent]:
        """A poll is a resync — clear the fingerprint so the next one re-emits."""
        self._last.pop(str(match_id), None)
        return None
