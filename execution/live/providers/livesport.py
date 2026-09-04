"""Livesport / Flashscore as a second live provider.

WHY A SECOND LEG AT ALL
    SofaScore challenges an IP once it has pulled enough API traffic, and no TLS
    trick undoes it — the address is what is being refused. When that happens
    the board has no live source, which has already happened once. Livesport
    runs on entirely separate infrastructure and answers from an address
    SofaScore has burned, so it is a genuine failover rather than a mirror.

    Until now `FailoverManager` managed exactly one provider, which makes it a
    health tracker rather than a failover. This is the second leg that makes it
    real.

WHAT THIS PROVIDER CAN AND CANNOT DO — stated plainly, because the limits
change how the rest of the pipeline must treat it.

    ✓ live scores, per-set games, tiebreaks, sets won
    ✓ the CURRENT GAME's point score (0/15/30/40/A)
    ✗ point-by-point HISTORY. Verified again 2026-09-04: the legacy
      `df_pbp_1_<id>` feed returns one byte — empty — on www.livesport.com,
      local-global.flashscore.ninja and d.flashscore.com alike. The site renders
      point-by-point from a persisted-query GraphQL API
      (700.ds.lsapp.eu/pq_graphql) whose operation hash rotates with each client
      release. Depending on it would mean an integration that breaks on their
      deploy schedule, so this provider polls instead and produces points from
      the moment it starts watching.
    ✗ the SERVER. Flashscore renders it as an icon, not a feed field. It is
      emitted as None rather than guessed, and `derive_transitions` will
      therefore report GAME_END without BREAK. That is deliberate: momentum and
      the signal engine weight breaks heavily, and a fabricated break is worse
      than a missing one.

NO SEQUENCE GUARANTEE
    A polling source has no sequence of record, so `sequence` here is a local
    counter and `SequenceTracker` can never observe a gap from it. That is
    honest rather than convenient: this provider genuinely cannot tell you it
    missed a point.

    Integrity comes from the other direction instead. Two points inside one poll
    interval collapse into a score JUMP, and the jump is what is detectable —
    `events.derive_transitions` already refuses to attribute a game winner when
    the games score moves by more than one, so a missed game produces no
    phantom break. Poll faster to make misses rarer; do not pretend they are
    impossible.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Optional

from execution.live.events import EventType, LiveEvent, Score
from execution.live.provider import TennisDataProvider

# Points arrive roughly every 30s; 8s keeps two-points-per-interval rare without
# hammering a source that is doing us a favour by answering at all.
DEFAULT_POLL_S = 8.0

_POINT_MAP = {"0": "0", "15": "15", "30": "30", "40": "40", "A": "A", "AD": "A"}


def _points(m) -> tuple[str, str]:
    """Current game score, or love-all when the detail feed has not filled in."""
    h = _POINT_MAP.get(str(m.point_home or "0").upper(), "0")
    a = _POINT_MAP.get(str(m.point_away or "0").upper(), "0")
    return h, a


def match_to_event(m, *, sequence: int, received_ms: Optional[int] = None) -> LiveEvent:
    """One Flashscore `Match` to one canonical `LiveEvent`.

    The set index is 1-based and `home_games` is a per-set list, so the CURRENT
    set's games are the last entry — using the first would freeze the scoreboard
    at set one, which is the kind of off-by-one that looks like a stale feed.
    """
    hg = m.home_games or [0]
    ag = m.away_games or [0]
    idx = max(0, min(len(hg), len(ag)) - 1)

    # A tiebreak is flagged from the tiebreak map OR structurally from 6-6.
    # The map only populates once tiebreak points are played, so a set sitting
    # at 6-6 with the tiebreak about to start reported tiebreak=False — and
    # both `setengine` and the model's significance tiering key off this flag,
    # so the most important game of the set was being priced as an ordinary one.
    tb = bool(m.home_tb.get(m.set_index) is not None
              or m.away_tb.get(m.set_index) is not None)
    if not tb and int(hg[idx]) >= 6 and int(hg[idx]) == int(ag[idx]):
        tb = True

    now = received_ms if received_ms is not None else int(time.time() * 1000)
    return LiveEvent(
        match_id=str(m.fs_id),
        sequence=sequence,
        event_type=EventType.POINT,
        # Flashscore stamps no per-update time, so provider time is our receive
        # time. Reported honestly rather than invented — it makes the measured
        # provider latency zero for this leg, which is true of what we can see.
        provider_ts=now,
        received_ts=now,
        score=Score(
            sets=(int(m.home_sets or 0), int(m.away_sets or 0)),
            games=(int(hg[idx]), int(ag[idx])),
            points=_points(m),
            tiebreak=tb,
        ),
        server=None,                    # not exposed — never guessed
        point_winner=None,
        raw={"tour": m.tour, "tournament": m.tournament, "surface": m.surface,
             "home": m.home, "away": m.away, "status": m.status},
    )


class LivesportProvider(TennisDataProvider):
    """Polls Livesport's live feed and emits canonical events.

    `client` is injectable so this is testable without the network — the cases
    worth pinning (a score jump, a match ending, an empty poll) cannot be
    requested from a live source.
    """

    name = "livesport"

    # Read by the runtime: this provider cannot detect dropped events, so a
    # consumer should not read "no gaps" from it as evidence of completeness.
    has_sequence_guarantee = False

    def __init__(self, *, client=None, poll_s: float = DEFAULT_POLL_S,
                 with_points: bool = True):
        self._client = client
        self.poll_s = poll_s
        self.with_points = with_points
        self.subscriptions: set[str] = set()
        self.connected = False
        self.polls = 0
        self.last_error: Optional[str] = None
        self._seq: dict[str, int] = {}
        self._last: dict[str, tuple] = {}

    def _ensure_client(self):
        if self._client is None:
            from execution.flashscore import FlashscoreClient
            self._client = FlashscoreClient()
        return self._client

    async def connect(self) -> None:
        if self.connected:
            return                       # idempotent — reconnect calls this again
        self._ensure_client()
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def subscribe(self, match_id: str) -> None:
        self.subscriptions.add(str(match_id))

    async def unsubscribe(self, match_id: str) -> None:
        self.subscriptions.discard(str(match_id))

    def _poll(self) -> list:
        """One synchronous poll. Isolated so tests can drive it directly."""
        self.polls += 1
        return self._ensure_client().live_matches(with_points=self.with_points) or []

    def _fingerprint(self, m) -> tuple:
        """What counts as a change worth emitting.

        Only the scoreboard. Re-emitting an identical state every 8s would fill
        the pipeline with events that move nothing, and the model would burn a
        reprice on each.
        """
        hg = m.home_games or [0]
        ag = m.away_games or [0]
        return (int(m.home_sets or 0), int(m.away_sets or 0),
                tuple(hg), tuple(ag), m.point_home, m.point_away)

    def poll_events(self) -> list:
        """Convert one poll into the events that represent real changes."""
        try:
            matches = self._poll()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            return []

        out = []
        for m in matches:
            mid = str(m.fs_id)
            if self.subscriptions and mid not in self.subscriptions:
                continue                 # §20: only pay for matches being watched
            fp = self._fingerprint(m)
            if self._last.get(mid) == fp:
                continue                 # nothing moved
            self._last[mid] = fp
            self._seq[mid] = self._seq.get(mid, 0) + 1
            out.append(match_to_event(m, sequence=self._seq[mid]))
        return out

    async def events(self) -> AsyncIterator[LiveEvent]:  # type: ignore[override]
        loop = asyncio.get_running_loop()
        while self.connected:
            # The client is blocking requests; keep it off the event loop or a
            # slow poll stalls every other match's delivery.
            for ev in await loop.run_in_executor(None, self.poll_events):
                yield ev
            await asyncio.sleep(self.poll_s)

    async def resync(self, match_id: str) -> Optional[LiveEvent]:
        """A poll IS a resync: the next one carries the authoritative score.

        Clearing the fingerprint forces the next poll to re-emit even if the
        score has not moved, so a consumer that lost state gets a fresh one.
        """
        self._last.pop(str(match_id), None)
        return None
