"""Live match state (PRD §8) — the authoritative scoreboard we price from.

WHY A STORE INTERFACE RATHER THAN REDIS DIRECTLY
    The PRD specifies Redis. At the volumes this product actually sees — its
    own estimate is ~2,400 events/hour across 20 matches — a dict in one
    process is not the bottleneck, and a second stateful system is a second
    thing to operate, monitor and lose data in.

    So the interface is what the engine depends on, and `InMemoryStateStore` is
    the default. `RedisStateStore` exists for the day one process is no longer
    enough; swapping it is a constructor argument, not a rewrite. Redis stays a
    deployment decision instead of an architectural commitment.

WHAT STATE IS AND IS NOT
    This holds what the FEED told us: score, server, sequence, freshness.
    It holds no probabilities. Prices and model output live beside it
    (`odds.py`, `engine.py`) precisely so a stale scoreboard cannot silently
    drag a fresh price along with it — they age independently and must be
    checked independently.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Iterator, Optional, Protocol

from execution.live.events import EventType, LiveEvent, P1, P2, Score, derive_transitions
from execution.live.feed import FeedStatus, Health, SequenceTracker


@dataclass
class MatchState:
    """Everything we know about one live match from the feed alone."""

    match_id: str
    player1: str = ""
    player2: str = ""
    surface: str = "Hard"
    best_of: int = 3

    score: Score = field(default_factory=Score)
    server: Optional[str] = None
    last_sequence: int = -1
    last_event_ms: int = 0
    started: bool = False
    finished: bool = False

    # Ordered game results, oldest first: (winner, server) per completed game.
    # This is the tape `momentum.py` consumes — it is a function of recent
    # history, so it must survive across events rather than be recomputed.
    games: list = field(default_factory=list)

    # Points won, for serve-percentage blending in inplay.py.
    serve_points: dict = field(default_factory=lambda: {P1: 0, P2: 0})
    serve_points_won: dict = field(default_factory=lambda: {P1: 0, P2: 0})

    def as_dict(self) -> dict:
        d = asdict(self)
        d["score"] = self.score.as_dict()
        return d

    @property
    def age_ms(self) -> int:
        return max(0, int(time.time() * 1000) - self.last_event_ms)


class StateStore(Protocol):
    """The only surface the engine uses. Implementations may be remote."""

    def get(self, match_id: str) -> Optional[MatchState]: ...
    def put(self, state: MatchState) -> None: ...
    def delete(self, match_id: str) -> None: ...
    def keys(self) -> Iterator[str]: ...


class InMemoryStateStore:
    """Default store. One process, one dict."""

    def __init__(self) -> None:
        self._d: dict[str, MatchState] = {}

    def get(self, match_id: str) -> Optional[MatchState]:
        return self._d.get(match_id)

    def put(self, state: MatchState) -> None:
        self._d[state.match_id] = state

    def delete(self, match_id: str) -> None:
        self._d.pop(match_id, None)

    def keys(self) -> Iterator[str]:
        return iter(list(self._d))


class RedisStateStore:
    """Redis-backed store, for when one process is no longer enough.

    Keys follow the PRD: `match:{id}:state`. Values are JSON rather than a
    hash so the whole state is written and read atomically — a partially
    updated scoreboard is worse than a slightly old one, because it is not a
    position that ever existed.
    """

    def __init__(self, client, prefix: str = "match", ttl_s: int = 6 * 3600):
        self._r = client
        self._prefix = prefix
        self._ttl = ttl_s

    def _key(self, match_id: str) -> str:
        return f"{self._prefix}:{match_id}:state"

    def get(self, match_id: str) -> Optional[MatchState]:
        raw = self._r.get(self._key(match_id))
        if not raw:
            return None
        d = json.loads(raw)
        sc = d.pop("score", {}) or {}
        st = MatchState(**d)
        st.score = Score(
            sets=tuple(sc.get("sets", (0, 0))),
            games=tuple(sc.get("games", (0, 0))),
            points=tuple(sc.get("points", ("0", "0"))),
            tiebreak=bool(sc.get("tiebreak", False)),
        )
        return st

    def put(self, state: MatchState) -> None:
        # TTL rather than explicit cleanup: a match that stops emitting should
        # not occupy memory forever because a shutdown path was missed.
        self._r.setex(self._key(state.match_id), self._ttl, json.dumps(state.as_dict()))

    def delete(self, match_id: str) -> None:
        self._r.delete(self._key(match_id))

    def keys(self) -> Iterator[str]:
        for k in self._r.scan_iter(f"{self._prefix}:*:state"):
            yield k.decode().split(":")[1] if isinstance(k, bytes) else k.split(":")[1]


class MatchStateMachine:
    """Applies events to state, and owns that match's sequence tracker.

    One instance per match. The tracker and the state have to move together:
    a state updated from an event the tracker rejected is precisely the
    corruption the tracker exists to prevent.
    """

    def __init__(self, match_id: str, *, store: Optional[StateStore] = None,
                 player1: str = "", player2: str = "", surface: str = "Hard",
                 best_of: int = 3, now_ms=None):
        self.store: StateStore = store or InMemoryStateStore()
        self.tracker = SequenceTracker(now_ms=now_ms)
        self._prev_event: Optional[LiveEvent] = None
        st = MatchState(match_id=match_id, player1=player1, player2=player2,
                        surface=surface, best_of=best_of)
        self.store.put(st)

    def apply(self, event: LiveEvent) -> tuple[MatchState, FeedStatus, list[EventType]]:
        """Fold one event into state.

        Returns the state, the feed status, and the transitions derived from
        this event (§6). The caller decides what to do with a DEGRADED status —
        this does not refuse the update, because holding a state one event
        behind is not safer than holding it current-but-flagged, and the flag
        is what the publish gate reads.
        """
        status = self.tracker.observe(event)
        st = self.store.get(event.match_id)
        if st is None:                       # match we have not seen before
            st = MatchState(match_id=event.match_id)

        # A duplicate replay must not be folded in twice: it would double-count
        # the point in the serve tallies and in the momentum tape.
        if status.duplicates and event.sequence <= st.last_sequence:
            return st, status, []

        transitions = derive_transitions(self._prev_event, event)

        # Record completed games for the momentum tape before overwriting the
        # scoreboard they were derived from.
        if EventType.GAME_END in transitions and self._prev_event is not None:
            winner = _game_winner(self._prev_event.score.games, event.score.games)
            if winner is not None:
                st.games.append({
                    "winner": winner,
                    "server": self._prev_event.server,
                    "break": EventType.BREAK in transitions,
                    "sequence": event.sequence,
                })

        if event.event_type == EventType.POINT and event.server in (P1, P2):
            st.serve_points[event.server] = st.serve_points.get(event.server, 0) + 1
            if event.point_winner == event.server:
                st.serve_points_won[event.server] = st.serve_points_won.get(event.server, 0) + 1

        st.score = event.score
        st.server = event.server if event.server is not None else st.server
        st.last_sequence = max(st.last_sequence, event.sequence)
        st.last_event_ms = max(st.last_event_ms, event.received_ts)
        if event.event_type == EventType.MATCH_START or EventType.MATCH_START in transitions:
            st.started = True
        if event.event_type == EventType.MATCH_END:
            st.finished = True

        self._prev_event = event
        self.store.put(st)
        return st, status, transitions

    def status(self) -> FeedStatus:
        return self.tracker.status()


def _game_winner(prev_games, curr_games) -> Optional[str]:
    d1 = curr_games[0] - prev_games[0]
    d2 = curr_games[1] - prev_games[1]
    if d1 == 1 and d2 == 0:
        return P1
    if d2 == 1 and d1 == 0:
        return P2
    return None


# Events that justify the expensive tier of the model (§13). Everything else
# gets the cheap path — a Monte Carlo per point is latency and compute spent to
# refine a number the market will not have moved on.
SIGNIFICANT = frozenset({
    EventType.BREAK,
    EventType.SET_END,
    EventType.SET_START,
    EventType.MATCH_START,
    EventType.MATCH_END,
    EventType.SUSPENSION,
    EventType.RESUMPTION,
})


def is_significant(transitions: list[EventType], state: MatchState) -> bool:
    """Whether this update warrants the expensive model tier.

    Break/set/match boundaries always qualify. So does any point at 40-x or
    x-40 with a break in play: those are the points where win probability
    actually moves, and refusing to spend compute there defeats the purpose of
    having a live model at all.
    """
    if any(t in SIGNIFICANT for t in transitions):
        return True
    pts = state.score.points
    if state.score.tiebreak:
        return True
    # Game point or break point for either side.
    return "40" in pts or "A" in pts
