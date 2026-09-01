"""Canonical live-event schema — the seam between any provider and our model.

WHY THIS EXISTS SEPARATELY FROM THE MODEL
    Every tennis feed emits a different shape, and the engines in this package
    (`inplay.py`, `momentum.py`, `edgescore.py`) already know how to price a
    match. What we have never had is a provider-independent thing to hand them.
    Without it, provider JSON leaks into the model and swapping feeds means
    rewriting the pricing path — which is exactly how a "temporary" scraper
    becomes load-bearing.

    So: providers produce `LiveEvent`, and nothing downstream of the normalizer
    ever sees a provider's own field names.

WHAT IS DELIBERATELY NOT HERE
    No pricing, no edge, no signals. Those exist already. This module is the
    transport vocabulary and nothing else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    """Event kinds we accept from any provider.

    Providers differ wildly in what they emit — most give POINT and score, few
    give GAME_START, almost none give BREAK as its own event. The normalizer
    derives what it can (see `derive_transitions`) rather than requiring a
    provider to supply everything, because requiring it would mean supporting
    exactly one provider.
    """

    MATCH_START = "MATCH_START"
    POINT = "POINT"
    GAME_START = "GAME_START"
    GAME_END = "GAME_END"
    BREAK = "BREAK"
    SET_START = "SET_START"
    SET_END = "SET_END"
    MATCH_END = "MATCH_END"
    SUSPENSION = "SUSPENSION"
    RESUMPTION = "RESUMPTION"
    ODDS_UPDATE = "ODDS_UPDATE"


# Which side. Kept as a plain string ("p1"/"p2") rather than an int so a
# serialized event is readable in a log without a decoder ring.
P1 = "p1"
P2 = "p2"


@dataclass(frozen=True)
class Score:
    """Scoreboard at the moment the event happened.

    `points` are strings ("0", "15", "30", "40", "A") rather than numbers,
    because tennis point scores are not numbers — advantage has no integer and
    a tiebreak counts differently. Storing "40" as 3 forces every consumer to
    know which of the two schemes is in play.
    """

    sets: tuple[int, int] = (0, 0)
    games: tuple[int, int] = (0, 0)
    points: tuple[str, str] = ("0", "0")
    tiebreak: bool = False

    def as_dict(self) -> dict:
        return {
            "sets": list(self.sets),
            "games": list(self.games),
            "points": list(self.points),
            "tiebreak": self.tiebreak,
        }


@dataclass(frozen=True)
class LiveEvent:
    """One normalized event from any provider.

    `sequence` is mandatory and provider-assigned. It is the only thing that
    lets us tell "nothing has happened" apart from "we lost the connection and
    missed four points", and those two must never look alike to a pricing
    engine. See `feed.SequenceTracker`.
    """

    match_id: str
    sequence: int
    event_type: EventType

    # §21 latency accounting. `provider_ts` is when the provider says it
    # happened; `received_ts` is when it reached us. Both in epoch ms. The gap
    # between them is the one latency term we cannot optimise, only measure.
    provider_ts: int
    received_ts: int = field(default_factory=lambda: int(time.time() * 1000))

    score: Score = field(default_factory=Score)
    server: Optional[str] = None          # P1 / P2 / None when the feed omits it
    point_winner: Optional[str] = None    # set on POINT events
    event_id: str = ""

    # Anything provider-specific that we do not model but may want when
    # debugging a disagreement between two feeds.
    raw: dict = field(default_factory=dict)

    @property
    def provider_latency_ms(self) -> int:
        """How far behind the provider we are. Never negative: a provider clock
        running ahead of ours would otherwise report a negative latency and
        poison any average built on it."""
        return max(0, self.received_ts - self.provider_ts)

    def as_payload(self) -> dict:
        """The compact shape the WebSocket gateway sends on (§17).

        Deliberately small and flat. The browser is not a database client; it
        needs what changed, not the event's full provenance.
        """
        return {
            "match_id": self.match_id,
            "sequence": self.sequence,
            "type": self.event_type.value,
            "score": self.score.as_dict(),
            "server": self.server,
            "ts": self.provider_ts,
        }


def derive_transitions(prev: Optional[LiveEvent], curr: LiveEvent) -> list[EventType]:
    """Infer the structural events a provider did not send.

    Most feeds emit POINT and a score and nothing else. The engines want to
    know when a game ended, when serve was broken, when a set closed — so we
    derive them from consecutive scoreboards rather than requiring the feed to
    be generous.

    A break is a game won by the RETURNER, so it can only be derived when the
    feed tells us who was serving. When `server` is None we emit GAME_END
    without BREAK rather than guessing: a fabricated break is worse than a
    missing one, because momentum and the signal engine both weight it heavily.
    """
    if prev is None:
        return [EventType.MATCH_START] if curr.event_type == EventType.POINT else []

    out: list[EventType] = []
    p_games, c_games = prev.score.games, curr.score.games
    p_sets, c_sets = prev.score.sets, curr.score.sets

    if c_sets != p_sets:
        # A set closed. SET_END implies the game that won it also ended.
        out.append(EventType.GAME_END)
        if prev.server is not None and _game_winner(p_games, c_games) != prev.server:
            out.append(EventType.BREAK)
        out.append(EventType.SET_END)
        out.append(EventType.SET_START)
        return out

    if c_games != p_games:
        out.append(EventType.GAME_END)
        winner = _game_winner(p_games, c_games)
        if prev.server is not None and winner is not None and winner != prev.server:
            out.append(EventType.BREAK)
        out.append(EventType.GAME_START)

    return out


def _game_winner(prev_games: tuple[int, int], curr_games: tuple[int, int]) -> Optional[str]:
    """Which side's game count went up. None if the change is not a clean +1 —
    a jump of two means we missed a game, and inventing a winner for it would
    hand the momentum engine a fictional result."""
    d1 = curr_games[0] - prev_games[0]
    d2 = curr_games[1] - prev_games[1]
    if d1 == 1 and d2 == 0:
        return P1
    if d2 == 1 and d1 == 0:
        return P2
    return None
