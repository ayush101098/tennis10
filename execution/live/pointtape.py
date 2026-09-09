"""Reconstructing the server and the point tape when the feed gives neither.

WHY THIS EXISTS
    SofaScore has challenged this IP: every path 403s and sofa_proxy falls back
    to Flashscore for everything. Flashscore carries the score but not the
    server (it renders it as an icon) and has no point-by-point feed at all —
    re-verified, `df_pbp_1_<id>` returns one byte.

    So the two things live analytics need most are both missing, and no amount
    of provider-switching produces them. They can, however, be RECONSTRUCTED
    from endpoints that do answer:

    THE SERVER, from `statistics`
        "Service Points Won" reports `(won/served)` per player. The denominator
        is how many points that player has SERVED. Between two polls, whoever's
        denominator grew is the one serving. Measured live 2026-09-09: away +2
        while home +0, and the reverse on another match — clean, unambiguous.

        One anchor is enough. Serve alternates every completed game, so after
        anchoring once the server is known by alternation, and re-anchoring
        occasionally corrects any drift. That keeps this to a handful of extra
        requests rather than one per match per poll.

    THE POINT TAPE, from the point score
        The live feed carries the current game's point score. Each transition
        is a point, and the player whose score advanced won it. This yields the
        tape from the moment we start watching — not the match's earlier
        history, which is simply not recoverable from these sources.

WHAT IT WILL NOT DO
    Invent a server. Until the anchor lands the server is None, and everything
    serve-conditioned stays silent — the alternative is a hold probability
    computed for the wrong player, which is the bug this replaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from execution.live.events import P1, P2

# Re-anchor every N completed games. Alternation is exact, so this only guards
# against a missed game boundary; too often and it is wasted requests.
REANCHOR_EVERY_GAMES = 6

_SERVED_RE = re.compile(r"\((\d+)\s*/\s*(\d+)\)")


def served_counts(stats_payload: dict) -> tuple[Optional[int], Optional[int]]:
    """Points served by (home, away), from a `statistics` payload.

    Reads the denominator of "Service Points Won (won/served)". Returns
    (None, None) when the shape is not what we expect, so a feed change
    degrades to "no anchor" instead of a wrong one.
    """
    for group in (stats_payload or {}).get("statistics", []) or []:
        for grp in group.get("groups", []) or []:
            for item in grp.get("statisticsItems", []) or []:
                if item.get("name") != "Service Points Won":
                    continue
                h = _SERVED_RE.search(str(item.get("home", "")))
                a = _SERVED_RE.search(str(item.get("away", "")))
                if h and a:
                    return int(h.group(2)), int(a.group(2))
    return None, None


@dataclass
class ServeState:
    """What we know about who is serving, and how sure we are."""

    server: Optional[str] = None          # P1 / P2 / None
    anchored: bool = False
    games_since_anchor: int = 0
    last_served: tuple = (None, None)     # (home, away) points served
    source: str = "unknown"               # "statistics" | "alternation"


class ServeTracker:
    """Infers the server per match and keeps it current by alternation."""

    def __init__(self, *, reanchor_every: int = REANCHOR_EVERY_GAMES):
        self.reanchor_every = reanchor_every
        self._state: dict[str, ServeState] = {}

    def state(self, match_id: str) -> ServeState:
        return self._state.setdefault(match_id, ServeState())

    def needs_anchor(self, match_id: str) -> bool:
        """Whether it is worth spending a statistics request on this match."""
        st = self.state(match_id)
        return (not st.anchored) or st.games_since_anchor >= self.reanchor_every

    def observe_statistics(self, match_id: str, stats_payload: dict) -> Optional[str]:
        """Feed a statistics payload. Returns the server when it can tell.

        Needs two samples: the first records the counters, the second sees
        which grew. That is the whole trick, and it is why a match that has
        played no points between polls stays unknown rather than guessing.
        """
        st = self.state(match_id)
        h, a = served_counts(stats_payload)
        if h is None or a is None:
            return st.server

        prev_h, prev_a = st.last_served
        st.last_served = (h, a)
        if prev_h is None or prev_a is None:
            return st.server                     # first sample: baseline only

        dh, da = h - prev_h, a - prev_a
        if dh > da:
            st.server, st.anchored, st.source = P1, True, "statistics"
            st.games_since_anchor = 0
        elif da > dh:
            st.server, st.anchored, st.source = P2, True, "statistics"
            st.games_since_anchor = 0
        # dh == da (usually both 0): nothing was served, so nothing is learned.
        return st.server

    def observe_game_completed(self, match_id: str) -> Optional[str]:
        """A game ended, so serve passes to the other player.

        Alternation is exact in tennis, which is what lets one anchor carry a
        whole match. A tiebreak counts as one game here: serve rotates inside
        it, but the game after it continues the match alternation.
        """
        st = self.state(match_id)
        if not st.anchored or st.server is None:
            return None
        st.server = P2 if st.server == P1 else P1
        st.games_since_anchor += 1
        st.source = "alternation"
        return st.server

    def server(self, match_id: str) -> Optional[str]:
        return self.state(match_id).server

    def forget(self, match_id: str) -> None:
        self._state.pop(match_id, None)


# ── point tape ───────────────────────────────────────────────────────────────

# Ordering within a game. "A" (advantage) sits above 40; a drop back to 40 is
# the opponent winning the point, not this player losing ground on their own.
_ORDER = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4}


@dataclass
class TapePoint:
    """One reconstructed point."""

    winner: str                    # P1 / P2
    server: Optional[str]
    score_before: tuple
    score_after: tuple
    game_point: bool = False


@dataclass
class _MatchTape:
    last_points: Optional[tuple] = None
    last_games: Optional[tuple] = None
    points: list = field(default_factory=list)


class PointTape:
    """Rebuilds the point sequence from successive point scores.

    Only as complete as the polling: points before the first observation, and
    two points landing inside one interval, are not recoverable. That is stated
    rather than hidden — `gaps` counts the times it happened, so a consumer can
    tell a complete tape from a sampled one.
    """

    def __init__(self):
        self._tapes: dict[str, _MatchTape] = {}
        self.gaps = 0

    def observe(self, match_id: str, points: tuple, games: tuple,
                server: Optional[str] = None) -> Optional[TapePoint]:
        """Record a scoreboard reading. Returns a point when one was won."""
        tape = self._tapes.setdefault(match_id, _MatchTape())
        prev_pts, prev_games = tape.last_points, tape.last_games
        tape.last_points, tape.last_games = points, games

        if prev_pts is None or prev_games is None:
            return None                       # first sighting: baseline only

        # A game boundary resets the point score, so the transition is not a
        # point in the current game — the game-completion path owns it.
        if games != prev_games:
            return None
        if points == prev_pts:
            return None

        a0, b0 = _ORDER.get(prev_pts[0]), _ORDER.get(prev_pts[1])
        a1, b1 = _ORDER.get(points[0]), _ORDER.get(points[1])
        if None in (a0, b0, a1, b1):
            # A tiebreak carries integer points, not 0/15/30/40 — handled by
            # numeric comparison instead.
            return self._numeric(tape, match_id, prev_pts, points, server)

        if a1 > a0 and b1 <= b0:
            winner = P1
        elif b1 > b0 and a1 <= a0:
            winner = P2
        elif a0 == 4 and a1 == 3:
            winner = P2                       # advantage lost -> back to deuce
        elif b0 == 4 and b1 == 3:
            winner = P1
        else:
            self.gaps += 1                    # more than one point in a poll
            return None

        pt = TapePoint(winner=winner, server=server,
                       score_before=prev_pts, score_after=points,
                       game_point=(a1 == 4 or b1 == 4 or a1 == 3 or b1 == 3))
        tape.points.append(pt)
        return pt

    def _numeric(self, tape, match_id, prev_pts, points, server):
        try:
            a0, b0 = int(prev_pts[0]), int(prev_pts[1])
            a1, b1 = int(points[0]), int(points[1])
        except (TypeError, ValueError):
            return None
        if a1 == a0 + 1 and b1 == b0:
            winner = P1
        elif b1 == b0 + 1 and a1 == a0:
            winner = P2
        else:
            self.gaps += 1
            return None
        pt = TapePoint(winner=winner, server=server,
                       score_before=prev_pts, score_after=points)
        tape.points.append(pt)
        return pt

    def points(self, match_id: str) -> list:
        t = self._tapes.get(match_id)
        return list(t.points) if t else []

    def forget(self, match_id: str) -> None:
        self._tapes.pop(match_id, None)
