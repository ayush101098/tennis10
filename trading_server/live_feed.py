"""
Live Match Feed
===============
Unified live tennis data feed that auto-detects score changes and emits
point-by-point events to the trading engine.

Data Sources (cascading priority):
  1. api-tennis.com (paid) — point-level score + server + stats
  2. Sofascore (free)      — point-level score + server
  3. ESPN (free)            — game-level score only (infers points)

The feed polls the best available source, diffs against the previous
state, and when a score change is detected, emits ScoreChange events
that the trading loop can consume.

Usage:
    feed = LiveMatchFeed()
    feed.attach("Sinner", "Alcaraz")  # finds the match automatically

    # Poll loop
    while True:
        changes = feed.poll()
        for change in changes:
            # change.point_winner, change.new_score, change.server, etc.
            session.register_point(change.point_winner)
        await asyncio.sleep(1)
"""

from __future__ import annotations

import os
import re
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum

logger = logging.getLogger(__name__)


# ─── Data Models ──────────────────────────────────────────────────────────────

class DataSource(str, Enum):
    API_TENNIS = "api-tennis"
    SOFASCORE = "sofascore"
    ESPN = "espn"
    MANUAL = "manual"


@dataclass
class MatchScore:
    """Snapshot of a match score at a point in time."""
    sets_p1: int = 0
    sets_p2: int = 0
    games_p1: int = 0
    games_p2: int = 0
    points_p1: int = 0      # Internal index: 0,1,2,3,4... (0=0, 1=15, 2=30, 3=40, 4=AD)
    points_p2: int = 0
    point_text: str = "0-0"  # Display: "30-15", "DEUCE", "AD-IN"
    server: int = 1          # 1 = p1 serving, 2 = p2 serving
    is_tiebreak: bool = False
    set_scores: List[Tuple[int, int]] = field(default_factory=list)
    is_live: bool = False
    is_finished: bool = False
    timestamp: float = 0.0

    def score_key(self) -> str:
        """Unique key for de-duplication."""
        return f"{self.sets_p1}-{self.sets_p2}|{self.games_p1}-{self.games_p2}|{self.points_p1}-{self.points_p2}"


@dataclass
class MatchInfo:
    """Metadata about a live match."""
    p1_name: str = ""
    p2_name: str = ""
    tournament: str = ""
    round: str = ""
    surface: str = "Hard"
    tour: str = "ATP"
    source: DataSource = DataSource.ESPN
    source_match_id: str = ""   # ID in the source system
    betfair_market_id: str = ""  # Betfair market ID if found


@dataclass
class ScoreChange:
    """Emitted when a score change is detected."""
    point_winner: str  # "SERVER" or "RECEIVER"
    new_score: MatchScore
    previous_score: MatchScore
    server: int
    is_break: bool = False
    source: DataSource = DataSource.ESPN
    timestamp: float = 0.0

    # Optional stats (from api-tennis point-by-point)
    is_ace: bool = False
    is_double_fault: bool = False
    is_first_serve_in: bool = True
    is_break_point: bool = False


@dataclass
class LiveMatch:
    """A match being tracked by the feed."""
    info: MatchInfo
    current_score: MatchScore = field(default_factory=MatchScore)
    last_poll_time: float = 0.0
    poll_count: int = 0
    changes_detected: int = 0


# ─── Score Diffing Logic ─────────────────────────────────────────────────────

POINT_MAP = {0: "0", 1: "15", 2: "30", 3: "40"}


def infer_point_winner(old: MatchScore, new: MatchScore) -> Optional[str]:
    """
    Given two consecutive score snapshots, infer who won the point(s).
    Returns "SERVER" or "RECEIVER" or None if can't determine.
    """
    server = new.server if new.server else old.server

    # Case 1: Point-level change (within same game)
    if (old.sets_p1 == new.sets_p1 and old.sets_p2 == new.sets_p2 and
        old.games_p1 == new.games_p1 and old.games_p2 == new.games_p2):

        # Direct point comparison
        if new.points_p1 > old.points_p1 and new.points_p2 <= old.points_p2:
            return "SERVER" if server == 1 else "RECEIVER"
        if new.points_p2 > old.points_p2 and new.points_p1 <= old.points_p1:
            return "RECEIVER" if server == 1 else "SERVER"

        # Deuce: AD lost → back to deuce
        if old.points_p1 > old.points_p2 and new.points_p1 == new.points_p2:
            return "RECEIVER" if server == 1 else "SERVER"
        if old.points_p2 > old.points_p1 and new.points_p1 == new.points_p2:
            return "SERVER" if server == 1 else "RECEIVER"

    # Case 2: Game changed (points reset to 0-0)
    if (new.points_p1 == 0 and new.points_p2 == 0 and
        old.sets_p1 == new.sets_p1 and old.sets_p2 == new.sets_p2):

        if new.games_p1 > old.games_p1:
            # P1 won the game
            return "SERVER" if server == 1 else "RECEIVER"
        if new.games_p2 > old.games_p2:
            return "RECEIVER" if server == 1 else "SERVER"

    # Case 3: Set changed
    if new.sets_p1 > old.sets_p1:
        return "SERVER" if server == 1 else "RECEIVER"
    if new.sets_p2 > old.sets_p2:
        return "RECEIVER" if server == 1 else "SERVER"

    return None


def infer_game_level_points(old: MatchScore, new: MatchScore, server: int) -> List[ScoreChange]:
    """
    When we only have game-level data (ESPN), and a game changed,
    infer the minimum points that must have happened.
    e.g., game went from 3-2 to 4-2 → server held → at least 4 points won by server.
    We emit a single ScoreChange for the game-deciding point.
    """
    changes = []

    if new.games_p1 > old.games_p1 or new.games_p2 > old.games_p2 or \
       new.sets_p1 > old.sets_p1 or new.sets_p2 > old.sets_p2:

        # Determine who won
        if new.games_p1 > old.games_p1 or new.sets_p1 > old.sets_p1:
            winner = "SERVER" if server == 1 else "RECEIVER"
            is_break = server != 1  # p1 won but p2 was serving
        else:
            winner = "RECEIVER" if server == 1 else "SERVER"
            is_break = server != 2

        changes.append(ScoreChange(
            point_winner=winner,
            new_score=new,
            previous_score=old,
            server=server,
            is_break=is_break,
            timestamp=time.time(),
        ))

    return changes


# ─── Data Source Adapters ────────────────────────────────────────────────────

class SofascoreAdapter:
    """Fetches live tennis data from Sofascore's public API (free).
    Covers ATP, WTA, ITF, and Challenger tours."""

    BASE = "https://api.sofascore.com/api/v1"
    source = DataSource.SOFASCORE

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
            "Accept": "application/json",
        })

    def get_live_matches(self) -> List[Dict]:
        """Get all live tennis matches."""
        try:
            resp = self.session.get(
                f"{self.BASE}/sport/tennis/events/live",
                timeout=10,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            events = data.get("events", [])
            matches = []
            for ev in events:
                match = self._parse_event(ev)
                if match:
                    matches.append(match)
            return matches
        except Exception as e:
            logger.debug("Sofascore live fetch failed: %s", e)
            return []

    def get_match_score(self, event_id: str) -> Optional[MatchScore]:
        """Get current score for a specific match."""
        try:
            resp = self.session.get(
                f"{self.BASE}/event/{event_id}",
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            ev = data.get("event", data)
            return self._parse_score(ev)
        except Exception as e:
            logger.debug("Sofascore match fetch failed: %s", e)
            return None

    def _parse_event(self, ev: Dict) -> Optional[Dict]:
        """Parse a Sofascore event into a normalized dict."""
        home = ev.get("homeTeam", {})
        away = ev.get("awayTeam", {})
        status = ev.get("status", {})
        tournament = ev.get("tournament", {})

        p1_name = home.get("name", "")
        p2_name = away.get("name", "")
        if not p1_name or not p2_name:
            return None

        status_code = status.get("code", 0)
        is_live = status_code in (6, 7, 8, 9, 10)  # covers all in-play set states
        is_finished = status_code == 100

        score = self._parse_score(ev)

        # Detect tour from tournament category
        tour = self._detect_tour(tournament)

        return {
            "source_id": str(ev.get("id", "")),
            "p1_name": p1_name,
            "p2_name": p2_name,
            "tournament": tournament.get("name", ""),
            "round": ev.get("roundInfo", {}).get("name", ""),
            "tour": tour,
            "surface": self._detect_surface(tournament),
            "is_live": is_live,
            "is_finished": is_finished,
            "score": score,
        }

    @staticmethod
    def _detect_tour(tournament: Dict) -> str:
        """Detect ATP/WTA/ITF/Challenger from Sofascore tournament metadata."""
        cat = tournament.get("category", {})
        cat_name = cat.get("name", "").lower()
        cat_slug = cat.get("slug", "").lower()
        t_name = tournament.get("name", "").lower()
        unique_t = tournament.get("uniqueTournament", {})
        ut_name = unique_t.get("name", "").lower() if unique_t else ""

        all_text = f"{cat_name} {cat_slug} {t_name} {ut_name}"

        if "wta" in all_text:
            return "WTA"
        if "itf" in all_text:
            if "women" in all_text or "w15" in all_text or "w25" in all_text or "w35" in all_text or "w40" in all_text or "w50" in all_text or "w60" in all_text or "w75" in all_text or "w100" in all_text:
                return "ITF-W"
            return "ITF-M"
        if "challenger" in all_text:
            return "Challenger"
        if "atp" in all_text:
            return "ATP"
        # Default heuristic: category name
        if "women" in all_text:
            return "WTA"
        return "ATP"

    @staticmethod
    def _detect_surface(tournament: Dict) -> str:
        """Infer surface from tournament name."""
        t_name = tournament.get("name", "").lower()
        ut = tournament.get("uniqueTournament", {})
        ut_name = ut.get("name", "").lower() if ut else ""
        combined = f"{t_name} {ut_name}"
        if any(k in combined for k in ["roland garros", "rome", "madrid", "clay",
                                        "monte carlo", "barcelona", "hamburg",
                                        "buenos aires", "rio", "marrakech",
                                        "kitzbuhel", "umag", "bastad", "gstaad"]):
            return "Clay"
        if any(k in combined for k in ["wimbledon", "queens", "halle", "grass",
                                        "eastbourne", "stuttgart", "s-hertogenbosch",
                                        "mallorca", "newport"]):
            return "Grass"
        return "Hard"

    def _parse_score(self, ev: Dict) -> MatchScore:
        """Parse score from a Sofascore event."""
        home_score = ev.get("homeScore", {})
        away_score = ev.get("awayScore", {})

        # Period scores (sets)
        set_scores = []
        for i in range(1, 6):
            p_key = f"period{i}"
            if p_key in home_score and p_key in away_score:
                try:
                    set_scores.append((
                        int(home_score[p_key]),
                        int(away_score[p_key]),
                    ))
                except (ValueError, TypeError):
                    pass

        # Count completed sets
        sets_p1 = sum(1 for s1, s2 in set_scores[:-1]
                      if (s1 > s2 and (s1 >= 6 or s1 == 7)))
        sets_p2 = sum(1 for s1, s2 in set_scores[:-1]
                      if (s2 > s1 and (s2 >= 6 or s2 == 7)))

        # Current games
        games_p1 = set_scores[-1][0] if set_scores else 0
        games_p2 = set_scores[-1][1] if set_scores else 0

        # Point score
        point_p1 = home_score.get("point", "0")
        point_p2 = away_score.get("point", "0")

        point_map = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4}
        points_p1 = point_map.get(str(point_p1), 0)
        points_p2 = point_map.get(str(point_p2), 0)

        # Server
        home_serving = ev.get("homeTeamServing", None)
        server = 1 if home_serving else 2

        is_tiebreak = games_p1 == 6 and games_p2 == 6

        status_code = ev.get("status", {}).get("code", 0)

        return MatchScore(
            sets_p1=sets_p1,
            sets_p2=sets_p2,
            games_p1=games_p1,
            games_p2=games_p2,
            points_p1=points_p1,
            points_p2=points_p2,
            point_text=f"{point_p1}-{point_p2}",
            server=server,
            is_tiebreak=is_tiebreak,
            set_scores=set_scores,
            is_live=status_code in (6, 7, 8, 9, 10),
            timestamp=time.time(),
        )


class ApiTennisAdapter:
    """Fetches live data from api-tennis.com (paid, point-level).
    Covers all tours: ATP, WTA, ITF, Challenger."""

    BASE_URL = "https://api.api-tennis.com/tennis/"
    source = DataSource.API_TENNIS

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("API_TENNIS_KEY", "")
        self.session = requests.Session()

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 5)

    @staticmethod
    def detect_tour(match: Dict) -> str:
        """Detect tour from api-tennis tournament name."""
        t_name = (match.get("tournament_name", "") + " " +
                  match.get("league_name", "")).lower()
        if "wta" in t_name:
            return "WTA"
        if "itf" in t_name:
            if "women" in t_name or "w15" in t_name or "w25" in t_name or "w35" in t_name or "w50" in t_name or "w75" in t_name or "w100" in t_name:
                return "ITF-W"
            return "ITF-M"
        if "challenger" in t_name:
            return "Challenger"
        if "atp" in t_name:
            return "ATP"
        return "ATP"

    def get_live_matches(self) -> List[Dict]:
        if not self.is_available:
            return []
        try:
            resp = self.session.get(self.BASE_URL, params={
                "method": "get_livescore",
                "APIkey": self.api_key,
            }, timeout=15)
            data = resp.json()
            if data.get("success") != 1:
                return []
            return data.get("result", [])
        except Exception as e:
            logger.debug("api-tennis live fetch failed: %s", e)
            return []

    def get_match_detail(self, event_key: str) -> Optional[Dict]:
        if not self.is_available:
            return None
        try:
            resp = self.session.get(self.BASE_URL, params={
                "method": "get_livescore",
                "APIkey": self.api_key,
                "match_key": event_key,
            }, timeout=15)
            data = resp.json()
            if data.get("success") != 1:
                return None
            results = data.get("result", [])
            return results[0] if results else None
        except Exception as e:
            logger.debug("api-tennis detail fetch failed: %s", e)
            return None

    def parse_score(self, match: Dict) -> MatchScore:
        """Parse api-tennis match into MatchScore."""
        scores = match.get("scores", [])
        set_scores = []
        for s in scores:
            set_scores.append((
                int(s.get("score_first", 0)),
                int(s.get("score_second", 0)),
            ))

        sets_p1 = sum(1 for s1, s2 in set_scores[:-1] if s1 > s2 and s1 >= 6)
        sets_p2 = sum(1 for s1, s2 in set_scores[:-1] if s2 > s1 and s2 >= 6)

        games_p1 = set_scores[-1][0] if set_scores else 0
        games_p2 = set_scores[-1][1] if set_scores else 0

        # Point score from event_game_result
        game_result = match.get("event_game_result", "0 - 0")
        point_map = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4}
        points_p1, points_p2 = 0, 0
        if game_result and " - " in game_result:
            parts = game_result.split(" - ")
            points_p1 = point_map.get(parts[0].strip(), 0)
            points_p2 = point_map.get(parts[1].strip(), 0)

        server = 1 if match.get("event_serve") == "First Player" else 2
        is_tiebreak = games_p1 == 6 and games_p2 == 6

        return MatchScore(
            sets_p1=sets_p1,
            sets_p2=sets_p2,
            games_p1=games_p1,
            games_p2=games_p2,
            points_p1=points_p1,
            points_p2=points_p2,
            point_text=game_result,
            server=server,
            is_tiebreak=is_tiebreak,
            set_scores=set_scores,
            is_live=match.get("event_live", "0") == "1",
            timestamp=time.time(),
        )


class ESPNAdapter:
    """Fetches live data from ESPN (free, game-level only).
    Covers ATP and WTA tours."""

    BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"
    source = DataSource.ESPN

    # ESPN tour slugs: atp, wta. ESPN doesn't have ITF/Challenger
    # but those are covered by Sofascore and api-tennis.
    TOUR_SLUGS = ["atp", "wta"]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
        })

    def get_live_matches(self) -> List[Dict]:
        all_matches = []
        for tour in self.TOUR_SLUGS:
            try:
                resp = self.session.get(
                    f"{self.BASE}/{tour}/scoreboard",
                    timeout=12,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for ev in data.get("events", []):
                    # Some events have groupings, some have competitions directly
                    competitions = []
                    for g in ev.get("groupings", []):
                        competitions.extend(g.get("competitions", []))
                    if not competitions:
                        competitions = ev.get("competitions", [])
                    for c in competitions:
                        parsed = self._parse_competition(c, ev, tour)
                        if parsed and parsed.get("is_live"):
                            all_matches.append(parsed)
            except Exception as e:
                logger.debug("ESPN fetch failed for %s: %s", tour, e)
        return all_matches

    def _parse_competition(self, comp: Dict, event: Dict, tour: str) -> Optional[Dict]:
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            return None

        p1_data = competitors[0]
        p2_data = competitors[1]
        for p in competitors:
            if p.get("order") == 1:
                p1_data = p
            elif p.get("order") == 2:
                p2_data = p

        p1_ath = p1_data.get("athlete", {})
        p2_ath = p2_data.get("athlete", {})
        if isinstance(p1_ath, str):
            p1_ath = {"displayName": p1_ath}
        if isinstance(p2_ath, str):
            p2_ath = {"displayName": p2_ath}

        p1_name = p1_ath.get("displayName", "")
        p2_name = p2_ath.get("displayName", "")
        if not p1_name or not p2_name:
            return None

        status = comp.get("status", {}).get("type", {})
        is_live = status.get("state") == "in"
        is_finished = status.get("state") == "post"

        p1_linescores = p1_data.get("linescores", [])
        p2_linescores = p2_data.get("linescores", [])
        set_scores = []
        for i in range(max(len(p1_linescores), len(p2_linescores))):
            s1 = int(p1_linescores[i].get("value", 0)) if i < len(p1_linescores) else 0
            s2 = int(p2_linescores[i].get("value", 0)) if i < len(p2_linescores) else 0
            set_scores.append((s1, s2))

        sets_p1 = sum(1 for i, (s1, s2) in enumerate(set_scores[:-1])
                      if s1 > s2)
        sets_p2 = sum(1 for i, (s1, s2) in enumerate(set_scores[:-1])
                      if s2 > s1)

        games_p1 = set_scores[-1][0] if set_scores else 0
        games_p2 = set_scores[-1][1] if set_scores else 0

        return {
            "source_id": f"{event.get('id')}_{comp.get('id')}",
            "p1_name": p1_name,
            "p2_name": p2_name,
            "tournament": event.get("name", ""),
            "surface": self._detect_surface(event),
            "tour": tour.upper(),
            "is_live": is_live,
            "is_finished": is_finished,
            "score": MatchScore(
                sets_p1=sets_p1, sets_p2=sets_p2,
                games_p1=games_p1, games_p2=games_p2,
                points_p1=0, points_p2=0,
                point_text="",
                server=0,  # ESPN doesn't provide server
                set_scores=set_scores,
                is_live=is_live,
                is_finished=is_finished,
                timestamp=time.time(),
            ),
        }

    def _detect_surface(self, event: Dict) -> str:
        name = (event.get("name", "") + " " +
                event.get("venue", {}).get("fullName", "")).lower()
        if any(k in name for k in ["roland garros", "rome", "madrid", "clay",
                                    "monte carlo", "barcelona", "hamburg"]):
            return "Clay"
        if any(k in name for k in ["wimbledon", "queens", "halle", "grass",
                                    "eastbourne"]):
            return "Grass"
        return "Hard"


# ─── Main Feed ────────────────────────────────────────────────────────────────

class LiveMatchFeed:
    """
    Unified live match data feed.

    Polls the best available source, detects score changes, and emits
    ScoreChange events that the trading loop consumes.

    Sources are tried in priority order:
      1. api-tennis (paid) — has point-level + server
      2. Sofascore (free) — has point-level + server
      3. ESPN (free) — game-level only

    Usage:
        feed = LiveMatchFeed()

        # List all live matches
        matches = feed.get_live_matches()

        # Attach to a specific match
        feed.attach(player1="Sinner", player2="Alcaraz")

        # Poll for changes
        changes = feed.poll()
        for c in changes:
            print(f"{c.point_winner} won point → {c.new_score.point_text}")
    """

    def __init__(self, api_tennis_key: str = ""):
        self.api_tennis = ApiTennisAdapter(api_key=api_tennis_key)
        self.sofascore = SofascoreAdapter()
        self.espn = ESPNAdapter()
        self.preferred_source: Optional[DataSource] = None

        self._tracked: Optional[LiveMatch] = None
        self._previous_score: Optional[MatchScore] = None
        self._callbacks: List[Callable[[ScoreChange], None]] = []

    @property
    def adapters(self):
        """All available adapters (for status reporting)."""
        adapters = []
        if self.api_tennis.is_available:
            adapters.append(self.api_tennis)
        adapters.append(self.sofascore)
        adapters.append(self.espn)
        return adapters

    @property
    def active_source(self) -> Optional[DataSource]:
        if self._tracked:
            return self._tracked.info.source
        return None

    @property
    def is_attached(self) -> bool:
        return self._tracked is not None

    @property
    def current_match(self) -> Optional[LiveMatch]:
        return self._tracked

    # ─── Discovery ────────────────────────────────────────────────────────

    def get_live_matches(self) -> List[Dict]:
        """
        Get all live tennis matches from all available sources.
        Returns normalized dicts with source info.
        """
        all_matches = []
        seen = set()  # de-dup by player names

        # 1. api-tennis (best data — all tours incl. ITF/Challenger)
        if self.api_tennis.is_available:
            for m in self.api_tennis.get_live_matches():
                p1 = m.get("event_first_player", "")
                p2 = m.get("event_second_player", "")
                key = self._name_key(p1, p2)
                if key and key not in seen:
                    seen.add(key)
                    score = self.api_tennis.parse_score(m)
                    tour = self.api_tennis.detect_tour(m)
                    all_matches.append({
                        "p1_name": p1,
                        "p2_name": p2,
                        "tournament": m.get("tournament_name", ""),
                        "round": m.get("tournament_round", ""),
                        "source": DataSource.API_TENNIS,
                        "source_id": m.get("event_key", ""),
                        "score": score,
                        "surface": "Hard",
                        "tour": tour,
                    })

        # 2. Sofascore
        for m in self.sofascore.get_live_matches():
            key = self._name_key(m["p1_name"], m["p2_name"])
            if key and key not in seen:
                seen.add(key)
                m["source"] = DataSource.SOFASCORE
                all_matches.append(m)

        # 3. ESPN
        for m in self.espn.get_live_matches():
            key = self._name_key(m["p1_name"], m["p2_name"])
            if key and key not in seen:
                seen.add(key)
                m["source"] = DataSource.ESPN
                all_matches.append(m)

        return all_matches

    def attach(
        self,
        player1: str = "",
        player2: str = "",
        source_id: str = "",
    ) -> Optional[LiveMatch]:
        """
        Attach to a live match by player names or source ID.
        Searches all sources and picks the best one with point-level data.
        """
        matches = self.get_live_matches()

        if source_id:
            for m in matches:
                if m.get("source_id") == source_id:
                    return self._start_tracking(m)

        if player1 or player2:
            p1_lower = player1.lower().split()[-1] if player1 else ""
            p2_lower = player2.lower().split()[-1] if player2 else ""

            # Score matches by how well they fit
            best = None
            best_priority = -1

            for m in matches:
                mp1 = m.get("p1_name", "").lower()
                mp2 = m.get("p2_name", "").lower()

                name_match = False
                if p1_lower and p2_lower:
                    name_match = (p1_lower in mp1 and p2_lower in mp2) or \
                                 (p1_lower in mp2 and p2_lower in mp1)
                elif p1_lower:
                    name_match = p1_lower in mp1 or p1_lower in mp2
                elif p2_lower:
                    name_match = p2_lower in mp1 or p2_lower in mp2

                if not name_match:
                    continue

                # Priority by source quality
                priority = {
                    DataSource.API_TENNIS: 3,
                    DataSource.SOFASCORE: 2,
                    DataSource.ESPN: 1,
                }.get(m.get("source", DataSource.ESPN), 0)

                if priority > best_priority:
                    best = m
                    best_priority = priority

            if best:
                return self._start_tracking(best)

        logger.warning("No live match found for %s vs %s", player1, player2)
        return None

    def detach(self) -> None:
        """Stop tracking the current match."""
        self._tracked = None
        self._previous_score = None

    def on_change(self, callback: Callable[[ScoreChange], None]) -> None:
        """Register a callback for score changes."""
        self._callbacks.append(callback)

    # ─── Polling ──────────────────────────────────────────────────────────

    def poll(self) -> List[ScoreChange]:
        """
        Poll the data source for the tracked match.
        Returns a list of ScoreChange events (usually 0 or 1).
        """
        if not self._tracked:
            return []

        new_score = self._fetch_current_score()
        if not new_score:
            return []

        self._tracked.last_poll_time = time.time()
        self._tracked.poll_count += 1

        changes = self._diff_scores(self._previous_score, new_score)

        if changes:
            self._tracked.current_score = new_score
            self._previous_score = new_score
            self._tracked.changes_detected += len(changes)

            for c in changes:
                for cb in self._callbacks:
                    try:
                        cb(c)
                    except Exception as e:
                        logger.error("Score change callback error: %s", e)

        return changes

    # ─── Internal ─────────────────────────────────────────────────────────

    def _start_tracking(self, match_data: Dict) -> LiveMatch:
        """Begin tracking a match."""
        score = match_data.get("score", MatchScore())
        info = MatchInfo(
            p1_name=match_data.get("p1_name", ""),
            p2_name=match_data.get("p2_name", ""),
            tournament=match_data.get("tournament", ""),
            round=match_data.get("round", ""),
            surface=match_data.get("surface", "Hard"),
            tour=match_data.get("tour", "ATP"),
            source=match_data.get("source", DataSource.ESPN),
            source_match_id=match_data.get("source_id", ""),
        )
        self._tracked = LiveMatch(info=info, current_score=score)
        self._previous_score = score
        logger.info(
            "📡 Attached to: %s vs %s (%s via %s) | Score: %s-%s %s-%s %s",
            info.p1_name, info.p2_name, info.tournament, info.source.value,
            score.sets_p1, score.sets_p2, score.games_p1, score.games_p2,
            score.point_text,
        )
        return self._tracked

    def _fetch_current_score(self) -> Optional[MatchScore]:
        """Fetch latest score from the appropriate source."""
        if not self._tracked:
            return None

        source = self._tracked.info.source
        source_id = self._tracked.info.source_match_id

        if source == DataSource.API_TENNIS and source_id:
            detail = self.api_tennis.get_match_detail(source_id)
            if detail:
                return self.api_tennis.parse_score(detail)

        if source == DataSource.SOFASCORE and source_id:
            return self.sofascore.get_match_score(source_id)

        if source == DataSource.ESPN:
            # Re-fetch all live and find our match
            matches = self.espn.get_live_matches()
            p1 = self._tracked.info.p1_name.lower().split()[-1]
            for m in matches:
                if p1 in m.get("p1_name", "").lower() or p1 in m.get("p2_name", "").lower():
                    return m.get("score")

        return None

    def _diff_scores(
        self, old: Optional[MatchScore], new: MatchScore
    ) -> List[ScoreChange]:
        """Compare two scores and produce change events."""
        if not old:
            return []

        if old.score_key() == new.score_key():
            return []

        source = self._tracked.info.source if self._tracked else DataSource.ESPN

        # If we have point-level data (api-tennis or sofascore)
        if source in (DataSource.API_TENNIS, DataSource.SOFASCORE):
            winner = infer_point_winner(old, new)
            if winner:
                server = new.server if new.server else old.server
                # Is this a break point situation?
                is_break = False
                if winner == "RECEIVER":
                    # Receiver won while the other was serving
                    is_break = True

                return [ScoreChange(
                    point_winner=winner,
                    new_score=new,
                    previous_score=old,
                    server=server,
                    is_break=is_break,
                    source=source,
                    timestamp=time.time(),
                )]

        # Game-level only (ESPN) — infer from game/set changes
        server = new.server if new.server else (old.server if old.server else 1)
        return infer_game_level_points(old, new, server)

    @staticmethod
    def _name_key(p1: str, p2: str) -> str:
        """Create a de-dup key from player names."""
        names = sorted([p1.lower().strip(), p2.lower().strip()])
        return "|".join(names) if all(names) else ""
