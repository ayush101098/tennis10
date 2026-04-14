"""
Free Live Tennis Data Service (ESPN)
=====================================
100% FREE — no API key, no signup, no payment.
Uses ESPN's public JSON endpoints for live scores, schedules, rankings, and player data.

Endpoints used:
  - Scoreboard: live/scheduled matches for ATP and WTA
  - Rankings: ATP/WTA singles (Top 150)
  - Athlete profiles: player details via ID

All data is cached to be polite to ESPN's servers.
"""

import requests
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple


class FreeLiveTennisService:
    """
    100% free live tennis data — no API key, no signup, no payment.
    Uses ESPN public JSON endpoints.
    """

    BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"
    ATHLETE_BASE = "https://site.api.espn.com/apis/common/v3/sports/tennis"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        })
        # In-memory cache: key -> (timestamp, data)
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = {
            "live": 20,        # Refresh live data every 20s
            "schedule": 300,   # Schedule every 5 min
            "rankings": 3600,  # Rankings every hour
            "player": 3600,    # Player profiles every hour
        }

    # ------------------------------------------------------------------
    # Cache helper
    # ------------------------------------------------------------------
    def _cached_get(self, url: str, category: str,
                    params: Optional[Dict] = None) -> Optional[Dict]:
        """GET with caching. Returns parsed JSON or None."""
        cache_key = url + (json.dumps(params, sort_keys=True) if params else "")
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            ttl = self._cache_ttl.get(category, 120)
            if time.time() - ts < ttl:
                return data

        try:
            resp = self.session.get(url, params=params, timeout=12)
            if resp.status_code != 200:
                return None
            data = resp.json()
            self._cache[cache_key] = (time.time(), data)
            return data
        except Exception:
            return None

    # ==================================================================
    #  LIVE MATCHES
    # ==================================================================
    def get_live_matches(self) -> List[Dict]:
        """
        Get all currently live tennis matches across ATP and WTA.
        Returns a list of normalized match dicts.
        """
        all_matches = []
        for tour in ["atp", "wta"]:
            data = self._cached_get(f"{self.BASE}/{tour}/scoreboard", "live")
            if data:
                matches = self._extract_matches(data, tour)
                # Filter to only in-progress matches
                for m in matches:
                    if m["is_live"]:
                        all_matches.append(m)
        return all_matches

    # ==================================================================
    #  TODAY'S SCHEDULE
    # ==================================================================
    def get_todays_matches(self) -> List[Dict]:
        """Get all tennis matches for today (ATP + WTA)."""
        today = datetime.now().strftime("%Y%m%d")
        return self.get_schedule_for_date(today)

    def get_schedule_for_date(self, date_str: str) -> List[Dict]:
        """
        Get all tennis matches for a date.
        date_str: YYYYMMDD format
        """
        all_matches = []
        for tour in ["atp", "wta"]:
            data = self._cached_get(
                f"{self.BASE}/{tour}/scoreboard",
                "schedule",
                params={"dates": date_str},
            )
            if data:
                matches = self._extract_matches(data, tour)
                all_matches.extend(matches)
        return all_matches

    # ==================================================================
    #  RANKINGS
    # ==================================================================
    def get_rankings(self, tour: str = "ATP") -> List[Dict]:
        """
        Get ATP or WTA singles rankings (Top 150).
        tour: 'ATP' or 'WTA'
        """
        tour_path = tour.lower()
        data = self._cached_get(f"{self.BASE}/{tour_path}/rankings", "rankings")
        if not data:
            return []

        rankings = data.get("rankings", [])
        if not rankings:
            return []

        rk = rankings[0]
        ranks = rk.get("ranks", [])

        result = []
        for entry in ranks:
            ath = entry.get("athlete", {})
            if isinstance(ath, str):
                ath = {"displayName": ath, "id": ""}

            result.append({
                "rank": entry.get("current", 0),
                "previous_rank": entry.get("previous", 0),
                "player": ath.get("displayName", "Unknown"),
                "player_id": ath.get("id", ""),
                "country": self._get_flag(ath),
                "points": entry.get("points", 0),
                "trend": entry.get("trend", ""),
            })
        return result

    # ==================================================================
    #  PLAYER SEARCH (from rankings)
    # ==================================================================
    def search_player(self, name: str) -> List[Dict]:
        """Search for a player by name across ATP and WTA rankings."""
        name_lower = name.lower()
        results = []
        for tour in ["ATP", "WTA"]:
            rankings = self.get_rankings(tour)
            for r in rankings:
                if name_lower in r["player"].lower():
                    results.append({
                        "id": r["player_id"],
                        "name": r["player"],
                        "rank": r["rank"],
                        "points": r["points"],
                        "country": r["country"],
                        "tour": tour,
                    })
        return results

    # ==================================================================
    #  PLAYER PROFILE
    # ==================================================================
    def get_player_profile(self, player_id: str,
                           tour: str = "atp") -> Optional[Dict]:
        """Get player profile by ESPN athlete ID."""
        url = f"{self.ATHLETE_BASE}/{tour}/athletes/{player_id}"
        data = self._cached_get(url, "player")
        if not data:
            return None

        ath = data.get("athlete", data)
        return {
            "id": ath.get("id"),
            "name": ath.get("displayName", "Unknown"),
            "first_name": ath.get("firstName", ""),
            "last_name": ath.get("lastName", ""),
            "country": self._get_flag(ath),
            "height": ath.get("height", ""),
            "weight": ath.get("weight", ""),
            "age": ath.get("age", ""),
            "birthdate": ath.get("dateOfBirth", ""),
            "hand": ath.get("hand", ""),
        }

    # ==================================================================
    #  MATCH DETAIL
    # ==================================================================
    def get_match_detail(self, event_id: str, comp_id: str,
                         tour: str = "atp") -> Optional[Dict]:
        """
        Get details of a specific match.
        event_id: ESPN event ID (tournament)
        comp_id: competition ID (specific match)
        tour: 'atp' or 'wta'
        """
        data = self._cached_get(f"{self.BASE}/{tour}/scoreboard", "live")
        if not data:
            return None

        # Find the specific competition
        for ev in data.get("events", []):
            if str(ev.get("id")) == str(event_id):
                for g in ev.get("groupings", []):
                    for c in g.get("competitions", []):
                        if str(c.get("id")) == str(comp_id):
                            return self._parse_competition(c, ev, tour)
        return None

    # ==================================================================
    #  HIGH-LEVEL: ONE-CALL DATA FOR CALCULATOR
    # ==================================================================
    def get_calculator_ready_data(self, match: Dict) -> Dict:
        """
        Prepare a match dict with everything the calculator needs.
        Pass in a match from get_live_matches() or get_todays_matches().
        """
        result = dict(match)
        result.setdefault("serve_pct_p1", 0.65)
        result.setdefault("serve_pct_p2", 0.65)
        result.setdefault("bp_save_p1", 0.60)
        result.setdefault("bp_save_p2", 0.60)
        return result

    # ==================================================================
    #  INTERNAL PARSERS
    # ==================================================================
    def _extract_matches(self, data: Dict, tour: str) -> List[Dict]:
        """Extract all matches from an ESPN scoreboard response."""
        matches = []
        for ev in data.get("events", []):
            for g in ev.get("groupings", []):
                for c in g.get("competitions", []):
                    parsed = self._parse_competition(c, ev, tour)
                    if parsed:
                        matches.append(parsed)
        return matches

    def _parse_competition(self, comp: Dict, event: Dict,
                           tour: str) -> Optional[Dict]:
        """Parse a single ESPN competition into our normalized format."""
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            return None

        # Find home (order 1) and away (order 2) properly
        p1_data = competitors[0]
        p2_data = competitors[1]
        for p in competitors:
            if p.get("homeAway") == "home" or p.get("order") == 1:
                p1_data = p
            elif p.get("homeAway") == "away" or p.get("order") == 2:
                p2_data = p

        p1_ath = p1_data.get("athlete", {})
        p2_ath = p2_data.get("athlete", {})

        if isinstance(p1_ath, str):
            p1_ath = {"displayName": p1_ath}
        if isinstance(p2_ath, str):
            p2_ath = {"displayName": p2_ath}

        p1_name = p1_ath.get("displayName", "Player 1")
        p2_name = p2_ath.get("displayName", "Player 2")

        # Skip empty/TBD matches
        if not p1_name or not p2_name:
            return None
        if p1_name in ("?", "TBD") and p2_name in ("?", "TBD"):
            return None

        # Status
        status = comp.get("status", {})
        status_type = status.get("type", {})
        status_state = status_type.get("state", "")
        status_desc = status_type.get("description", "Scheduled")

        is_live = status_state == "in"
        is_finished = status_state == "post"

        # Set scores from linescores
        p1_linescores = p1_data.get("linescores", [])
        p2_linescores = p2_data.get("linescores", [])

        set_scores = []
        for i in range(max(len(p1_linescores), len(p2_linescores))):
            s1 = int(p1_linescores[i].get("value", 0)) if i < len(p1_linescores) else 0
            s2 = int(p2_linescores[i].get("value", 0)) if i < len(p2_linescores) else 0
            set_scores.append((s1, s2))

        # Count sets won
        sets_p1, sets_p2 = 0, 0
        for idx, (s1, s2) in enumerate(set_scores):
            completed = is_finished or idx < len(set_scores) - 1
            if completed:
                if s1 > s2:
                    sets_p1 += 1
                elif s2 > s1:
                    sets_p2 += 1
            elif is_live:
                # Current set might still be in progress
                if (s1 >= 6 and s1 - s2 >= 2) or s1 == 7:
                    sets_p1 += 1
                elif (s2 >= 6 and s2 - s1 >= 2) or s2 == 7:
                    sets_p2 += 1

        # Current games in set
        games_p1, games_p2 = 0, 0
        if set_scores and is_live:
            games_p1, games_p2 = set_scores[-1]

        # Winner
        p1_winner = p1_data.get("winner", False)
        p2_winner = p2_data.get("winner", False)

        # Surface
        surface = self._detect_surface(event)

        # Tournament info
        tournament = event.get("name", "Unknown Tournament")
        round_info = comp.get("round", {})
        round_name = ""
        if isinstance(round_info, dict):
            round_name = round_info.get("displayName", round_info.get("name", ""))
        elif isinstance(round_info, str):
            round_name = round_info

        # Score display
        if set_scores:
            set_strs = [f"{s1}-{s2}" for s1, s2 in set_scores]
            score_display = " ".join(set_strs)
        else:
            score_display = "Not started"

        # Seed
        p1_seed = p1_data.get("seed", "")
        p2_seed = p2_data.get("seed", "")

        # Time
        start_time = comp.get("startDate", comp.get("date", ""))
        event_time = ""
        event_date = ""
        if start_time:
            try:
                dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                event_time = dt.strftime("%H:%M")
                event_date = dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        return {
            "event_id": str(event.get("id", "")),
            "comp_id": str(comp.get("id", "")),
            "event_key": f"{event.get('id', '')}_{comp.get('id', '')}",
            "tour": tour.upper(),
            "p1_name": p1_name,
            "p2_name": p2_name,
            "p1_id": str(p1_ath.get("id", "")),
            "p2_id": str(p2_ath.get("id", "")),
            "p1_country": self._get_flag(p1_ath),
            "p2_country": self._get_flag(p2_ath),
            "p1_seed": str(p1_seed) if p1_seed else "",
            "p2_seed": str(p2_seed) if p2_seed else "",
            "tournament": tournament,
            "round": round_name,
            "surface": surface,
            "status": status_desc,
            "status_state": status_state,
            "is_live": is_live,
            "is_finished": is_finished,
            "p1_winner": bool(p1_winner),
            "p2_winner": bool(p2_winner),
            "server": 0,
            "sets_p1": sets_p1,
            "sets_p2": sets_p2,
            "games_p1": games_p1,
            "games_p2": games_p2,
            "points_p1": 0,
            "points_p2": 0,
            "set_scores": set_scores,
            "score_display": score_display,
            "event_time": event_time,
            "event_date": event_date,
            "serve_pct_p1": 0.65,
            "serve_pct_p2": 0.65,
            "bp_save_p1": 0.60,
            "bp_save_p2": 0.60,
            "aces_p1": 0, "aces_p2": 0,
            "dfs_p1": 0, "dfs_p2": 0,
            "winners_p1": 0, "winners_p2": 0,
            "ues_p1": 0, "ues_p2": 0,
        }

    def _detect_surface(self, event: Dict) -> str:
        """Detect court surface from tournament name."""
        name = event.get("name", "").lower()
        venue = event.get("venue", {}).get("fullName", "").lower()
        combined = name + " " + venue

        clay_keywords = ["roland garros", "rome", "madrid open",
                         "monte carlo", "monte-carlo", "barcelona",
                         "rio open", "buenos aires", "clay",
                         "hamburg", "lyon", "umag", "kitzbuhel",
                         "swedish open", "swiss open"]
        grass_keywords = ["wimbledon", "queens", "queen's", "halle",
                          "eastbourne", "grass", "s-hertogenbosch",
                          "mallorca", "newport"]
        hard_keywords = ["australian open", "us open", "indian wells",
                         "miami open", "shanghai", "canada",
                         "cincinnati", "dubai", "doha"]

        for kw in clay_keywords:
            if kw in combined:
                return "Clay"
        for kw in grass_keywords:
            if kw in combined:
                return "Grass"
        for kw in hard_keywords:
            if kw in combined:
                return "Hard"

        return "Hard"

    def _get_flag(self, ath: Dict) -> str:
        """Extract country from athlete data."""
        flag = ath.get("flag", {})
        if isinstance(flag, dict):
            return flag.get("alt", flag.get("text", ""))
        return str(flag) if flag else ""

    # ==================================================================
    #  UTILITY
    # ==================================================================
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()

    def is_configured(self) -> bool:
        """Always True — no API key needed."""
        return True

    def get_status(self) -> str:
        """Return service status info."""
        return "✅ ESPN Free API — No key required"


# ======================================================================
# Global singleton
# ======================================================================
_service: Optional[FreeLiveTennisService] = None


def get_free_service() -> FreeLiveTennisService:
    """Get or create the global FreeLiveTennisService instance."""
    global _service
    if _service is None:
        _service = FreeLiveTennisService()
    return _service
