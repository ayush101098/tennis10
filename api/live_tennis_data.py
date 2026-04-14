"""
Live Tennis Data Service
========================
Unified interface to real-time tennis data via api-tennis.com (v2.9.4)

Provides:
  - Live scores with point-by-point data
  - Today's fixtures & results
  - Player profiles & season stats
  - Head-to-head records
  - ATP/WTA rankings
  - Pre-match odds (bet365, bwin, 1xbet, etc.)
  - Live in-play odds
  
Usage:
    svc = LiveTennisDataService(api_key="YOUR_KEY")
    live = svc.get_live_matches()
    player = svc.get_player_profile(player_key=1905)
"""

import requests
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import lru_cache


class LiveTennisDataService:
    """Real-time tennis data via api-tennis.com"""

    BASE_URL = "https://api.api-tennis.com/tennis/"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("API_TENNIS_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TennisBettingHub/2.0",
            "Accept": "application/json",
        })
        # Simple in-memory cache: key -> (timestamp, data)
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = {
            "livescore": 20,       # 20 s
            "live_odds": 15,       # 15 s
            "fixtures": 300,       # 5 min
            "players": 3600,       # 1 hr
            "standings": 3600,     # 1 hr
            "odds": 120,           # 2 min
            "h2h": 600,            # 10 min
        }

    # ------------------------------------------------------------------
    # Low-level API call
    # ------------------------------------------------------------------
    def _call(self, method: str, extra_params: Optional[Dict] = None,
              cache_category: Optional[str] = None) -> Dict:
        """Make an API call with caching."""
        params = {"method": method, "APIkey": self.api_key}
        if extra_params:
            params.update(extra_params)

        cache_key = json.dumps(params, sort_keys=True)

        # Check cache
        if cache_category and cache_key in self._cache:
            ts, data = self._cache[cache_key]
            ttl = self._cache_ttl.get(cache_category, 60)
            if time.time() - ts < ttl:
                return data

        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            return {"success": 0, "error": str(e)}
        except json.JSONDecodeError:
            return {"success": 0, "error": "Invalid JSON response"}

        if cache_category:
            self._cache[cache_key] = (time.time(), data)
        return data

    def is_configured(self) -> bool:
        """Check whether an API key has been set."""
        return bool(self.api_key and len(self.api_key) > 5)

    # ------------------------------------------------------------------
    # LIVE SCORES
    # ------------------------------------------------------------------
    def get_live_matches(self, event_type_key: Optional[str] = None,
                         tournament_key: Optional[str] = None) -> List[Dict]:
        """
        Get all currently live matches.
        Returns list of match dicts with scores, point-by-point, serve info.
        """
        params: Dict[str, str] = {}
        if event_type_key:
            params["event_type_key"] = event_type_key
        if tournament_key:
            params["tournament_key"] = tournament_key

        data = self._call("get_livescore", params, cache_category="livescore")
        if data.get("success") != 1:
            return []
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    def get_live_match_detail(self, match_key: str) -> Optional[Dict]:
        """Get detailed live data for a single match."""
        data = self._call("get_livescore", {"match_key": match_key},
                          cache_category="livescore")
        if data.get("success") != 1:
            return None
        result = data.get("result", [])
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return None

    # ------------------------------------------------------------------
    # FIXTURES (today / date range)
    # ------------------------------------------------------------------
    def get_todays_matches(self, event_type_key: Optional[str] = None) -> List[Dict]:
        """Get all matches scheduled for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_fixtures(today, today, event_type_key=event_type_key)

    def get_fixtures(self, date_start: str, date_stop: str,
                     event_type_key: Optional[str] = None,
                     tournament_key: Optional[str] = None) -> List[Dict]:
        """Get fixtures for a date range (yyyy-mm-dd)."""
        params: Dict[str, str] = {
            "date_start": date_start,
            "date_stop": date_stop,
        }
        if event_type_key:
            params["event_type_key"] = event_type_key
        if tournament_key:
            params["tournament_key"] = tournament_key

        data = self._call("get_fixtures", params, cache_category="fixtures")
        if data.get("success") != 1:
            return []
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # PLAYER PROFILES
    # ------------------------------------------------------------------
    def get_player_profile(self, player_key: int) -> Optional[Dict]:
        """Get player profile with season-by-season stats."""
        data = self._call("get_players", {"player_key": str(player_key)},
                          cache_category="players")
        if data.get("success") != 1:
            return None
        result = data.get("result", [])
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return None

    def search_player_by_tournament(self, tournament_key: str) -> List[Dict]:
        """Get players in a specific tournament."""
        data = self._call("get_players", {"tournament_key": tournament_key},
                          cache_category="players")
        if data.get("success") != 1:
            return []
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # HEAD-TO-HEAD
    # ------------------------------------------------------------------
    def get_h2h(self, player1_key: int, player2_key: int) -> Dict:
        """
        Get H2H record between two players.
        Returns dict with keys: H2H, firstPlayerResults, secondPlayerResults
        """
        data = self._call("get_H2H", {
            "first_player_key": str(player1_key),
            "second_player_key": str(player2_key),
        }, cache_category="h2h")
        if data.get("success") != 1:
            return {"H2H": [], "firstPlayerResults": [], "secondPlayerResults": []}
        return data.get("result", {})

    # ------------------------------------------------------------------
    # RANKINGS / STANDINGS
    # ------------------------------------------------------------------
    def get_rankings(self, tour: str = "ATP") -> List[Dict]:
        """Get current ATP or WTA rankings."""
        data = self._call("get_standings", {"event_type": tour},
                          cache_category="standings")
        if data.get("success") != 1:
            return []
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # ODDS (pre-match)
    # ------------------------------------------------------------------
    def get_match_odds(self, match_key: str) -> Dict:
        """Get pre-match odds for a specific match from multiple bookmakers."""
        data = self._call("get_odds", {"match_key": match_key},
                          cache_category="odds")
        if data.get("success") != 1:
            return {}
        result = data.get("result", {})
        if isinstance(result, dict):
            # result is {match_key: {market: {outcome: {book: odds}}}}
            return result.get(str(match_key), result)
        return {}

    def get_odds_for_date(self, date: str) -> Dict:
        """Get odds for all matches on a date (yyyy-mm-dd)."""
        data = self._call("get_odds", {
            "date_start": date, "date_stop": date
        }, cache_category="odds")
        if data.get("success") != 1:
            return {}
        return data.get("result", {})

    # ------------------------------------------------------------------
    # LIVE ODDS (in-play)
    # ------------------------------------------------------------------
    def get_live_odds(self, match_key: Optional[str] = None) -> Dict:
        """Get live in-play odds. Optionally filter by match_key."""
        params: Dict[str, str] = {}
        if match_key:
            params["match_key"] = match_key
        data = self._call("get_live_odds", params, cache_category="live_odds")
        if data.get("success") != 1:
            return {}
        return data.get("result", {})

    # ------------------------------------------------------------------
    # TOURNAMENTS & EVENT TYPES
    # ------------------------------------------------------------------
    def get_tournaments(self) -> List[Dict]:
        """Get all available tournaments."""
        data = self._call("get_tournaments", cache_category="fixtures")
        if data.get("success") != 1:
            return []
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    def get_event_types(self) -> List[Dict]:
        """Get event types (ATP Singles, WTA Singles, etc.)."""
        data = self._call("get_events", cache_category="fixtures")
        if data.get("success") != 1:
            return []
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    # ==================================================================
    # HIGH-LEVEL HELPERS  (parsed data ready for the calculator)
    # ==================================================================

    def parse_live_score(self, match: Dict) -> Dict:
        """
        Parse a live match dict into a clean structure for the calculator.
        Returns sets, games, points, server, and stats.
        """
        scores = match.get("scores", [])
        sets_p1, sets_p2 = 0, 0
        set_scores = []
        for s in scores:
            s1 = int(s.get("score_first", 0))
            s2 = int(s.get("score_second", 0))
            set_scores.append((s1, s2))
            if s1 > s2 and (s1 >= 6 or (s1 == 7)):
                sets_p1 += 1
            elif s2 > s1 and (s2 >= 6 or (s2 == 7)):
                sets_p2 += 1

        # Current set games (last entry in scores is current set)
        games_p1, games_p2 = 0, 0
        if set_scores:
            last = set_scores[-1]
            # If the last set is still in play, these are current games
            if match.get("event_status", "").startswith("Set"):
                games_p1, games_p2 = last
            else:
                # Finished set
                games_p1, games_p2 = last

        # Current game score from event_game_result ("30 - 15")
        game_result = match.get("event_game_result", "0 - 0")
        points_p1, points_p2 = 0, 0
        if game_result and " - " in game_result:
            parts = game_result.split(" - ")
            try:
                pt1 = parts[0].strip()
                pt2 = parts[1].strip()
                # Map tennis point names to index
                point_map = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4}
                points_p1 = point_map.get(pt1, 0)
                points_p2 = point_map.get(pt2, 0)
            except (ValueError, IndexError):
                pass

        # Server
        server = 1 if match.get("event_serve") == "First Player" else 2

        # Point by point stats
        pbp = match.get("pointbypoint", [])
        aces_p1, aces_p2, dfs_p1, dfs_p2 = 0, 0, 0, 0
        bp_p1, bp_p2, bp_won_p1, bp_won_p2 = 0, 0, 0, 0
        total_games = 0
        serve_games_p1, serve_games_p2 = 0, 0
        serve_won_p1, serve_won_p2 = 0, 0

        for game in pbp:
            total_games += 1
            served_by = game.get("player_served", "")
            won_by = game.get("serve_winner", "")
            if served_by == "First Player":
                serve_games_p1 += 1
                if won_by == "First Player":
                    serve_won_p1 += 1
            elif served_by == "Second Player":
                serve_games_p2 += 1
                if won_by == "Second Player":
                    serve_won_p2 += 1

            for pt in game.get("points", []):
                if pt.get("break_point"):
                    if served_by == "First Player":
                        bp_p1 += 1
                        if won_by != "First Player":
                            bp_won_p2 += 1
                    else:
                        bp_p2 += 1
                        if won_by != "Second Player":
                            bp_won_p1 += 1

        # Derive serve win percentages
        serve_pct_p1 = (serve_won_p1 / max(1, serve_games_p1))
        serve_pct_p2 = (serve_won_p2 / max(1, serve_games_p2))

        return {
            "p1_name": match.get("event_first_player", "Player 1"),
            "p2_name": match.get("event_second_player", "Player 2"),
            "p1_key": match.get("first_player_key"),
            "p2_key": match.get("second_player_key"),
            "status": match.get("event_status", ""),
            "event_key": match.get("event_key"),
            "tournament": match.get("tournament_name", ""),
            "round": match.get("tournament_round", ""),
            "event_type": match.get("event_type_type", ""),
            "server": server,
            "sets_p1": sets_p1,
            "sets_p2": sets_p2,
            "games_p1": games_p1,
            "games_p2": games_p2,
            "points_p1": points_p1,
            "points_p2": points_p2,
            "set_scores": set_scores,
            "game_result_str": game_result,
            "serve_pct_p1": serve_pct_p1,
            "serve_pct_p2": serve_pct_p2,
            "serve_games_p1": serve_games_p1,
            "serve_games_p2": serve_games_p2,
            "serve_won_p1": serve_won_p1,
            "serve_won_p2": serve_won_p2,
            "aces_p1": aces_p1,
            "aces_p2": aces_p2,
            "dfs_p1": dfs_p1,
            "dfs_p2": dfs_p2,
            "bp_faced_p1": bp_p1,
            "bp_faced_p2": bp_p2,
            "bp_won_p1": bp_won_p1,
            "bp_won_p2": bp_won_p2,
            "total_games": total_games,
            "event_live": match.get("event_live", "0"),
            "event_date": match.get("event_date", ""),
            "event_time": match.get("event_time", ""),
        }

    def parse_odds(self, odds_data: Dict) -> Dict:
        """
        Parse odds dict into a simple structure.
        Returns best available Home/Away odds from major bookmakers.
        """
        ha = odds_data.get("Home/Away", {})
        home_odds = ha.get("Home", {})
        away_odds = ha.get("Away", {})

        # Priority: bet365 > bwin > 1xbet > any
        priority = ["bet365", "bwin", "1xbet", "Betsson", "Sportingbet", "Betcris"]

        p1_odds = None
        p2_odds = None
        best_book = None

        for book in priority:
            if book in home_odds and p1_odds is None:
                try:
                    p1_odds = float(home_odds[book])
                    p2_odds = float(away_odds.get(book, "0"))
                    best_book = book
                except (ValueError, TypeError):
                    continue

        # Fallback to first available
        if p1_odds is None and home_odds:
            for book, val in home_odds.items():
                try:
                    p1_odds = float(val)
                    p2_odds = float(away_odds.get(book, "0"))
                    best_book = book
                    break
                except (ValueError, TypeError):
                    continue

        # Set odds (if available)
        set_betting = odds_data.get("Set Betting", {})
        first_set_ha = odds_data.get("Home/Away (1st Set)", {})

        return {
            "p1_match_odds": p1_odds or 1.90,
            "p2_match_odds": p2_odds or 1.90,
            "bookmaker": best_book or "N/A",
            "all_p1_odds": {k: float(v) for k, v in home_odds.items()
                            if self._is_float(v)},
            "all_p2_odds": {k: float(v) for k, v in away_odds.items()
                            if self._is_float(v)},
            "set_betting": set_betting,
            "first_set": first_set_ha,
        }

    def parse_player_stats(self, player: Dict, surface: str = "hard") -> Dict:
        """
        Parse player profile into stats the calculator needs.
        Uses latest singles season stats.
        """
        stats = player.get("stats", [])
        name = player.get("player_name", "Unknown")
        country = player.get("player_country", "")

        # Find latest singles season
        latest = None
        for s in stats:
            if s.get("type") == "singles":
                if latest is None or int(s.get("season", 0)) > int(latest.get("season", 0)):
                    latest = s

        if not latest:
            return {
                "name": name,
                "country": country,
                "rank": 999,
                "matches_won": 0,
                "matches_lost": 0,
                "win_rate": 0.50,
                "surface_win_rate": 0.50,
                "titles": 0,
            }

        won = int(latest.get("matches_won", 0) or 0)
        lost = int(latest.get("matches_lost", 0) or 0)
        total = won + lost
        win_rate = won / max(1, total)

        # Surface-specific stats
        surface_map = {
            "hard": ("hard_won", "hard_lost"),
            "clay": ("clay_won", "clay_lost"),
            "grass": ("grass_won", "grass_lost"),
        }
        sw_key, sl_key = surface_map.get(surface.lower(), ("hard_won", "hard_lost"))
        s_won = int(latest.get(sw_key, 0) or 0)
        s_lost = int(latest.get(sl_key, 0) or 0)
        s_total = s_won + s_lost
        surface_win_rate = s_won / max(1, s_total) if s_total > 0 else win_rate

        return {
            "name": name,
            "country": country,
            "rank": int(latest.get("rank", 999) or 999),
            "season": latest.get("season", ""),
            "matches_won": won,
            "matches_lost": lost,
            "win_rate": win_rate,
            "titles": int(latest.get("titles", 0) or 0),
            "surface_won": s_won,
            "surface_lost": s_lost,
            "surface_win_rate": surface_win_rate,
        }

    def get_calculator_ready_data(self, match_key: str,
                                  surface: str = "hard") -> Optional[Dict]:
        """
        One-call method: get everything needed to auto-populate the calculator.
        Returns parsed live score, odds, and player stats.
        """
        match = self.get_live_match_detail(match_key)
        if not match:
            return None

        parsed = self.parse_live_score(match)

        # Odds
        odds_raw = self.get_match_odds(match_key)
        parsed["odds"] = self.parse_odds(odds_raw)

        # Player profiles (if keys available)
        p1_key = parsed.get("p1_key")
        p2_key = parsed.get("p2_key")
        if p1_key:
            p1_profile = self.get_player_profile(int(p1_key))
            if p1_profile:
                parsed["p1_profile"] = self.parse_player_stats(p1_profile, surface)
        if p2_key:
            p2_profile = self.get_player_profile(int(p2_key))
            if p2_profile:
                parsed["p2_profile"] = self.parse_player_stats(p2_profile, surface)

        # Live odds
        live_odds = self.get_live_odds(match_key)
        if live_odds:
            parsed["live_odds"] = live_odds

        return parsed

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _is_float(val) -> bool:
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict:
        """Return cache statistics."""
        now = time.time()
        total = len(self._cache)
        expired = sum(
            1 for _, (ts, _) in self._cache.items()
            if now - ts > 300
        )
        return {"total_entries": total, "expired": expired, "active": total - expired}


# ======================================================================
# Convenience: create a global singleton
# ======================================================================
_global_service: Optional[LiveTennisDataService] = None


def get_service(api_key: Optional[str] = None) -> LiveTennisDataService:
    """Get or create the global LiveTennisDataService instance."""
    global _global_service
    if _global_service is None or (api_key and api_key != _global_service.api_key):
        _global_service = LiveTennisDataService(api_key)
    return _global_service
