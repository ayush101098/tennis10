"""
Match Schedule + Base Probability
=================================
Fetches today's and tomorrow's tennis matches from Sofascore and ESPN,
and computes a base (pre-match) win probability for each using ranking
differentials and surface adjustments.

The probability model is lightweight — no ML, no serve stats needed.
It uses an Elo-like ranking→probability conversion that works purely
from player ranking numbers, with surface and best-of adjustments.
"""

from __future__ import annotations

import math
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .live_feed import SofascoreAdapter, ESPNAdapter, ApiTennisAdapter, DataSource

logger = logging.getLogger(__name__)


# ─── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class ScheduledMatch:
    """A match on today's or tomorrow's schedule."""
    id: str = ""
    player1: str = ""
    player2: str = ""
    p1_rank: int = 0          # 0 = unknown
    p2_rank: int = 0
    p1_seed: int = 0
    p2_seed: int = 0
    tournament: str = ""
    round: str = ""
    tour: str = "ATP"         # ATP, WTA, ITF-M, ITF-W, Challenger
    surface: str = "Hard"
    best_of: int = 3
    source: str = "sofascore"
    status: str = "scheduled"  # scheduled, live, finished, cancelled
    start_time: str = ""       # ISO or human-readable
    start_timestamp: float = 0.0

    # Base probability (computed)
    p1_win_prob: float = 0.5
    p2_win_prob: float = 0.5
    prob_method: str = "ranking"  # ranking, elo, seed


# ─── Base Probability from Rankings ──────────────────────────────────────────

def ranking_to_elo(rank: int) -> float:
    """
    Convert ATP/WTA ranking to approximate Elo rating.
    Based on empirical mapping: #1 ≈ 2400, #100 ≈ 1800, #300 ≈ 1500.
    """
    if rank <= 0:
        return 1700.0  # unknown player default
    if rank == 1:
        return 2400.0
    # Logarithmic decay: Elo = 2400 - 180 * ln(rank)
    return max(1300.0, 2400.0 - 180.0 * math.log(rank))


def elo_win_probability(elo1: float, elo2: float) -> float:
    """Standard Elo win probability: P(1 beats 2) = 1 / (1 + 10^((elo2-elo1)/400))"""
    return 1.0 / (1.0 + 10.0 ** ((elo2 - elo1) / 400.0))


def surface_adjustment(p: float, p1_rank: int, p2_rank: int, surface: str) -> float:
    """
    Small surface adjustment. Higher-ranked players tend to
    overperform on hard/grass relative to ranking, while clay
    compresses the field slightly.
    """
    if surface == "Clay":
        # Clay is more unpredictable, slight regression toward 50%
        return p * 0.92 + 0.04
    if surface == "Grass":
        # Grass favors big servers / higher ranked slightly more
        return p * 1.03 - 0.015
    return p  # Hard — neutral


def best_of_adjustment(p_match: float, best_of: int) -> float:
    """
    Convert a best-of-3 win probability to best-of-5.
    The better player benefits from more sets.
    """
    if best_of != 5:
        return p_match
    # Approximate: p5 = p3^1.25 / (p3^1.25 + (1-p3)^1.25)
    # This amplifies the favorite's edge
    p = p_match
    q = 1.0 - p
    p5 = (p ** 1.22) / (p ** 1.22 + q ** 1.22)
    return p5


def compute_base_probability(
    p1_rank: int,
    p2_rank: int,
    surface: str = "Hard",
    best_of: int = 3,
    p1_seed: int = 0,
    p2_seed: int = 0,
) -> Tuple[float, float, str]:
    """
    Compute base win probability for player 1 and player 2.
    Returns (p1_prob, p2_prob, method).
    """
    method = "ranking"

    # If both rankings are known, use ranking→Elo
    if p1_rank > 0 and p2_rank > 0:
        elo1 = ranking_to_elo(p1_rank)
        elo2 = ranking_to_elo(p2_rank)
        p1 = elo_win_probability(elo1, elo2)
    elif p1_seed > 0 and p2_seed > 0:
        # Fall back to seeds (treat seed as approximate rank)
        method = "seed"
        elo1 = ranking_to_elo(p1_seed)
        elo2 = ranking_to_elo(p2_seed)
        p1 = elo_win_probability(elo1, elo2)
    elif p1_rank > 0 or p2_rank > 0:
        # One known, one unknown → give unknown a default ~150
        r1 = p1_rank if p1_rank > 0 else 150
        r2 = p2_rank if p2_rank > 0 else 150
        elo1 = ranking_to_elo(r1)
        elo2 = ranking_to_elo(r2)
        p1 = elo_win_probability(elo1, elo2)
    else:
        # Both unknown → 50/50
        return 0.50, 0.50, "unknown"

    # Surface adjustment
    p1 = surface_adjustment(p1, p1_rank, p2_rank, surface)

    # Best-of-5 adjustment (Grand Slams)
    p1 = best_of_adjustment(p1, best_of)

    # Clamp
    p1 = max(0.03, min(0.97, p1))
    p2 = 1.0 - p1

    return round(p1, 4), round(p2, 4), method


# ─── Schedule Fetchers ───────────────────────────────────────────────────────

def _sofascore_scheduled(date_str: str) -> List[Dict]:
    """
    Fetch all tennis matches for a given date from Sofascore.
    date_str: "YYYY-MM-DD"
    Returns matches with status: scheduled, live, or finished.
    """
    adapter = SofascoreAdapter()
    matches = []
    try:
        resp = adapter.session.get(
            f"{adapter.BASE}/sport/tennis/scheduled-events/{date_str}",
            timeout=15,
        )
        if resp.status_code != 200:
            logger.debug("Sofascore schedule %s returned %d", date_str, resp.status_code)
            return []
        data = resp.json()
        events = data.get("events", [])
        for ev in events:
            parsed = _parse_sofascore_scheduled(ev)
            if parsed:
                matches.append(parsed)
    except Exception as e:
        logger.debug("Sofascore schedule fetch failed: %s", e)
    return matches


def _parse_sofascore_scheduled(ev: Dict) -> Optional[Dict]:
    """Parse a Sofascore scheduled event."""
    home = ev.get("homeTeam", {})
    away = ev.get("awayTeam", {})
    tournament = ev.get("tournament", {})

    p1_name = home.get("name", "")
    p2_name = away.get("name", "")
    if not p1_name or not p2_name:
        return None

    # Status
    status_obj = ev.get("status", {})
    status_code = status_obj.get("code", 0)
    status_desc = status_obj.get("description", "").lower()
    if status_code == 0:
        status = "scheduled"
    elif status_code in (6, 7, 8, 9, 10):
        status = "live"
    elif status_code == 100:
        status = "finished"
    elif status_code in (60, 70, 80, 90):
        status = "cancelled"
    else:
        status = "scheduled"

    # Rankings
    p1_rank = home.get("ranking", 0) or 0
    p2_rank = away.get("ranking", 0) or 0

    # Seeds — sometimes in subTeam field
    p1_seed = 0
    p2_seed = 0
    home_sub = home.get("subTeams", [])
    away_sub = away.get("subTeams", [])
    if not p1_rank and home_sub:
        p1_rank = home_sub[0].get("ranking", 0) or 0

    # Tour detection
    tour = SofascoreAdapter._detect_tour(tournament)

    # Surface
    surface = SofascoreAdapter._detect_surface(tournament)

    # Best-of: Grand Slams (ATP) are best of 5
    best_of = 3
    ut = tournament.get("uniqueTournament", {})
    ut_name = (ut.get("name", "") if ut else "").lower()
    t_name = tournament.get("name", "").lower()
    if any(gs in f"{t_name} {ut_name}" for gs in [
        "australian open", "roland garros", "wimbledon", "us open"
    ]):
        if tour == "ATP":
            best_of = 5

    # Start time
    start_ts = ev.get("startTimestamp", 0)
    start_time = ""
    if start_ts:
        try:
            start_time = datetime.fromtimestamp(start_ts).strftime("%H:%M")
        except Exception:
            pass

    # Round
    round_info = ev.get("roundInfo", {})
    round_name = round_info.get("name", "")

    source_id = str(ev.get("id", ""))

    return {
        "id": f"sf_{source_id}",
        "player1": p1_name,
        "player2": p2_name,
        "p1_rank": p1_rank,
        "p2_rank": p2_rank,
        "p1_seed": p1_seed,
        "p2_seed": p2_seed,
        "tournament": tournament.get("name", ""),
        "round": round_name,
        "tour": tour,
        "surface": surface,
        "best_of": best_of,
        "source": "sofascore",
        "status": status,
        "start_time": start_time,
        "start_timestamp": float(start_ts),
    }


def _espn_scheduled(date_str: str) -> List[Dict]:
    """
    Fetch all tennis matches for a given date from ESPN.
    date_str: "YYYY-MM-DD" → ESPN wants "YYYYMMDD"
    """
    adapter = ESPNAdapter()
    espn_date = date_str.replace("-", "")
    matches = []

    for tour in adapter.TOUR_SLUGS:
        try:
            resp = adapter.session.get(
                f"{adapter.BASE}/{tour}/scoreboard",
                params={"dates": espn_date},
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            for ev in data.get("events", []):
                competitions = []
                for g in ev.get("groupings", []):
                    competitions.extend(g.get("competitions", []))
                if not competitions:
                    competitions = ev.get("competitions", [])
                for c in competitions:
                    parsed = _parse_espn_scheduled(c, ev, tour)
                    if parsed:
                        matches.append(parsed)
        except Exception as e:
            logger.debug("ESPN schedule fetch for %s failed: %s", tour, e)

    return matches


def _parse_espn_scheduled(comp: Dict, event: Dict, tour: str) -> Optional[Dict]:
    """Parse an ESPN competition into a scheduled match."""
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

    # Status
    status_obj = comp.get("status", {}).get("type", {})
    state = status_obj.get("state", "pre")
    if state == "in":
        status = "live"
    elif state == "post":
        status = "finished"
    else:
        status = "scheduled"

    # Rankings from ESPN (sometimes in athlete.rankings)
    p1_rank = 0
    p2_rank = 0
    if isinstance(p1_ath, dict):
        rankings = p1_ath.get("rankings", [])
        if rankings:
            p1_rank = rankings[0].get("current", 0)
    if isinstance(p2_ath, dict):
        rankings = p2_ath.get("rankings", [])
        if rankings:
            p2_rank = rankings[0].get("current", 0)

    # Seeds
    p1_seed = int(p1_data.get("seed", 0) or 0)
    p2_seed = int(p2_data.get("seed", 0) or 0)

    # If ranks still unknown, try to infer from seed
    if not p1_rank and p1_seed:
        p1_rank = p1_seed  # rough approximation
    if not p2_rank and p2_seed:
        p2_rank = p2_seed

    # Surface
    surface = ESPNAdapter._detect_surface(event)

    # Best-of
    best_of = 3
    event_name = event.get("name", "").lower()
    if any(gs in event_name for gs in [
        "australian open", "french open", "roland garros", "wimbledon", "us open"
    ]):
        if tour.upper() == "ATP":
            best_of = 5

    # Start time
    start_time = ""
    start_ts = 0.0
    date_str_raw = comp.get("date", "") or event.get("date", "")
    if date_str_raw:
        try:
            dt = datetime.fromisoformat(date_str_raw.replace("Z", "+00:00"))
            start_time = dt.strftime("%H:%M")
            start_ts = dt.timestamp()
        except Exception:
            pass

    # Round
    round_name = ""
    notes = comp.get("notes", [])
    if notes:
        round_name = notes[0].get("headline", "")
    if not round_name:
        round_name = comp.get("description", "")

    source_id = f"{event.get('id', '')}_{comp.get('id', '')}"

    return {
        "id": f"espn_{source_id}",
        "player1": p1_name,
        "player2": p2_name,
        "p1_rank": p1_rank,
        "p2_rank": p2_rank,
        "p1_seed": p1_seed,
        "p2_seed": p2_seed,
        "tournament": event.get("name", ""),
        "round": round_name,
        "tour": tour.upper(),
        "surface": surface,
        "best_of": best_of,
        "source": "espn",
        "status": status,
        "start_time": start_time,
        "start_timestamp": start_ts,
    }


# ─── Main Schedule Function ─────────────────────────────────────────────────

def fetch_schedule(
    include_finished: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Fetch today's and tomorrow's tennis matches from all sources.
    Computes base probability for each match.

    Returns: {
        "today": [...],
        "tomorrow": [...],
        "today_date": "2026-04-14",
        "tomorrow_date": "2026-04-15",
    }
    """
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    # Fetch from Sofascore + ESPN in parallel
    today_matches: List[Dict] = []
    tomorrow_matches: List[Dict] = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_sofascore_scheduled, today): ("today", "sofascore"),
            pool.submit(_sofascore_scheduled, tomorrow): ("tomorrow", "sofascore"),
            pool.submit(_espn_scheduled, today): ("today", "espn"),
            pool.submit(_espn_scheduled, tomorrow): ("tomorrow", "espn"),
        }
        for future in as_completed(futures):
            day, src = futures[future]
            try:
                results = future.result()
                if day == "today":
                    today_matches.extend(results)
                else:
                    tomorrow_matches.extend(results)
            except Exception as e:
                logger.debug("Schedule fetch failed (%s/%s): %s", day, src, e)

    # De-duplicate by player names (prefer Sofascore for rankings)
    today_matches = _dedup_matches(today_matches)
    tomorrow_matches = _dedup_matches(tomorrow_matches)

    # Filter out finished if requested
    if not include_finished:
        today_matches = [m for m in today_matches if m["status"] != "finished"]
        tomorrow_matches = [m for m in tomorrow_matches if m["status"] != "finished"]

    # Compute base probabilities
    for m in today_matches + tomorrow_matches:
        p1_prob, p2_prob, method = compute_base_probability(
            p1_rank=m.get("p1_rank", 0),
            p2_rank=m.get("p2_rank", 0),
            surface=m.get("surface", "Hard"),
            best_of=m.get("best_of", 3),
            p1_seed=m.get("p1_seed", 0),
            p2_seed=m.get("p2_seed", 0),
        )
        m["p1_win_prob"] = p1_prob
        m["p2_win_prob"] = p2_prob
        m["prob_method"] = method

    # Sort by start time, then tournament
    today_matches.sort(key=lambda m: (m.get("start_timestamp", 0) or 9999999999, m.get("tournament", "")))
    tomorrow_matches.sort(key=lambda m: (m.get("start_timestamp", 0) or 9999999999, m.get("tournament", "")))

    return {
        "today": today_matches,
        "tomorrow": tomorrow_matches,
        "today_date": today,
        "tomorrow_date": tomorrow,
    }


def _dedup_matches(matches: List[Dict]) -> List[Dict]:
    """De-duplicate matches by player names, preferring sources with ranking data."""
    seen: Dict[str, Dict] = {}
    for m in matches:
        names = sorted([
            m.get("player1", "").lower().strip(),
            m.get("player2", "").lower().strip(),
        ])
        key = "|".join(names)
        if not all(names):
            continue

        existing = seen.get(key)
        if not existing:
            seen[key] = m
        else:
            # Prefer the one with ranking data
            existing_has_rank = (existing.get("p1_rank", 0) > 0 or existing.get("p2_rank", 0) > 0)
            new_has_rank = (m.get("p1_rank", 0) > 0 or m.get("p2_rank", 0) > 0)
            if new_has_rank and not existing_has_rank:
                seen[key] = m
            elif new_has_rank and existing_has_rank:
                # Merge: take the one from Sofascore (usually better rankings)
                if m.get("source") == "sofascore":
                    seen[key] = m

    return list(seen.values())
