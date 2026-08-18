"""Live-odds provider (Sofascore) → de-vigged fair probability for in-play matches.

Sofascore's public API is Cloudflare-protected, so we use tls_client (already a
project dep, used by sofa_proxy.py). For each live tennis singles match we read
the "Full time" (match-winner) market, convert fractional odds to decimal, strip
the bookmaker margin (de-vig), and expose the fair probability that a given
player wins. That fair probability — independent of Polymarket — is a genuine
edge signal when compared against the Polymarket price.

    prov = SofascoreOdds()
    p = prov.fair_prob("Thiago Monteiro", "Pierre-Hugues Herbert")
    #  -> probability that player1 (Monteiro) wins, or None if not matched/live

Fixture matching is deliberately conservative: both surnames must appear in the
same Sofascore event (either home/away orientation), else we return None rather
than risk pricing the wrong match.
"""

import re
import time
import unicodedata
from typing import Optional

SOFA_BASE = "https://www.sofascore.com/api/v1"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "Accept": "*/*",
}
_CACHE_TTL = 15  # seconds


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def _surname(full_name: str) -> str:
    parts = _norm(full_name).split()
    return parts[-1] if parts else ""


def _decimal_from_fractional(frac: str) -> Optional[float]:
    """'4/6' -> 1.667 decimal odds."""
    try:
        a, b = frac.split("/")
        return float(a) / float(b) + 1.0
    except (ValueError, ZeroDivisionError, AttributeError):
        return None


class SofascoreOdds:
    def __init__(self):
        self._session = None
        self._index: dict = {}       # (surnameA, surnameB) frozenset -> entry
        self._fetched_at = 0.0
        self._ok = False

    # ── low level ────────────────────────────────────────────────────────────

    def _get(self, path: str):
        import tls_client  # optional dep; only needed for live odds
        if self._session is None:
            self._session = tls_client.Session(
                client_identifier="chrome_120", random_tls_extension_order=True)
        r = self._session.get(f"{SOFA_BASE}/{path}", headers=_HEADERS)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except Exception:
            return None

    def _match_winner_prob(self, event_id: int) -> Optional[float]:
        """De-vigged fair probability that the HOME player wins."""
        od = self._get(f"event/{event_id}/odds/1/all")
        if not od:
            return None
        market = next((m for m in od.get("markets", [])
                       if m.get("marketName") == "Full time"), None)
        if not market:
            return None
        choices = market.get("choices") or []
        home = next((c for c in choices if c.get("name") == "1"), None)
        away = next((c for c in choices if c.get("name") == "2"), None)
        if not home or not away:
            return None
        dh = _decimal_from_fractional(home.get("fractionalValue"))
        da = _decimal_from_fractional(away.get("fractionalValue"))
        if not dh or not da:
            return None
        ih, ia = 1.0 / dh, 1.0 / da       # implied (with margin)
        total = ih + ia
        if total <= 0:
            return None
        return ih / total                 # de-vigged fair prob for home

    # ── index refresh ──────────────────────────────────────────────────────

    def refresh(self, force: bool = False) -> None:
        if not force and (time.time() - self._fetched_at) < _CACHE_TTL:
            return
        data = self._get("sport/tennis/events/live")
        self._fetched_at = time.time()
        if not data:
            self._ok = False
            return
        index = {}
        for e in data.get("events", []):
            status = (e.get("status") or {}).get("type")
            if status != "inprogress":
                continue
            home = (e.get("homeTeam") or {}).get("name", "")
            away = (e.get("awayTeam") or {}).get("name", "")
            if "/" in home or "/" in away:      # skip doubles
                continue
            sh, sa = _surname(home), _surname(away)
            if not sh or not sa or sh == sa:
                continue
            index[frozenset((sh, sa))] = {
                "event_id": e.get("id"), "home": home, "away": away,
                "home_surname": sh, "away_surname": sa,
                "hs": e.get("homeScore") or {}, "as": e.get("awayScore") or {},
                "last_period": e.get("lastPeriod") or "",
                "desc": (e.get("status") or {}).get("description", ""),
            }
        self._index = index
        self._ok = True

    @property
    def ok(self) -> bool:
        return self._ok

    def live_count(self) -> int:
        return len(self._index)

    # ── public ───────────────────────────────────────────────────────────────

    def state(self, player1: str, player2: str) -> Optional[dict]:
        """Live score oriented to player1: {sets, games, point, desc}. None if no match."""
        self.refresh()
        s1, s2 = _surname(player1), _surname(player2)
        entry = self._index.get(frozenset((s1, s2))) if s1 and s2 else None
        if not entry:
            return None
        p1_home = entry["home_surname"] == s1
        H, A = (entry["hs"], entry["as"]) if p1_home else (entry["as"], entry["hs"])
        try:
            cur = int(entry["last_period"].replace("period", "")) if entry["last_period"] else 1
        except ValueError:
            cur = 1
        games = " ".join(f"{H.get(f'period{i}', 0) or 0}-{A.get(f'period{i}', 0) or 0}"
                         for i in range(1, max(cur, 1) + 1))
        point = ""
        if H.get("point") is not None or A.get("point") is not None:
            point = f"{H.get('point', '')}-{A.get('point', '')}"
        return {
            "sets": f"{H.get('current', 0)}-{A.get('current', 0)}",
            "games": games, "point": point, "desc": entry["desc"],
        }

    @staticmethod
    def _frac(val: str):
        """'7/14 (50%)' -> (7.0, 14.0)."""
        try:
            a, b = str(val).split()[0].split("/")
            return float(a), float(b)
        except (ValueError, AttributeError, IndexError):
            return None, None

    def _serve_stats(self, event_id: int) -> Optional[dict]:
        """In-match serve-points-won % + serve points played, per side (home/away)."""
        j = self._get(f"event/{event_id}/statistics")
        if not j:
            return None
        total, won = {}, {}
        for grp in j.get("statistics", []):
            if grp.get("period") != "ALL":
                continue
            for g in grp.get("groups", []):
                for it in g.get("statisticsItems", []):
                    nm = it.get("name", "")
                    if nm == "First serve":          # denom = total serve points
                        for side in ("home", "away"):
                            _, den = self._frac(it.get(side, ""))
                            if den:
                                total[side] = den
                    elif nm == "Service points won":  # plain count
                        for side in ("home", "away"):
                            try:
                                won[side] = float(str(it.get(side)).split()[0])
                            except (ValueError, AttributeError):
                                pass
        out = {}
        for side in ("home", "away"):
            t = total.get(side)
            if t and t > 0:
                out[side] = (won.get(side, 0.0) / t, int(t))
        return out or None

    def serve_stats_for(self, player1: str, player2: str) -> Optional[dict]:
        """Live serve% + points played oriented to player1/player2. None if unavailable."""
        self.refresh()
        s1, s2 = _surname(player1), _surname(player2)
        entry = self._index.get(frozenset((s1, s2))) if s1 and s2 else None
        if not entry:
            return None
        st = self._serve_stats(entry["event_id"])
        if not st:
            return None
        p1_home = entry["home_surname"] == s1
        h, a = st.get("home"), st.get("away")
        p1s, p2s = (h, a) if p1_home else (a, h)
        return {"p1": p1s, "p2": p2s}   # each: (serve_pct, serve_points) or None

    def game_sequence(self, player1: str, player2: str, max_games: int = 16):
        """Oriented completed/in-progress game log for the live match, oldest->newest.

        Each item: {set, game, server, winner, points} where server/winner are
        1 (player1) / 2 (player2) / None, and points is a list of (p1_pt, p2_pt)
        running-score strings. Returns [] if live but no games yet, None if the
        fixture isn't a live Sofascore singles match. Feeds LiveMomentumEngine.
        """
        self.refresh()
        s1, s2 = _surname(player1), _surname(player2)
        entry = self._index.get(frozenset((s1, s2))) if s1 and s2 else None
        if not entry:
            return None
        pbp = self._get(f"event/{entry['event_id']}/point-by-point")
        sets = (pbp or {}).get("pointByPoint") or []
        p1_home = entry["home_surname"] == s1

        def orient(v):  # 1(home)/2(away) -> 1(p1)/2(p2)
            return v if p1_home else (3 - v) if v in (1, 2) else None

        rows = []
        for st in sets:
            set_no = st.get("set") or 0
            for g in (st.get("games") or []):
                sc = g.get("score") or {}
                scoring = sc.get("scoring")
                pts = [((p.get("homePoint"), p.get("awayPoint")) if p1_home
                        else (p.get("awayPoint"), p.get("homePoint")))
                       for p in (g.get("points") or [])]
                rows.append({
                    "set": set_no, "game": g.get("game") or 0,
                    "server": orient(sc.get("serving")),
                    "winner": orient(scoring) if scoring in (1, 2) else None,
                    "points": pts,
                })
        # deterministic oldest -> newest regardless of API ordering
        rows.sort(key=lambda r: (r["set"], r["game"]))
        return rows[-max_games:]

    def _current_server(self, event_id: int) -> Optional[int]:
        """1 (home) or 2 (away) currently serving, from the latest point-by-point game."""
        pbp = self._get(f"event/{event_id}/point-by-point")
        sets = (pbp or {}).get("pointByPoint") or []
        if not sets:
            return None
        games = (sets[0] or {}).get("games") or []   # sets[0] = current/most recent
        if not games:
            return None
        srv = games[0].get("score", {}).get("serving") or games[0].get("serving")
        return int(srv) if srv in (1, 2, "1", "2") else None

    def live_state(self, player1: str, player2: str) -> Optional[dict]:
        """Structured live score oriented to player1, incl. who's serving now.
        Returns None if the fixture isn't a live Sofascore singles match."""
        self.refresh()
        s1, s2 = _surname(player1), _surname(player2)
        entry = self._index.get(frozenset((s1, s2))) if s1 and s2 else None
        if not entry:
            return None
        p1_home = entry["home_surname"] == s1
        H, A = (entry["hs"], entry["as"]) if p1_home else (entry["as"], entry["hs"])
        try:
            cur = int(entry["last_period"].replace("period", "")) if entry["last_period"] else 1
        except ValueError:
            cur = 1
        serving = self._current_server(entry["event_id"])
        p1_serving = None
        if serving is not None:
            p1_serving = (serving == 1) if p1_home else (serving == 2)
        return {
            "sets_p1": int(H.get("current", 0) or 0),
            "sets_p2": int(A.get("current", 0) or 0),
            "games_p1": int(H.get(f"period{cur}", 0) or 0),
            "games_p2": int(A.get(f"period{cur}", 0) or 0),
            "p1_serving": p1_serving, "current_set": cur,
        }

    def fair_prob(self, player1: str, player2: str) -> Optional[float]:
        """Fair probability that `player1` wins, from de-vigged live odds. None if
        the fixture isn't a live Sofascore singles match or odds are missing."""
        self.refresh()
        s1, s2 = _surname(player1), _surname(player2)
        if not s1 or not s2:
            return None
        entry = self._index.get(frozenset((s1, s2)))
        if not entry:
            return None
        p_home = self._match_winner_prob(entry["event_id"])
        if p_home is None or not 0.0 < p_home < 1.0:
            return None
        # Orient to player1: is player1 the home side?
        return p_home if entry["home_surname"] == s1 else (1.0 - p_home)
