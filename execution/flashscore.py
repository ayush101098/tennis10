"""
Flashscore live tennis provider — the fallback for when SofaScore locks us out.

WHY THIS EXISTS
  SofaScore challenges an IP on /api/v1/* once it has pulled enough API traffic,
  and no TLS trick undoes it — the address is the thing being refused. That left
  the board with no live source at all. Flashscore serves the same tours
  (ATP / WTA / Challenger / ITF M / ITF W) from a completely separate
  infrastructure and answers normally from an address SofaScore has burned, so
  it is a genuine second leg rather than a mirror of the first.

WHAT IT GIVES US, HONESTLY
  ✓ live scores, per-set games, tiebreaks, sets won        (main feed)
  ✓ the CURRENT GAME's point score — 0/15/30/40/A          (dc_ feed)
  ✓ serve/return splits: aces, 1st-serve %, break points   (df_st_ feed)
  ✗ true point-by-point history. Flashscore does not expose one: probing
    df_pbp_1_<id> across every live match returned empty, every time. What the
    momentum engine needs is the point *sequence*, so `PointStream` below
    reconstructs it by polling the current-game score and recording each
    transition. That yields points from the moment we start watching — not the
    match's earlier history, which is simply not recoverable from this source.
  ✗ the current server. Flashscore renders it as an icon, not a feed field.
    Left as None rather than guessed; the terminal already treats an unknown
    server as unknown instead of assuming P1.

WIRE FORMAT
  Flashscore answers in its own delimited encoding, not JSON:
      records separated by ~, fields by ¬, key÷value inside a field
  `parse_feed` turns that into dicts. Field meanings were decoded against live
  matches and are documented at each mapping below.

USAGE
    from execution.flashscore import FlashscoreClient
    fs = FlashscoreClient()
    for m in fs.live_matches():
        print(m.tour, m.home, m.away, m.score_line, m.point_score)
"""

from __future__ import annotations

import re
import time
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta

import tls_client

FEED_BASE = "https://www.flashscore.com/x/feed"
# Flashscore requires this static signature header on every feed request;
# without it the CDN answers 401.
FSIGN = "SW9D1eZo"

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.flashscore.com/",
    "X-Fsign": FSIGN,
}

# Tennis is sport 2 in Flashscore's numbering.
SPORT_TENNIS = 2


# ─── Wire decoding ───────────────────────────────────────────────────────────

def parse_feed(text: str) -> list[dict]:
    """Flashscore's `key÷value¬key÷value~…` encoding → a list of record dicts."""
    out = []
    for rec in text.split("~"):
        d = {}
        for f in rec.split("¬"):
            if "÷" in f:
                k, v = f.split("÷", 1)
                d[k] = v
        if d:
            out.append(d)
    return out


# ─── Tournament header parsing ───────────────────────────────────────────────
# Header looks like:  "CHALLENGER MEN - SINGLES: Roehampton (United Kingdom), hard"
_HEADER_RE = re.compile(
    r"^(?P<tour>[^:]+?)\s*:\s*(?P<name>[^(,]+?)\s*(?:\((?P<country>[^)]*)\))?\s*(?:,\s*(?P<surface>[a-z ]+))?$",
    re.I,
)

# Flashscore tour label → the SofaScore category slug the rest of the stack
# already speaks. Keeping SofaScore's vocabulary means this provider can back
# the existing proxy without the terminal or push_sofa changing at all.
_TOUR_TO_SLUG = [
    ("ATP - SINGLES", "atp"),
    ("WTA 125", "wta-125"),
    ("WTA - SINGLES", "wta"),
    ("CHALLENGER MEN", "challenger"),
    ("CHALLENGER WOMEN", "wta-125"),
    ("ITF MEN", "itf-men"),
    ("ITF WOMEN", "itf-women"),
]

_SLUG_TO_CATEGORY_ID = {
    "atp": 3, "wta": 6, "challenger": 72, "itf-men": 785, "itf-women": 213,
    "wta-125": 6,
}
_CATEGORY_ID_TO_SLUG = {3: "atp", 6: "wta", 72: "challenger", 785: "itf-men", 213: "itf-women"}

_SLUG_TO_TOUR = {"atp": "ATP", "wta": "WTA", "challenger": "CHALLENGER",
                 "itf-men": "ITF M", "itf-women": "ITF W", "wta-125": "W125"}


def _classify(header: str) -> tuple[str | None, str, str, str]:
    """(category_slug, tournament_name, country, surface) from a tournament header.

    Returns slug=None for anything we deliberately skip — doubles, team events,
    and any tour outside the five the pipeline trades.
    """
    h = (header or "").strip()
    upper = h.upper()
    if "DOUBLES" in upper:
        return None, "", "", ""

    slug = None
    for label, s in _TOUR_TO_SLUG:
        if upper.startswith(label):
            slug = s
            break
    if slug is None:
        return None, "", "", ""

    m = _HEADER_RE.match(h)
    if not m:
        return slug, h, "", "Hard"
    name = (m.group("name") or "").strip()
    country = (m.group("country") or "").strip()
    surface = (m.group("surface") or "hard").strip().lower()
    surface = {"hard": "Hard", "clay": "Clay", "grass": "Grass",
               "indoors": "Hard", "hard indoors": "Hard"}.get(surface, "Hard")
    return slug, name, country, surface


# ─── Stable numeric ids ──────────────────────────────────────────────────────
# Flashscore ids are short strings ("0EbdDcdl"); everything downstream —
# the terminal's `sofa_<cat>_<id>` keys, the /event/<id>/… paths — expects a
# SofaScore-style integer. Hash to a stable 31-bit int and keep the reverse map
# so a later /event/<id>/statistics call can find its way back.
_id_map: dict[int, str] = {}
_id_lock = threading.Lock()


def numeric_id(fs_id: str) -> int:
    n = 0
    for ch in fs_id:
        n = (n * 131 + ord(ch)) & 0x7FFFFFFF
    n |= 1 << 30          # keep it comfortably large so it can't collide with a real small id
    with _id_lock:
        _id_map[n] = fs_id
    return n


def flashscore_id(num: int) -> str | None:
    with _id_lock:
        return _id_map.get(num)


# ─── Match model ─────────────────────────────────────────────────────────────

@dataclass
class Match:
    fs_id: str
    id: int
    slug: str                     # sofa category slug
    tour: str                     # ATP / WTA / CHALLENGER / ITF M / ITF W
    tournament: str
    country: str
    surface: str
    home: str
    away: str
    status: str                   # scheduled | live | finished
    start_ts: int
    set_index: int                # 1-based current/last set
    home_sets: int
    away_sets: int
    home_games: list[int] = field(default_factory=list)
    away_games: list[int] = field(default_factory=list)
    home_tb: dict[int, int] = field(default_factory=dict)   # set index → tiebreak points
    away_tb: dict[int, int] = field(default_factory=dict)
    point_home: str | None = None   # "0" "15" "30" "40" "A"
    point_away: str | None = None
    server: int | None = None       # not exposed by Flashscore — see module docstring

    @property
    def score_line(self) -> str:
        return " ".join(f"{h}-{a}" for h, a in zip(self.home_games, self.away_games))

    @property
    def point_score(self) -> str:
        if self.point_home is None:
            return ""
        return f"{self.point_home}-{self.point_away}"


# Main-feed field meanings, decoded against live matches:
#   AA event id      AD start ts       AB 1=scheduled 2=live 3=finished
#   AC 16+set number while live (17 = 1st set, 18 = 2nd, …); 3 when finished
#   AE/AF player names            AG/AH sets won
#   BA/BB set1 games  BC/BD set2  BE/BF set3  BG/BH set4  BI/BJ set5
#   DA/DB set1 tiebreak  DC/DD set2  DE/DF set3  DG/DH set4  DI/DJ set5
_SET_KEYS = [("BA", "BB"), ("BC", "BD"), ("BE", "BF"), ("BG", "BH"), ("BI", "BJ")]
_TB_KEYS = [("DA", "DB"), ("DC", "DD"), ("DE", "DF"), ("DG", "DH"), ("DI", "DJ")]

_STATUS = {"1": "scheduled", "2": "live", "3": "finished"}


def _int(d: dict, k: str, default: int = 0) -> int:
    try:
        return int(d.get(k, default))
    except (TypeError, ValueError):
        return default


def _build_match(header: str, d: dict) -> Match | None:
    slug, tname, country, surface = _classify(header)
    if slug is None:
        return None
    home, away = d.get("AE", ""), d.get("AF", "")
    if not home or not away:
        return None

    status = _STATUS.get(d.get("AB", ""), "scheduled")
    ac = _int(d, "AC")
    set_index = (ac - 16) if status == "live" and ac >= 17 else 0

    hg, ag = [], []
    for hk, ak in _SET_KEYS:
        if hk not in d and ak not in d:
            continue
        hg.append(_int(d, hk))
        ag.append(_int(d, ak))

    htb, atb = {}, {}
    for i, (hk, ak) in enumerate(_TB_KEYS, start=1):
        # Tiebreak keys only appear when that set actually went to one.
        if hk in d or ak in d:
            htb[i] = _int(d, hk)
            atb[i] = _int(d, ak)

    fs_id = d["AA"]
    return Match(
        fs_id=fs_id, id=numeric_id(fs_id), slug=slug, tour=_SLUG_TO_TOUR.get(slug, slug.upper()),
        tournament=tname, country=country, surface=surface,
        home=home, away=away, status=status, start_ts=_int(d, "AD"),
        set_index=set_index or max(1, len(hg)),
        home_sets=_int(d, "AG"), away_sets=_int(d, "AH"),
        home_games=hg, away_games=ag, home_tb=htb, away_tb=atb,
    )


# ─── Client ──────────────────────────────────────────────────────────────────

class FlashscoreClient:
    """Reads Flashscore's public feeds. No key, no account.

    Optional `egress` routes requests through a proxy, same as sofa_proxy —
    useful if Flashscore ever starts rate-limiting the address too.
    """

    def __init__(self, egress: str | None = None, timeout: int = 25):
        self.timeout = timeout
        self.egress = egress
        self._session = self._new_session()
        self._lock = threading.Lock()

    def _new_session(self):
        s = tls_client.Session(client_identifier="chrome_133",
                               random_tls_extension_order=True)
        s.headers.update(_HEADERS)
        if self.egress:
            s.proxies = {"http": self.egress, "https": self.egress}
        return s

    def _get(self, feed: str) -> str:
        with self._lock:
            r = self._session.get(f"{FEED_BASE}/{feed}", timeout_seconds=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"flashscore {feed} → HTTP {r.status_code}")
        return r.text

    # ── feeds ────────────────────────────────────────────────────────────────

    def _day_feed(self, day_offset: int = 0) -> list[Match]:
        """f_<sport>_<dayOffset>_<tz>_<lang>_1 — every tennis match for a day."""
        text = self._get(f"f_{SPORT_TENNIS}_{day_offset}_3_en_1")
        out: list[Match] = []
        header = ""
        for d in parse_feed(text):
            if "ZA" in d:
                header = d["ZA"]
            if "AA" in d:
                m = _build_match(header, d)
                if m:
                    out.append(m)
        return out

    def matches(self, day_offset: int = 0) -> list[Match]:
        """All singles matches for a day (0 = today, 1 = tomorrow, -1 = yesterday)."""
        return self._day_feed(day_offset)

    def live_matches(self, with_points: bool = True) -> list[Match]:
        """In-progress singles matches. `with_points` fills the current game score."""
        live = [m for m in self._day_feed(0) if m.status == "live"]
        if with_points:
            for m in live:
                try:
                    self.fill_point_score(m)
                except Exception:
                    pass          # a missing point score must not drop the match
        return live

    def fill_point_score(self, m: Match) -> Match:
        """Add the current game's point score from the detail feed.

        dc_ fields: DP = home point, DQ = away point ("0"/"15"/"30"/"40"/"A"),
        DB = 16+set number, DE/DG = home sets, DF/DH = away sets.
        """
        d = {}
        for rec in parse_feed(self._get(f"dc_1_{m.fs_id}")):
            d.update(rec)
        if "DP" in d:
            m.point_home = d.get("DP")
            m.point_away = d.get("DQ")
        db = _int(d, "DB")
        if db >= 17:
            m.set_index = db - 16
        return m

    def statistics(self, fs_id: str) -> dict[str, dict[str, str]]:
        """Serve/return splits, grouped by section.

        {"Service": {"Aces": {"home": "5", "away": "1"}, …}, "Return": {…}}
        SG = stat name, SH = home value, SI = away value, SF = section.
        """
        out: dict[str, dict[str, dict[str, str]]] = {}
        section = "Match"
        for d in parse_feed(self._get(f"df_st_1_{fs_id}")):
            if "SF" in d:
                section = d["SF"]
            if "SG" in d:
                out.setdefault(section, {})[d["SG"]] = {
                    "home": d.get("SH", ""), "away": d.get("SI", ""),
                }
        return out

    def summary(self, fs_id: str) -> list[dict]:
        """Per-set summary rows (games, tiebreak, duration)."""
        return parse_feed(self._get(f"df_sur_1_{fs_id}"))


# ─── Reconstructed point stream ──────────────────────────────────────────────

_POINT_ORDER = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4}


@dataclass
class Point:
    ts: float
    set_index: int
    home_games: int
    away_games: int
    point_home: str
    point_away: str
    winner: int | None      # 1 = home won this point, 2 = away, None = first observation


class PointStream:
    """Turns repeated polls of the current game score into a point sequence.

    Flashscore has no point-by-point feed (see module docstring), so this is the
    substitute: on each poll, compare the game score to the previous one and
    record who gained. It is only as complete as the polling — points before the
    first poll, and any two points landing inside one interval, are not
    recoverable. Poll every 5–10s to keep misses rare.
    """

    def __init__(self, client: FlashscoreClient, poll_s: float = 8.0):
        self.client = client
        self.poll_s = poll_s
        self._last: dict[str, tuple] = {}       # fs_id → (set, hg, ag, ph, pa)
        self.points: dict[str, list[Point]] = {}

    def observe(self, m: Match) -> Point | None:
        """Fold one observation in; returns the Point if one was scored."""
        if m.point_home is None:
            return None
        hg = m.home_games[m.set_index - 1] if len(m.home_games) >= m.set_index else 0
        ag = m.away_games[m.set_index - 1] if len(m.away_games) >= m.set_index else 0
        cur = (m.set_index, hg, ag, m.point_home, m.point_away)
        prev = self._last.get(m.fs_id)
        self._last[m.fs_id] = cur

        if prev is None:
            return None                      # nothing to diff against yet
        if prev == cur:
            return None                      # no change since last poll

        winner = self._who_scored(prev, cur)
        p = Point(ts=time.time(), set_index=m.set_index, home_games=hg, away_games=ag,
                  point_home=m.point_home, point_away=m.point_away, winner=winner)
        self.points.setdefault(m.fs_id, []).append(p)
        return p

    @staticmethod
    def _who_scored(prev: tuple, cur: tuple) -> int | None:
        """Infer the point winner from how the score moved."""
        _, phg, pag, pph, ppa = prev
        _, chg, cag, cph, cpa = cur

        # A game ended: whoever's game count went up won the last point.
        if chg > phg:
            return 1
        if cag > pag:
            return 2

        # Same game — compare point ranks.
        dh = _POINT_ORDER.get(cph, -1) - _POINT_ORDER.get(pph, -1)
        da = _POINT_ORDER.get(cpa, -1) - _POINT_ORDER.get(ppa, -1)
        if dh > 0 and da <= 0:
            return 1
        if da > 0 and dh <= 0:
            return 2
        # Deuce reset (40-A → 40-40) tells us someone scored but not who.
        return None

    def poll_once(self) -> list[tuple[Match, Point]]:
        """One sweep over every live match; returns the points that landed."""
        scored = []
        for m in self.client.live_matches(with_points=True):
            p = self.observe(m)
            if p:
                scored.append((m, p))
        return scored

    def run(self, iterations: int | None = None, on_point=None) -> None:
        """Poll until stopped. `on_point(match, point)` fires per scored point."""
        n = 0
        while iterations is None or n < iterations:
            try:
                for m, p in self.poll_once():
                    if on_point:
                        on_point(m, p)
            except Exception as e:
                print(f"[flashscore] poll error: {str(e)[:100]}")
            n += 1
            if iterations is None or n < iterations:
                time.sleep(self.poll_s)


# ─── SofaScore-shaped translation ────────────────────────────────────────────
# The whole stack — terminal, push_sofa, execution — already speaks SofaScore's
# event schema. Emitting that shape means Flashscore can back the existing proxy
# with nothing downstream changing. Mapping documented in scheduleService.ts.

def _status_block(m: Match) -> dict:
    if m.status == "live":
        # 6–10 = set 1–5 in SofaScore's numbering.
        code = min(10, 5 + max(1, m.set_index))
        return {"code": code, "type": "inprogress",
                "description": f"{m.set_index}{_ordinal(m.set_index)} Set"}
    if m.status == "finished":
        return {"code": 100, "type": "finished", "description": "Ended"}
    return {"code": 0, "type": "notstarted", "description": "Not started"}


def _ordinal(n: int) -> str:
    return {1: "st", 2: "nd", 3: "rd"}.get(n, "th")


def _score_block(games: list[int], sets_won: int, tb: dict[int, int],
                 point: str | None) -> dict:
    out: dict = {"current": sets_won, "display": sets_won}
    for i, g in enumerate(games, start=1):
        out[f"period{i}"] = g
    for i, v in tb.items():
        out[f"period{i}TieBreak"] = v
    if point is not None:
        out["point"] = point
    return out


def to_sofa_event(m: Match) -> dict:
    """One Match → one SofaScore-shaped event object."""
    return {
        "id": m.id,
        "startTimestamp": m.start_ts,
        "homeTeam": {"name": m.home, "slug": m.home, "type": 1, "id": numeric_id(m.fs_id + "H")},
        "awayTeam": {"name": m.away, "slug": m.away, "type": 1, "id": numeric_id(m.fs_id + "A")},
        "tournament": {
            "name": m.tournament,
            "slug": m.tournament.lower().replace(" ", "-"),
            "category": {
                "name": m.tour, "slug": m.slug,
                "id": _SLUG_TO_CATEGORY_ID.get(m.slug, 0),
                "sport": {"name": "Tennis", "slug": "tennis", "id": 5},
            },
            "uniqueTournament": {
                "name": m.tournament,
                "groundType": m.surface,
                "id": numeric_id(m.tournament + m.slug),
            },
        },
        "status": _status_block(m),
        "homeScore": _score_block(m.home_games, m.home_sets, m.home_tb, m.point_home),
        "awayScore": _score_block(m.away_games, m.away_sets, m.away_tb, m.point_away),
        # Flashscore does not publish the server; omitting the key is how the
        # terminal is told "unknown" (it explicitly avoids guessing P1).
        **({"firstToServe": m.server} if m.server else {}),
        "roundInfo": {"name": ""},
        "source": "flashscore",
    }


def sofa_live_payload(client: FlashscoreClient) -> dict:
    """SofaScore-shaped body for `sport/tennis/events/live`."""
    return {"events": [to_sofa_event(m) for m in client.live_matches(with_points=True)]}


def sofa_scheduled_payload(client: FlashscoreClient, category_id: int,
                           day: str | None = None) -> dict:
    """SofaScore-shaped body for `category/<id>/scheduled-events/<date>`."""
    slug = _CATEGORY_ID_TO_SLUG.get(category_id)
    if slug is None:
        return {"events": []}
    target = day or date.today().isoformat()
    offset = (date.fromisoformat(target) - date.today()).days
    if abs(offset) > 7:
        return {"events": []}
    matches = [m for m in client.matches(offset) if m.slug == slug]
    for m in matches:
        if m.status == "live":
            try:
                client.fill_point_score(m)
            except Exception:
                pass
    return {"events": [to_sofa_event(m) for m in matches]}


def sofa_statistics_payload(client: FlashscoreClient, fs_id: str) -> dict:
    """SofaScore-shaped body for `event/<id>/statistics`."""
    stats = client.statistics(fs_id)
    groups = []
    for section, rows in stats.items():
        groups.append({
            "groupName": section,
            "statisticsItems": [
                {"name": name, "home": v["home"], "away": v["away"]}
                for name, v in rows.items()
            ],
        })
    return {"statistics": [{"period": "ALL", "groups": groups}]} if groups else {"statistics": []}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Flashscore live tennis provider")
    ap.add_argument("--live", action="store_true", help="list live matches")
    ap.add_argument("--day", type=int, default=None, help="list a day's matches (0=today)")
    ap.add_argument("--stats", metavar="FS_ID", help="statistics for a match")
    ap.add_argument("--stream", action="store_true", help="reconstruct the point stream")
    ap.add_argument("--polls", type=int, default=None, help="stop after N polls")
    ap.add_argument("--interval", type=float, default=8.0)
    args = ap.parse_args()

    fs = FlashscoreClient()

    if args.stats:
        for section, rows in fs.statistics(args.stats).items():
            print(f"\n{section}")
            for name, v in rows.items():
                print(f"  {name:<28} {v['home']:>12}  {v['away']:>12}")
        return

    if args.stream:
        print(f"[flashscore] point stream, polling every {args.interval}s — Ctrl-C to stop")
        stream = PointStream(fs, poll_s=args.interval)
        def show(m: Match, p: Point):
            who = {1: m.home, 2: m.away, None: "?"}[p.winner]
            print(f"  {m.tour:<10} {m.home[:16]:<16} v {m.away[:16]:<16} "
                  f"set {p.set_index} {p.home_games}-{p.away_games} "
                  f"[{p.point_home}-{p.point_away}]  point → {who}")
        try:
            stream.run(iterations=args.polls, on_point=show)
        except KeyboardInterrupt:
            total = sum(len(v) for v in stream.points.values())
            print(f"\nstopped. {total} point(s) captured across "
                  f"{len(stream.points)} match(es).")
        return

    matches = fs.live_matches() if args.live or args.day is None else fs.matches(args.day)
    if args.live or args.day is None:
        matches = [m for m in matches if m.status == "live"]
    print(f"{len(matches)} match(es)\n")
    for m in matches:
        pt = f"  [{m.point_score}]" if m.point_score else ""
        print(f"  {m.tour:<10} {m.tournament[:22]:<22} {m.surface:<6} "
              f"{m.home[:18]:<18} v {m.away[:18]:<18} "
              f"{m.home_sets}-{m.away_sets}  {m.score_line}{pt}")


if __name__ == "__main__":
    _main()
