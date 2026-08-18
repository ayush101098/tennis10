"""Pre-match probability model for the execution pipeline.

Wraps the trained Sackmann logistic model (ml_models/logistic_regression_trained.pkl)
with the exact 14-feature pipeline from trading_server.live_features.LiveFeatureEngine.
Career averages come from tennis_data.db; live in-match stats are left empty so the
feature engine uses pure pre-match priors.

Public API:
    mp = MatchModel()
    p = mp.predict_match("Cameron Norrie", "Mariano Navone", "Clay", level="")
    #   -> P(player1 wins the match), or None if either player is unknown
    p_set = mp.set_prob(p)          # single-set win prob consistent with p (best-of-3)
"""

import pickle
import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trading_server.live_features import LiveFeatureEngine, PlayerProfile  # noqa: E402
from trading_server.live_stats import PlayerMatchStats  # noqa: E402

DB_PATH = REPO_ROOT / "tennis_data.db"
MODEL_PATH = REPO_ROOT / "ml_models" / "logistic_regression_trained.pkl"

# Career-average query: unions a player's per-match serve/return stats from
# whichever side (winner or loser) they were on, most recent first.
_CAREER_SQL = """
WITH pm AS (
  SELECT tournament_date,
         w_1stIn*1.0/NULLIF(w_svpt,0)              AS fs_pct,
         w_1stWon*1.0/NULLIF(w_1stIn,0)            AS first_win,
         w_2ndWon*1.0/NULLIF(w_svpt-w_1stIn,0)     AS second_win,
         w_bpSaved*1.0/NULLIF(w_bpFaced,0)         AS bp_save,
         w_ace*1.0/NULLIF(w_SvGms,0)               AS aces_pg,
         w_df*1.0/NULLIF(w_SvGms,0)                AS df_pg,
         (w_1stWon+w_2ndWon)*1.0/NULLIF(w_svpt,0)  AS serve_won,
         1                                          AS won,
         winner_rank AS rk, winner_rank_points AS pts
  FROM matches WHERE winner_id = ? AND w_svpt > 0
  UNION ALL
  SELECT tournament_date,
         l_1stIn*1.0/NULLIF(l_svpt,0),
         l_1stWon*1.0/NULLIF(l_1stIn,0),
         l_2ndWon*1.0/NULLIF(l_svpt-l_1stIn,0),
         l_bpSaved*1.0/NULLIF(l_bpFaced,0),
         l_ace*1.0/NULLIF(l_SvGms,0),
         l_df*1.0/NULLIF(l_SvGms,0),
         (l_1stWon+l_2ndWon)*1.0/NULLIF(l_svpt,0),
         0,
         loser_rank, loser_rank_points
  FROM matches WHERE loser_id = ? AND l_svpt > 0)
SELECT COUNT(*) n,
       AVG(fs_pct), AVG(first_win), AVG(second_win), AVG(bp_save),
       AVG(aces_pg), AVG(df_pg), AVG(serve_won), AVG(won)
FROM pm
"""

# Career return-points-won %: the fraction of points this player wins when
# RETURNING, derived from the opponent's serve stats in each of the player's
# matches (opponent serving = this player returning). Needed for the
# opponent-adjusted point-based model (Klaassen & Magnus combined serve/return).
_RETURN_SQL = """
WITH pm AS (
  SELECT (l_1stWon+l_2ndWon)*1.0/NULLIF(l_svpt,0) AS opp_serve_won
    FROM matches WHERE winner_id = ? AND l_svpt > 0
  UNION ALL
  SELECT (w_1stWon+w_2ndWon)*1.0/NULLIF(w_svpt,0)
    FROM matches WHERE loser_id = ? AND w_svpt > 0)
SELECT COUNT(*) n, AVG(opp_serve_won) FROM pm
"""

# Most-recent rank / ranking points for a player (either side).
_RANK_SQL = """
SELECT rk, pts FROM (
  SELECT tournament_date, winner_rank rk, winner_rank_points pts
    FROM matches WHERE winner_id = ? AND winner_rank IS NOT NULL
  UNION ALL
  SELECT tournament_date, loser_rank, loser_rank_points
    FROM matches WHERE loser_id = ? AND loser_rank IS NOT NULL
) ORDER BY tournament_date DESC LIMIT 1
"""


class MatchModel:
    def __init__(self, db_path: Path = DB_PATH, model_path: Path = MODEL_PATH):
        self.conn = sqlite3.connect(str(db_path))
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.features = data["features"]

    # ── player lookup ─────────────────────────────────────────────────────────

    def find_player(self, name: str):
        """Return (player_id, player_name) for a player name, or None.

        Surname alone is ambiguous ("Djokovic" matches Novak AND his brother
        Marko), so we match on the FIRST name too: exact full name first, then
        first-name match, then first-initial, and only then fall back to the
        shortest surname match. Getting this wrong silently feeds the wrong
        player's serve/return stats into true_p.
        """
        name = name.strip()
        parts = name.split()
        if not parts:
            return None
        surname, first = parts[-1], (parts[0] if len(parts) > 1 else "")
        cands = self.conn.execute(
            "SELECT player_id, player_name FROM players WHERE player_name LIKE ?",
            (f"%{surname}%",)).fetchall()
        if not cands:
            return None
        nl = name.lower()
        for pid, pn in cands:                       # exact full name
            if pn.lower() == nl:
                return (pid, pn)
        if first:
            fl = first.lower()
            for pid, pn in cands:                   # first name matches
                if pn.split()[0].lower() == fl:
                    return (pid, pn)
            for pid, pn in cands:                   # first initial matches
                if pn.split()[0][:1].lower() == fl[:1]:
                    return (pid, pn)
        return min(cands, key=lambda r: len(r[1]))  # fallback: shortest

    def _profile(self, player_id: int) -> PlayerProfile:
        r = self.conn.execute(_CAREER_SQL, (player_id, player_id)).fetchone()
        n = r[0] or 0
        d = PlayerProfile()  # sensible defaults if a stat is NULL
        if n > 0:
            d.first_serve_pct = r[1] if r[1] is not None else d.first_serve_pct
            d.first_serve_win_pct = r[2] if r[2] is not None else d.first_serve_win_pct
            d.second_serve_win_pct = r[3] if r[3] is not None else d.second_serve_win_pct
            d.bp_save_pct = r[4] if r[4] is not None else d.bp_save_pct
            d.aces_per_game = r[5] if r[5] is not None else d.aces_per_game
            d.df_per_game = r[6] if r[6] is not None else d.df_per_game
            d.serve_pct = r[7] if r[7] is not None else d.serve_pct
            d.win_rate = r[8] if r[8] is not None else d.win_rate
        rk = self.conn.execute(_RANK_SQL, (player_id, player_id)).fetchone()
        if rk:
            if rk[0] is not None:
                d.rank = int(rk[0])
            if rk[1] is not None:
                d.ranking_points = int(rk[1])
        return d

    def return_pct(self, name: str) -> float | None:
        """Career return-points-won % for a player (fraction of points won when
        returning). None if the player is unknown or has no serve/return data."""
        found = self.find_player(name)
        if not found:
            return None
        r = self.conn.execute(_RETURN_SQL, (found[0], found[0])).fetchone()
        if not r or not r[0] or r[1] is None:
            return None
        ret = 1.0 - float(r[1])
        return ret if 0.1 < ret < 0.6 else None

    # ── prediction ──────────────────────────────────────────────────────────

    def predict_match(self, name1: str, name2: str, surface: str,
                      level: str = "") -> float | None:
        """P(player1 wins the match). None if either player is unknown."""
        p1, p2 = self.find_player(name1), self.find_player(name2)
        if not p1 or not p2 or p1[0] == p2[0]:
            return None
        prof1, prof2 = self._profile(p1[0]), self._profile(p2[0])
        engine = LiveFeatureEngine(prof1, prof2, surface=surface,
                                   tournament_level=level)
        X = engine.build_sackmann_features(PlayerMatchStats(), PlayerMatchStats())
        Xdf = pd.DataFrame(X, columns=self.features)
        Xs = self.scaler.transform(Xdf)
        return float(self.model.predict_proba(Xs)[0][1])

    # ── match <-> set conversion (Markov-consistent) ──────────────────────────

    @staticmethod
    def _match_from_set(p_set: float, best_of: int = 3) -> float:
        q = 1.0 - p_set
        if best_of == 5:
            return p_set**3 * (1 + 3 * q + 6 * q * q)
        return p_set**2 * (3 - 2 * p_set)  # best-of-3

    def set_prob(self, p_match: float, best_of: int = 3) -> float:
        """Invert the match<->set relationship: single-set win prob giving p_match."""
        lo, hi = 0.01, 0.99
        for _ in range(60):
            mid = (lo + hi) / 2
            if self._match_from_set(mid, best_of) < p_match:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def close(self):
        self.conn.close()
