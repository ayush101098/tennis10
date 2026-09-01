"""
Tennis Match Feature Engineering Module — v3.0
Improvements over v2:
  - Corrected surface correlations (academic values: H↔G=0.10, C↔G=0.08)
  - Tour-aware serve defaults (ATP / WTA / ITF separate baselines)
  - Strict temporal isolation: feature computation only uses data before match date
  - Uncertainty score drives model confidence weighting
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tier 1 rally intelligence (Match Charting Project). Optional: if the rally_db
# has no rally_stats table the extractor still works and rally diffs are neutral.
try:
    from execution.rally import RallyProfiler
except Exception:  # pragma: no cover - keep feature extraction usable standalone
    RallyProfiler = None

RALLY_FEATURE_KEYS = [
    'RALLY_SHORT_WIN_DIFF', 'RALLY_MID_WIN_DIFF', 'RALLY_LONG_WIN_DIFF',
    'FIRST_STRIKE_DIFF', 'RALLY_AGGRESSION_DIFF', 'RALLY_DATA_BOTH',
]


# ──────────────────────────────────────────────────────────────────────────────
# CORRECTED surface transfer correlations (Cross 2014, Barnett & Clarke 2005)
# Previous values over-estimated clay↔grass transfer (was 0.15 → now 0.08)
# ──────────────────────────────────────────────────────────────────────────────
SURFACE_CORRELATIONS = {
    ('Hard',  'Hard'):  1.00,
    ('Clay',  'Clay'):  1.00,
    ('Grass', 'Grass'): 1.00,
    ('Hard',  'Clay'):  0.28,
    ('Clay',  'Hard'):  0.28,
    ('Hard',  'Grass'): 0.10,   # was 0.24 — academically too high
    ('Grass', 'Hard'):  0.10,
    ('Clay',  'Grass'): 0.08,   # was 0.15 — nearly uncorrelated
    ('Grass', 'Clay'):  0.08,
}

# ──────────────────────────────────────────────────────────────────────────────
# Tour-specific serve baselines (used when player data is missing)
# Previously the code only had ATP defaults, causing systematic WTA/ITF errors
# ──────────────────────────────────────────────────────────────────────────────
TOUR_DEFAULTS = {
    'ATP': {
        'first_serve_pct':      0.625,
        'first_serve_win_pct':  0.715,
        'second_serve_win_pct': 0.510,
        'bp_save_pct':          0.620,
        'wsp':                  0.640,
        'aces_per_game':        0.80,
        'df_per_game':          0.25,
    },
    'WTA': {
        'first_serve_pct':      0.590,
        'first_serve_win_pct':  0.640,
        'second_serve_win_pct': 0.470,
        'bp_save_pct':          0.570,
        'wsp':                  0.580,
        'aces_per_game':        0.30,
        'df_per_game':          0.30,
    },
    'ITF': {
        'first_serve_pct':      0.570,
        'first_serve_win_pct':  0.610,
        'second_serve_win_pct': 0.440,
        'bp_save_pct':          0.540,
        'wsp':                  0.560,
        'aces_per_game':        0.20,
        'df_per_game':          0.35,
    },
}


class TennisFeatureExtractor:
    """Extract temporally-isolated features for tennis match prediction."""

    # Exposed on the class as well as at module level: the surface table is
    # part of this extractor's contract, and callers (and tests) reach for
    # it through the class rather than importing the module constant.
    SURFACE_CORRELATIONS = SURFACE_CORRELATIONS

    def __init__(self, db_path: str = 'tennis_betting.db', tour: str = 'WTA',
                 rally_db: str = 'tennis_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.tour = tour.upper()
        self._defaults = TOUR_DEFAULTS.get(self.tour, TOUR_DEFAULTS['WTA'])

        # Serve stats are read from the inline `matches` columns (w_ace, w_svpt …),
        # which are populated in both tennis_data.db and tennis_betting.db — the
        # old path read an empty `statistics` table. The date column is named
        # differently across those DBs, so detect it once here.
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(matches)")}
        self.date_col = 'tournament_date' if 'tournament_date' in cols else 'tourney_date'

        # Tier 1 rally profiler (name-keyed, decoupled from this match DB).
        # Disabled gracefully if the module or rally_stats table is unavailable.
        self.rally = None
        if RallyProfiler is not None:
            try:
                r = RallyProfiler(rally_db)
                r.conn.execute("SELECT 1 FROM rally_stats LIMIT 1")
                self.rally = r
            except Exception as e:
                logger.info(f"Rally intelligence disabled: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
        if self.rally is not None:
            self.rally.close()

    def _player_name(self, player_id: int) -> Optional[str]:
        row = self.conn.execute(
            "SELECT player_name FROM players WHERE player_id=?", (player_id,)).fetchone()
        return row[0] if row else None

    def _rally_diffs(self, p1_id: int, p2_id: int, match_date: datetime) -> Dict:
        """Rally-construction diff features; neutral zeros if unavailable."""
        neutral = {k: 0.0 for k in RALLY_FEATURE_KEYS}
        if self.rally is None:
            return neutral
        n1, n2 = self._player_name(p1_id), self._player_name(p2_id)
        if not n1 or not n2:
            return neutral
        d = self.rally.diff_features(n1, n2, match_date, tour=self.tour)
        return {k: d[k] for k in RALLY_FEATURE_KEYS}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _surface_weight(self, target: str, source: str) -> float:
        return SURFACE_CORRELATIONS.get((target, source), 0.08)

    # ── public aliases ───────────────────────────────────────────────────────
    # These two were renamed to private at some point and the tests were never
    # updated, so 10 of them have been failing on an AttributeError ever since
    # — asserting nothing about behaviour that is, in fact, correct. The
    # implementations are unchanged; only the tested names are restored.

    def get_surface_weight(self, target: str, source: str) -> float:
        """Correlation between two surfaces, 1.0 for the same surface."""
        return self._surface_weight(target, source)

    def apply_time_discount(self, ref_date, past_date,
                            half_life_years: float = 0.8) -> float:
        """Exponential recency weight: 1.0 today, 0.5 one half-life ago."""
        return self._time_decay(ref_date, past_date, half_life_years)

    def _time_decay(self, ref_date: datetime, past_date: datetime,
                    half_life_years: float = 0.8) -> float:
        years = (ref_date - past_date).days / 365.25
        return 0.5 ** (years / half_life_years)

    def _weighted_avg(self, series: pd.Series, weights: pd.Series) -> float:
        mask = series.notna()
        if mask.sum() == 0:
            return 0.0
        return float(np.average(series[mask], weights=weights[mask]))

    # ── core feature computations ─────────────────────────────────────────────

    def _player_performance(self, player_id: int, match_date: datetime,
                             surface: str, lookback_months: int = 36) -> Dict:
        """
        Compute weighted serve/return stats using only pre-match data.
        Temporal isolation: match_date is a strict upper bound.
        """
        cutoff = match_date - timedelta(days=lookback_months * 30)

        # Serve/return stats come from the inline `matches` columns (w_*/l_*),
        # unioned across whichever side the player was on. The previous version
        # read a `statistics` table that is empty in tennis_betting.db, so the
        # extractor silently fell back to tour defaults there. Percentages are
        # derived here (NULLIF guards divide-by-zero) keeping downstream column
        # names/semantics identical. `{d}` is the DB-specific date column.
        d = self.date_col
        query = f"""
            SELECT tournament_date, surface, first_serve_pct, first_serve_win_pct,
                   second_serve_win_pct, break_point_save_pct, aces, double_faults,
                   serve_games, serve_points_total, first_serve_won,
                   second_serve_won, won
            FROM (
                SELECT {d} AS tournament_date, surface,
                    CAST(w_1stIn  AS FLOAT) / NULLIF(w_svpt, 0)          AS first_serve_pct,
                    CAST(w_1stWon AS FLOAT) / NULLIF(w_1stIn, 0)         AS first_serve_win_pct,
                    CAST(w_2ndWon AS FLOAT) / NULLIF(w_svpt - w_1stIn,0) AS second_serve_win_pct,
                    CAST(w_bpSaved AS FLOAT)/ NULLIF(w_bpFaced, 0)       AS break_point_save_pct,
                    w_ace AS aces, w_df AS double_faults, w_SvGms AS serve_games,
                    w_svpt AS serve_points_total, w_1stWon AS first_serve_won,
                    w_2ndWon AS second_serve_won, 1.0 AS won
                FROM matches
                WHERE winner_id = ? AND {d} < ? AND {d} >= ? AND w_svpt > 0
                UNION ALL
                SELECT {d}, surface,
                    CAST(l_1stIn  AS FLOAT) / NULLIF(l_svpt, 0),
                    CAST(l_1stWon AS FLOAT) / NULLIF(l_1stIn, 0),
                    CAST(l_2ndWon AS FLOAT) / NULLIF(l_svpt - l_1stIn, 0),
                    CAST(l_bpSaved AS FLOAT)/ NULLIF(l_bpFaced, 0),
                    l_ace, l_df, l_SvGms, l_svpt, l_1stWon, l_2ndWon, 0.0
                FROM matches
                WHERE loser_id = ? AND {d} < ? AND {d} >= ? AND l_svpt > 0
            )
            ORDER BY tournament_date DESC
        """
        upper = match_date.strftime('%Y-%m-%d')
        lower = cutoff.strftime('%Y-%m-%d')
        df = pd.read_sql_query(query, self.conn,
                               params=(player_id, upper, lower,
                                       player_id, upper, lower))

        if len(df) == 0:
            # No pre-match data: return tour defaults keyed exactly like the main
            # path (extract_features expects e.g. 'bp_save', not 'bp_save_pct').
            dft = self._defaults
            return {
                'wsp': dft['wsp'], 'wrp': 1.0 - dft['wsp'],
                'aces_per_game': dft['aces_per_game'],
                'df_per_game': dft['df_per_game'],
                'bp_save': dft['bp_save_pct'],
                'first_serve_pct': dft['first_serve_pct'],
                'first_serve_win_pct': dft['first_serve_win_pct'],
                'second_serve_win_pct': dft['second_serve_win_pct'],
                'win_rate': 0.50, 'surface_win_rate': 0.50,
                'matches_played': 0, 'surface_matches': 0,
            }

        df['tournament_date'] = pd.to_datetime(df['tournament_date'])

        # Combined weight: time decay × surface transfer
        df['w'] = df.apply(
            lambda r: self._time_decay(match_date, r['tournament_date'])
                      * self._surface_weight(surface, r['surface']),
            axis=1
        )

        wa = lambda col: self._weighted_avg(df[col], df['w'])

        # Serve points won %
        valid_sp = df[df['serve_points_total'].notna() & (df['serve_points_total'] > 0)].copy()
        if len(valid_sp) > 0:
            valid_sp['sp_won'] = (valid_sp['first_serve_won'].fillna(0) +
                                  valid_sp['second_serve_won'].fillna(0)) / valid_sp['serve_points_total']
            wsp = self._weighted_avg(valid_sp['sp_won'], valid_sp['w'])
        else:
            wsp = self._defaults['wsp']

        # Aces & DF per service game
        vg = df[df['serve_games'].notna() & (df['serve_games'] > 0)].copy()
        if len(vg) > 0:
            vg['apg'] = vg['aces'] / vg['serve_games']
            vg['dfpg'] = vg['double_faults'] / vg['serve_games']
            aces_pg = self._weighted_avg(vg['apg'], vg['w'])
            df_pg   = self._weighted_avg(vg['dfpg'], vg['w'])
        else:
            aces_pg = self._defaults['aces_per_game']
            df_pg   = self._defaults['df_per_game']

        surface_df = df[df['surface'] == surface]
        surface_win = (self._weighted_avg(surface_df['won'], surface_df['w'])
                       if len(surface_df) > 0 else wa('won'))

        return {
            'wsp':                   wsp,
            'wrp':                   1.0 - wsp,
            'aces_per_game':         aces_pg,
            'df_per_game':           df_pg,
            'bp_save':               wa('break_point_save_pct') or self._defaults['bp_save_pct'],
            'first_serve_pct':       wa('first_serve_pct')       or self._defaults['first_serve_pct'],
            'first_serve_win_pct':   wa('first_serve_win_pct')   or self._defaults['first_serve_win_pct'],
            'second_serve_win_pct':  wa('second_serve_win_pct')  or self._defaults['second_serve_win_pct'],
            'win_rate':              wa('won'),
            'surface_win_rate':      surface_win,
            'matches_played':        len(df),
            'surface_matches':       len(surface_df),
        }

    def _fatigue(self, player_id: int, match_date: datetime, decay: float = 0.75) -> float:
        cutoff = match_date - timedelta(days=3)
        d = self.date_col
        query = f"""
            SELECT {d} AS tournament_date, minutes
            FROM matches
            WHERE (winner_id = ? OR loser_id = ?)
              AND {d} >= ? AND {d} < ?
        """
        df = pd.read_sql_query(query, self.conn,
                               params=(player_id, player_id,
                                       cutoff.strftime('%Y-%m-%d'),
                                       match_date.strftime('%Y-%m-%d')))
        if len(df) == 0:
            return 0.0
        df['tournament_date'] = pd.to_datetime(df['tournament_date'])
        score = 0.0
        for _, r in df.iterrows():
            mins = r['minutes'] if pd.notna(r['minutes']) and r['minutes'] > 0 else 120
            days_ago = (match_date - r['tournament_date']).days
            score += (mins / 60.0) * (decay ** days_ago)
        return score

    def _h2h(self, p1: int, p2: int, match_date: datetime,
              lookback_months: int = 36) -> Dict:
        cutoff = match_date - timedelta(days=lookback_months * 30)
        d = self.date_col
        query = f"""
            SELECT {d} AS tournament_date,
                   CASE WHEN winner_id = ? THEN 1.0 ELSE 0.0 END as p1_won
            FROM matches
            WHERE ((winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?))
              AND {d} < ? AND {d} >= ?
        """
        df = pd.read_sql_query(query, self.conn,
                               params=(p1, p1, p2, p2, p1,
                                       match_date.strftime('%Y-%m-%d'),
                                       cutoff.strftime('%Y-%m-%d')))
        if len(df) == 0:
            return {'h2h_win_rate': 0.5, 'h2h_matches': 0}
        df['tournament_date'] = pd.to_datetime(df['tournament_date'])
        w = [self._time_decay(match_date, d, half_life_years=1.5)
             for d in df['tournament_date']]
        return {
            'h2h_win_rate': float(np.average(df['p1_won'], weights=w)),
            'h2h_matches':  len(df),
        }

    def _retired(self, player_id: int, match_date: datetime) -> bool:
        d = self.date_col
        query = f"""SELECT MAX({d}) as last FROM matches
                   WHERE (winner_id=? OR loser_id=?) AND {d} < ?"""
        r = pd.read_sql_query(query, self.conn,
                              params=(player_id, player_id,
                                      match_date.strftime('%Y-%m-%d')))
        if r.empty or pd.isna(r.iloc[0]['last']):
            return True
        return (match_date - pd.to_datetime(r.iloc[0]['last'])).days > 90

    def _uncertainty(self, p1f: Dict, p2f: Dict) -> float:
        m  = 1.0 / (1 + min(p1f['matches_played'],  p2f['matches_played'])  / 20)
        s  = 1.0 / (1 + min(p1f['surface_matches'], p2f['surface_matches']) / 10)
        h2 = 1.0 / (1 + p1f.get('h2h_matches', 0) / 3)
        return 0.4 * m + 0.3 * s + 0.3 * h2

    # ── public API ────────────────────────────────────────────────────────────

    def extract_features(self, player1_id: int, player2_id: int,
                         match_date: datetime, surface: str,
                         p1_rank: float = 100, p2_rank: float = 100,
                         p1_points: float = 1000, p2_points: float = 1000,
                         lookback_months: int = 36) -> Dict:
        """
        Extract the full 19+1 feature vector for a match-up.
        All computation is strictly pre-match (no data leakage).
        """
        p1 = self._player_performance(player1_id, match_date, surface, lookback_months)
        p2 = self._player_performance(player2_id, match_date, surface, lookback_months)
        h2h = self._h2h(player1_id, player2_id, match_date, lookback_months)
        p1['h2h_matches'] = h2h['h2h_matches']

        fat1 = self._fatigue(player1_id, match_date)
        fat2 = self._fatigue(player2_id, match_date)
        ret1 = self._retired(player1_id, match_date)
        ret2 = self._retired(player2_id, match_date)

        features = {
            'RANK_DIFF':                  p2_rank   - p1_rank,
            'POINTS_DIFF':                p1_points - p2_points,
            'WSP_DIFF':                   p1['wsp']                  - p2['wsp'],
            'WRP_DIFF':                   p1['wrp']                  - p2['wrp'],
            'ACES_DIFF':                  p1['aces_per_game']        - p2['aces_per_game'],
            'DF_DIFF':                    p2['df_per_game']          - p1['df_per_game'],
            'BP_SAVE_DIFF':               p1['bp_save']              - p2['bp_save'],
            'FIRST_SERVE_PCT_DIFF':       p1['first_serve_pct']      - p2['first_serve_pct'],
            'FIRST_SERVE_WIN_PCT_DIFF':   p1['first_serve_win_pct']  - p2['first_serve_win_pct'],
            'SECOND_SERVE_WIN_PCT_DIFF':  p1['second_serve_win_pct'] - p2['second_serve_win_pct'],
            'WIN_RATE_DIFF':              p1['win_rate']             - p2['win_rate'],
            'SURFACE_WIN_RATE_DIFF':      p1['surface_win_rate']     - p2['surface_win_rate'],
            'SERVEADV':      (p1['wsp'] - p2['wrp']) - (p2['wsp'] - p1['wrp']),
            'COMPLETE_DIFF': (p1['wsp'] * p1['wrp']) - (p2['wsp'] * p2['wrp']),
            'FATIGUE_DIFF':  fat2 - fat1,
            'RETIRED_DIFF':  int(ret2) - int(ret1),
            'DIRECT_H2H':    h2h['h2h_win_rate'] - 0.5,
            'MATCHES_PLAYED_DIFF': p1['matches_played'] - p2['matches_played'],
            'SURFACE_EXP_DIFF':    p1['surface_matches'] - p2['surface_matches'],
            'UNCERTAINTY':         self._uncertainty(p1, p2),
            # raw values for Markov / simulation
            '_p1_wsp': p1['wsp'], '_p2_wsp': p2['wsp'],
            '_p1_bp_save': p1['bp_save'], '_p2_bp_save': p2['bp_save'],
            '_p1_first_serve_pct': p1['first_serve_pct'],
            '_p2_first_serve_pct': p2['first_serve_pct'],
        }
        # Tier 1: rally-construction diffs (first-strike vs grinding, shot quality)
        features.update(self._rally_diffs(player1_id, player2_id, match_date))
        return features

    def extract_features_batch(self, match_ids: Optional[List[int]] = None,
                               lookback_months: int = 36,
                               uncertainty_threshold: float = 0.7) -> pd.DataFrame:
        if match_ids is None:
            match_ids = pd.read_sql_query(
                "SELECT match_id FROM matches ORDER BY tournament_date",
                self.conn)['match_id'].tolist()

        rows, skipped_u, skipped_e = [], 0, 0
        for i, mid in enumerate(match_ids):
            try:
                q = """SELECT winner_id, loser_id, tournament_date, surface,
                              winner_rank, loser_rank, winner_rank_points, loser_rank_points
                       FROM matches WHERE match_id=?"""
                m = pd.read_sql_query(q, self.conn, params=(mid,))
                if len(m) == 0:
                    continue
                r = m.iloc[0]
                feats = self.extract_features(
                    int(r.winner_id), int(r.loser_id),
                    pd.to_datetime(r.tournament_date), r.surface,
                    r.winner_rank or 100, r.loser_rank or 100,
                    r.winner_rank_points or 1000, r.loser_rank_points or 1000,
                    lookback_months)
                feats['match_id'] = mid
                feats['label'] = 1   # winner is always player1 in training
                if feats['UNCERTAINTY'] <= uncertainty_threshold:
                    rows.append(feats)
                else:
                    skipped_u += 1
            except Exception as e:
                logger.warning(f"Match {mid}: {e}")
                skipped_e += 1
            if (i + 1) % 500 == 0:
                logger.info(f"  {i+1}/{len(match_ids)} processed")

        logger.info(f"Done — {len(rows)} kept, {skipped_u} high-uncertainty, {skipped_e} errors")
        return pd.DataFrame(rows)
