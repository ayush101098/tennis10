"""
🎾 Tennis Betting Intelligence Hub
===================================
A beautiful, easy-to-navigate betting analysis platform

Run: streamlit run live_betting_app.py
"""

import streamlit as st
import numpy as np
from scipy.special import comb
import sqlite3
import os
from pathlib import Path
from datetime import datetime
import json

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🎾 Tennis Betting Hub",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PREMIUM CSS ====================
st.markdown("""
<style>
    /* Navigation pills */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 10px;
        padding: 20px;
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        margin-bottom: 20px;
    }
    
    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }
    .hero-header h1 {
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .hero-subtitle {
        opacity: 0.9;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    
    /* Score display */
    .live-score {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.4);
        letter-spacing: 3px;
    }
    .server-indicator {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 10px;
    }
    
    /* Value cards */
    .value-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.4);
        transition: transform 0.3s ease;
    }
    .value-card:hover { transform: translateY(-5px); }
    .value-card .label { font-size: 0.9rem; opacity: 0.9; }
    .value-card .value { font-size: 2rem; font-weight: bold; margin: 10px 0; }
    .value-card .subtext { font-size: 0.85rem; opacity: 0.8; }
    
    .no-value-card {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(203, 45, 62, 0.4);
    }
    .no-value-card .label { font-size: 0.9rem; opacity: 0.9; }
    .no-value-card .value { font-size: 2rem; font-weight: bold; margin: 10px 0; }
    
    .neutral-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Stat cards */
    .stat-box {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    .stat-box .number { font-size: 2.5rem; font-weight: bold; }
    .stat-box .label { font-size: 0.9rem; opacity: 0.8; margin-top: 5px; }
    
    /* Break point alert */
    .break-alert {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 15px 0;
        animation: pulse 2s infinite;
        box-shadow: 0 0 30px rgba(255, 65, 108, 0.6);
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 30px rgba(255, 65, 108, 0.6); }
        50% { transform: scale(1.02); box-shadow: 0 0 50px rgba(255, 65, 108, 0.8); }
    }
    
    /* Bet history */
    .bet-won {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.2) 0%, rgba(56, 239, 125, 0.2) 100%);
        border-left: 4px solid #38ef7d;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .bet-lost {
        background: linear-gradient(135deg, rgba(203, 45, 62, 0.2) 0%, rgba(239, 71, 58, 0.2) 100%);
        border-left: 4px solid #ef473a;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .bet-pending {
        background: rgba(255,255,255,0.05);
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Momentum indicator */
    .momentum-bar {
        height: 10px;
        background: linear-gradient(90deg, #ef473a 0%, #ffc107 50%, #38ef7d 100%);
        border-radius: 5px;
        position: relative;
        margin: 20px 0;
    }
    .momentum-marker {
        width: 20px;
        height: 20px;
        background: white;
        border-radius: 50%;
        position: absolute;
        top: -5px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        transition: left 0.3s ease;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# ==================== DATABASE FUNCTIONS ====================
DB_PATH = Path(__file__).parent / "tennis_data.db"

@st.cache_resource
def get_db_connection():
    if os.path.exists(DB_PATH):
        return sqlite3.connect(str(DB_PATH), check_same_thread=False)
    return None

def search_player(name: str, limit: int = 5):
    conn = get_db_connection()
    if not conn or len(name) < 2:
        return []
    try:
        query = """
        SELECT DISTINCT p.player_id, p.name, p.country
        FROM players p
        WHERE LOWER(p.name) LIKE LOWER(?)
        LIMIT ?
        """
        cursor = conn.execute(query, (f"%{name}%", limit))
        return [{'id': r[0], 'name': r[1], 'country': r[2] or ''} for r in cursor.fetchall()]
    except:
        return []

def get_player_serve_stats(player_id: int, surface: str = None):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        surface_filter = f"AND m.surface = '{surface}'" if surface and surface != "All" else ""
        query = f"""
        SELECT 
            AVG(s.first_serve_pct) as first_serve_pct,
            AVG(s.first_serve_win_pct) as first_serve_win_pct,
            AVG(s.second_serve_win_pct) as second_serve_win_pct,
            AVG(CAST(s.aces AS FLOAT) / NULLIF(s.serve_games, 0)) as aces_per_game,
            AVG(CAST(s.double_faults AS FLOAT) / NULLIF(s.serve_games, 0)) as df_per_game,
            COUNT(*) as match_count
        FROM statistics s
        JOIN matches m ON s.match_id = m.match_id
        WHERE s.player_id = ?
            AND m.tournament_date >= date('now', '-365 days')
            AND s.first_serve_pct IS NOT NULL
            {surface_filter}
        """
        cursor = conn.execute(query, (player_id,))
        row = cursor.fetchone()
        if row and row[5] >= 3:
            first_pct = row[0] or 0.62
            first_win = row[1] or 0.70
            second_win = row[2] or 0.50
            serve_point_pct = first_pct * first_win + (1 - first_pct) * 0.95 * second_win
            return {
                'first_serve_pct': row[0], 'first_serve_win_pct': row[1],
                'second_serve_win_pct': row[2], 'aces_per_game': row[3],
                'df_per_game': row[4], 'match_count': row[5],
                'serve_point_pct': serve_point_pct
            }
        return None
    except:
        return None

def get_h2h_record(player1_id: int, player2_id: int):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        query = """
        SELECT 
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as p1_wins,
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as p2_wins
        FROM matches
        WHERE (winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)
        """
        cursor = conn.execute(query, (player1_id, player2_id, player1_id, player2_id, player2_id, player1_id))
        row = cursor.fetchone()
        if row and (row[0] or row[1]):
            return {'p1_wins': row[0] or 0, 'p2_wins': row[1] or 0}
        return None
    except:
        return None

def get_db_stats():
    conn = get_db_connection()
    if not conn:
        return None
    try:
        stats = {}
        stats['matches'] = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        stats['players'] = conn.execute("SELECT COUNT(DISTINCT player_id) FROM players").fetchone()[0]
        return stats
    except:
        return None


def get_recent_form(player_id: int, limit: int = 5):
    """Get last N matches for a player with results."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        query = """
        SELECT 
            m.tournament_name,
            m.surface,
            CASE WHEN m.winner_id = ? THEN 'W' ELSE 'L' END as result,
            CASE WHEN m.winner_id = ? THEN p2.name ELSE p1.name END as opponent,
            m.score,
            m.tournament_date
        FROM matches m
        JOIN players p1 ON m.winner_id = p1.player_id
        JOIN players p2 ON m.loser_id = p2.player_id
        WHERE m.winner_id = ? OR m.loser_id = ?
        ORDER BY m.tournament_date DESC
        LIMIT ?
        """
        cursor = conn.execute(query, (player_id, player_id, player_id, player_id, limit))
        rows = cursor.fetchall()
        if rows:
            return [{
                'tournament': r[0] or 'Unknown',
                'surface': r[1] or 'Hard',
                'result': r[2],
                'opponent': r[3],
                'score': r[4] or '',
                'date': r[5]
            } for r in rows]
        return None
    except:
        return None


def get_surface_win_rates(player_id: int):
    """Get win rates by surface."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        query = """
        SELECT 
            m.surface,
            SUM(CASE WHEN m.winner_id = ? THEN 1 ELSE 0 END) as wins,
            COUNT(*) as total
        FROM matches m
        WHERE (m.winner_id = ? OR m.loser_id = ?)
            AND m.tournament_date >= date('now', '-730 days')
        GROUP BY m.surface
        HAVING total >= 3
        """
        cursor = conn.execute(query, (player_id, player_id, player_id))
        rows = cursor.fetchall()
        if rows:
            return {r[0]: {'wins': r[1], 'total': r[2], 'pct': r[1]/r[2] if r[2] > 0 else 0} for r in rows if r[0]}
        return None
    except:
        return None


def get_hold_break_stats(player_id: int, surface: str = None):
    """Get average hold and break percentages."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        surface_filter = f"AND m.surface = '{surface}'" if surface and surface != "All" else ""
        query = f"""
        SELECT 
            AVG(s.service_games_won_pct) as hold_pct,
            AVG(s.return_games_won_pct) as break_pct,
            AVG(s.break_points_saved_pct) as bp_saved_pct,
            AVG(s.break_points_converted_pct) as bp_converted_pct,
            COUNT(*) as matches
        FROM statistics s
        JOIN matches m ON s.match_id = m.match_id
        WHERE s.player_id = ?
            AND m.tournament_date >= date('now', '-365 days')
            {surface_filter}
        """
        cursor = conn.execute(query, (player_id,))
        row = cursor.fetchone()
        if row and row[4] >= 3:
            return {
                'hold_pct': row[0],
                'break_pct': row[1],
                'bp_saved_pct': row[2],
                'bp_converted_pct': row[3],
                'matches': row[4]
            }
        return None
    except:
        return None


def get_h2h_details(player1_id: int, player2_id: int):
    """Get detailed H2H with recent matches."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        # Overall record
        query = """
        SELECT 
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as p1_wins,
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as p2_wins
        FROM matches
        WHERE (winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)
        """
        cursor = conn.execute(query, (player1_id, player2_id, player1_id, player2_id, player2_id, player1_id))
        row = cursor.fetchone()
        
        if not row or (not row[0] and not row[1]):
            return None
        
        result = {'p1_wins': row[0] or 0, 'p2_wins': row[1] or 0, 'matches': []}
        
        # Recent H2H matches
        query2 = """
        SELECT 
            m.tournament_name,
            m.surface,
            p1.name as winner,
            m.score,
            m.tournament_date
        FROM matches m
        JOIN players p1 ON m.winner_id = p1.player_id
        WHERE (m.winner_id = ? AND m.loser_id = ?) OR (m.winner_id = ? AND m.loser_id = ?)
        ORDER BY m.tournament_date DESC
        LIMIT 5
        """
        cursor2 = conn.execute(query2, (player1_id, player2_id, player2_id, player1_id))
        for r in cursor2.fetchall():
            result['matches'].append({
                'tournament': r[0] or 'Unknown',
                'surface': r[1] or 'Hard',
                'winner': r[2],
                'score': r[3] or '',
                'date': r[4]
            })
        
        return result
    except:
        return None


# ==================== PROBABILITY FUNCTIONS ====================

def p_game_from_points(server_pts: int, returner_pts: int, p_point: float) -> float:
    if server_pts >= 4 and server_pts >= returner_pts + 2:
        return 1.0
    if returner_pts >= 4 and returner_pts >= server_pts + 2:
        return 0.0
    
    if server_pts >= 3 and returner_pts >= 3:
        p_d = (p_point ** 2) / (p_point ** 2 + (1 - p_point) ** 2)
        if server_pts == returner_pts:
            return p_d
        elif server_pts > returner_pts:
            return p_point + (1 - p_point) * p_d
        else:
            return p_point * p_d
    
    cache = {}
    def prob_from(s, r):
        if (s, r) in cache:
            return cache[(s, r)]
        if s == 4 and r <= 2:
            return 1.0
        if r == 4 and s <= 2:
            return 0.0
        if s >= 3 and r >= 3:
            p_d = (p_point ** 2) / (p_point ** 2 + (1 - p_point) ** 2)
            return p_d if s == r else (p_point + (1-p_point)*p_d if s > r else p_point*p_d)
        result = p_point * prob_from(min(s+1, 4), r) + (1-p_point) * prob_from(s, min(r+1, 4))
        cache[(s, r)] = result
        return result
    return prob_from(server_pts, returner_pts)


def p_win_game(p_point: float) -> float:
    p, q = p_point, 1 - p_point
    p_before_deuce = p**4 * (1 + 4*q + 10*q**2)
    p_deuce = comb(6, 3) * p**3 * q**3 * (p**2 / (p**2 + q**2))
    return p_before_deuce + p_deuce


def p_set_from_games(g1: int, g2: int, server: int, p1_serve: float, p2_serve: float) -> float:
    p_hold_p1 = p_win_game(p1_serve)
    p_hold_p2 = p_win_game(p2_serve)
    
    cache = {}
    def prob_from(g1, g2, srv):
        if (g1, g2, srv) in cache:
            return cache[(g1, g2, srv)]
        if g1 >= 6 and g1 >= g2 + 2: return 1.0
        if g2 >= 6 and g2 >= g1 + 2: return 0.0
        if g1 == 7: return 1.0
        if g2 == 7: return 0.0
        if g1 == 6 and g2 == 6:
            p_avg = (p1_serve + (1 - p2_serve)) / 2
            return (p_avg ** 2) / (p_avg ** 2 + (1 - p_avg) ** 2)
        p_p1_wins = p_hold_p1 if srv == 1 else (1 - p_hold_p2)
        result = p_p1_wins * prob_from(g1+1, g2, 3-srv) + (1-p_p1_wins) * prob_from(g1, g2+1, 3-srv)
        cache[(g1, g2, srv)] = result
        return result
    return prob_from(g1, g2, server)


def p_match_from_sets(s1: int, s2: int, p_set: float, best_of: int = 3) -> float:
    sets_to_win = (best_of + 1) // 2
    cache = {}
    def prob_from(s1, s2):
        if (s1, s2) in cache: return cache[(s1, s2)]
        if s1 >= sets_to_win: return 1.0
        if s2 >= sets_to_win: return 0.0
        result = p_set * prob_from(s1+1, s2) + (1-p_set) * prob_from(s1, s2+1)
        cache[(s1, s2)] = result
        return result
    return prob_from(s1, s2)


def adjust_serve_for_surface(base_serve: float, surface: str, serve_speed: str) -> float:
    surface_adj = {'Hard': 0, 'Clay': -0.03, 'Grass': +0.03, 'Indoor': +0.02, 'All': 0}
    speed_adj = {'Average': 0, 'Big Server (+4%)': +0.04, 'Weak Server (-3%)': -0.03}
    adjusted = base_serve + surface_adj.get(surface, 0) + speed_adj.get(serve_speed, 0)
    return np.clip(adjusted, 0.45, 0.80)


# ==================== SESSION STATE ====================
defaults = {
    'p1': 'Player 1', 'p2': 'Player 2',
    'p1_serve': 62, 'p2_serve': 62,
    'sets': [0, 0], 'games': [0, 0], 'points': [0, 0],
    'server': 1, 'best_of': 3,
    'surface': 'Hard',
    'p1_speed': 'Average', 'p2_speed': 'Average',
    'p1_id': None, 'p2_id': None,
    'p1_stats': None, 'p2_stats': None, 'h2h': None,
    'bankroll': 100.0,
    'game_hold_odds': 1.20, 'game_break_odds': 4.50,
    'set_p1_odds': 1.80, 'set_p2_odds': 2.00,
    'match_p1_odds': 1.50, 'match_p2_odds': 2.50,
    'bets': [],
    'total_staked': 0.0,
    'total_profit': 0.0,
    'point_history': [],
    'games_history': [],
    'current_page': 'Live Match',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==================== HELPER FUNCTIONS ====================

def get_point_display(pts):
    if pts == 0: return '0'
    elif pts == 1: return '15'
    elif pts == 2: return '30'
    elif pts == 3: return '40'
    else: return 'AD'

def get_score_string():
    p1_pt = get_point_display(st.session_state.points[0])
    p2_pt = get_point_display(st.session_state.points[1])
    
    if st.session_state.points[0] >= 3 and st.session_state.points[1] >= 3:
        if st.session_state.points[0] == st.session_state.points[1]:
            point_str = "DEUCE"
        elif st.session_state.points[0] > st.session_state.points[1]:
            point_str = f"AD-{st.session_state.p1[:3].upper()}"
        else:
            point_str = f"AD-{st.session_state.p2[:3].upper()}"
    else:
        point_str = f"{p1_pt}-{p2_pt}"
    
    return f"{st.session_state.sets[0]}-{st.session_state.sets[1]}  •  {st.session_state.games[0]}-{st.session_state.games[1]}  •  {point_str}"

def record_point(winner: int):
    st.session_state.point_history.append(winner)
    st.session_state.points[winner - 1] += 1
    
    p1, p2 = st.session_state.points
    game_won = None
    
    if p1 >= 4 and p1 >= p2 + 2:
        game_won = 1
    elif p2 >= 4 and p2 >= p1 + 2:
        game_won = 2
    
    if game_won:
        st.session_state.points = [0, 0]
        st.session_state.games[game_won - 1] += 1
        st.session_state.games_history.append({
            'winner': game_won,
            'server': st.session_state.server,
            'was_break': game_won != st.session_state.server
        })
        st.session_state.server = 3 - st.session_state.server
        
        g1, g2 = st.session_state.games
        if (g1 >= 6 and g1 >= g2 + 2) or g1 == 7:
            st.session_state.sets[0] += 1
            st.session_state.games = [0, 0]
        elif (g2 >= 6 and g2 >= g1 + 2) or g2 == 7:
            st.session_state.sets[1] += 1
            st.session_state.games = [0, 0]

def record_bet(bet_type: str, selection: str, odds: float, stake: float, model_prob: float):
    st.session_state.bets.append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'type': bet_type,
        'selection': selection,
        'odds': odds,
        'stake': stake,
        'model_prob': model_prob,
        'ev': (model_prob * odds - 1) * stake,
        'result': None,
        'profit': None
    })
    st.session_state.total_staked += stake

def settle_bet(idx: int, won: bool):
    bet = st.session_state.bets[idx]
    if won:
        bet['result'] = 'WON'
        bet['profit'] = bet['stake'] * (bet['odds'] - 1)
    else:
        bet['result'] = 'LOST'
        bet['profit'] = -bet['stake']
    st.session_state.total_profit += bet['profit']

def get_momentum():
    if len(st.session_state.point_history) < 3:
        return 0.5, "Even", "⚖️"
    recent = st.session_state.point_history[-5:]
    p1_won = sum(1 for p in recent if p == 1)
    ratio = p1_won / len(recent)
    if ratio >= 0.7:
        return ratio, st.session_state.p1, "🔥"
    elif ratio <= 0.3:
        return ratio, st.session_state.p2, "🔥"
    else:
        return ratio, "Even", "⚖️"

def is_break_point():
    server = st.session_state.server
    s_pts = st.session_state.points[server - 1]
    r_pts = st.session_state.points[2 - server]
    if r_pts >= 3 and r_pts > s_pts:
        return True
    return False

def get_break_point_count():
    server = st.session_state.server
    s_pts = st.session_state.points[server - 1]
    r_pts = st.session_state.points[2 - server]
    if r_pts < 3:
        return 0
    if s_pts < 3:
        return r_pts - 2
    return 1


# ==================== SIDEBAR NAVIGATION ====================
with st.sidebar:
    st.markdown("## 🎾 Navigation")
    
    pages = {
        "🎯 Live Match": "Live Match",
        "📊 Bet Analysis": "Bet Analysis", 
        "📈 Statistics": "Statistics",
        "⚙️ Settings": "Settings"
    }
    
    for label, page in pages.items():
        if st.button(label, use_container_width=True, 
                     type="primary" if st.session_state.current_page == page else "secondary"):
            st.session_state.current_page = page
            st.rerun()
    
    st.markdown("---")
    
    # Quick stats
    db_stats = get_db_stats()
    if db_stats:
        st.markdown("### 📚 Database")
        st.metric("Matches", f"{db_stats['matches']:,}")
        st.metric("Players", f"{db_stats['players']:,}")
    
    st.markdown("---")
    
    # Session P&L
    profit_color = "green" if st.session_state.total_profit >= 0 else "red"
    st.markdown(f"""
    ### 💰 Session P&L
    <h2 style='color: {profit_color};'>${st.session_state.total_profit:+.2f}</h2>
    <p style='opacity: 0.7;'>Staked: ${st.session_state.total_staked:.2f}</p>
    """, unsafe_allow_html=True)


# ==================== PAGE: LIVE MATCH ====================
if st.session_state.current_page == "Live Match":
    # Hero header
    st.markdown(f"""
    <div class="hero-header">
        <h1>🎾 {st.session_state.p1} vs {st.session_state.p2}</h1>
        <div class="hero-subtitle">{st.session_state.surface} Court • Best of {st.session_state.best_of}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # === PLAYER INSIGHTS FROM DATABASE ===
    if st.session_state.p1_id and st.session_state.p2_id:
        h2h = get_h2h_details(st.session_state.p1_id, st.session_state.p2_id)
        if h2h:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2d3436 0%, #000000 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: white; text-align: center; margin: 0;">⚔️ Head to Head</h3>
                <div style="display: flex; justify-content: center; align-items: center; gap: 30px; margin-top: 15px;">
                    <div style="text-align: center;">
                        <div style="color: #38ef7d; font-size: 2.5rem; font-weight: bold;">{h2h['p1_wins']}</div>
                        <div style="color: white; opacity: 0.8;">{st.session_state.p1[:15]}</div>
                    </div>
                    <div style="color: white; font-size: 1.5rem;">-</div>
                    <div style="text-align: center;">
                        <div style="color: #ff6b6b; font-size: 2.5rem; font-weight: bold;">{h2h['p2_wins']}</div>
                        <div style="color: white; opacity: 0.8;">{st.session_state.p2[:15]}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show player insights in expandable section
    with st.expander("📊 Player Database Insights", expanded=False):
        pi_col1, pi_col2 = st.columns(2)
        
        with pi_col1:
            st.markdown(f"#### {st.session_state.p1}")
            if st.session_state.p1_id:
                # Recent form
                form = get_recent_form(st.session_state.p1_id, 5)
                if form:
                    form_str = ' '.join(['🟢' if m['result'] == 'W' else '🔴' for m in form])
                    wins = sum(1 for m in form if m['result'] == 'W')
                    st.markdown(f"**Recent Form:** {form_str} ({wins}/5)")
                    for m in form[:3]:
                        st.caption(f"{m['result']} vs {m['opponent'][:15]} ({m['surface']})")
                
                # Surface win rates
                surface_stats = get_surface_win_rates(st.session_state.p1_id)
                if surface_stats:
                    st.markdown("**Surface Win Rates:**")
                    for surf, data in surface_stats.items():
                        emoji = '🏟️' if surf == 'Hard' else ('🧱' if surf == 'Clay' else '🌿')
                        st.markdown(f"{emoji} {surf}: **{data['pct']:.0%}** ({data['wins']}/{data['total']})")
                
                # Hold/Break stats
                hb_stats = get_hold_break_stats(st.session_state.p1_id, st.session_state.surface)
                if hb_stats:
                    st.markdown(f"**On {st.session_state.surface}:**")
                    if hb_stats['hold_pct']:
                        st.markdown(f"Hold: **{hb_stats['hold_pct']:.0%}** | Break: **{hb_stats['break_pct']:.0%}**")
                    if hb_stats['bp_saved_pct']:
                        st.markdown(f"BP Saved: {hb_stats['bp_saved_pct']:.0%} | BP Conv: {hb_stats['bp_converted_pct']:.0%}")
            else:
                st.caption("Player not in database - using manual stats")
        
        with pi_col2:
            st.markdown(f"#### {st.session_state.p2}")
            if st.session_state.p2_id:
                # Recent form
                form = get_recent_form(st.session_state.p2_id, 5)
                if form:
                    form_str = ' '.join(['🟢' if m['result'] == 'W' else '🔴' for m in form])
                    wins = sum(1 for m in form if m['result'] == 'W')
                    st.markdown(f"**Recent Form:** {form_str} ({wins}/5)")
                    for m in form[:3]:
                        st.caption(f"{m['result']} vs {m['opponent'][:15]} ({m['surface']})")
                
                # Surface win rates
                surface_stats = get_surface_win_rates(st.session_state.p2_id)
                if surface_stats:
                    st.markdown("**Surface Win Rates:**")
                    for surf, data in surface_stats.items():
                        emoji = '🏟️' if surf == 'Hard' else ('🧱' if surf == 'Clay' else '🌿')
                        st.markdown(f"{emoji} {surf}: **{data['pct']:.0%}** ({data['wins']}/{data['total']})")
                
                # Hold/Break stats
                hb_stats = get_hold_break_stats(st.session_state.p2_id, st.session_state.surface)
                if hb_stats:
                    st.markdown(f"**On {st.session_state.surface}:**")
                    if hb_stats['hold_pct']:
                        st.markdown(f"Hold: **{hb_stats['hold_pct']:.0%}** | Break: **{hb_stats['break_pct']:.0%}**")
                    if hb_stats['bp_saved_pct']:
                        st.markdown(f"BP Saved: {hb_stats['bp_saved_pct']:.0%} | BP Conv: {hb_stats['bp_converted_pct']:.0%}")
            else:
                st.caption("Player not in database - using manual stats")
        
        # H2H Details
        if st.session_state.p1_id and st.session_state.p2_id:
            h2h = get_h2h_details(st.session_state.p1_id, st.session_state.p2_id)
            if h2h and h2h['matches']:
                st.markdown("---")
                st.markdown("#### ⚔️ Recent H2H Matches")
                for m in h2h['matches']:
                    st.markdown(f"**{m['winner']}** won | {m['tournament']} ({m['surface']}) | {m['score']}")
    
    # Live score
    server_name = st.session_state.p1 if st.session_state.server == 1 else st.session_state.p2
    st.markdown(f"""
    <div class="live-score">
        {get_score_string()}
        <div class="server-indicator">🎾 {server_name} serving</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Break point alert
    if is_break_point():
        bp_count = get_break_point_count()
        returner = st.session_state.p2 if st.session_state.server == 1 else st.session_state.p1
        st.markdown(f"""
        <div class="break-alert">
            🔴 BREAK POINT{'S' if bp_count > 1 else ''} ×{bp_count} — {returner}!
        </div>
        """, unsafe_allow_html=True)
    
    # Quick point buttons
    st.markdown("### ⚡ Score a Point")
    btn_cols = st.columns([2, 2, 1, 1])
    
    with btn_cols[0]:
        if st.button(f"✅ {st.session_state.p1}", type="primary", use_container_width=True):
            record_point(1)
            st.rerun()
    
    with btn_cols[1]:
        if st.button(f"✅ {st.session_state.p2}", type="primary", use_container_width=True):
            record_point(2)
            st.rerun()
    
    with btn_cols[2]:
        if st.button("↩️ Undo", use_container_width=True):
            if st.session_state.point_history:
                last = st.session_state.point_history.pop()
                if st.session_state.points[last - 1] > 0:
                    st.session_state.points[last - 1] -= 1
            st.rerun()
    
    with btn_cols[3]:
        if st.button("🔄 Swap", use_container_width=True):
            st.session_state.server = 3 - st.session_state.server
            st.rerun()
    
    st.markdown("---")
    
    # === ODDS INPUT & VALUE ANALYSIS ===
    st.markdown("### 💰 Live Game Odds")
    
    server_name = st.session_state.p1 if st.session_state.server == 1 else st.session_state.p2
    returner_name = st.session_state.p2 if st.session_state.server == 1 else st.session_state.p1
    
    oc1, oc2 = st.columns(2)
    with oc1:
        st.session_state.game_hold_odds = st.number_input(
            f"🟢 {server_name} HOLD", 1.01, 20.0,
            float(st.session_state.game_hold_odds), 0.05, key="hold_odds"
        )
    with oc2:
        st.session_state.game_break_odds = st.number_input(
            f"🔴 {returner_name} BREAK", 1.01, 20.0,
            float(st.session_state.game_break_odds), 0.05, key="break_odds"
        )
    
    # Calculate probabilities
    p1_serve = st.session_state.p1_serve / 100
    p2_serve = st.session_state.p2_serve / 100
    
    if st.session_state.server == 1:
        server_pts, returner_pts = st.session_state.points
        p_serve = p1_serve
    else:
        server_pts, returner_pts = st.session_state.points[1], st.session_state.points[0]
        p_serve = p2_serve
    
    p_hold = p_game_from_points(server_pts, returner_pts, p_serve)
    p_break = 1 - p_hold
    
    # Fair odds & edge
    hold_implied = 1 / st.session_state.game_hold_odds
    break_implied = 1 / st.session_state.game_break_odds
    total_implied = hold_implied + break_implied
    hold_fair = hold_implied / total_implied
    break_fair = break_implied / total_implied
    
    edge_hold = p_hold - hold_fair
    edge_break = p_break - break_fair
    
    kelly_hold = max(0, (p_hold * st.session_state.game_hold_odds - 1) / (st.session_state.game_hold_odds - 1)) if edge_hold > 0 else 0
    kelly_break = max(0, (p_break * st.session_state.game_break_odds - 1) / (st.session_state.game_break_odds - 1)) if edge_break > 0 else 0
    
    # Value cards
    st.markdown("### 🎯 Value Analysis")
    val_cols = st.columns(2)
    
    with val_cols[0]:
        fair_hold = 1/p_hold if p_hold > 0.01 else 99
        if edge_hold > 0.03:
            stake = kelly_hold * 0.25 * st.session_state.bankroll
            st.markdown(f"""
            <div class="value-card">
                <div class="label">🟢 {server_name} HOLD</div>
                <div class="value">+{edge_hold:.1%} EDGE</div>
                <div class="subtext">Model: {p_hold:.1%} | Fair: {fair_hold:.2f} | Book: {st.session_state.game_hold_odds}</div>
                <div class="subtext" style="margin-top: 10px;">💵 Stake: ${stake:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📝 Record Hold Bet", key="rec_hold", use_container_width=True):
                record_bet("Game", f"{server_name} Hold", st.session_state.game_hold_odds, stake, p_hold)
                st.success(f"✅ Bet recorded: ${stake:.2f} @ {st.session_state.game_hold_odds}")
        elif edge_hold > 0:
            st.markdown(f"""
            <div class="neutral-card">
                <div class="label">🟢 {server_name} HOLD</div>
                <div class="value">+{edge_hold:.1%}</div>
                <div class="subtext">Marginal edge - small value</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="no-value-card">
                <div class="label">🟢 {server_name} HOLD</div>
                <div class="value">{edge_hold:+.1%}</div>
                <div class="subtext">No value</div>
            </div>
            """, unsafe_allow_html=True)
    
    with val_cols[1]:
        fair_break = 1/p_break if p_break > 0.01 else 99
        if edge_break > 0.03:
            stake = kelly_break * 0.25 * st.session_state.bankroll
            st.markdown(f"""
            <div class="value-card">
                <div class="label">🔴 {returner_name} BREAK</div>
                <div class="value">+{edge_break:.1%} EDGE</div>
                <div class="subtext">Model: {p_break:.1%} | Fair: {fair_break:.2f} | Book: {st.session_state.game_break_odds}</div>
                <div class="subtext" style="margin-top: 10px;">💵 Stake: ${stake:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📝 Record Break Bet", key="rec_break", use_container_width=True):
                record_bet("Game", f"{returner_name} Break", st.session_state.game_break_odds, stake, p_break)
                st.success(f"✅ Bet recorded: ${stake:.2f} @ {st.session_state.game_break_odds}")
        elif edge_break > 0:
            st.markdown(f"""
            <div class="neutral-card">
                <div class="label">🔴 {returner_name} BREAK</div>
                <div class="value">+{edge_break:.1%}</div>
                <div class="subtext">Marginal edge - small value</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="no-value-card">
                <div class="label">🔴 {returner_name} BREAK</div>
                <div class="value">{edge_break:+.1%}</div>
                <div class="subtext">No value</div>
            </div>
            """, unsafe_allow_html=True)


# ==================== PAGE: BET ANALYSIS ====================
elif st.session_state.current_page == "Bet Analysis":
    st.markdown("""
    <div class="hero-header">
        <h1>📊 Bet Analysis</h1>
        <div class="hero-subtitle">Deep dive into probabilities and optimal stakes</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different analyses
    analysis_tabs = st.tabs(["🎯 Game Bets", "📊 Set/Match", "📈 Probability Map"])
    
    with analysis_tabs[0]:
        st.markdown("### Current Game Analysis")
        
        p1_serve = st.session_state.p1_serve / 100
        p2_serve = st.session_state.p2_serve / 100
        
        if st.session_state.server == 1:
            server_pts, returner_pts = st.session_state.points
            p_serve = p1_serve
            server_name = st.session_state.p1
            returner_name = st.session_state.p2
        else:
            server_pts, returner_pts = st.session_state.points[1], st.session_state.points[0]
            p_serve = p2_serve
            server_name = st.session_state.p2
            returner_name = st.session_state.p1
        
        p_hold = p_game_from_points(server_pts, returner_pts, p_serve)
        p_break = 1 - p_hold
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Hold Prob", f"{p_hold:.1%}")
        with col2:
            st.metric("Break Prob", f"{p_break:.1%}")
        with col3:
            fair_hold = 1/p_hold if p_hold > 0.01 else 99
            st.metric("Fair Hold Odds", f"{fair_hold:.2f}")
        with col4:
            fair_break = 1/p_break if p_break > 0.01 else 99
            st.metric("Fair Break Odds", f"{fair_break:.2f}")
        
        st.markdown("---")
        
        # Kelly calculator
        st.markdown("### 🧮 Kelly Stake Calculator")
        
        kc1, kc2 = st.columns(2)
        with kc1:
            custom_prob = st.slider("Your probability estimate", 0.01, 0.99, p_hold, 0.01)
        with kc2:
            custom_odds = st.number_input("Bookmaker odds", 1.01, 20.0, 2.0, 0.05)
        
        if custom_prob * custom_odds > 1:
            kelly_full = (custom_prob * custom_odds - 1) / (custom_odds - 1)
            kelly_quarter = kelly_full * 0.25
            
            st.success(f"""
            **Value bet detected!**
            - Edge: **{(custom_prob - 1/custom_odds):.1%}**
            - Full Kelly: **{kelly_full:.1%}** of bankroll (${kelly_full * st.session_state.bankroll:.2f})
            - Quarter Kelly (recommended): **{kelly_quarter:.1%}** (${kelly_quarter * st.session_state.bankroll:.2f})
            """)
        else:
            st.warning("No value at these odds. Expected value is negative.")
    
    with analysis_tabs[1]:
        st.markdown("### Set & Match Probabilities")
        
        # Set odds input
        sc1, sc2 = st.columns(2)
        with sc1:
            st.session_state.set_p1_odds = st.number_input(f"{st.session_state.p1} wins SET", 1.01, 20.0, float(st.session_state.set_p1_odds), 0.05)
        with sc2:
            st.session_state.set_p2_odds = st.number_input(f"{st.session_state.p2} wins SET", 1.01, 20.0, float(st.session_state.set_p2_odds), 0.05)
        
        # Match odds input
        mc1, mc2 = st.columns(2)
        with mc1:
            st.session_state.match_p1_odds = st.number_input(f"{st.session_state.p1} wins MATCH", 1.01, 50.0, float(st.session_state.match_p1_odds), 0.05)
        with mc2:
            st.session_state.match_p2_odds = st.number_input(f"{st.session_state.p2} wins MATCH", 1.01, 50.0, float(st.session_state.match_p2_odds), 0.05)
        
        # Calculate
        p1_serve = st.session_state.p1_serve / 100
        p2_serve = st.session_state.p2_serve / 100
        
        p_set_win = p_set_from_games(st.session_state.games[0], st.session_state.games[1], st.session_state.server, p1_serve, p2_serve)
        p_p1_match = p_match_from_sets(st.session_state.sets[0], st.session_state.sets[1], p_set_win, st.session_state.best_of)
        
        st.markdown("---")
        
        # Display results
        res1, res2 = st.columns(2)
        
        with res1:
            st.markdown("#### Set Winner")
            set_fair = 1/st.session_state.set_p1_odds / (1/st.session_state.set_p1_odds + 1/st.session_state.set_p2_odds)
            edge_set = p_set_win - set_fair
            st.metric(f"{st.session_state.p1}", f"{p_set_win:.1%}", f"{edge_set:+.1%} edge")
            st.metric(f"{st.session_state.p2}", f"{1-p_set_win:.1%}", f"{-edge_set:+.1%} edge")
        
        with res2:
            st.markdown("#### Match Winner")
            match_fair = 1/st.session_state.match_p1_odds / (1/st.session_state.match_p1_odds + 1/st.session_state.match_p2_odds)
            edge_match = p_p1_match - match_fair
            st.metric(f"{st.session_state.p1}", f"{p_p1_match:.1%}", f"{edge_match:+.1%} edge")
            st.metric(f"{st.session_state.p2}", f"{1-p_p1_match:.1%}", f"{-edge_match:+.1%} edge")
    
    with analysis_tabs[2]:
        st.markdown("### 📈 Break Probability Map")
        
        server_name = st.session_state.p1 if st.session_state.server == 1 else st.session_state.p2
        p_serve = st.session_state.p1_serve/100 if st.session_state.server == 1 else st.session_state.p2_serve/100
        
        st.caption(f"{server_name} serving at {p_serve*100:.0f}% point win rate")
        
        current_s = st.session_state.points[st.session_state.server - 1]
        current_r = st.session_state.points[2 - st.session_state.server]
        
        # Create matrix
        st.markdown("**Server Points →**")
        
        headers = st.columns(5)
        headers[0].markdown("**Ret↓**")
        for i, h in enumerate(['0', '15', '30', '40']):
            headers[i+1].markdown(f"**{h}**")
        
        for r in range(4):
            row = st.columns(5)
            row[0].markdown(f"**{['0','15','30','40'][r]}**")
            for s in range(4):
                p_h = p_game_from_points(s, r, p_serve)
                p_b = 1 - p_h
                fair = 1/p_b if p_b > 0.05 else 20
                
                is_current = (s == current_s and r == current_r)
                is_bp = (r == 3 and s < 3)
                
                if is_current:
                    row[s+1].markdown(f"**→ {p_b:.0%}**")
                elif is_bp:
                    row[s+1].markdown(f"🔴 {p_b:.0%}")
                else:
                    row[s+1].markdown(f"{p_b:.0%}")


# ==================== PAGE: STATISTICS ====================
elif st.session_state.current_page == "Statistics":
    st.markdown("""
    <div class="hero-header">
        <h1>📈 Session Statistics</h1>
        <div class="hero-subtitle">Track your performance and match insights</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Session overview
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.metric("Total Bets", len(st.session_state.bets))
    
    with stat_cols[1]:
        won = sum(1 for b in st.session_state.bets if b['result'] == 'WON')
        settled = sum(1 for b in st.session_state.bets if b['result'])
        win_rate = won/settled*100 if settled > 0 else 0
        st.metric("Win Rate", f"{win_rate:.0f}%", f"{won}/{settled}")
    
    with stat_cols[2]:
        st.metric("Total Staked", f"${st.session_state.total_staked:.2f}")
    
    with stat_cols[3]:
        roi = (st.session_state.total_profit / st.session_state.total_staked * 100) if st.session_state.total_staked > 0 else 0
        st.metric("ROI", f"{roi:+.1f}%", f"${st.session_state.total_profit:+.2f}")
    
    st.markdown("---")
    
    # Match stats
    st.markdown("### 🎾 Match Statistics")
    
    mc1, mc2, mc3 = st.columns(3)
    
    with mc1:
        breaks_p1 = sum(1 for g in st.session_state.games_history if g['was_break'] and g['winner'] == 1)
        st.metric(f"{st.session_state.p1} Breaks", breaks_p1)
    
    with mc2:
        breaks_p2 = sum(1 for g in st.session_state.games_history if g['was_break'] and g['winner'] == 2)
        st.metric(f"{st.session_state.p2} Breaks", breaks_p2)
    
    with mc3:
        total_games = len(st.session_state.games_history)
        st.metric("Games Played", total_games)
    
    # Momentum
    st.markdown("---")
    st.markdown("### 📊 Momentum")
    
    mom_ratio, mom_name, mom_emoji = get_momentum()
    
    st.markdown(f"""
    <div style="text-align: center; padding: 20px;">
        <h2>{mom_emoji} {mom_name}</h2>
        <div class="momentum-bar">
            <div class="momentum-marker" style="left: {mom_ratio * 100}%;"></div>
        </div>
        <p style="opacity: 0.7;">{st.session_state.p1} ← → {st.session_state.p2}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Bet history
    st.markdown("---")
    st.markdown("### 📜 Bet History")
    
    if st.session_state.bets:
        for i, bet in enumerate(reversed(st.session_state.bets)):
            idx = len(st.session_state.bets) - 1 - i
            
            if bet['result'] == 'WON':
                card_class = 'bet-won'
            elif bet['result'] == 'LOST':
                card_class = 'bet-lost'
            else:
                card_class = 'bet-pending'
            
            profit_str = f"${bet['profit']:+.2f}" if bet['profit'] is not None else "Pending"
            
            st.markdown(f"""
            <div class="{card_class}">
                <strong>{bet['time']}</strong> | {bet['selection']} @ {bet['odds']:.2f} | ${bet['stake']:.2f} → {profit_str}
            </div>
            """, unsafe_allow_html=True)
            
            if not bet['result']:
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.button("✅ Won", key=f"win_{idx}", use_container_width=True):
                        settle_bet(idx, True)
                        st.rerun()
                with bc2:
                    if st.button("❌ Lost", key=f"lose_{idx}", use_container_width=True):
                        settle_bet(idx, False)
                        st.rerun()
    else:
        st.info("No bets recorded yet. Place bets from the Live Match page!")


# ==================== PAGE: SETTINGS ====================
elif st.session_state.current_page == "Settings":
    st.markdown("""
    <div class="hero-header">
        <h1>⚙️ Match Settings</h1>
        <div class="hero-subtitle">Configure players, surface, and betting parameters</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Player setup
    st.markdown("### 👥 Players")
    
    pc1, pc2 = st.columns(2)
    
    with pc1:
        st.markdown("#### Player 1")
        p1_input = st.text_input("Name", st.session_state.p1, key="p1_name")
        if p1_input and len(p1_input) >= 2:
            players = search_player(p1_input)
            if players and p1_input != st.session_state.p1:
                for p in players[:3]:
                    if st.button(f"📚 {p['name']} ({p['country']})", key=f"sp1_{p['id']}"):
                        st.session_state.p1 = p['name']
                        st.session_state.p1_id = p['id']
                        stats = get_player_serve_stats(p['id'], st.session_state.surface)
                        if stats:
                            st.session_state.p1_stats = stats
                            st.session_state.p1_serve = int(stats['serve_point_pct'] * 100)
                        st.rerun()
        st.session_state.p1 = p1_input or "Player 1"
        
        if st.session_state.p1_stats:
            st.success(f"✅ {st.session_state.p1_stats['match_count']} matches loaded")
    
    with pc2:
        st.markdown("#### Player 2")
        p2_input = st.text_input("Name", st.session_state.p2, key="p2_name")
        if p2_input and len(p2_input) >= 2:
            players = search_player(p2_input)
            if players and p2_input != st.session_state.p2:
                for p in players[:3]:
                    if st.button(f"📚 {p['name']} ({p['country']})", key=f"sp2_{p['id']}"):
                        st.session_state.p2 = p['name']
                        st.session_state.p2_id = p['id']
                        stats = get_player_serve_stats(p['id'], st.session_state.surface)
                        if stats:
                            st.session_state.p2_stats = stats
                            st.session_state.p2_serve = int(stats['serve_point_pct'] * 100)
                        st.rerun()
        st.session_state.p2 = p2_input or "Player 2"
        
        if st.session_state.p2_stats:
            st.success(f"✅ {st.session_state.p2_stats['match_count']} matches loaded")
    
    st.markdown("---")
    
    # Match settings
    st.markdown("### 🏟️ Match Settings")
    
    ms1, ms2, ms3 = st.columns(3)
    
    with ms1:
        st.session_state.surface = st.selectbox("Surface", ['Hard', 'Clay', 'Grass', 'Indoor'])
    
    with ms2:
        st.session_state.best_of = st.selectbox("Format", [3, 5], index=0 if st.session_state.best_of == 3 else 1)
    
    with ms3:
        server_options = [st.session_state.p1, st.session_state.p2]
        current_idx = st.session_state.server - 1
        selected = st.selectbox("Who serves first?", server_options, index=current_idx)
        st.session_state.server = 1 if selected == st.session_state.p1 else 2
    
    st.markdown("---")
    
    # Serve stats
    st.markdown("### 🎾 Serve Statistics")
    
    ss1, ss2 = st.columns(2)
    
    with ss1:
        st.session_state.p1_speed = st.selectbox(f"{st.session_state.p1} serve type", 
            ['Average', 'Big Server (+4%)', 'Weak Server (-3%)'], key="p1_type")
        
        base_p1 = st.session_state.p1_stats['serve_point_pct'] if st.session_state.p1_stats else 0.62
        adj_p1 = adjust_serve_for_surface(base_p1, st.session_state.surface, st.session_state.p1_speed)
        st.session_state.p1_serve = st.slider(f"{st.session_state.p1} serve point %", 50, 75, int(adj_p1 * 100))
    
    with ss2:
        st.session_state.p2_speed = st.selectbox(f"{st.session_state.p2} serve type",
            ['Average', 'Big Server (+4%)', 'Weak Server (-3%)'], key="p2_type")
        
        base_p2 = st.session_state.p2_stats['serve_point_pct'] if st.session_state.p2_stats else 0.62
        adj_p2 = adjust_serve_for_surface(base_p2, st.session_state.surface, st.session_state.p2_speed)
        st.session_state.p2_serve = st.slider(f"{st.session_state.p2} serve point %", 50, 75, int(adj_p2 * 100))
    
    st.markdown("---")
    
    # Bankroll
    st.markdown("### 💰 Bankroll")
    st.session_state.bankroll = st.number_input("Starting bankroll ($)", 10.0, 10000.0, float(st.session_state.bankroll), 10.0)
    
    st.markdown("---")
    
    # Reset
    if st.button("🔄 Reset Everything", type="secondary", use_container_width=True):
        for k in defaults:
            st.session_state[k] = defaults[k] if not isinstance(defaults[k], list) else []
        st.rerun()


# ==================== FOOTER ====================
st.markdown("---")
db_status = "✅ Connected" if os.path.exists(DB_PATH) else "❌ Not found"
st.caption(f"🎾 Tennis Betting Hub | Database: {db_status} | Built with Streamlit")
