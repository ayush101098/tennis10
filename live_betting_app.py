"""
🎾 Live Tennis Betting Tracker PRO
===================================
Enhanced features:
- Quick point-by-point tracking buttons
- Live bet tracking with P&L
- Visual value indicators
- Momentum tracking
- Break point alerts
- Session statistics
- Auto player lookup from ATP/WTA database

Run: streamlit run live_betting_app.py
"""

import streamlit as st
import numpy as np
from scipy.special import comb
import sqlite3
import os
from pathlib import Path
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🎾 Tennis Betting PRO",
    page_icon="🎾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .value-bet {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 18px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
    }
    .no-value {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    .marginal-bet {
        background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
        color: black;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    .score-display {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin: 10px 0;
    }
    .break-point-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .stat-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .quick-btn {
        font-size: 20px !important;
        padding: 15px 30px !important;
    }
    .profit { color: #28a745; font-weight: bold; }
    .loss { color: #dc3545; font-weight: bold; }
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


# ==================== PROBABILITY FUNCTIONS ====================

def p_game_from_points(server_pts: int, returner_pts: int, p_point: float) -> float:
    """Probability server wins game from current point score."""
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
    # Bet tracking
    'bets': [],
    'total_staked': 0.0,
    'total_profit': 0.0,
    # Point history for momentum
    'point_history': [],
    'games_history': [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==================== HELPER FUNCTIONS ====================

def get_point_display(pts):
    """Convert point number to tennis notation."""
    if pts == 0: return '0'
    elif pts == 1: return '15'
    elif pts == 2: return '30'
    elif pts == 3: return '40'
    else: return 'AD'

def get_score_string():
    """Get formatted score string."""
    p1_pt = get_point_display(st.session_state.points[0])
    p2_pt = get_point_display(st.session_state.points[1])
    
    # Handle deuce/advantage
    if st.session_state.points[0] >= 3 and st.session_state.points[1] >= 3:
        if st.session_state.points[0] == st.session_state.points[1]:
            point_str = "DEUCE"
        elif st.session_state.points[0] > st.session_state.points[1]:
            point_str = f"AD-{st.session_state.p1[:6]}"
        else:
            point_str = f"AD-{st.session_state.p2[:6]}"
    else:
        point_str = f"{p1_pt}-{p2_pt}"
    
    return f"{st.session_state.sets[0]}-{st.session_state.sets[1]} | {st.session_state.games[0]}-{st.session_state.games[1]} | {point_str}"

def record_point(winner: int):
    """Record a point and handle game/set transitions."""
    st.session_state.point_history.append(winner)
    st.session_state.points[winner - 1] += 1
    
    # Check for game won
    p1, p2 = st.session_state.points
    game_won = None
    
    # Regular game win
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
        
        # Check set won
        g1, g2 = st.session_state.games
        if (g1 >= 6 and g1 >= g2 + 2) or g1 == 7:
            st.session_state.sets[0] += 1
            st.session_state.games = [0, 0]
        elif (g2 >= 6 and g2 >= g1 + 2) or g2 == 7:
            st.session_state.sets[1] += 1
            st.session_state.games = [0, 0]

def record_bet(bet_type: str, selection: str, odds: float, stake: float, model_prob: float):
    """Record a placed bet."""
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
    """Settle a bet."""
    bet = st.session_state.bets[idx]
    if won:
        bet['result'] = 'WON'
        bet['profit'] = bet['stake'] * (bet['odds'] - 1)
    else:
        bet['result'] = 'LOST'
        bet['profit'] = -bet['stake']
    st.session_state.total_profit += bet['profit']

def get_momentum():
    """Calculate recent momentum (last 5 points)."""
    if len(st.session_state.point_history) < 3:
        return 0.5, "Even"
    recent = st.session_state.point_history[-5:]
    p1_won = sum(1 for p in recent if p == 1)
    ratio = p1_won / len(recent)
    if ratio >= 0.7:
        return ratio, f"🔥 {st.session_state.p1}"
    elif ratio <= 0.3:
        return ratio, f"🔥 {st.session_state.p2}"
    else:
        return ratio, "Even"

def is_break_point():
    """Check if current position is a break point."""
    server = st.session_state.server
    s_pts = st.session_state.points[server - 1]
    r_pts = st.session_state.points[2 - server]
    
    # Break point: returner at 40 (3+) and server not at game point
    if r_pts >= 3 and r_pts > s_pts:
        return True
    return False

def get_break_point_count():
    """Count break points (e.g., 0-40 = 3 BPs)."""
    server = st.session_state.server
    s_pts = st.session_state.points[server - 1]
    r_pts = st.session_state.points[2 - server]
    
    if r_pts < 3:
        return 0
    if s_pts < 3:
        return r_pts - 2  # 0-40 = 3 BPs, 15-40 = 2 BPs, 30-40 = 1 BP
    return 1  # Deuce situation, Ad-out = 1 BP


# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Match Setup")
    
    # Player 1
    p1_input = st.text_input("👤 Player 1", st.session_state.p1, key="p1_input")
    if p1_input and len(p1_input) >= 2:
        players = search_player(p1_input)
        if players and p1_input != st.session_state.p1:
            for p in players[:2]:
                if st.button(f"📚 {p['name']}", key=f"sp1_{p['id']}"):
                    st.session_state.p1 = p['name']
                    st.session_state.p1_id = p['id']
                    stats = get_player_serve_stats(p['id'], st.session_state.surface)
                    if stats:
                        st.session_state.p1_stats = stats
                        st.session_state.p1_serve = int(stats['serve_point_pct'] * 100)
                    st.rerun()
    st.session_state.p1 = p1_input or "Player 1"
    if st.session_state.p1_stats:
        st.caption(f"✅ {st.session_state.p1_stats['match_count']} matches loaded")
    
    # Player 2
    p2_input = st.text_input("👤 Player 2", st.session_state.p2, key="p2_input")
    if p2_input and len(p2_input) >= 2:
        players = search_player(p2_input)
        if players and p2_input != st.session_state.p2:
            for p in players[:2]:
                if st.button(f"📚 {p['name']}", key=f"sp2_{p['id']}"):
                    st.session_state.p2 = p['name']
                    st.session_state.p2_id = p['id']
                    stats = get_player_serve_stats(p['id'], st.session_state.surface)
                    if stats:
                        st.session_state.p2_stats = stats
                        st.session_state.p2_serve = int(stats['serve_point_pct'] * 100)
                    st.rerun()
    st.session_state.p2 = p2_input or "Player 2"
    if st.session_state.p2_stats:
        st.caption(f"✅ {st.session_state.p2_stats['match_count']} matches loaded")
    
    st.markdown("---")
    
    # Surface & Speed
    st.session_state.surface = st.selectbox("🏟️ Surface", ['Hard', 'Clay', 'Grass', 'Indoor'])
    
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.p1_speed = st.selectbox("P1 Serve", ['Average', 'Big Server (+4%)', 'Weak Server (-3%)'], key="spd1")
    with c2:
        st.session_state.p2_speed = st.selectbox("P2 Serve", ['Average', 'Big Server (+4%)', 'Weak Server (-3%)'], key="spd2")
    
    # Serve %
    base_p1 = st.session_state.p1_stats['serve_point_pct'] if st.session_state.p1_stats else 0.62
    base_p2 = st.session_state.p2_stats['serve_point_pct'] if st.session_state.p2_stats else 0.62
    adj_p1 = adjust_serve_for_surface(base_p1, st.session_state.surface, st.session_state.p1_speed)
    adj_p2 = adjust_serve_for_surface(base_p2, st.session_state.surface, st.session_state.p2_speed)
    
    st.session_state.p1_serve = st.slider(f"{st.session_state.p1[:10]} serve %", 50, 75, int(adj_p1 * 100))
    st.session_state.p2_serve = st.slider(f"{st.session_state.p2[:10]} serve %", 50, 75, int(adj_p2 * 100))
    
    st.markdown("---")
    st.session_state.best_of = st.radio("Best of", [3, 5], horizontal=True)
    st.session_state.bankroll = st.number_input("💰 Bankroll ($)", value=float(st.session_state.bankroll), min_value=10.0)
    
    if st.button("🔄 Reset Match", type="secondary"):
        for k in ['sets', 'games', 'points', 'point_history', 'games_history', 'bets', 'total_staked', 'total_profit']:
            st.session_state[k] = defaults[k] if not isinstance(defaults[k], list) else []
        st.session_state.server = 1
        st.rerun()


# ==================== MAIN DISPLAY ====================
st.title("🎾 Live Tennis Betting Tracker PRO")

# ==================== LIVE SCORE DISPLAY ====================
server_emoji = "🎾"
server_name = st.session_state.p1 if st.session_state.server == 1 else st.session_state.p2

# Big score display
st.markdown(f"""
<div class="score-display">
    {st.session_state.p1} vs {st.session_state.p2}<br>
    <span style="font-size: 64px;">{get_score_string()}</span><br>
    <span style="font-size: 20px;">{server_emoji} {server_name} serving</span>
</div>
""", unsafe_allow_html=True)

# Break point alert
if is_break_point():
    bp_count = get_break_point_count()
    returner = st.session_state.p2 if st.session_state.server == 1 else st.session_state.p1
    st.markdown(f"""
    <div class="break-point-alert">
        🔴 BREAK POINT{'S' if bp_count > 1 else ''} x{bp_count} for {returner}!
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== QUICK POINT BUTTONS ====================
st.subheader("⚡ Quick Score Update")

btn_cols = st.columns([2, 2, 1, 1])

with btn_cols[0]:
    if st.button(f"✅ Point {st.session_state.p1}", type="primary", use_container_width=True):
        record_point(1)
        st.rerun()

with btn_cols[1]:
    if st.button(f"✅ Point {st.session_state.p2}", type="primary", use_container_width=True):
        record_point(2)
        st.rerun()

with btn_cols[2]:
    if st.button("↩️ Undo", use_container_width=True):
        if st.session_state.point_history:
            # Simple undo - just reset points (basic implementation)
            if st.session_state.points[0] > 0 or st.session_state.points[1] > 0:
                last = st.session_state.point_history.pop()
                if st.session_state.points[last - 1] > 0:
                    st.session_state.points[last - 1] -= 1
            st.rerun()

with btn_cols[3]:
    if st.button("🔄 Swap Server", use_container_width=True):
        st.session_state.server = 3 - st.session_state.server
        st.rerun()

# Manual score adjustment
with st.expander("📝 Manual Score Adjustment"):
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.session_state.sets[0] = st.number_input(f"{st.session_state.p1[:8]} Sets", 0, 3, st.session_state.sets[0])
    with mc2:
        st.session_state.sets[1] = st.number_input(f"{st.session_state.p2[:8]} Sets", 0, 3, st.session_state.sets[1])
    with mc3:
        st.session_state.games[0] = st.number_input(f"{st.session_state.p1[:8]} Games", 0, 7, st.session_state.games[0])
    with mc4:
        st.session_state.games[1] = st.number_input(f"{st.session_state.p2[:8]} Games", 0, 7, st.session_state.games[1])
    
    pc1, pc2 = st.columns(2)
    pts_opts = ['0', '15', '30', '40', 'AD']
    with pc1:
        p1_pt_sel = st.selectbox(f"{st.session_state.p1[:8]} Points", pts_opts, index=min(st.session_state.points[0], 4))
        st.session_state.points[0] = pts_opts.index(p1_pt_sel)
    with pc2:
        p2_pt_sel = st.selectbox(f"{st.session_state.p2[:8]} Points", pts_opts, index=min(st.session_state.points[1], 4))
        st.session_state.points[1] = pts_opts.index(p2_pt_sel)

st.markdown("---")

# ==================== LIVE ODDS INPUT ====================
st.header("💰 Live Odds")

odds_tabs = st.tabs(["🎯 GAME (Primary)", "📊 SET", "🏆 MATCH"])

with odds_tabs[0]:
    server_name = st.session_state.p1 if st.session_state.server == 1 else st.session_state.p2
    returner_name = st.session_state.p2 if st.session_state.server == 1 else st.session_state.p1
    
    oc1, oc2 = st.columns(2)
    with oc1:
        st.session_state.game_hold_odds = st.number_input(
            f"🟢 {server_name} HOLDS", 1.01, 20.0, 
            float(st.session_state.game_hold_odds), 0.05, key="hold"
        )
    with oc2:
        st.session_state.game_break_odds = st.number_input(
            f"🔴 {returner_name} BREAKS", 1.01, 20.0,
            float(st.session_state.game_break_odds), 0.05, key="break"
        )

with odds_tabs[1]:
    sc1, sc2 = st.columns(2)
    with sc1:
        st.session_state.set_p1_odds = st.number_input(f"{st.session_state.p1} wins SET", 1.01, 20.0, float(st.session_state.set_p1_odds), 0.05)
    with sc2:
        st.session_state.set_p2_odds = st.number_input(f"{st.session_state.p2} wins SET", 1.01, 20.0, float(st.session_state.set_p2_odds), 0.05)

with odds_tabs[2]:
    mc1, mc2 = st.columns(2)
    with mc1:
        st.session_state.match_p1_odds = st.number_input(f"{st.session_state.p1} wins MATCH", 1.01, 50.0, float(st.session_state.match_p1_odds), 0.05)
    with mc2:
        st.session_state.match_p2_odds = st.number_input(f"{st.session_state.p2} wins MATCH", 1.01, 50.0, float(st.session_state.match_p2_odds), 0.05)

st.markdown("---")

# ==================== CALCULATIONS ====================
p1_serve = st.session_state.p1_serve / 100
p2_serve = st.session_state.p2_serve / 100

# Current game calculations
if st.session_state.server == 1:
    server_pts, returner_pts = st.session_state.points
    p_serve = p1_serve
else:
    server_pts, returner_pts = st.session_state.points[1], st.session_state.points[0]
    p_serve = p2_serve

p_hold = p_game_from_points(server_pts, returner_pts, p_serve)
p_break = 1 - p_hold

# Set & Match
p_p1_game = p_hold if st.session_state.server == 1 else p_break
p_set_win = p_set_from_games(st.session_state.games[0]+1, st.session_state.games[1], 3-st.session_state.server, p1_serve, p2_serve)
p_set_lose = p_set_from_games(st.session_state.games[0], st.session_state.games[1]+1, 3-st.session_state.server, p1_serve, p2_serve)
p_p1_set = p_p1_game * p_set_win + (1 - p_p1_game) * p_set_lose
p_p1_match = p_match_from_sets(st.session_state.sets[0], st.session_state.sets[1], p_p1_set, st.session_state.best_of)

# Odds analysis
hold_implied = 1 / st.session_state.game_hold_odds
break_implied = 1 / st.session_state.game_break_odds
hold_fair = hold_implied / (hold_implied + break_implied)
break_fair = break_implied / (hold_implied + break_implied)

edge_hold = p_hold - hold_fair
edge_break = p_break - break_fair

kelly_hold = max(0, (p_hold * st.session_state.game_hold_odds - 1) / (st.session_state.game_hold_odds - 1)) if edge_hold > 0 else 0
kelly_break = max(0, (p_break * st.session_state.game_break_odds - 1) / (st.session_state.game_break_odds - 1)) if edge_break > 0 else 0

# ==================== GAME BET RECOMMENDATION ====================
st.header("🎯 GAME BET ANALYSIS")

server_name = st.session_state.p1 if st.session_state.server == 1 else st.session_state.p2
returner_name = st.session_state.p2 if st.session_state.server == 1 else st.session_state.p1

col1, col2 = st.columns(2)

with col1:
    fair_hold = 1/p_hold if p_hold > 0.01 else 99
    st.markdown(f"### 🟢 {server_name} HOLDS")
    st.markdown(f"**Model: {p_hold:.1%}** | Fair: {fair_hold:.2f} | Book: {st.session_state.game_hold_odds}")
    
    if edge_hold > 0.03:
        stake = kelly_hold * 0.25 * st.session_state.bankroll
        st.markdown(f"""
        <div class="value-bet">
            ✅ VALUE BET<br>
            Edge: <b>+{edge_hold:.1%}</b><br>
            Stake: <b>${stake:.2f}</b>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"📝 Record HOLD Bet", key="rec_hold"):
            record_bet("Game", f"{server_name} Hold", st.session_state.game_hold_odds, stake, p_hold)
            st.success(f"Bet recorded: ${stake:.2f} on Hold @ {st.session_state.game_hold_odds}")
    elif edge_hold > 0:
        st.markdown(f'<div class="marginal-bet">⚠️ Marginal +{edge_hold:.1%}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="no-value">❌ No Value ({edge_hold:+.1%})</div>', unsafe_allow_html=True)

with col2:
    fair_break = 1/p_break if p_break > 0.01 else 99
    st.markdown(f"### 🔴 {returner_name} BREAKS")
    st.markdown(f"**Model: {p_break:.1%}** | Fair: {fair_break:.2f} | Book: {st.session_state.game_break_odds}")
    
    if edge_break > 0.03:
        stake = kelly_break * 0.25 * st.session_state.bankroll
        st.markdown(f"""
        <div class="value-bet">
            ✅ VALUE BET<br>
            Edge: <b>+{edge_break:.1%}</b><br>
            Stake: <b>${stake:.2f}</b>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"📝 Record BREAK Bet", key="rec_break"):
            record_bet("Game", f"{returner_name} Break", st.session_state.game_break_odds, stake, p_break)
            st.success(f"Bet recorded: ${stake:.2f} on Break @ {st.session_state.game_break_odds}")
    elif edge_break > 0:
        st.markdown(f'<div class="marginal-bet">⚠️ Marginal +{edge_break:.1%}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="no-value">❌ No Value ({edge_break:+.1%})</div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== BREAK PROBABILITY MAP ====================
st.header("📈 Break Probability by Point Score")

st.caption(f"{server_name} serving at {p_serve*100:.0f}%")

# Current position highlighted
current_s = st.session_state.points[st.session_state.server - 1]
current_r = st.session_state.points[2 - st.session_state.server]

map_cols = st.columns(5)
headers = ['', '0', '15', '30', '40']
for i, h in enumerate(headers):
    map_cols[i].markdown(f"**{h}**")

for r in range(4):
    row_cols = st.columns(5)
    row_cols[0].markdown(f"**{['0','15','30','40'][r]}**")
    for s in range(4):
        p_h = p_game_from_points(s, r, p_serve)
        p_b = 1 - p_h
        fair = 1/p_b if p_b > 0.05 else 20
        
        # Highlight current position
        is_current = (s == current_s and r == current_r)
        is_bp = (r == 3 and s < 3)
        
        if is_current:
            row_cols[s+1].markdown(f"**→ {p_b:.0%}** @ {fair:.2f}")
        elif is_bp:
            row_cols[s+1].markdown(f"🔥 {p_b:.0%} @ {fair:.2f}")
        else:
            row_cols[s+1].markdown(f"{p_b:.0%} @ {fair:.2f}")

st.markdown("---")

# ==================== SESSION STATS ====================
st.header("📊 Session Statistics")

stat_cols = st.columns(4)

# Momentum
mom_ratio, mom_text = get_momentum()
with stat_cols[0]:
    st.metric("Momentum", mom_text)
    if len(st.session_state.point_history) >= 3:
        recent_p1 = sum(1 for p in st.session_state.point_history[-5:] if p == 1)
        st.caption(f"Last 5 pts: {recent_p1}-{len(st.session_state.point_history[-5:]) - recent_p1}")

# Games/Breaks
with stat_cols[1]:
    breaks_p1 = sum(1 for g in st.session_state.games_history if g['was_break'] and g['winner'] == 1)
    breaks_p2 = sum(1 for g in st.session_state.games_history if g['was_break'] and g['winner'] == 2)
    st.metric("Breaks", f"{breaks_p1}-{breaks_p2}")
    st.caption(f"{st.session_state.p1[:6]} - {st.session_state.p2[:6]}")

# Betting P&L
with stat_cols[2]:
    profit_class = "profit" if st.session_state.total_profit >= 0 else "loss"
    st.metric("Bets P&L", f"${st.session_state.total_profit:+.2f}")
    st.caption(f"Staked: ${st.session_state.total_staked:.2f}")

with stat_cols[3]:
    st.metric("Total Bets", len(st.session_state.bets))
    won = sum(1 for b in st.session_state.bets if b['result'] == 'WON')
    settled = sum(1 for b in st.session_state.bets if b['result'])
    if settled > 0:
        st.caption(f"Won: {won}/{settled}")

# Bet History
if st.session_state.bets:
    with st.expander("📜 Bet History"):
        for i, bet in enumerate(reversed(st.session_state.bets)):
            idx = len(st.session_state.bets) - 1 - i
            status = bet['result'] or "PENDING"
            profit_str = f"${bet['profit']:+.2f}" if bet['profit'] is not None else ""
            
            bc1, bc2, bc3 = st.columns([3, 1, 1])
            with bc1:
                st.markdown(f"**{bet['time']}** | {bet['selection']} @ {bet['odds']:.2f} | ${bet['stake']:.2f}")
            with bc2:
                st.markdown(f"**{status}** {profit_str}")
            with bc3:
                if not bet['result']:
                    if st.button("✅ Won", key=f"win_{idx}"):
                        settle_bet(idx, True)
                        st.rerun()
                    if st.button("❌ Lost", key=f"lose_{idx}"):
                        settle_bet(idx, False)
                        st.rerun()

# ==================== SET & MATCH ANALYSIS ====================
with st.expander("📊 SET & MATCH Analysis"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Set Winner")
        set_p1_fair = (1/st.session_state.set_p1_odds) / ((1/st.session_state.set_p1_odds) + (1/st.session_state.set_p2_odds))
        edge_set = p_p1_set - set_p1_fair
        st.markdown(f"{st.session_state.p1}: **{p_p1_set:.1%}** | Edge: {edge_set:+.1%}")
        st.markdown(f"{st.session_state.p2}: **{1-p_p1_set:.1%}** | Edge: {-edge_set:+.1%}")
    
    with col2:
        st.markdown("### Match Winner")
        match_p1_fair = (1/st.session_state.match_p1_odds) / ((1/st.session_state.match_p1_odds) + (1/st.session_state.match_p2_odds))
        edge_match = p_p1_match - match_p1_fair
        st.markdown(f"{st.session_state.p1}: **{p_p1_match:.1%}** | Edge: {edge_match:+.1%}")
        st.markdown(f"{st.session_state.p2}: **{1-p_p1_match:.1%}** | Edge: {-edge_match:+.1%}")

# Footer
st.markdown("---")
db_status = "✅" if os.path.exists(DB_PATH) else "❌"
st.caption(f"🎾 Tennis Betting PRO | {st.session_state.surface} | DB: {db_status} | Bankroll: ${st.session_state.bankroll:.0f}")
