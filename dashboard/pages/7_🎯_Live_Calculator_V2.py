"""
Live Match Calculator V2 - Compact Professional Interface
==========================================================
Single-page comprehensive layout with data persistence and bet tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from live_match_persistence import (
    save_live_match, load_live_match, finish_live_match,
    save_selected_bet, get_pending_bets, get_all_selected_bets
)

# Import player stats if available
try:
    from player_stats_integration import get_player_stats_for_calculator
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

st.set_page_config(page_title="Pro Live Calculator V2", page_icon="🎯", layout="wide")

# Compact CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    .value-bet-alert {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 0.9em;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
    h1 { font-size: 1.8em !important; margin-bottom: 0.5em !important; }
    h2 { font-size: 1.4em !important; margin-top: 0.5em !important; }
    h3 { font-size: 1.2em !important; }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'><h1 style='margin:0; color:white;'>🎯 Pro Live Calculator V2</h1></div>", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS (Imported from original)
# ============================================================================

def calculate_markov_probabilities(p1_serve_pct, p2_serve_pct):
    """Calculate match probabilities using Markov Chain"""
    def game_prob(p_serve):
        prob = p_serve**4
        prob += 4 * p_serve**4 * (1-p_serve)
        prob += 10 * p_serve**4 * (1-p_serve)**2
        deuce_win = (p_serve**2) / (1 - 2*p_serve*(1-p_serve)) if p_serve != 0.5 else 0.5
        prob += 20 * p_serve**3 * (1-p_serve)**3 * deuce_win
        return prob
    
    p1_game_on_serve = game_prob(p1_serve_pct)
    p2_game_on_serve = game_prob(p2_serve_pct)
    
    def set_prob(p_game_serve, p_game_return):
        p_win = 0
        for i in range(6):
            for j in range(min(i, 5)):
                prob_state = 1
                for k in range(i+j):
                    if k % 2 == 0:
                        prob_state *= p_game_serve if k < i else (1 - p_game_serve)
                    else:
                        prob_state *= (1 - p_game_return) if k < i else p_game_return
                if i == 6:
                    p_win += prob_state
        return max(0.05, min(0.95, p_win))
    
    p1_set = set_prob(p1_game_on_serve, 1 - p2_game_on_serve)
    p1_match = p1_set ** 2
    
    return {
        'p1_match': p1_match,
        'p2_match': 1 - p1_match,
        'p1_set': p1_set,
        'p2_set': 1 - p1_set,
        'p1_game_on_serve': p1_game_on_serve,
        'p2_game_on_serve': p2_game_on_serve
    }

def apply_advanced_parameters(base_serve, base_return, params):
    """Apply advanced parameter adjustments"""
    if not params:
        return base_serve, base_return
    
    momentum_adj = (params.get('momentum', 0.5) - 0.5) * 0.15
    surface_adj = (params.get('surface_mastery', 0.5) - 0.5) * 0.10
    clutch_adj = (params.get('clutch', 0.5) - 0.5) * 0.08
    bp_defense_adj = (params.get('bp_defense', 0.6) - 0.6) * 0.10
    consistency_adj = (params.get('consistency', 0.5) - 0.5) * 0.05
    first_serve_adj = (params.get('first_serve_pct', 0.62) - 0.62) * 0.08
    
    total_adj = sum([momentum_adj, surface_adj, clutch_adj, bp_defense_adj, consistency_adj, first_serve_adj])
    
    adjusted_serve = max(0.4, min(0.85, base_serve + total_adj))
    adjusted_return = max(0.2, min(0.55, base_return + total_adj * 0.5))
    
    return adjusted_serve, adjusted_return

def calculate_ml_probabilities(p1_serve, p2_serve, p1_sets, p2_sets, p1_games, p2_games):
    """Simplified ML probability (logistic regression approximation)"""
    serve_diff = p1_serve - p2_serve
    set_diff = p1_sets - p2_sets
    game_diff = p1_games - p2_games
    
    score = (serve_diff * 2.5) + (set_diff * 0.3) + (game_diff * 0.1)
    prob = 1 / (1 + np.exp(-score * 5))
    return max(0.05, min(0.95, prob))

def calculate_nn_probabilities(p1_serve, p2_serve, p1_sets, p2_sets, p1_games, p2_games, p1_points, p2_points):
    """Simplified NN probability approximation"""
    ml_prob = calculate_ml_probabilities(p1_serve, p2_serve, p1_sets, p2_sets, p1_games, p2_games)
    point_factor = (p1_points - p2_points) * 0.02
    return max(0.05, min(0.95, ml_prob + point_factor))

def adjust_for_current_score(base_prob, sets_won, games_won, points_won, total_points, games_history=None):
    """Adjust probability based on current match state"""
    set_bonus = sets_won * 0.18
    game_bonus = (games_won / 6) * 0.12
    point_bonus = (points_won / 4) * 0.06
    momentum = min(total_points / 100, 1.0) * 0.05
    
    recent_game_momentum = 0
    if games_history and len(games_history) >= 3:
        recent_games = games_history[-3:]
        recent_wins = sum(1 for g in recent_games if g == 'won')
        recent_game_momentum = (recent_wins / 3 - 0.5) * 0.1
    
    adjusted = base_prob + set_bonus + game_bonus + point_bonus + momentum + recent_game_momentum
    return max(0.05, min(0.95, adjusted))

def get_point_display(points):
    """Convert numeric points to tennis score"""
    if points == 0: return "0"
    elif points == 1: return "15"
    elif points == 2: return "30"
    elif points == 3: return "40"
    elif points == 4: return "AD"
    else: return str(points)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Match setup defaults
if 'player1_name' not in st.session_state:
    st.session_state.player1_name = "Player 1"
if 'player2_name' not in st.session_state:
    st.session_state.player2_name = "Player 2"
if 'surface' not in st.session_state:
    st.session_state.surface = "Hard"

# Serve/return percentages
if 'p1_serve_win' not in st.session_state:
    st.session_state.p1_serve_win = 0.68
if 'p2_serve_win' not in st.session_state:
    st.session_state.p2_serve_win = 0.65
if 'p1_return_win' not in st.session_state:
    st.session_state.p1_return_win = 0.35
if 'p2_return_win' not in st.session_state:
    st.session_state.p2_return_win = 0.32

# Match score
if 'p1_sets' not in st.session_state:
    st.session_state.p1_sets = 0
if 'p2_sets' not in st.session_state:
    st.session_state.p2_sets = 0
if 'p1_games' not in st.session_state:
    st.session_state.p1_games = 0
if 'p2_games' not in st.session_state:
    st.session_state.p2_games = 0
if 'p1_point_score' not in st.session_state:
    st.session_state.p1_point_score = 0
if 'p2_point_score' not in st.session_state:
    st.session_state.p2_point_score = 0
if 'total_points' not in st.session_state:
    st.session_state.total_points = 0
if 'break_count_p1' not in st.session_state:
    st.session_state.break_count_p1 = 0
if 'break_count_p2' not in st.session_state:
    st.session_state.break_count_p2 = 0

# History tracking
if 'probability_history' not in st.session_state:
    st.session_state.probability_history = []
if 'score_history' not in st.session_state:
    st.session_state.score_history = []
if 'point_winner_history' not in st.session_state:
    st.session_state.point_winner_history = []
if 'p1_games_won_history' not in st.session_state:
    st.session_state.p1_games_won_history = []
if 'p2_games_won_history' not in st.session_state:
    st.session_state.p2_games_won_history = []

# Advanced parameters
if 'p1_adv_params' not in st.session_state:
    st.session_state.p1_adv_params = {'momentum': 0.5, 'surface_mastery': 0.5, 'clutch': 0.5, 'bp_defense': 0.6, 'consistency': 0.5, 'first_serve_pct': 0.62}
if 'p2_adv_params' not in st.session_state:
    st.session_state.p2_adv_params = {'momentum': 0.5, 'surface_mastery': 0.5, 'clutch': 0.5, 'bp_defense': 0.6, 'consistency': 0.5, 'first_serve_pct': 0.62}

# Pre-match bookmaker odds
if 'p1_prematch_odds' not in st.session_state:
    st.session_state.p1_prematch_odds = 1.85
if 'p2_prematch_odds' not in st.session_state:
    st.session_state.p2_prematch_odds = 2.10

# Match snapshots (odds and probabilities at each point/game)
if 'match_snapshots' not in st.session_state:
    st.session_state.match_snapshots = []

# Match ID for database tracking
if 'current_match_id' not in st.session_state:
    st.session_state.current_match_id = None

# ============================================================================
# COMPACT INPUT PANEL (Single Row)
# ============================================================================

with st.container():
    col_p1, col_p2, col_surf, col_load = st.columns([2, 2, 1, 1])
    
    with col_p1:
        p1_name = st.text_input("Player 1", value=st.session_state.player1_name, key="p1_input", label_visibility="collapsed", placeholder="Player 1 Name")
        if p1_name != st.session_state.player1_name:
            st.session_state.player1_name = p1_name
    
    with col_p2:
        p2_name = st.text_input("Player 2", value=st.session_state.player2_name, key="p2_input", label_visibility="collapsed", placeholder="Player 2 Name")
        if p2_name != st.session_state.player2_name:
            st.session_state.player2_name = p2_name
    
    with col_surf:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], index=0, key="surf_select")
        st.session_state.surface = surface
    
    with col_load:
        st.write("")  # Spacing
        if st.button("📂 Load Match", key="load_btn"):
            loaded = load_live_match(st.session_state.player1_name, st.session_state.player2_name)
            if loaded:
                # Restore all session state from database
                st.session_state.p1_sets = loaded['p1_sets']
                st.session_state.p2_sets = loaded['p2_sets']
                st.session_state.p1_games = loaded['p1_games']
                st.session_state.p2_games = loaded['p2_games']
                st.session_state.p1_point_score = loaded['p1_points']
                st.session_state.p2_point_score = loaded['p2_points']
                st.session_state.total_points = loaded['total_points']
                st.session_state.break_count_p1 = loaded['p1_breaks']
                st.session_state.break_count_p2 = loaded['p2_breaks']
                st.session_state.probability_history = loaded['probability_history']
                st.session_state.score_history = loaded['score_history']
                st.session_state.point_winner_history = loaded['point_winner_history']
                st.session_state.p1_games_won_history = loaded['p1_games_won_history']
                st.session_state.p2_games_won_history = loaded['p2_games_won_history']
                st.session_state.current_match_id = loaded['match_id']
                st.success(f"✅ Loaded match: {loaded['total_points']} points tracked")
                st.rerun()
            else:
                st.info("No saved match found for these players")

# Compact serve/return stats
with st.expander("⚙️ Player Stats & Advanced Parameters", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{st.session_state.player1_name}**")
        st.session_state.p1_serve_win = st.slider("Serve Win %", 50, 85, int(st.session_state.p1_serve_win*100), key="p1s") / 100
        st.session_state.p1_return_win = st.slider("Return Win %", 20, 50, int(st.session_state.p1_return_win*100), key="p1r") / 100
        
        st.caption("Advanced Parameters:")
        c1a, c1b, c1c = st.columns(3)
        with c1a:
            st.session_state.p1_adv_params['momentum'] = st.slider("Momentum", 0.0, 1.0, 0.5, 0.05, key="p1m")
            st.session_state.p1_adv_params['clutch'] = st.slider("Clutch", 0.0, 1.0, 0.5, 0.05, key="p1c")
        with c1b:
            st.session_state.p1_adv_params['surface_mastery'] = st.slider("Surface", 0.0, 1.0, 0.5, 0.05, key="p1sm")
            st.session_state.p1_adv_params['bp_defense'] = st.slider("BP Def", 0.0, 1.0, 0.6, 0.05, key="p1bp")
        with c1c:
            st.session_state.p1_adv_params['consistency'] = st.slider("Consistency", 0.0, 1.0, 0.5, 0.05, key="p1cons")
            st.session_state.p1_adv_params['first_serve_pct'] = st.slider("1st Srv %", 0.0, 1.0, 0.62, 0.01, key="p1fs")
    
    with col2:
        st.markdown(f"**{st.session_state.player2_name}**")
        st.session_state.p2_serve_win = st.slider("Serve Win % ", 50, 85, int(st.session_state.p2_serve_win*100), key="p2s") / 100
        st.session_state.p2_return_win = st.slider("Return Win % ", 20, 50, int(st.session_state.p2_return_win*100), key="p2r") / 100
        
        st.caption("Advanced Parameters:")
        c2a, c2b, c2c = st.columns(3)
        with c2a:
            st.session_state.p2_adv_params['momentum'] = st.slider("Momentum ", 0.0, 1.0, 0.5, 0.05, key="p2m")
            st.session_state.p2_adv_params['clutch'] = st.slider("Clutch ", 0.0, 1.0, 0.5, 0.05, key="p2c")
        with c2b:
            st.session_state.p2_adv_params['surface_mastery'] = st.slider("Surface ", 0.0, 1.0, 0.5, 0.05, key="p2sm")
            st.session_state.p2_adv_params['bp_defense'] = st.slider("BP Def ", 0.0, 1.0, 0.6, 0.05, key="p2bp")
        with c2c:
            st.session_state.p2_adv_params['consistency'] = st.slider("Consistency ", 0.0, 1.0, 0.5, 0.05, key="p2cons")
            st.session_state.p2_adv_params['first_serve_pct'] = st.slider("1st Srv % ", 0.0, 1.0, 0.62, 0.01, key="p2fs")

# Pre-match bookmaker odds section
with st.expander("💰 Pre-Match Bookmaker Odds", expanded=False):
    st.caption("Enter the odds BEFORE the match started (for tracking value)")
    col_pm1, col_pm2 = st.columns(2)
    with col_pm1:
        st.session_state.p1_prematch_odds = st.number_input(
            f"{st.session_state.player1_name} Pre-Match Odds",
            min_value=1.01, max_value=50.0,
            value=st.session_state.p1_prematch_odds,
            step=0.05, key="pm_p1_odds"
        )
        st.caption(f"Implied prob: {1/st.session_state.p1_prematch_odds:.1%}")
    with col_pm2:
        st.session_state.p2_prematch_odds = st.number_input(
            f"{st.session_state.player2_name} Pre-Match Odds",
            min_value=1.01, max_value=50.0,
            value=st.session_state.p2_prematch_odds,
            step=0.05, key="pm_p2_odds"
        )
        st.caption(f"Implied prob: {1/st.session_state.p2_prematch_odds:.1%}")

st.markdown("---")

# ============================================================================
# TABBED INTERFACE FOR COMPACT LAYOUT
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎾 Live Tracker", "📊 Probability & Bets", "💰 My Selected Bets", "📈 Analytics", "📸 Match Snapshots"])

# TAB 1: LIVE POINT TRACKING
# ============================================================================
with tab1:
    # Point tracking function
    def handle_point_won(winner):
        """Handle point won with auto-save to database and match snapshots"""
        st.session_state.point_winner_history.append(winner)
        st.session_state.total_points += 1
        
        if winner == 1:
            st.session_state.p1_point_score += 1
        else:
            st.session_state.p2_point_score += 1
        
        # Check if game won
        game_won = False
        set_won = False
        
        if st.session_state.p1_point_score >= 4 and st.session_state.p1_point_score - st.session_state.p2_point_score >= 2:
            st.session_state.p1_games += 1
            st.session_state.p1_games_won_history.append('won')
            st.session_state.p2_games_won_history.append('lost')
            game_won = True
            
            games_before = (st.session_state.p1_games - 1) + st.session_state.p2_games
            if games_before % 2 == 1:
                st.session_state.break_count_p1 += 1
            
            st.session_state.p1_point_score = 0
            st.session_state.p2_point_score = 0
            
            if st.session_state.p1_games >= 6 and st.session_state.p1_games - st.session_state.p2_games >= 2:
                st.session_state.p1_sets += 1
                st.session_state.p1_games = 0
                st.session_state.p2_games = 0
                set_won = True
                
        elif st.session_state.p2_point_score >= 4 and st.session_state.p2_point_score - st.session_state.p1_point_score >= 2:
            st.session_state.p2_games += 1
            st.session_state.p1_games_won_history.append('lost')
            st.session_state.p2_games_won_history.append('won')
            game_won = True
            
            games_before = st.session_state.p1_games + (st.session_state.p2_games - 1)
            if games_before % 2 == 0:
                st.session_state.break_count_p2 += 1
            
            st.session_state.p1_point_score = 0
            st.session_state.p2_point_score = 0
            
            if st.session_state.p2_games >= 6 and st.session_state.p2_games - st.session_state.p1_games >= 2:
                st.session_state.p2_sets += 1
                st.session_state.p1_games = 0
                st.session_state.p2_games = 0
                set_won = True
        
        # Create match snapshot (save state + odds at important moments)
        snapshot_type = 'point'
        if set_won:
            snapshot_type = 'set'
        elif game_won:
            snapshot_type = 'game'
        
        snapshot = {
            'type': snapshot_type,
            'timestamp': datetime.now().isoformat(),
            'point_number': st.session_state.total_points,
            'score': {
                'sets': f"{st.session_state.p1_sets}-{st.session_state.p2_sets}",
                'games': f"{st.session_state.p1_games}-{st.session_state.p2_games}",
                'points': f"{st.session_state.p1_point_score}-{st.session_state.p2_point_score}"
            },
            'probabilities': {
                'p1_win_prob': st.session_state.probability_history[-1]['p1'] if st.session_state.probability_history else 0.5,
                'p2_win_prob': st.session_state.probability_history[-1]['p2'] if st.session_state.probability_history else 0.5
            },
            'odds': {
                'match': {'p1': None, 'p2': None},  # Will be filled from user inputs
                'set': {'p1': None, 'p2': None},
                'game': {'p1': None, 'p2': None}
            }
        }
        st.session_state.match_snapshots.append(snapshot)
        
        # Auto-save to database
        match_data = {
            'player1_name': st.session_state.player1_name,
            'player2_name': st.session_state.player2_name,
            'surface': st.session_state.surface,
            'p1_serve_win': st.session_state.p1_serve_win,
            'p2_serve_win': st.session_state.p2_serve_win,
            'p1_return_win': st.session_state.p1_return_win,
            'p2_return_win': st.session_state.p2_return_win,
            'p1_sets': st.session_state.p1_sets,
            'p2_sets': st.session_state.p2_sets,
            'p1_games': st.session_state.p1_games,
            'p2_games': st.session_state.p2_games,
            'p1_points': st.session_state.p1_point_score,
            'p2_points': st.session_state.p2_point_score,
            'total_points': st.session_state.total_points,
            'p1_breaks': st.session_state.break_count_p1,
            'p2_breaks': st.session_state.break_count_p2,
            'probability_history': st.session_state.probability_history,
            'score_history': st.session_state.score_history,
            'point_winner_history': st.session_state.point_winner_history,
            'p1_games_won_history': st.session_state.p1_games_won_history,
            'p2_games_won_history': st.session_state.p2_games_won_history,
            'advanced_params': {'p1': st.session_state.p1_adv_params, 'p2': st.session_state.p2_adv_params},
            'match_snapshots': st.session_state.match_snapshots,
            'prematch_odds': {'p1': st.session_state.p1_prematch_odds, 'p2': st.session_state.p2_prematch_odds}
        }
        match_id = save_live_match(match_data)
        st.session_state.current_match_id = match_id
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])
    
    with col_btn1:
        if st.button(f"🟢 {st.session_state.player1_name} WINS POINT", key="p1_pt", use_container_width=True, type="primary"):
            handle_point_won(1)
            st.rerun()
    
    with col_btn2:
        if st.button(f"🔵 {st.session_state.player2_name} WINS POINT", key="p2_pt", use_container_width=True, type="primary"):
            handle_point_won(2)
            st.rerun()
    
    with col_btn3:
        if st.button("🔄 RESET", key="reset_btn", use_container_width=True):
            for key in ['p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'p1_point_score', 'p2_point_score',
                        'total_points', 'break_count_p1', 'break_count_p2', 'probability_history',
                        'score_history', 'point_winner_history', 'p1_games_won_history', 'p2_games_won_history']:
                if key in st.session_state:
                    st.session_state[key] = 0 if 'count' in key or 'total' in key or 'score' in key or 'sets' in key or 'games' in key or 'points' in key else []
            st.rerun()
    
    # Current score display
    st.markdown("### Current Score")
    col_s1, col_s2, col_s3 = st.columns([2, 2, 1])
    
    with col_s1:
        st.markdown(f"**{st.session_state.player1_name}**")
        st.markdown(f"<h2 style='color: #28a745;'>{st.session_state.p1_sets} | {st.session_state.p1_games} | {get_point_display(st.session_state.p1_point_score)}</h2>", unsafe_allow_html=True)
    
    with col_s2:
        st.markdown(f"**{st.session_state.player2_name}**")
        st.markdown(f"<h2 style='color: #007bff;'>{st.session_state.p2_sets} | {st.session_state.p2_games} | {get_point_display(st.session_state.p2_point_score)}</h2>", unsafe_allow_html=True)
    
    with col_s3:
        st.metric("Points", st.session_state.total_points)
        st.caption(f"Breaks: {st.session_state.break_count_p1}-{st.session_state.break_count_p2}")

# TAB 2: PROBABILITIES & VALUE BETS
# ============================================================================
with tab2:
    # Calculate current probabilities
    p1_serve_adj, p1_return_adj = apply_advanced_parameters(
        st.session_state.p1_serve_win, st.session_state.p1_return_win, st.session_state.p1_adv_params
    )
    p2_serve_adj, p2_return_adj = apply_advanced_parameters(
        st.session_state.p2_serve_win, st.session_state.p2_return_win, st.session_state.p2_adv_params
    )
    
    probs = calculate_markov_probabilities(p1_serve_adj, p2_serve_adj)
    
    # Adjust for current score
    point_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    p1_point_num = point_map.get(st.session_state.p1_point_score, 0)
    p2_point_num = point_map.get(st.session_state.p2_point_score, 0)
    
    p1_markov = adjust_for_current_score(
        probs['p1_match'], st.session_state.p1_sets, st.session_state.p1_games,
        p1_point_num, st.session_state.total_points, st.session_state.p1_games_won_history
    )
    
    p1_lr = calculate_ml_probabilities(
        p1_serve_adj, p2_serve_adj,
        st.session_state.p1_sets, st.session_state.p2_sets,
        st.session_state.p1_games, st.session_state.p2_games
    )
    
    p1_nn = calculate_nn_probabilities(
        p1_serve_adj, p2_serve_adj,
        st.session_state.p1_sets, st.session_state.p2_sets,
        st.session_state.p1_games, st.session_state.p2_games,
        p1_point_num, p2_point_num
    )
    
    # Ensemble
    p1_win_prob = (p1_markov * 0.4) + (p1_lr * 0.25) + (p1_nn * 0.35)
    p2_win_prob = 1 - p1_win_prob
    
    # Log to history if new point
    if st.session_state.total_points > 0 and len(st.session_state.probability_history) < st.session_state.total_points:
        st.session_state.probability_history.append({'p1': p1_win_prob, 'p2': p2_win_prob})
        current_score = f"{st.session_state.p1_sets}-{st.session_state.p2_sets}, {st.session_state.p1_games}-{st.session_state.p2_games}"
        st.session_state.score_history.append(current_score)
    
    # Display probabilities
    col_prob1, col_prob2 = st.columns(2)
    
    with col_prob1:
        # Calculate momentum
        p1_momentum = 0
        if len(st.session_state.probability_history) >= 2:
            recent_window = min(5, len(st.session_state.probability_history))
            old_p1 = st.session_state.probability_history[-recent_window]['p1']
            p1_momentum = (p1_win_prob - old_p1) * 100
        
        momentum_color = "🟢" if p1_momentum > 0 else "🔴" if p1_momentum < 0 else "⚪"
        momentum_text = f"{momentum_color} {p1_momentum:+.1f}%" if p1_momentum != 0 else "⚪ Stable"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white; margin: 0;'>{st.session_state.player1_name}</h3>
            <h1 style='color: white; font-size: 2.5em; margin: 10px 0;'>{p1_win_prob:.1%}</h1>
            <p style='color: white; font-size: 1.1em; margin: 0;'>Momentum: {momentum_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_prob2:
        p2_momentum = 0
        if len(st.session_state.probability_history) >= 2:
            recent_window = min(5, len(st.session_state.probability_history))
            old_p2 = st.session_state.probability_history[-recent_window]['p2']
            p2_momentum = (p2_win_prob - old_p2) * 100
        
        momentum_color = "🟢" if p2_momentum > 0 else "🔴" if p2_momentum < 0 else "⚪"
        momentum_text = f"{momentum_color} {p2_momentum:+.1f}%" if p2_momentum != 0 else "⚪ Stable"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); 
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white; margin: 0;'>{st.session_state.player2_name}</h3>
            <h1 style='color: white; font-size: 2.5em; margin: 10px 0;'>{p2_win_prob:.1%}</h1>
            <p style='color: white; font-size: 1.1em; margin: 0;'>Momentum: {momentum_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # VALUE BETS SECTION WITH SAVE BUTTONS
    st.subheader("💰 Recommended Value Bets")
    
    # MATCH WINNER ODDS
    st.markdown("**Match Winner Odds:**")
    col_odds1, col_odds2 = st.columns(2)
    
    with col_odds1:
        p1_match_odds = st.number_input(f"{st.session_state.player1_name} Match", min_value=1.01, max_value=50.0, value=1.85, step=0.05, key="p1_odds")
    
    with col_odds2:
        p2_match_odds = st.number_input(f"{st.session_state.player2_name} Match", min_value=1.01, max_value=50.0, value=2.10, step=0.05, key="p2_odds")
    
    # SET WINNER ODDS (current set)
    st.markdown("**Current Set Winner Odds:**")
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        p1_set_odds = st.number_input(f"{st.session_state.player1_name} Set", min_value=1.01, max_value=50.0, value=1.75, step=0.05, key="p1_set_odds")
    
    with col_set2:
        p2_set_odds = st.number_input(f"{st.session_state.player2_name} Set", min_value=1.01, max_value=50.0, value=2.20, step=0.05, key="p2_set_odds")
    
    # NEXT GAME WINNER ODDS
    st.markdown("**Next Game Winner Odds:**")
    col_game1, col_game2 = st.columns(2)
    
    with col_game1:
        p1_game_odds = st.number_input(f"{st.session_state.player1_name} Game", min_value=1.01, max_value=10.0, value=1.50, step=0.05, key="p1_game_odds")
    
    with col_game2:
        p2_game_odds = st.number_input(f"{st.session_state.player2_name} Game", min_value=1.01, max_value=10.0, value=2.75, step=0.05, key="p2_game_odds")
    
    # Calculate value bets for all markets
    st.markdown("---")
    st.subheader("🎯 Value Bet Alerts")
    
    # MATCH WINNER VALUE BETS
    p1_implied = 1 / p1_match_odds
    p2_implied = 1 / p2_match_odds
    p1_edge = p1_win_prob - p1_implied
    p2_edge = p2_win_prob - p2_implied
    p1_ev = (p1_win_prob * (p1_match_odds - 1)) - (1 - p1_win_prob)
    p2_ev = (p2_win_prob * (p2_match_odds - 1)) - (1 - p2_win_prob)
    p1_stake = min(p1_edge * 400, 150) if p1_edge > 0.025 else 0
    p2_stake = min(p2_edge * 400, 150) if p2_edge > 0.025 else 0
    
    # SET WINNER VALUE BETS (using current win prob as proxy)
    p1_set_implied = 1 / p1_set_odds
    p2_set_implied = 1 / p2_set_odds
    p1_set_edge = p1_win_prob - p1_set_implied  # Simplified
    p2_set_edge = p2_win_prob - p2_set_implied
    p1_set_ev = (p1_win_prob * (p1_set_odds - 1)) - (1 - p1_win_prob)
    p2_set_ev = (p2_win_prob * (p2_set_odds - 1)) - (1 - p2_win_prob)
    p1_set_stake = min(p1_set_edge * 300, 100) if p1_set_edge > 0.025 else 0
    p2_set_stake = min(p2_set_edge * 300, 100) if p2_set_edge > 0.025 else 0
    
    # GAME WINNER VALUE BETS
    p1_game_implied = 1 / p1_game_odds
    p2_game_implied = 1 / p2_game_odds
    p1_game_edge = p1_win_prob - p1_game_implied  # Simplified
    p2_game_edge = p2_win_prob - p2_game_implied
    p1_game_ev = (p1_win_prob * (p1_game_odds - 1)) - (1 - p1_win_prob)
    p2_game_ev = (p2_win_prob * (p2_game_odds - 1)) - (1 - p2_win_prob)
    p1_game_stake = min(p1_game_edge * 200, 50) if p1_game_edge > 0.025 else 0
    p2_game_stake = min(p2_game_edge * 200, 50) if p2_game_edge > 0.025 else 0
    
    # Display all value bets
    col_bet1, col_bet2 = st.columns(2)
    
    with col_bet1:
        st.markdown(f"**{st.session_state.player1_name} Bets:**")
        
        # Match Winner
        if p1_edge > 0.025:
            st.markdown(f"""
            <div class='value-bet-alert'>
                ✅ <strong>MATCH WINNER</strong><br/>
                💰 Stake: <strong>${p1_stake:.0f}</strong> @ {p1_match_odds:.2f}<br/>
                📈 EV: <strong>{p1_ev:+.2%}</strong> | Edge: <strong>{p1_edge:+.2%}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"💾 SAVE MATCH BET", key="save_p1_match"):
                bet_data = {
                    'match_id': st.session_state.current_match_id,
                    'player1_name': st.session_state.player1_name,
                    'player2_name': st.session_state.player2_name,
                    'bet_type': 'Match Winner',
                    'selection': st.session_state.player1_name,
                    'odds': p1_match_odds,
                    'probability': p1_win_prob,
                    'edge': p1_edge,
                    'expected_value': p1_ev,
                    'recommended_stake': p1_stake,
                    'model_confidence': 0.85,
                    'current_score': f"{st.session_state.p1_sets}-{st.session_state.p2_sets}, {st.session_state.p1_games}-{st.session_state.p2_games}",
                    'notes': f"Match bet @ {datetime.now().strftime('%H:%M')}"
                }
                bet_id = save_selected_bet(bet_data)
                st.success(f"✅ Match bet saved! ID: {bet_id}")
        else:
            st.info(f"❌ Match: No value at {p1_match_odds:.2f}")
        
        # Set Winner
        if p1_set_edge > 0.025:
            st.markdown(f"""
            <div class='value-bet-alert'>
                ✅ <strong>SET WINNER</strong><br/>
                💰 Stake: <strong>${p1_set_stake:.0f}</strong> @ {p1_set_odds:.2f}<br/>
                📈 EV: <strong>{p1_set_ev:+.2%}</strong> | Edge: <strong>{p1_set_edge:+.2%}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"💾 SAVE SET BET", key="save_p1_set"):
                bet_data = {
                    'match_id': st.session_state.current_match_id,
                    'player1_name': st.session_state.player1_name,
                    'player2_name': st.session_state.player2_name,
                    'bet_type': 'Set Winner',
                    'selection': st.session_state.player1_name,
                    'odds': p1_set_odds,
                    'probability': p1_win_prob,
                    'edge': p1_set_edge,
                    'expected_value': p1_set_ev,
                    'recommended_stake': p1_set_stake,
                    'model_confidence': 0.80,
                    'current_score': f"Set {st.session_state.p1_sets + st.session_state.p2_sets + 1}: {st.session_state.p1_games}-{st.session_state.p2_games}",
                    'notes': f"Set bet @ {datetime.now().strftime('%H:%M')}"
                }
                bet_id = save_selected_bet(bet_data)
                st.success(f"✅ Set bet saved! ID: {bet_id}")
        else:
            st.info(f"❌ Set: No value at {p1_set_odds:.2f}")
        
        # Game Winner
        if p1_game_edge > 0.025:
            st.markdown(f"""
            <div class='value-bet-alert'>
                ✅ <strong>GAME WINNER</strong><br/>
                💰 Stake: <strong>${p1_game_stake:.0f}</strong> @ {p1_game_odds:.2f}<br/>
                📈 EV: <strong>{p1_game_ev:+.2%}</strong> | Edge: <strong>{p1_game_edge:+.2%}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"💾 SAVE GAME BET", key="save_p1_game"):
                bet_data = {
                    'match_id': st.session_state.current_match_id,
                    'player1_name': st.session_state.player1_name,
                    'player2_name': st.session_state.player2_name,
                    'bet_type': 'Game Winner',
                    'selection': st.session_state.player1_name,
                    'odds': p1_game_odds,
                    'probability': p1_win_prob,
                    'edge': p1_game_edge,
                    'expected_value': p1_game_ev,
                    'recommended_stake': p1_game_stake,
                    'model_confidence': 0.75,
                    'current_score': f"Game {st.session_state.p1_games + st.session_state.p2_games + 1}",
                    'notes': f"Next game bet @ {datetime.now().strftime('%H:%M')}"
                }
                bet_id = save_selected_bet(bet_data)
                st.success(f"✅ Game bet saved! ID: {bet_id}")
        else:
            st.info(f"❌ Game: No value at {p1_game_odds:.2f}")
    
    with col_bet2:
        st.markdown(f"**{st.session_state.player2_name} Bets:**")
        
        # Match Winner
        if p2_edge > 0.025:
            st.markdown(f"""
            <div class='value-bet-alert'>
                ✅ <strong>MATCH WINNER</strong><br/>
                💰 Stake: <strong>${p2_stake:.0f}</strong> @ {p2_match_odds:.2f}<br/>
                📈 EV: <strong>{p2_ev:+.2%}</strong> | Edge: <strong>{p2_edge:+.2%}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"💾 SAVE MATCH BET ", key="save_p2_match"):
                bet_data = {
                    'match_id': st.session_state.current_match_id,
                    'player1_name': st.session_state.player1_name,
                    'player2_name': st.session_state.player2_name,
                    'bet_type': 'Match Winner',
                    'selection': st.session_state.player2_name,
                    'odds': p2_match_odds,
                    'probability': p2_win_prob,
                    'edge': p2_edge,
                    'expected_value': p2_ev,
                    'recommended_stake': p2_stake,
                    'model_confidence': 0.85,
                    'current_score': f"{st.session_state.p1_sets}-{st.session_state.p2_sets}, {st.session_state.p1_games}-{st.session_state.p2_games}",
                    'notes': f"Match bet @ {datetime.now().strftime('%H:%M')}"
                }
                bet_id = save_selected_bet(bet_data)
                st.success(f"✅ Match bet saved! ID: {bet_id}")
        else:
            st.info(f"❌ Match: No value at {p2_match_odds:.2f}")
        
        # Set Winner
        if p2_set_edge > 0.025:
            st.markdown(f"""
            <div class='value-bet-alert'>
                ✅ <strong>SET WINNER</strong><br/>
                💰 Stake: <strong>${p2_set_stake:.0f}</strong> @ {p2_set_odds:.2f}<br/>
                📈 EV: <strong>{p2_set_ev:+.2%}</strong> | Edge: <strong>{p2_set_edge:+.2%}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"💾 SAVE SET BET ", key="save_p2_set"):
                bet_data = {
                    'match_id': st.session_state.current_match_id,
                    'player1_name': st.session_state.player1_name,
                    'player2_name': st.session_state.player2_name,
                    'bet_type': 'Set Winner',
                    'selection': st.session_state.player2_name,
                    'odds': p2_set_odds,
                    'probability': p2_win_prob,
                    'edge': p2_set_edge,
                    'expected_value': p2_set_ev,
                    'recommended_stake': p2_set_stake,
                    'model_confidence': 0.80,
                    'current_score': f"Set {st.session_state.p1_sets + st.session_state.p2_sets + 1}: {st.session_state.p1_games}-{st.session_state.p2_games}",
                    'notes': f"Set bet @ {datetime.now().strftime('%H:%M')}"
                }
                bet_id = save_selected_bet(bet_data)
                st.success(f"✅ Set bet saved! ID: {bet_id}")
        else:
            st.info(f"❌ Set: No value at {p2_set_odds:.2f}")
        
        # Game Winner
        if p2_game_edge > 0.025:
            st.markdown(f"""
            <div class='value-bet-alert'>
                ✅ <strong>GAME WINNER</strong><br/>
                💰 Stake: <strong>${p2_game_stake:.0f}</strong> @ {p2_game_odds:.2f}<br/>
                📈 EV: <strong>{p2_game_ev:+.2%}</strong> | Edge: <strong>{p2_game_edge:+.2%}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"💾 SAVE GAME BET ", key="save_p2_game"):
                bet_data = {
                    'match_id': st.session_state.current_match_id,
                    'player1_name': st.session_state.player1_name,
                    'player2_name': st.session_state.player2_name,
                    'bet_type': 'Game Winner',
                    'selection': st.session_state.player2_name,
                    'odds': p2_game_odds,
                    'probability': p2_win_prob,
                    'edge': p2_game_edge,
                    'expected_value': p2_game_ev,
                    'recommended_stake': p2_game_stake,
                    'model_confidence': 0.75,
                    'current_score': f"Game {st.session_state.p1_games + st.session_state.p2_games + 1}",
                    'notes': f"Next game bet @ {datetime.now().strftime('%H:%M')}"
                }
                bet_id = save_selected_bet(bet_data)
                st.success(f"✅ Game bet saved! ID: {bet_id}")
        else:
            st.info(f"❌ Game: No value at {p2_game_odds:.2f}")

# TAB 3: MY SELECTED BETS
# ============================================================================
with tab3:
    st.subheader("💰 My Selected Bets")
    
    pending_bets = get_pending_bets(limit=50)
    
    if pending_bets:
        df = pd.DataFrame(pending_bets)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Display as table
        st.dataframe(
            df[['created_at', 'bet_type', 'selection', 'odds', 'edge', 'expected_value', 
                'recommended_stake', 'current_score', 'notes']].rename(columns={
                'created_at': 'Time',
                'bet_type': 'Type',
                'selection': 'Selection',
                'odds': 'Odds',
                'edge': 'Edge',
                'expected_value': 'EV',
                'recommended_stake': 'Stake ($)',
                'current_score': 'Score',
                'notes': 'Notes'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption(f"**Total Bets:** {len(pending_bets)} | **Total Stake:** ${df['recommended_stake'].sum():.0f}")
    else:
        st.info("No bets saved yet. Go to 'Probability & Bets' tab to save recommended bets!")

# TAB 4: ANALYTICS
# ============================================================================
with tab4:
    if len(st.session_state.probability_history) > 1:
        st.subheader("📈 Probability Evolution")
        
        points = list(range(1, len(st.session_state.probability_history) + 1))
        p1_probs = [p['p1'] * 100 for p in st.session_state.probability_history]
        p2_probs = [p['p2'] * 100 for p in st.session_state.probability_history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=points, y=p1_probs,
            mode='lines+markers',
            name=st.session_state.player1_name,
            line=dict(color='#28a745', width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=points, y=p2_probs,
            mode='lines+markers',
            name=st.session_state.player2_name,
            line=dict(color='#007bff', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Win Probability Throughout Match",
            xaxis_title="Point Number",
            yaxis_title="Win Probability (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key momentum shifts
        st.subheader("🔥 Key Momentum Shifts")
        shifts = []
        for i in range(1, len(st.session_state.probability_history)):
            p1_change = (st.session_state.probability_history[i]['p1'] - st.session_state.probability_history[i-1]['p1']) * 100
            if abs(p1_change) >= 3:
                winner = st.session_state.player1_name if st.session_state.point_winner_history[i-1] == 1 else st.session_state.player2_name
                shifts.append({
                    'point': i,
                    'winner': winner,
                    'change': p1_change
                })
        
        if shifts:
            for shift in shifts[-5:]:
                arrow = "📈" if shift['change'] > 0 else "📉"
                st.caption(f"**Point {shift['point']}**: {shift['winner']} won → {arrow} {abs(shift['change']):.1f}% shift")
        else:
            st.caption("No significant momentum shifts yet (3%+ changes)")
    else:
        st.info("Start tracking points to see analytics!")

# TAB 5: MATCH SNAPSHOTS (Odds/Probability History)
# ============================================================================
with tab5:
    st.subheader("📸 Match Instance Snapshots")
    st.caption("Complete history of score, probabilities, and odds at every point/game/set")
    
    if st.session_state.match_snapshots:
        # Filter options
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_type = st.selectbox("Filter by:", ["All", "Sets Only", "Games Only", "Every Point"], key="snap_filter")
        with col_f2:
            show_count = st.number_input("Show last N snapshots:", min_value=5, max_value=200, value=20, step=5)
        
        # Filter snapshots
        filtered_snaps = st.session_state.match_snapshots
        if filter_type == "Sets Only":
            filtered_snaps = [s for s in filtered_snaps if s['type'] == 'set']
        elif filter_type == "Games Only":
            filtered_snaps = [s for s in filtered_snaps if s['type'] == 'game']
        
        # Show last N
        display_snaps = filtered_snaps[-show_count:]
        
        # Create dataframe
        snapshot_data = []
        for snap in display_snaps:
            snapshot_data.append({
                'Point': snap['point_number'],
                'Type': snap['type'].upper(),
                'Sets': snap['score']['sets'],
                'Games': snap['score']['games'],
                'Points': snap['score']['points'],
                f"{st.session_state.player1_name} Prob": f"{snap['probabilities']['p1_win_prob']:.1%}",
                f"{st.session_state.player2_name} Prob": f"{snap['probabilities']['p2_win_prob']:.1%}",
                'Time': snap['timestamp'].split('T')[1].split('.')[0] if 'T' in snap['timestamp'] else snap['timestamp']
            })
        
        df_snapshots = pd.DataFrame(snapshot_data)
        st.dataframe(df_snapshots, use_container_width=True, hide_index=True)
        
        st.caption(f"**Total Snapshots:** {len(st.session_state.match_snapshots)} | **Shown:** {len(display_snaps)}")
        
        # Show pre-match odds comparison
        st.markdown("---")
        st.subheader("📊 Pre-Match vs Current Odds Movement")
        col_pm1, col_pm2 = st.columns(2)
        
        with col_pm1:
            st.metric(
                f"{st.session_state.player1_name}",
                f"Pre: {st.session_state.p1_prematch_odds:.2f}",
                help="Pre-match odds you entered"
            )
        
        with col_pm2:
            st.metric(
                f"{st.session_state.player2_name}",
                f"Pre: {st.session_state.p2_prematch_odds:.2f}",
                help="Pre-match odds you entered"
            )
        
        st.info("💡 **Tip**: Enter current odds in 'Probability & Bets' tab to track odds movements. Snapshots will record odds changes at important match moments.")
    else:
        st.info("No snapshots yet. Start tracking points to create match history!")

st.markdown("---")
st.caption("💾 Data auto-saves after each point | 🔄 Use 'Load Match' to restore progress | 📸 Snapshots track complete match state")
