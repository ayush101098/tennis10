"""
COMPREHENSIVE LIVE TENNIS TRACKER & EDGE DETECTOR
==================================================
Pre-match setup → Live tracking → True P calculation → ALL edge detection
Uses 112,384 real matches + ML models (94-98% accuracy)
Enhanced with Qian et al. (2025) research paper innovations:
  - 4-parameter Performance Model (E, S, P, R) with Entropy Weights
  - EWMA Momentum (λ=0.6467, Pearson r=0.731)
  - Logistic Regression Point Prediction (70% accuracy)
  - Monte Carlo Match Simulation for real-time win probability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import pickle
import os
import sys
import math
import random
from datetime import datetime
from pathlib import Path

# Add project root to path for API import
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from api.free_live_data import FreeLiveTennisService, get_free_service
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False

st.set_page_config(page_title="Live Tennis Tracker Pro", page_icon="🎾", layout="wide")

# Enhanced CSS
st.markdown("""
<style>
    .edge-critical {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        font-weight: bold;
        border-left: 5px solid #ff0000;
    }
    .edge-high {
        background: linear-gradient(135deg, #fd7e14 0%, #e8590c 100%);
        color: white;
        padding: 10px;
        border-radius: 6px;
        margin: 6px 0;
        font-weight: bold;
    }
    .edge-medium {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 8px;
        margin: 6px 0;
        color: #856404;
    }
    .value-bet {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px;
        margin: 8px 0;
        border-radius: 4px;
    }
    .score-box {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid #dee2e6;
    }
    .prob-display {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 6px;
        margin: 5px 0;
    }
    .stats-card {
        background: white;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎾 Live Tennis Tracker & Edge Detector Pro")
st.caption("📊 Real-time tracking | 🤖 ML predictions (94-98% accuracy) | 🎯 ALL edge detection")

# ============================================================================
# LOAD ML MODELS & REAL DATA
# ============================================================================

@st.cache_resource
def load_ml_models():
    """Load trained ML models"""
    try:
        import os
        # Get the absolute path to the project root
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(current_dir, 'ml_models')
        
        lr_path = os.path.join(models_dir, 'logistic_regression_advanced.pkl')
        rf_path = os.path.join(models_dir, 'random_forest_advanced.pkl')
        scaler_path = os.path.join(models_dir, 'scaler_advanced.pkl')
        features_path = os.path.join(models_dir, 'feature_names_advanced.pkl')
        
        if not os.path.exists(lr_path):
            st.warning(f"ML models not found at {models_dir}")
            return None, None, None, None
        
        with open(lr_path, 'rb') as f:
            lr_model = pickle.load(f)
        with open(rf_path, 'rb') as f:
            rf_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        
        return lr_model, rf_model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading ML models: {str(e)}")
        return None, None, None, None

@st.cache_data
def load_real_bp_stats():
    """Load real break point statistics from database"""
    try:
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(current_dir, 'tennis_betting.db')
        
        if not os.path.exists(db_path):
            st.error(f"Database not found at {db_path}")
            return pd.DataFrame(), pd.DataFrame()
        
        conn = sqlite3.connect(db_path)
        
        # Get surface-specific BP stats
        query = '''
        SELECT 
            surface,
            AVG(CAST(w_bpSaved AS FLOAT) / NULLIF(w_bpFaced, 0)) as winner_bp_save,
            AVG(CAST(l_bpSaved AS FLOAT) / NULLIF(l_bpFaced, 0)) as loser_bp_save,
            AVG(w_bpFaced) as avg_bp_faced_winner,
            AVG(l_bpFaced) as avg_bp_faced_loser,
            COUNT(*) as match_count
        FROM matches
        WHERE w_bpSaved IS NOT NULL AND surface IN ('Hard', 'Clay', 'Grass')
        GROUP BY surface
        '''
        stats = pd.read_sql_query(query, conn)
        
        # Get critical game stats
        critical_query = '''
        SELECT 
            AVG(CASE WHEN w_bpFaced >= 10 THEN 1 ELSE 0 END) as high_bp_faced_win_rate,
            AVG(CASE WHEN l_bpFaced >= 10 THEN 0 ELSE 1 END) as high_bp_faced_loss_rate
        FROM matches
        WHERE w_bpFaced IS NOT NULL
        '''
        critical_stats = pd.read_sql_query(critical_query, conn)
        
        conn.close()
        return stats, critical_stats
    except Exception as e:
        st.error(f"Error loading break point stats: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Load models and data
lr_model, rf_model, scaler, feature_names = load_ml_models()
bp_stats, critical_stats = load_real_bp_stats()
models_loaded = lr_model is not None
data_loaded = not bp_stats.empty

# Display model status in sidebar
st.sidebar.markdown("### 🤖 System Status")
if models_loaded:
    st.sidebar.success("✅ ML Models Loaded (LR: 94.29%, RF: 98.04%)")
else:
    st.sidebar.warning("⚠️ ML Models not found - using Markov only")

st.sidebar.markdown("### 📊 Real Data Stats")
if data_loaded:
    for _, row in bp_stats.iterrows():
        st.sidebar.caption(f"**{row['surface']}:** W:{row['winner_bp_save']:.1%} / L:{row['loser_bp_save']:.1%} ({int(row['match_count']):,} matches)")
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Edge:** Winners save 63% BP, losers 49%. 14.4% gap!")
else:
    st.sidebar.error("⚠️ Database not loaded - check tennis_betting.db")

# API status in sidebar
st.sidebar.markdown("### 🌐 Live Data API")
if _API_AVAILABLE:
    if st.session_state.get("api_match_key"):
        st.sidebar.success(f"🔴 Connected — Match #{st.session_state.api_match_key}")
        if st.session_state.get("api_auto_refresh"):
            st.sidebar.caption("🔄 Auto-refreshing every 30s")
    else:
        st.sidebar.info("✅ ESPN Free API — no key needed")
else:
    st.sidebar.caption("API module not available")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Match setup
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False
if 'p1_name' not in st.session_state:
    st.session_state.p1_name = "Player 1"
if 'p2_name' not in st.session_state:
    st.session_state.p2_name = "Player 2"
if 'surface' not in st.session_state:
    st.session_state.surface = "Hard"
if 'best_of' not in st.session_state:
    st.session_state.best_of = 3

# Pre-match probabilities
if 'pre_p1_prob' not in st.session_state:
    st.session_state.pre_p1_prob = 0.50
if 'p1_match_odds' not in st.session_state:
    st.session_state.p1_match_odds = 1.85
if 'p2_match_odds' not in st.session_state:
    st.session_state.p2_match_odds = 2.10

# Match conditions
if 'court_speed' not in st.session_state:
    st.session_state.court_speed = 50  # 0-100 scale
if 'temperature' not in st.session_state:
    st.session_state.temperature = 20
if 'altitude' not in st.session_state:
    st.session_state.altitude = 0
if 'indoor' not in st.session_state:
    st.session_state.indoor = False

# Score tracking
if 'sets1' not in st.session_state:
    st.session_state.sets1, st.session_state.sets2 = 0, 0
if 'games1' not in st.session_state:
    st.session_state.games1, st.session_state.games2 = 0, 0
if 'points1' not in st.session_state:
    st.session_state.points1, st.session_state.points2 = 0, 0
if 'server' not in st.session_state:
    st.session_state.server = 1

# Live stats tracking
if 'total_bp_p1' not in st.session_state:
    st.session_state.total_bp_p1, st.session_state.total_bp_p2 = 0, 0
if 'bp_converted_p1' not in st.session_state:
    st.session_state.bp_converted_p1, st.session_state.bp_converted_p2 = 0, 0
if 'aces_p1' not in st.session_state:
    st.session_state.aces_p1, st.session_state.aces_p2 = 0, 0
if 'dfs_p1' not in st.session_state:
    st.session_state.dfs_p1, st.session_state.dfs_p2 = 0, 0
if 'winners_p1' not in st.session_state:
    st.session_state.winners_p1, st.session_state.winners_p2 = 0, 0
if 'ues_p1' not in st.session_state:
    st.session_state.ues_p1, st.session_state.ues_p2 = 0, 0

# Probability history
if 'prob_history' not in st.session_state:
    st.session_state.prob_history = []

# Live bookmaker odds tracking
if 'live_p1_match_odds' not in st.session_state:
    st.session_state.live_p1_match_odds = 1.85
if 'live_p2_match_odds' not in st.session_state:
    st.session_state.live_p2_match_odds = 2.10
if 'live_p1_set_odds' not in st.session_state:
    st.session_state.live_p1_set_odds = 1.70
if 'live_p2_set_odds' not in st.session_state:
    st.session_state.live_p2_set_odds = 2.20
if 'live_p1_game_odds' not in st.session_state:
    st.session_state.live_p1_game_odds = 1.50
if 'live_p2_game_odds' not in st.session_state:
    st.session_state.live_p2_game_odds = 2.60

# Pre-match odds storage (for variance tracking)
if 'pre_match_p1_odds' not in st.session_state:
    st.session_state.pre_match_p1_odds = 1.85
if 'pre_match_p2_odds' not in st.session_state:
    st.session_state.pre_match_p2_odds = 2.10

# Momentum learning & match intelligence
if 'momentum_score' not in st.session_state:
    st.session_state.momentum_score = 0  # -10 to +10, positive = P1
if 'p1_clutch_points_won' not in st.session_state:
    st.session_state.p1_clutch_points_won = 0
if 'p1_clutch_points_total' not in st.session_state:
    st.session_state.p1_clutch_points_total = 0
if 'p2_clutch_points_won' not in st.session_state:
    st.session_state.p2_clutch_points_won = 0
if 'p2_clutch_points_total' not in st.session_state:
    st.session_state.p2_clutch_points_total = 0
if 'recent_games_p1' not in st.session_state:
    st.session_state.recent_games_p1 = []  # Track last 5 games won/lost
if 'recent_games_p2' not in st.session_state:
    st.session_state.recent_games_p2 = []

# Bet tracking system
if 'active_bets' not in st.session_state:
    st.session_state.active_bets = []  # List of {market, player, odds, stake, time, score}
if 'settled_bets' not in st.session_state:
    st.session_state.settled_bets = []  # List of settled bets with P&L
if 'total_staked' not in st.session_state:
    st.session_state.total_staked = 0.0
if 'total_returns' not in st.session_state:
    st.session_state.total_returns = 0.0
if 'total_profit' not in st.session_state:
    st.session_state.total_profit = 0.0

# ============================================================================
# RESEARCH PAPER STATE: Qian et al. (2025) - EWMA Momentum & Performance
# ============================================================================

# Point-by-point history for EWMA momentum calculation
if 'point_history' not in st.session_state:
    st.session_state.point_history = []  # List of {winner: 1|2, server: 1|2, rally_count: int, ...}

# EWMA Momentum for each player (Eq 9: M_t = λ·P_t + (1-λ)·M_{t-1})
if 'ewma_momentum_p1' not in st.session_state:
    st.session_state.ewma_momentum_p1 = 50.0  # 0-100 scale, starts neutral
if 'ewma_momentum_p2' not in st.session_state:
    st.session_state.ewma_momentum_p2 = 50.0

# Performance history (4-param model: E, S, P, R)
if 'performance_history_p1' not in st.session_state:
    st.session_state.performance_history_p1 = []
if 'performance_history_p2' not in st.session_state:
    st.session_state.performance_history_p2 = []

# Momentum history for visualization (like Fig 5)
if 'momentum_history' not in st.session_state:
    st.session_state.momentum_history = []

# Winning streaks
if 'current_streak_p1' not in st.session_state:
    st.session_state.current_streak_p1 = 0
if 'current_streak_p2' not in st.session_state:
    st.session_state.current_streak_p2 = 0

# Return game tracking (last 5 return games)
if 'return_games_p1' not in st.session_state:
    st.session_state.return_games_p1 = []  # List of {points_won, points_total}
if 'return_games_p2' not in st.session_state:
    st.session_state.return_games_p2 = []

# Serve tracking per game
if 'serve_faults_p1' not in st.session_state:
    st.session_state.serve_faults_p1 = 0
if 'serve_faults_p2' not in st.session_state:
    st.session_state.serve_faults_p2 = 0
if 'total_serves_p1' not in st.session_state:
    st.session_state.total_serves_p1 = 0
if 'total_serves_p2' not in st.session_state:
    st.session_state.total_serves_p2 = 0

# Simulation results cache
if 'sim_win_prob_p1' not in st.session_state:
    st.session_state.sim_win_prob_p1 = 0.5
if 'sim_history' not in st.session_state:
    st.session_state.sim_history = []

# API-fed live data state (ESPN free — no key needed)
if 'api_match_key' not in st.session_state:
    st.session_state.api_match_key = None
if 'api_auto_refresh' not in st.session_state:
    st.session_state.api_auto_refresh = False
if 'api_last_refresh' not in st.session_state:
    st.session_state.api_last_refresh = None

# ============================================================================
# CORE PROBABILITY CALCULATION FUNCTIONS
# ============================================================================

# ============================================================================
# RESEARCH PAPER MODEL: Qian et al. (2025) - Performance & Momentum
# Ref: Journal of Computers Vol. 36 No. 1, Feb 2025
# ============================================================================

# Entropy-derived weights from Table 7 (averaged across matches)
PERF_WEIGHTS = {
    'scoring_efficiency': 0.1307,  # E - rally-based scoring efficiency
    'winning_streak':     0.1184,  # S - consecutive points won
    'serve_efficiency':   0.1358,  # P - serve success rate
    'returner_win_rate':  0.6151,  # R - points won as returner (dominant factor)
}

# Optimized EWMA decay factor from simulated annealing (Table 5)
EWMA_LAMBDA = 0.6467  # Pearson r = 0.731 with this λ

def calculate_scoring_efficiency(won_point, rally_count):
    """
    Scoring Efficiency (E) from paper Section 2.1.
    Maps rally count to efficiency: short rallies = high |E|, long rallies → 0.
    Sign indicates won (+) or lost (-).
    For double faults (rally=0), treat as rally=1.
    """
    r = max(1, rally_count)
    # Efficiency decays with rally length: E = sign / (1 + log(r))
    efficiency = 1.0 / (1.0 + math.log(r))
    return efficiency if won_point else -efficiency

def calculate_winning_streak(player_num):
    """
    Winning Streak (S) from paper Section 2.1.
    Returns current consecutive points won, normalized to 0-1 scale.
    """
    if player_num == 1:
        streak = st.session_state.current_streak_p1
    else:
        streak = st.session_state.current_streak_p2
    # Normalize: cap at 10 consecutive points for scaling
    return min(streak, 10) / 10.0

def calculate_serve_efficiency(player_num):
    """
    Serve Efficiency (P) from paper Section 2.1.
    Gauges how frequently a player successfully serves.
    P = (total_serves - faults) / total_serves
    """
    if player_num == 1:
        total = st.session_state.total_serves_p1
        faults = st.session_state.serve_faults_p1
    else:
        total = st.session_state.total_serves_p2
        faults = st.session_state.serve_faults_p2
    
    if total == 0:
        return 0.85  # Default serve efficiency
    return max(0.0, (total - faults) / total)

def calculate_returner_win_rate(player_num):
    """
    Returner's Win Rate (R) from paper Section 2.1.
    Points won while playing as returner in last 5 return games.
    This is the most heavily weighted indicator (w=0.6151).
    """
    if player_num == 1:
        games = st.session_state.return_games_p1[-5:]
    else:
        games = st.session_state.return_games_p2[-5:]
    
    if not games:
        return 0.30  # Default: servers win ~70%, so returner wins ~30%
    
    total_won = sum(g.get('points_won', 0) for g in games)
    total_pts = sum(g.get('points_total', 1) for g in games)
    
    return total_won / max(1, total_pts)

def calculate_point_performance(player_num, won_point, rally_count=3, is_server=True):
    """
    Calculate Performance P_j for a single point using the 4-parameter linear model.
    
    From paper Eq. (8):
    P_j = w1·E(j) + w2·P(j) + w3·S(j) + w4·R(j) - K_j
    
    K_j is a constant to normalize serving vs returning advantage.
    The paper found servers have ~70% win rate, so K adjusts for this.
    
    Returns performance on 0-100 scale.
    """
    w = PERF_WEIGHTS
    
    E = calculate_scoring_efficiency(won_point, rally_count)
    S = calculate_winning_streak(player_num)
    P = calculate_serve_efficiency(player_num)
    R = calculate_returner_win_rate(player_num)
    
    # Raw performance
    raw_perf = (w['scoring_efficiency'] * E +
                w['winning_streak'] * S +
                w['serve_efficiency'] * P +
                w['returner_win_rate'] * R)
    
    # K adjustment: remove serving advantage bias (paper Section 2.1)
    # Server avg performance ~0.3 higher, so subtract when serving, add when returning
    K = 0.15 if is_server else -0.15
    adjusted_perf = raw_perf - K
    
    # Map to 0-100 scale (raw range is approximately -0.6 to +0.8)
    scaled = (adjusted_perf + 0.6) / 1.4 * 100
    return max(0, min(100, scaled))

def update_ewma_momentum(player_num, performance):
    """
    Update EWMA Momentum from paper Eq. (9):
    M_t = λ · P_t + (1-λ) · M_{t-1}
    
    λ = 0.6467 (optimized via simulated annealing, Pearson r=0.731)
    
    Recent performance is weighted more heavily, older data decays exponentially.
    """
    lam = EWMA_LAMBDA
    
    if player_num == 1:
        old_m = st.session_state.ewma_momentum_p1
        new_m = lam * performance + (1 - lam) * old_m
        st.session_state.ewma_momentum_p1 = new_m
    else:
        old_m = st.session_state.ewma_momentum_p2
        new_m = lam * performance + (1 - lam) * old_m
        st.session_state.ewma_momentum_p2 = new_m
    
    return new_m

def momentum_to_point_win_prob(momentum_p1, momentum_p2):
    """
    Convert momentum to point win probability using logistic regression.
    
    From paper Eq. (11):
    P(win) = 1 / (1 + exp(-(β0 + β1·M_t)))
    
    With optimized params: C=100, penalty='l1', solver='liblinear'
    The paper achieved 70% accuracy on theoretical point prediction.
    
    We use the momentum differential as the input feature.
    """
    # Momentum differential (normalized to roughly -1 to +1)
    m_diff = (momentum_p1 - momentum_p2) / 100.0
    
    # Logistic regression coefficients (calibrated from paper's findings)
    # β0 = 0 (centered), β1 = 2.5 (moderate sensitivity)
    beta0 = 0.0
    beta1 = 2.5
    
    logit = beta0 + beta1 * m_diff
    prob = 1.0 / (1.0 + math.exp(-logit))
    
    return max(0.05, min(0.95, prob))

def simulate_match_from_position(p1_point_prob, sets1, sets2, games1, games2, 
                                  points1, points2, server, best_of=3, 
                                  n_simulations=1000):
    """
    Monte Carlo Match Simulation from paper Section 6 (Fig. 7).
    
    Starting from current score position, simulate all probable outcomes
    by traversing the match tree. Uses point-level prediction from momentum
    to simulate each point, then rolls up to games, sets, and match.
    
    Returns P1 match win probability.
    """
    wins_p1 = 0
    p2_point_prob = 1 - p1_point_prob
    
    for _ in range(n_simulations):
        # Copy current state
        s1, s2 = sets1, sets2
        g1, g2 = games1, games2
        pt1, pt2 = points1, points2
        srv = server
        sets_to_win = (best_of + 1) // 2
        
        # Simulate until match ends
        max_points = 500  # Safety cap
        point_count = 0
        
        while s1 < sets_to_win and s2 < sets_to_win and point_count < max_points:
            point_count += 1
            
            # Determine point win probability based on who's serving
            if srv == 1:
                p_win = p1_point_prob + 0.10  # Server advantage ~10%
            else:
                p_win = p1_point_prob - 0.10  # Returner disadvantage
            p_win = max(0.15, min(0.85, p_win))
            
            # Simulate point
            if random.random() < p_win:
                pt1 += 1
            else:
                pt2 += 1
            
            # Check game completion
            in_tiebreak = (g1 == 6 and g2 == 6)
            
            if in_tiebreak:
                # Tiebreak: first to 7 with 2 ahead
                if pt1 >= 7 and pt1 - pt2 >= 2:
                    g1 += 1; pt1, pt2 = 0, 0
                elif pt2 >= 7 and pt2 - pt1 >= 2:
                    g2 += 1; pt1, pt2 = 0, 0
                elif (pt1 + pt2) % 2 == 1:
                    srv = 2 if srv == 1 else 1  # Alternate serves in TB
            else:
                # Normal game: need 4 pts with 2 ahead (deuce rules)
                if pt1 >= 4 and pt1 - pt2 >= 2:
                    g1 += 1; pt1, pt2 = 0, 0
                    srv = 2 if srv == 1 else 1
                elif pt2 >= 4 and pt2 - pt1 >= 2:
                    g2 += 1; pt1, pt2 = 0, 0
                    srv = 2 if srv == 1 else 1
            
            # Check set completion
            if g1 >= 6 and g1 - g2 >= 2:
                s1 += 1; g1, g2 = 0, 0
            elif g2 >= 6 and g2 - g1 >= 2:
                s2 += 1; g1, g2 = 0, 0
            elif g1 == 7 and g2 == 6:
                s1 += 1; g1, g2 = 0, 0
            elif g2 == 7 and g1 == 6:
                s2 += 1; g1, g2 = 0, 0
        
        if s1 >= sets_to_win:
            wins_p1 += 1
    
    return wins_p1 / n_simulations

def process_point_result(winner, rally_count=3, is_ace=False, is_df=False):
    """
    Process a single point result through the paper's full pipeline:
    1. Calculate 4-parameter performance for both players
    2. Update EWMA momentum 
    3. Update winning streaks
    4. Store in history
    
    Call this whenever a point is tracked.
    """
    server = st.session_state.server
    
    # Update winning streaks
    if winner == 1:
        st.session_state.current_streak_p1 += 1
        st.session_state.current_streak_p2 = 0
    else:
        st.session_state.current_streak_p2 += 1
        st.session_state.current_streak_p1 = 0
    
    # Update serve tracking
    if server == 1:
        st.session_state.total_serves_p1 += 1
        if is_df:
            st.session_state.serve_faults_p1 += 2  # Double fault = 2 faults
        else:
            st.session_state.serve_faults_p1 += 0  # Approximate: no fault data
    else:
        st.session_state.total_serves_p2 += 1
        if is_df:
            st.session_state.serve_faults_p2 += 2
    
    # Calculate performance for both players
    perf_p1 = calculate_point_performance(
        1, winner == 1, rally_count, is_server=(server == 1)
    )
    perf_p2 = calculate_point_performance(
        2, winner == 2, rally_count, is_server=(server == 2)
    )
    
    # Update EWMA momentum
    m1 = update_ewma_momentum(1, perf_p1)
    m2 = update_ewma_momentum(2, perf_p2)
    
    # Store performance history
    st.session_state.performance_history_p1.append(perf_p1)
    st.session_state.performance_history_p2.append(perf_p2)
    
    # Store momentum history for visualization
    point_num = len(st.session_state.point_history) + 1
    st.session_state.momentum_history.append({
        'point': point_num,
        'momentum_p1': m1,
        'momentum_p2': m2,
        'perf_p1': perf_p1,
        'perf_p2': perf_p2,
        'winner': winner,
        'server': server,
    })
    
    # Store point in history
    st.session_state.point_history.append({
        'winner': winner,
        'server': server,
        'rally_count': rally_count,
        'is_ace': is_ace,
        'is_df': is_df,
        'perf_p1': perf_p1,
        'perf_p2': perf_p2,
        'momentum_p1': m1,
        'momentum_p2': m2,
    })
    
    # Run simulation with updated momentum
    point_prob = momentum_to_point_win_prob(m1, m2)
    sim_prob = simulate_match_from_position(
        point_prob,
        st.session_state.sets1, st.session_state.sets2,
        st.session_state.games1, st.session_state.games2,
        st.session_state.points1, st.session_state.points2,
        st.session_state.server,
        st.session_state.match_conditions.get('best_of', 3),
        n_simulations=500
    )
    st.session_state.sim_win_prob_p1 = sim_prob
    st.session_state.sim_history.append({
        'point': point_num,
        'sim_p1': sim_prob,
        'sim_p2': 1 - sim_prob,
    })


# ============================================================================
# ORIGINAL MODEL FUNCTIONS (Markov + ML)
# ============================================================================

def calculate_true_p_ml(p1_stats, p2_stats, match_conditions):
    """
    Calculate TRUE P using ML models
    
    Args:
        p1_stats: Dict of player 1 stats
        p2_stats: Dict of player 2 stats
        match_conditions: Dict of match conditions
    
    Returns:
        Dictionary with probabilities from both models
    """
    if not models_loaded:
        return None
    
    try:
        # Build feature vector matching the trained model features exactly
        # Trained features: p1_serve_pct, p2_serve_pct, p1_bp_save, p2_bp_save, surface_hard, surface_clay
        all_features = {
            'p1_serve_pct': p1_stats['serve_pct'],
            'p2_serve_pct': p2_stats['serve_pct'],
            'p1_bp_save': p1_stats['bp_save'],
            'p2_bp_save': p2_stats['bp_save'],
            'surface_hard': 1 if match_conditions['surface'] == 'Hard' else 0,
            'surface_clay': 1 if match_conditions['surface'] == 'Clay' else 0,
        }
        
        # Only use features the model was trained on
        features = {k: all_features[k] for k in feature_names if k in all_features}
        
        # Convert to DataFrame matching training feature order
        X = pd.DataFrame([features])[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Get predictions from both models
        lr_prob = lr_model.predict_proba(X_scaled)[0][1]
        rf_prob = rf_model.predict_proba(X_scaled)[0][1]
        ensemble_prob = (lr_prob * 0.4) + (rf_prob * 0.6)  # Weight RF more (higher accuracy)
        
        return {
            'logistic_regression': lr_prob,
            'random_forest': rf_prob,
            'ensemble': ensemble_prob
        }
    except Exception as e:
        st.error(f"ML Model Error: {str(e)}")
        return None

def calculate_game_prob(p_serve_pt):
    """Calculate probability of holding serve using Markov model"""
    p = max(0.01, min(0.99, p_serve_pt))  # Clamp to avoid division issues
    q = 1 - p
    
    # Standard game probabilities
    prob_40_0 = p**3
    prob_40_15 = 3 * (p**3) * q
    prob_40_30 = 6 * (p**3) * (q**2)
    prob_deuce = 20 * (p**3) * (q**3)
    
    # Deuce calculation with safety check
    denominator = 1 - 2*p*q
    if abs(denominator) < 0.001:  # Avoid division by very small numbers
        prob_win_deuce = 0.5
    else:
        prob_win_deuce = (p**2) / denominator
    
    total_prob = prob_40_0 + prob_40_15 + prob_40_30 + (prob_deuce * prob_win_deuce)
    return max(0.01, min(0.99, total_prob))

def calculate_break_probability_real(server_bp_save, returner_bp_conv, surface='Hard'):
    """
    Calculate break probability using REAL data patterns
    """
    global bp_stats, data_loaded
    
    if not data_loaded or bp_stats.empty:
        # Fallback to estimate if data not loaded
        base_break_prob = 0.20
        adjusted_prob = base_break_prob * (returner_bp_conv / server_bp_save) if server_bp_save > 0 else 0.20
        return max(0.05, min(0.85, adjusted_prob))
    
    # Get baseline from real data
    surface_data = bp_stats[bp_stats.surface == surface]
    if surface_data.empty:
        surface_data = bp_stats[bp_stats.surface == 'Hard']
    
    if surface_data.empty:
        # Ultimate fallback
        baseline_save = 0.63
    else:
        baseline_save = surface_data.iloc[0]['winner_bp_save']
    
    # Adjust for player-specific rates
    server_factor = server_bp_save / baseline_save if baseline_save > 0 else 1
    returner_factor = returner_bp_conv / (1 - baseline_save) if baseline_save < 1 else 1
    
    # Base break probability (from real data: ~20% of games are broken)
    base_break_prob = 1 - baseline_save
    adjusted_prob = base_break_prob * (returner_factor / server_factor) if server_factor > 0 else base_break_prob
    
    return max(0.05, min(0.85, adjusted_prob))

def calculate_set_prob(p_hold, p_break_opponent):
    """Calculate set win probability"""
    games_advantage = (p_hold - 0.5) + (p_break_opponent - 0.5)
    set_prob = 0.5 + (games_advantage * 2.5)
    return max(0.05, min(0.95, set_prob))

def calculate_match_prob(p_set, best_of=3):
    """Calculate match win probability"""
    if best_of == 3:
        return (p_set**2) * (1 + 2*(1-p_set))
    else:  # best of 5
        return (p_set**3) * (1 + 3*(1-p_set) + 6*((1-p_set)**2))

def apply_match_conditions(base_prob, conditions, player_surface_mastery):
    """
    Adjust probabilities based on real match conditions
    
    Args:
        base_prob: Base probability
        conditions: Dict with court_speed, temperature, altitude, indoor
        player_surface_mastery: -10 to +10 scale
    
    Returns:
        Adjusted probability
    """
    adjustment = 0
    
    # Court speed (fast court favors servers)
    if conditions['court_speed'] > 60:  # Fast court
        adjustment += 0.02
    elif conditions['court_speed'] < 40:  # Slow court
        adjustment -= 0.02
    
    # Temperature (heat can affect stamina)
    if conditions['temperature'] > 30:  # Hot
        adjustment -= 0.01  # Favors fitter player
    elif conditions['temperature'] < 10:  # Cold
        adjustment -= 0.01  # Ball bounces less
    
    # Altitude (high altitude = faster ball)
    if conditions['altitude'] > 1000:
        adjustment += 0.015
    
    # Indoor (more predictable, less variables)
    if conditions['indoor']:
        adjustment += 0.005
    
    # Surface mastery
    adjustment += (player_surface_mastery / 100)
    
    return max(0.05, min(0.95, base_prob + adjustment))

# ============================================================================
# EDGE DETECTION SYSTEM - DETECT ALL WINNING EDGES
# ============================================================================

def detect_all_edges(p1_stats, p2_stats, live_stats, current_probs, match_conditions):
    """
    Comprehensive edge detection system
    Returns list of all detected edges with severity levels
    
    Edge Types:
    1. Break Point Opportunities (BP edges)
    2. Momentum Shifts
    3. Clutch Situations
    4. Service Vulnerability
    5. Fatigue/Consistency Edges
    6. Match Conditions Exploitation
    """
    edges = []
    
    # 1. BREAK POINT EDGES
    if live_stats['total_bp_p1'] > 0:
        actual_save_p1 = 1 - (live_stats['bp_converted_p2'] / live_stats['total_bp_p1'])
        expected_save_p1 = p1_stats['bp_save']
        
        if actual_save_p1 < expected_save_p1 - 0.15:
            edges.append({
                'type': 'BREAK_OPPORTUNITY',
                'severity': 'CRITICAL',
                'player': st.session_state.p2_name,
                'message': f"{st.session_state.p1_name} saving only {actual_save_p1:.0%} BP (expected {expected_save_p1:.0%})",
                'edge_value': (expected_save_p1 - actual_save_p1) * 100,
                'action': f"BET on {st.session_state.p2_name} to break"
            })
        elif actual_save_p1 < expected_save_p1 - 0.10:
            edges.append({
                'type': 'BREAK_OPPORTUNITY',
                'severity': 'HIGH',
                'player': st.session_state.p2_name,
                'message': f"{st.session_state.p1_name} underperforming on BP by {(expected_save_p1 - actual_save_p1):.0%}",
                'edge_value': (expected_save_p1 - actual_save_p1) * 100,
                'action': f"Consider betting {st.session_state.p2_name} next game"
            })
    
    if live_stats['total_bp_p2'] > 0:
        actual_save_p2 = 1 - (live_stats['bp_converted_p1'] / live_stats['total_bp_p2'])
        expected_save_p2 = p2_stats['bp_save']
        
        if actual_save_p2 < expected_save_p2 - 0.15:
            edges.append({
                'type': 'BREAK_OPPORTUNITY',
                'severity': 'CRITICAL',
                'player': st.session_state.p1_name,
                'message': f"{st.session_state.p2_name} saving only {actual_save_p2:.0%} BP (expected {expected_save_p2:.0%})",
                'edge_value': (expected_save_p2 - actual_save_p2) * 100,
                'action': f"BET on {st.session_state.p1_name} to break"
            })
        elif actual_save_p2 < expected_save_p2 - 0.10:
            edges.append({
                'type': 'BREAK_OPPORTUNITY',
                'severity': 'HIGH',
                'player': st.session_state.p1_name,
                'message': f"{st.session_state.p2_name} underperforming on BP by {(expected_save_p2 - actual_save_p2):.0%}",
                'edge_value': (expected_save_p2 - actual_save_p2) * 100,
                'action': f"Consider betting {st.session_state.p1_name} next game"
            })
    
    # 2. HIGH BP FACED = DANGER ZONE (Real data: 10+ BP faced = 65% loss rate)
    if live_stats['total_bp_p1'] >= 10:
        edges.append({
            'type': 'CRITICAL_BP_COUNT',
            'severity': 'CRITICAL',
            'player': st.session_state.p2_name,
            'message': f"{st.session_state.p1_name} faced {live_stats['total_bp_p1']} BP (Real data: 65% loss rate at 10+)",
            'edge_value': 15,
            'action': f"STRONG BET on {st.session_state.p2_name} to win match"
        })
    elif live_stats['total_bp_p1'] >= 8:
        edges.append({
            'type': 'CRITICAL_BP_COUNT',
            'severity': 'HIGH',
            'player': st.session_state.p2_name,
            'message': f"{st.session_state.p1_name} faced {live_stats['total_bp_p1']} BP (approaching danger zone)",
            'edge_value': 10,
            'action': f"Monitor for betting opportunity on {st.session_state.p2_name}"
        })
    
    if live_stats['total_bp_p2'] >= 10:
        edges.append({
            'type': 'CRITICAL_BP_COUNT',
            'severity': 'CRITICAL',
            'player': st.session_state.p1_name,
            'message': f"{st.session_state.p2_name} faced {live_stats['total_bp_p2']} BP (Real data: 65% loss rate at 10+)",
            'edge_value': 15,
            'action': f"STRONG BET on {st.session_state.p1_name} to win match"
        })
    elif live_stats['total_bp_p2'] >= 8:
        edges.append({
            'type': 'CRITICAL_BP_COUNT',
            'severity': 'HIGH',
            'player': st.session_state.p1_name,
            'message': f"{st.session_state.p2_name} faced {live_stats['total_bp_p2']} BP (approaching danger zone)",
            'edge_value': 10,
            'action': f"Monitor for betting opportunity on {st.session_state.p1_name}"
        })
    
    # 3. MOMENTUM SHIFTS (Legacy + EWMA-enhanced)
    if 'momentum' in p1_stats and 'momentum' in p2_stats:
        momentum_diff = p1_stats['momentum'] - p2_stats['momentum']
        if momentum_diff >= 6:
            edges.append({
                'type': 'MOMENTUM_SHIFT',
                'severity': 'HIGH',
                'player': st.session_state.p1_name,
                'message': f"{st.session_state.p1_name} has +{momentum_diff} momentum advantage",
                'edge_value': momentum_diff,
                'action': f"Bet {st.session_state.p1_name} next set/game"
            })
        elif momentum_diff <= -6:
            edges.append({
                'type': 'MOMENTUM_SHIFT',
                'severity': 'HIGH',
                'player': st.session_state.p2_name,
                'message': f"{st.session_state.p2_name} has +{abs(momentum_diff)} momentum advantage",
                'edge_value': abs(momentum_diff),
                'action': f"Bet {st.session_state.p2_name} next set/game"
            })
    
    # 3b. EWMA MOMENTUM EDGES (Qian et al. 2025)
    ewma_p1 = st.session_state.ewma_momentum_p1
    ewma_p2 = st.session_state.ewma_momentum_p2
    ewma_gap = ewma_p1 - ewma_p2
    
    if len(st.session_state.momentum_history) >= 3:
        # Strong momentum divergence
        if ewma_gap > 15:
            edges.append({
                'type': 'EWMA_MOMENTUM_SURGE',
                'severity': 'CRITICAL',
                'player': st.session_state.p1_name,
                'message': f"EWMA momentum: {st.session_state.p1_name} {ewma_p1:.1f} vs {st.session_state.p2_name} {ewma_p2:.1f} (gap: {ewma_gap:.1f})",
                'edge_value': ewma_gap,
                'action': f"STRONG BET on {st.session_state.p1_name} (momentum surge, λ=0.6467)"
            })
        elif ewma_gap > 8:
            edges.append({
                'type': 'EWMA_MOMENTUM_BUILDING',
                'severity': 'HIGH',
                'player': st.session_state.p1_name,
                'message': f"EWMA momentum building: {st.session_state.p1_name} {ewma_p1:.1f} vs {ewma_p2:.1f}",
                'edge_value': ewma_gap,
                'action': f"Bet {st.session_state.p1_name} (momentum building)"
            })
        elif ewma_gap < -15:
            edges.append({
                'type': 'EWMA_MOMENTUM_SURGE',
                'severity': 'CRITICAL',
                'player': st.session_state.p2_name,
                'message': f"EWMA momentum: {st.session_state.p2_name} {ewma_p2:.1f} vs {st.session_state.p1_name} {ewma_p1:.1f} (gap: {abs(ewma_gap):.1f})",
                'edge_value': abs(ewma_gap),
                'action': f"STRONG BET on {st.session_state.p2_name} (momentum surge, λ=0.6467)"
            })
        elif ewma_gap < -8:
            edges.append({
                'type': 'EWMA_MOMENTUM_BUILDING',
                'severity': 'HIGH',
                'player': st.session_state.p2_name,
                'message': f"EWMA momentum building: {st.session_state.p2_name} {ewma_p2:.1f} vs {ewma_p1:.1f}",
                'edge_value': abs(ewma_gap),
                'action': f"Bet {st.session_state.p2_name} (momentum building)"
            })
        
        # Momentum turning point detection (direction change)
        if len(st.session_state.momentum_history) >= 5:
            recent = st.session_state.momentum_history[-3:]
            older = st.session_state.momentum_history[-5:-3]
            recent_trend_p1 = recent[-1]['momentum_p1'] - recent[0]['momentum_p1']
            older_trend_p1 = older[-1]['momentum_p1'] - older[0]['momentum_p1']
            
            if older_trend_p1 < -3 and recent_trend_p1 > 3:
                edges.append({
                    'type': 'MOMENTUM_TURNING_POINT',
                    'severity': 'HIGH',
                    'player': st.session_state.p1_name,
                    'message': f"{st.session_state.p1_name} momentum REVERSING upward (was falling, now rising)",
                    'edge_value': abs(recent_trend_p1),
                    'action': f"BET {st.session_state.p1_name} (momentum turning point!)"
                })
            elif older_trend_p1 > 3 and recent_trend_p1 < -3:
                edges.append({
                    'type': 'MOMENTUM_TURNING_POINT',
                    'severity': 'HIGH',
                    'player': st.session_state.p2_name,
                    'message': f"{st.session_state.p1_name} momentum REVERSING downward (was rising, now falling)",
                    'edge_value': abs(recent_trend_p1),
                    'action': f"BET {st.session_state.p2_name} ({st.session_state.p1_name} losing momentum)"
                })
    
    # 4. CLUTCH SITUATIONS (5-5 in set, 4-4, tiebreaks)
    games_total = st.session_state.games1 + st.session_state.games2
    if (st.session_state.games1 >= 4 and st.session_state.games2 >= 4) or games_total == 12:
        # Critical games - check clutch stats
        if 'clutch' in p1_stats and 'clutch' in p2_stats:
            clutch_diff = p1_stats['clutch'] - p2_stats['clutch']
            if clutch_diff >= 4:
                edges.append({
                    'type': 'CLUTCH_SITUATION',
                    'severity': 'HIGH',
                    'player': st.session_state.p1_name,
                    'message': f"Critical game! {st.session_state.p1_name} clutch: +{clutch_diff}",
                    'edge_value': clutch_diff * 1.5,
                    'action': f"BET {st.session_state.p1_name} this game (clutch advantage)"
                })
            elif clutch_diff <= -4:
                edges.append({
                    'type': 'CLUTCH_SITUATION',
                    'severity': 'HIGH',
                    'player': st.session_state.p2_name,
                    'message': f"Critical game! {st.session_state.p2_name} clutch: +{abs(clutch_diff)}",
                    'edge_value': abs(clutch_diff) * 1.5,
                    'action': f"BET {st.session_state.p2_name} this game (clutch advantage)"
                })
    
    # 5. CONSISTENCY EDGES (High UE ratio)
    total_points_p1 = live_stats.get('winners_p1', 0) + live_stats.get('ues_p1', 0)
    total_points_p2 = live_stats.get('winners_p2', 0) + live_stats.get('ues_p2', 0)
    
    if total_points_p1 > 20:
        ue_ratio_p1 = live_stats.get('ues_p1', 0) / total_points_p1
        if ue_ratio_p1 > 0.4:  # High error rate
            edges.append({
                'type': 'CONSISTENCY_EDGE',
                'severity': 'MEDIUM',
                'player': st.session_state.p2_name,
                'message': f"{st.session_state.p1_name} UE ratio: {ue_ratio_p1:.0%} (high errors)",
                'edge_value': (ue_ratio_p1 - 0.3) * 100,
                'action': f"Bet {st.session_state.p2_name} (opponent making errors)"
            })
    
    if total_points_p2 > 20:
        ue_ratio_p2 = live_stats.get('ues_p2', 0) / total_points_p2
        if ue_ratio_p2 > 0.4:
            edges.append({
                'type': 'CONSISTENCY_EDGE',
                'severity': 'MEDIUM',
                'player': st.session_state.p1_name,
                'message': f"{st.session_state.p2_name} UE ratio: {ue_ratio_p2:.0%} (high errors)",
                'edge_value': (ue_ratio_p2 - 0.3) * 100,
                'action': f"Bet {st.session_state.p1_name} (opponent making errors)"
            })
    
    # 6. SERVICE GAME VULNERABILITY
    df_rate_p1 = live_stats.get('dfs_p1', 0) / max(1, st.session_state.games1 * 2)
    df_rate_p2 = live_stats.get('dfs_p2', 0) / max(1, st.session_state.games2 * 2)
    
    if df_rate_p1 > 0.15:  # High DF rate (>15% per service game)
        edges.append({
            'type': 'SERVICE_VULNERABILITY',
            'severity': 'HIGH',
            'player': st.session_state.p2_name,
            'message': f"{st.session_state.p1_name} DF rate: {df_rate_p1:.0%} (struggling to serve)",
            'edge_value': (df_rate_p1 - 0.10) * 100,
            'action': f"BET {st.session_state.p2_name} to break when {st.session_state.p1_name} serves"
        })
    
    if df_rate_p2 > 0.15:
        edges.append({
            'type': 'SERVICE_VULNERABILITY',
            'severity': 'HIGH',
            'player': st.session_state.p1_name,
            'message': f"{st.session_state.p2_name} DF rate: {df_rate_p2:.0%} (struggling to serve)",
            'edge_value': (df_rate_p2 - 0.10) * 100,
            'action': f"BET {st.session_state.p1_name} to break when {st.session_state.p2_name} serves"
        })
    
    # 7. MATCH CONDITIONS EXPLOITATION
    if 'surface_mastery' in p1_stats and 'surface_mastery' in p2_stats:
        surface_diff = p1_stats['surface_mastery'] - p2_stats['surface_mastery']
        if surface_diff >= 5:
            edges.append({
                'type': 'SURFACE_MASTERY',
                'severity': 'MEDIUM',
                'player': st.session_state.p1_name,
                'message': f"{st.session_state.p1_name} surface mastery: +{surface_diff} on {match_conditions['surface']}",
                'edge_value': surface_diff,
                'action': f"Long-term bet on {st.session_state.p1_name} (surface advantage)"
            })
        elif surface_diff <= -5:
            edges.append({
                'type': 'SURFACE_MASTERY',
                'severity': 'MEDIUM',
                'player': st.session_state.p2_name,
                'message': f"{st.session_state.p2_name} surface mastery: +{abs(surface_diff)} on {match_conditions['surface']}",
                'edge_value': abs(surface_diff),
                'action': f"Long-term bet on {st.session_state.p2_name} (surface advantage)"
            })
    
    # Sort edges by severity
    severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2}
    edges.sort(key=lambda x: (severity_order.get(x['severity'], 3), -x['edge_value']))
    
    return edges

# ============================================================================
# API AUTO-FILL HELPER
# ============================================================================

def _apply_api_data_to_setup(data: dict):
    """
    Apply parsed API match data to session state so the pre-match form
    is pre-filled with real values.  Called from the auto-fill UI.
    """
    # Player names
    if data.get("p1_name"):
        st.session_state.p1_name = data["p1_name"]
    if data.get("p2_name"):
        st.session_state.p2_name = data["p2_name"]

    # Odds
    odds = data.get("odds", {})
    if odds.get("p1_match_odds"):
        st.session_state.p1_match_odds = odds["p1_match_odds"]
    if odds.get("p2_match_odds"):
        st.session_state.p2_match_odds = odds["p2_match_odds"]

    # Score (if match is live)
    if data.get("sets_p1", 0) or data.get("sets_p2", 0):
        st.session_state.sets1 = data.get("sets_p1", 0)
        st.session_state.sets2 = data.get("sets_p2", 0)
    if data.get("games_p1", 0) or data.get("games_p2", 0):
        st.session_state.games1 = data.get("games_p1", 0)
        st.session_state.games2 = data.get("games_p2", 0)
    if data.get("points_p1", 0) or data.get("points_p2", 0):
        st.session_state.points1 = data.get("points_p1", 0)
        st.session_state.points2 = data.get("points_p2", 0)
    if data.get("server"):
        st.session_state.server = data["server"]

    # Surface (ESPN service already detects this)
    if data.get("surface"):
        st.session_state.surface = data["surface"]

    # Player profiles → serve stats
    p1_prof = data.get("p1_profile", {})
    p2_prof = data.get("p2_profile", {})

    # Store flag so setup form knows data was loaded
    st.session_state._api_data_loaded = True
    st.session_state._api_data = data


# ============================================================================
# PRE-MATCH SETUP SECTION
# ============================================================================

if not st.session_state.setup_complete:
    st.markdown("## 📋 Pre-Match Setup")

    # ── API AUTO-FILL ────────────────────────────────────────────────
    if _API_AVAILABLE:
        with st.expander("🔴 AUTO-FILL FROM LIVE DATA (ESPN — Free, No Key)", expanded=True):
            st.caption("Automatically populate all fields from a live or scheduled match — 100% FREE")
            st.success("✅ ESPN Free API — no API key required!")

            # Check if data was loaded from Live Matches page
            api_loaded = st.session_state.get("api_loaded_match")
            if api_loaded:
                st.info(f"📦 Match loaded from Live Matches page: **{api_loaded['p1_name']} vs {api_loaded['p2_name']}**")
                if st.button("🎯 Use This Match Data", key="use_loaded_match"):
                    _apply_api_data_to_setup(api_loaded)
                    st.rerun()
                st.markdown("---")

            svc = get_free_service()

            fetch_mode = st.radio("Fetch mode",
                ["🔴 Live matches now", "📅 Today's schedule"],
                horizontal=True, key="api_fetch_mode")

            if fetch_mode == "🔴 Live matches now":
                if st.button("🔄 Fetch Live Matches", key="fetch_live_btn"):
                    with st.spinner("Fetching live matches from ESPN..."):
                        live = svc.get_live_matches()
                    if live:
                        st.session_state._api_live_matches = live
                    else:
                        st.warning("No live matches right now. Try Today's schedule.")

                live_cached = st.session_state.get("_api_live_matches", [])
                if live_cached:
                    options = {
                        f"{m['p1_name']} vs {m['p2_name']} "
                        f"({m['tournament']}) [{m['score_display']}]": m
                        for m in live_cached
                    }
                    selected = st.selectbox("Select a match", list(options.keys()), key="api_live_sel")
                    if selected and st.button("📥 Load This Match", key="load_live_btn"):
                        match = options[selected]
                        full_data = svc.get_calculator_ready_data(match)
                        st.session_state.api_match_key = match.get("event_key", "")
                        _apply_api_data_to_setup(full_data)
                        st.rerun()

            elif fetch_mode == "📅 Today's schedule":
                if st.button("🔄 Fetch Today's Matches", key="fetch_today_btn"):
                    with st.spinner("Fetching today's matches from ESPN..."):
                        fixtures = svc.get_todays_matches()
                    if fixtures:
                        st.session_state._api_today_matches = fixtures
                        st.success(f"Found {len(fixtures)} matches today!")
                    else:
                        st.warning("No matches found for today.")

                today_cached = st.session_state.get("_api_today_matches", [])
                if today_cached:
                    options = {
                        f"{m['p1_name']} vs {m['p2_name']} "
                        f"({m['tournament']}) [{m['round']} — {m['status']}]": m
                        for m in today_cached
                        if m['p1_name'] not in ('?', 'TBD', '') and m['p2_name'] not in ('?', 'TBD', '')
                    }
                    selected = st.selectbox("Select a match", list(options.keys()), key="api_today_sel")
                    if selected and st.button("📥 Load This Match", key="load_today_btn"):
                        match = options[selected]
                        full_data = svc.get_calculator_ready_data(match)
                        st.session_state.api_match_key = match.get("event_key", "")
                        _apply_api_data_to_setup(full_data)
                        st.rerun()
    else:
        st.caption("💡 Install api/free_live_data.py for live auto-fill")

    st.markdown("---")
    st.markdown("### ✏️ Manual Setup (or edit auto-filled values)")

    with st.expander("⚙️ MATCH INFORMATION", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.p1_name = st.text_input("Player 1 Name", st.session_state.p1_name)
            p1_rank = st.number_input("Player 1 Ranking", 1, 500, 50)
        with col2:
            st.session_state.p2_name = st.text_input("Player 2 Name", st.session_state.p2_name)
            p2_rank = st.number_input("Player 2 Ranking", 1, 500, 50)
        
        col3, col4, col5 = st.columns(3)
        with col3:
            st.session_state.surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
        with col4:
            st.session_state.best_of = st.selectbox("Best of", [3, 5])
        with col5:
            st.session_state.indoor = st.checkbox("Indoor Match")
    
    with st.expander("📊 PRE-MATCH PROBABILITIES", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.p1_match_odds = st.number_input(f"{st.session_state.p1_name} Odds", 1.01, 50.0, 1.85, 0.05)
            st.session_state.pre_p1_prob = 1 / st.session_state.p1_match_odds
            st.caption(f"Implied Probability: {st.session_state.pre_p1_prob:.1%}")
        with col2:
            st.session_state.p2_match_odds = st.number_input(f"{st.session_state.p2_name} Odds", 1.01, 50.0, 2.10, 0.05)
            pre_p2_prob = 1 / st.session_state.p2_match_odds
            st.caption(f"Implied Probability: {pre_p2_prob:.1%}")
        with col3:
            bookmaker_margin = (st.session_state.pre_p1_prob + pre_p2_prob - 1) * 100
            st.metric("Bookmaker Margin", f"{bookmaker_margin:.1f}%")
    
    with st.expander("🌡️ MATCH CONDITIONS", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.court_speed = st.slider("Court Speed", 0, 100, 50, help="0=Very Slow, 50=Medium, 100=Very Fast")
        with col2:
            st.session_state.temperature = st.number_input("Temperature (°C)", -10, 50, 20)
        with col3:
            st.session_state.altitude = st.number_input("Altitude (m)", 0, 3000, 0)
    
    with st.expander(f"🎾 {st.session_state.p1_name.upper()} - PLAYER STATS", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            p1_serve_pct = st.number_input(f"Serve Win %", 50, 85, 65, key="p1_serve")
            p1_return_pct = st.number_input(f"Return Win %", 20, 50, 35, key="p1_return")
        with col2:
            p1_bp_save = st.number_input(f"BP Save %", 30, 90, 65, key="p1_bp")
            p1_bp_conv = st.number_input(f"BP Conversion %", 20, 60, 40, key="p1_conv")
        with col3:
            p1_first_serve_pct = st.number_input(f"1st Serve %", 40, 80, 60, key="p1_fs")
            p1_aces_avg = st.number_input(f"Avg Aces/Match", 0, 30, 5, key="p1_aces")
        
        st.markdown("**Advanced Parameters (-10 to +10)**")
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            p1_momentum = st.number_input("Momentum", -10, 10, 0, key="p1_mom", help="Recent form")
        with col5:
            p1_surface_mastery = st.number_input("Surface Mastery", -10, 10, 0, key="p1_surf", help="Performance on this surface")
        with col6:
            p1_clutch = st.number_input("Clutch Factor", -10, 10, 0, key="p1_clutch", help="Performance in critical moments")
        with col7:
            p1_consistency = st.number_input("Consistency", -10, 10, 0, key="p1_cons", help="Error rate")
    
    with st.expander(f"🎾 {st.session_state.p2_name.upper()} - PLAYER STATS", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            p2_serve_pct = st.number_input(f"Serve Win % ", 50, 85, 65, key="p2_serve")
            p2_return_pct = st.number_input(f"Return Win % ", 20, 50, 35, key="p2_return")
        with col2:
            p2_bp_save = st.number_input(f"BP Save % ", 30, 90, 65, key="p2_bp")
            p2_bp_conv = st.number_input(f"BP Conversion % ", 20, 60, 40, key="p2_conv")
        with col3:
            p2_first_serve_pct = st.number_input(f"1st Serve % ", 40, 80, 60, key="p2_fs")
            p2_aces_avg = st.number_input(f"Avg Aces/Match ", 0, 30, 5, key="p2_aces")
        
        st.markdown("**Advanced Parameters (-10 to +10)**")
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            p2_momentum = st.number_input("Momentum ", -10, 10, 0, key="p2_mom", help="Recent form")
        with col5:
            p2_surface_mastery = st.number_input("Surface Mastery ", -10, 10, 0, key="p2_surf", help="Performance on this surface")
        with col6:
            p2_clutch = st.number_input("Clutch Factor ", -10, 10, 0, key="p2_clutch", help="Performance in critical moments")
        with col7:
            p2_consistency = st.number_input("Consistency ", -10, 10, 0, key="p2_cons", help="Error rate")
    
    # Pre-match betting odds input
    st.markdown("### 💰 Pre-Match Betting Odds (for variance tracking)")
    col_odds1, col_odds2 = st.columns(2)
    with col_odds1:
        pre_p1_odds = st.number_input(f"{st.session_state.p1_name} Match Odds", 1.01, 50.0, 1.85, 0.01, key="pre_p1_odds")
    with col_odds2:
        pre_p2_odds = st.number_input(f"{st.session_state.p2_name} Match Odds", 1.01, 50.0, 2.10, 0.01, key="pre_p2_odds")
    
    st.caption(f"Implied: {st.session_state.p1_name} {1/pre_p1_odds:.1%} | {st.session_state.p2_name} {1/pre_p2_odds:.1%}")
    
    if st.button("✅ START LIVE TRACKING", type="primary", use_container_width=True):
        # Store all setup data
        st.session_state.p1_stats = {
            'rank': p1_rank,
            'serve_pct': p1_serve_pct / 100,
            'return_pct': p1_return_pct / 100,
            'bp_save': p1_bp_save / 100,
            'bp_conv': p1_bp_conv / 100,
            'first_serve_pct': p1_first_serve_pct,
            'momentum': p1_momentum,
            'surface_mastery': p1_surface_mastery,
            'clutch': p1_clutch,
            'consistency': p1_consistency
        }
        st.session_state.p2_stats = {
            'rank': p2_rank,
            'serve_pct': p2_serve_pct / 100,
            'return_pct': p2_return_pct / 100,
            'bp_save': p2_bp_save / 100,
            'bp_conv': p2_bp_conv / 100,
            'first_serve_pct': p2_first_serve_pct,
            'momentum': p2_momentum,
            'surface_mastery': p2_surface_mastery,
            'clutch': p2_clutch,
            'consistency': p2_consistency
        }
        st.session_state.match_conditions = {
            'surface': st.session_state.surface,
            'best_of': st.session_state.best_of,
            'indoor': st.session_state.indoor,
            'court_speed': st.session_state.court_speed,
            'temperature': st.session_state.temperature,
            'altitude': st.session_state.altitude
        }
        # Store pre-match odds
        st.session_state.pre_match_p1_odds = pre_p1_odds
        st.session_state.pre_match_p2_odds = pre_p2_odds
        st.session_state.live_p1_match_odds = pre_p1_odds
        st.session_state.live_p2_match_odds = pre_p2_odds
        
        st.session_state.setup_complete = True
        st.rerun()
    
    st.stop()

# ============================================================================
# LIVE TRACKING MODE
# ============================================================================

# ── API LIVE REFRESH BAR ────────────────────────────────────────────
if _API_AVAILABLE and st.session_state.api_match_key:
    with st.container():
        col_api_r1, col_api_r2, col_api_r3, col_api_r4 = st.columns([2, 1, 1, 1])
        with col_api_r1:
            st.markdown(f"🔴 **Live ESPN Feed** — Match #{st.session_state.api_match_key}")
            if st.session_state.api_last_refresh:
                st.caption(f"Last refresh: {st.session_state.api_last_refresh}")
        with col_api_r2:
            if st.button("🔄 Refresh Now", key="api_refresh_now"):
                _svc = get_free_service()
                # Parse event_key back to event_id and comp_id
                _key_parts = str(st.session_state.api_match_key).split("_")
                if len(_key_parts) == 2:
                    _event_id, _comp_id = _key_parts
                    with st.spinner("Refreshing from ESPN..."):
                        for _tour in ["atp", "wta"]:
                            _match = _svc.get_match_detail(_event_id, _comp_id, _tour)
                            if _match:
                                break
                    if _match:
                        st.session_state.sets1 = _match["sets_p1"]
                        st.session_state.sets2 = _match["sets_p2"]
                        st.session_state.games1 = _match["games_p1"]
                        st.session_state.games2 = _match["games_p2"]
                        st.session_state.api_last_refresh = datetime.now().strftime("%H:%M:%S")
                        st.rerun()
                    else:
                        st.warning("Could not refresh — match may have ended")
                else:
                    st.warning("Invalid match key format")
        with col_api_r3:
            st.session_state.api_auto_refresh = st.checkbox(
                "Auto-refresh (30s)", value=st.session_state.api_auto_refresh,
                key="api_auto_toggle"
            )
        with col_api_r4:
            if st.button("❌ Disconnect", key="api_disconnect"):
                st.session_state.api_match_key = None
                st.session_state.api_auto_refresh = False
                st.rerun()
        st.markdown("---")

# Get stats from session state
p1_stats = st.session_state.p1_stats
p2_stats = st.session_state.p2_stats
match_conditions = st.session_state.match_conditions

# ============================================================================
# CALCULATE LIVE PROBABILITIES (DYNAMIC - UPDATES WITH SCORE)
# ============================================================================

# Get current score state
current_sets1 = st.session_state.sets1
current_sets2 = st.session_state.sets2
current_games1 = st.session_state.games1
current_games2 = st.session_state.games2
current_points1 = st.session_state.points1
current_points2 = st.session_state.points2

# Adjust serve percentages based on live performance (if enough points played)
total_points_played = current_sets1 * 48 + current_sets2 * 48 + current_games1 * 4 + current_games2 * 4 + current_points1 + current_points2
if total_points_played > 20:
    # Adjust serve % based on aces and DFs
    p1_serve_adj = p1_stats['serve_pct'] + (st.session_state.aces_p1 * 0.01) - (st.session_state.dfs_p1 * 0.01)
    p2_serve_adj = p2_stats['serve_pct'] + (st.session_state.aces_p2 * 0.01) - (st.session_state.dfs_p2 * 0.01)
    p1_serve_adj = max(0.5, min(0.75, p1_serve_adj))
    p2_serve_adj = max(0.5, min(0.75, p2_serve_adj))
else:
    p1_serve_adj = p1_stats['serve_pct']
    p2_serve_adj = p2_stats['serve_pct']

# Markov probabilities with current score state
p1_hold_markov = calculate_game_prob(p1_serve_adj)
p2_hold_markov = calculate_game_prob(p2_serve_adj)
p1_break_p2_markov = 1 - p2_hold_markov
p2_break_p1_markov = 1 - p1_hold_markov

# Real data break probabilities (updated based on live BP stats)
if st.session_state.total_bp_p2 > 3:
    live_p2_bp_save = 1 - (st.session_state.bp_converted_p1 / max(1, st.session_state.total_bp_p2))
    p2_bp_save_blended = 0.6 * p2_stats['bp_save'] + 0.4 * live_p2_bp_save
else:
    p2_bp_save_blended = p2_stats['bp_save']

if st.session_state.total_bp_p1 > 3:
    live_p1_bp_save = 1 - (st.session_state.bp_converted_p2 / max(1, st.session_state.total_bp_p1))
    p1_bp_save_blended = 0.6 * p1_stats['bp_save'] + 0.4 * live_p1_bp_save
else:
    p1_bp_save_blended = p1_stats['bp_save']

p1_break_p2_real = calculate_break_probability_real(
    p2_bp_save_blended, p1_stats['bp_conv'], match_conditions['surface']
)
p2_break_p1_real = calculate_break_probability_real(
    p1_bp_save_blended, p2_stats['bp_conv'], match_conditions['surface']
)

# Calculate point-by-point probability within current game
def calculate_point_prob(p_win_point, points_ahead, points_behind):
    """Calculate probability of winning game from current point score"""
    if points_ahead >= 4 and points_ahead - points_behind >= 2:
        return 1.0
    if points_behind >= 4 and points_behind - points_ahead >= 2:
        return 0.0
    
    # Simplified point-level calculation (could use full Markov here)
    if points_ahead > points_behind:
        return p_win_point + (points_ahead - points_behind) * 0.05
    elif points_behind > points_ahead:
        return p_win_point - (points_behind - points_ahead) * 0.05
    else:
        return p_win_point

# Adjust game probabilities based on current point score
if st.session_state.server == 1:
    p1_current_game_prob = calculate_point_prob(p1_hold_markov, current_points1, current_points2)
    p2_current_game_prob = 1 - p1_current_game_prob
else:
    p2_current_game_prob = calculate_point_prob(p2_hold_markov, current_points2, current_points1)
    p1_current_game_prob = 1 - p2_current_game_prob

# Set and match probabilities (incorporating current set score)
p1_set = calculate_set_prob(p1_hold_markov, p1_break_p2_real)
p2_set = 1 - p1_set

# Adjust set probability based on current games score
if current_games1 > current_games2:
    p1_set = min(0.95, p1_set + (current_games1 - current_games2) * 0.05)
elif current_games2 > current_games1:
    p1_set = max(0.05, p1_set - (current_games2 - current_games1) * 0.05)

p2_set = 1 - p1_set

p1_match_markov = calculate_match_prob(p1_set, match_conditions['best_of'])
p2_match_markov = 1 - p1_match_markov

# Adjust match probability based on current sets score
if current_sets1 > current_sets2:
    p1_match_markov = min(0.98, p1_match_markov + (current_sets1 - current_sets2) * 0.15)
elif current_sets2 > current_sets1:
    p1_match_markov = max(0.02, p1_match_markov - (current_sets2 - current_sets1) * 0.15)

p2_match_markov = 1 - p1_match_markov

# Apply match conditions
p1_match_adjusted = apply_match_conditions(
    p1_match_markov, match_conditions, p1_stats['surface_mastery']
)
p2_match_adjusted = 1 - p1_match_adjusted

# ML Model predictions (if available) - blended with live adjustments
ml_probs = None
if models_loaded:
    ml_probs = calculate_true_p_ml(p1_stats, p2_stats, match_conditions)

# ============================================================================
# ENHANCED TRUE P: Blend ML + Markov + EWMA Simulation (Qian et al. 2025)
# ============================================================================

# Get EWMA momentum-based simulation probability
sim_p1 = st.session_state.sim_win_prob_p1
has_momentum_data = len(st.session_state.point_history) >= 3

# Calculate point prediction from momentum (paper Eq. 11)
ewma_point_prob = momentum_to_point_win_prob(
    st.session_state.ewma_momentum_p1, 
    st.session_state.ewma_momentum_p2
)

# Build TRUE P as weighted ensemble of all models
if ml_probs and has_momentum_data:
    # Full ensemble: ML (35%) + Markov (25%) + Simulation (40%)
    # Simulation gets highest weight because it uses live momentum data (paper's key insight)
    true_p1 = (0.35 * ml_probs['ensemble'] + 
               0.25 * p1_match_adjusted + 
               0.40 * sim_p1)
    true_p2 = 1 - true_p1
elif ml_probs:
    # No momentum data yet: ML (60%) + Markov (40%)
    true_p1 = 0.6 * ml_probs['ensemble'] + 0.4 * p1_match_adjusted
    true_p2 = 1 - true_p1
elif has_momentum_data:
    # No ML models: Markov (50%) + Simulation (50%)
    true_p1 = 0.5 * p1_match_adjusted + 0.5 * sim_p1
    true_p2 = 1 - true_p1
else:
    true_p1 = p1_match_adjusted
    true_p2 = p2_match_adjusted

# Store in probability history
current_point = len(st.session_state.prob_history)
st.session_state.prob_history.append({
    'point': current_point,
    'p1_prob': true_p1,
    'p2_prob': true_p2,
    'score': f"{st.session_state.sets1}-{st.session_state.sets2}, {st.session_state.games1}-{st.session_state.games2}"
})

# Collect live stats
live_stats = {
    'total_bp_p1': st.session_state.total_bp_p1,
    'total_bp_p2': st.session_state.total_bp_p2,
    'bp_converted_p1': st.session_state.bp_converted_p1,
    'bp_converted_p2': st.session_state.bp_converted_p2,
    'aces_p1': st.session_state.aces_p1,
    'aces_p2': st.session_state.aces_p2,
    'dfs_p1': st.session_state.dfs_p1,
    'dfs_p2': st.session_state.dfs_p2,
    'winners_p1': st.session_state.winners_p1,
    'winners_p2': st.session_state.winners_p2,
    'ues_p1': st.session_state.ues_p1,
    'ues_p2': st.session_state.ues_p2
}

current_probs = {
    'p1_match': true_p1,
    'p2_match': true_p2,
    'p1_set': p1_set,
    'p2_set': p2_set,
    'p1_game': p1_current_game_prob,
    'p2_game': p2_current_game_prob
}

# ============================================================================
# DISPLAY HEADER & SCORES
# ============================================================================

# Match info bar
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    st.metric("Surface", match_conditions['surface'])
with col_info2:
    st.metric("Best of", match_conditions['best_of'])
with col_info3:
    st.metric("Conditions", f"{match_conditions['temperature']}°C, Speed:{match_conditions['court_speed']}")
with col_info4:
    if st.button("🔄 New Match", help="Reset and start new match"):
        st.session_state.setup_complete = False
        for key in list(st.session_state.keys()):
            if key not in ['p1_name', 'p2_name', 'surface']:
                del st.session_state[key]
        st.rerun()

st.markdown("---")

# Score display - MAIN SCOREBOARD
st.markdown("### 🎾 LIVE MATCH SCORE")

# Force refresh of score values from session state
current_sets1 = st.session_state.sets1
current_sets2 = st.session_state.sets2
current_games1 = st.session_state.games1
current_games2 = st.session_state.games2
current_points1 = st.session_state.points1
current_points2 = st.session_state.points2
current_server = st.session_state.server

col_score1, col_vscore, col_score2 = st.columns([5, 1, 5])

with col_score1:
    server_indicator = "🎾 " if current_server == 1 else ""
    # Determine who's leading for highlighting
    p1_leading = (current_sets1 > current_sets2) or \
                 (current_sets1 == current_sets2 and current_games1 > current_games2) or \
                 (current_sets1 == current_sets2 and current_games1 == current_games2 and current_points1 > current_points2)
    
    border_color = "#28a745" if p1_leading else "#dee2e6"
    bg_color = "#f0fff4" if p1_leading else "#f8f9fa"
    
    # Use metric for live updating display
    st.markdown(f'''
    <div class="score-box" style="border: 3px solid {border_color}; background: {bg_color};">
        <div style="font-size:24px; margin-bottom:8px;">{server_indicator}<strong>{st.session_state.p1_name}</strong></div>
        <div style="font-size:52px; font-weight:900; letter-spacing:12px; color:#000; text-align:center;">
            <span style="color:#dc3545; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">{current_sets1}</span>
            <span style="color:#666;"> - </span>
            <span style="color:#fd7e14; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">{current_games1}</span>
            <span style="color:#666;"> - </span>
            <span style="color:#28a745; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">{current_points1}</span>
        </div>
        <div class="prob-display" style="background:linear-gradient(135deg, #28a745 0%, #20c997 100%);color:white; margin-top:10px;">
            WIN: {true_p1:.1%}
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # P1 Live stats
    bp_save_p1_actual = 1 - (st.session_state.bp_converted_p2 / max(1, st.session_state.total_bp_p1)) if st.session_state.total_bp_p1 > 0 else 0
    st.caption(f"**BP:** {st.session_state.total_bp_p1} faced, {bp_save_p1_actual:.0%} saved | **Aces:** {st.session_state.aces_p1} | **DF:** {st.session_state.dfs_p1} | **W/UE:** {st.session_state.winners_p1}/{st.session_state.ues_p1}")

with col_vscore:
    st.markdown("<div style='text-align:center;font-size:48px;font-weight:bold;padding-top:80px;color:#6c757d;'>VS</div>", unsafe_allow_html=True)

with col_score2:
    server_indicator = "🎾 " if current_server == 2 else ""
    # Determine who's leading for highlighting
    p2_leading = (current_sets2 > current_sets1) or \
                 (current_sets2 == current_sets1 and current_games2 > current_games1) or \
                 (current_sets2 == current_sets1 and current_games2 == current_games1 and current_points2 > current_points1)
    
    border_color = "#007bff" if p2_leading else "#dee2e6"
    bg_color = "#e7f3ff" if p2_leading else "#f8f9fa"
    
    st.markdown(f'''
    <div class="score-box" style="border: 3px solid {border_color}; background: {bg_color};">
        <div style="font-size:24px; margin-bottom:8px;">{server_indicator}<strong>{st.session_state.p2_name}</strong></div>
        <div style="font-size:52px; font-weight:900; letter-spacing:12px; color:#000; text-align:center;">
            <span style="color:#dc3545; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">{current_sets2}</span>
            <span style="color:#666;"> - </span>
            <span style="color:#fd7e14; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">{current_games2}</span>
            <span style="color:#666;"> - </span>
            <span style="color:#28a745; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">{current_points2}</span>
        </div>
        <div class="prob-display" style="background:linear-gradient(135deg, #007bff 0%, #0056b3 100%);color:white; margin-top:10px;">
            WIN: {true_p2:.1%}
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # P2 Live stats
    bp_save_p2_actual = 1 - (st.session_state.bp_converted_p1 / max(1, st.session_state.total_bp_p2)) if st.session_state.total_bp_p2 > 0 else 0
    st.caption(f"**BP:** {st.session_state.total_bp_p2} faced, {bp_save_p2_actual:.0%} saved | **Aces:** {st.session_state.aces_p2} | **DF:** {st.session_state.dfs_p2} | **W/UE:** {st.session_state.winners_p2}/{st.session_state.ues_p2}")

# Debug info (can be removed after verification)
st.caption(f"🔍 Current Score State: {current_sets1}-{current_sets2} | {current_games1}-{current_games2} | {current_points1}-{current_points2} | Server: {current_server}")

st.markdown("---")

# ============================================================================
# MANUAL SCORE INPUT & LIVE STATS UPDATE
# ============================================================================

st.markdown("### � Live Bookmaker Odds (Update from betting sites)")
st.caption("Enter current live odds to calculate exact betting edges at any point")

col_odds1, col_odds2, col_odds3 = st.columns(3)

with col_odds1:
    st.markdown("**📊 Match Winner Odds**")
    new_p1_match_odds = st.number_input(f"{st.session_state.p1_name}", 1.01, 50.0, st.session_state.live_p1_match_odds, 0.01, key="live_match_p1")
    new_p2_match_odds = st.number_input(f"{st.session_state.p2_name}", 1.01, 50.0, st.session_state.live_p2_match_odds, 0.01, key="live_match_p2")
    implied_match_p1 = 1 / new_p1_match_odds
    implied_match_p2 = 1 / new_p2_match_odds
    st.caption(f"Implied: {implied_match_p1:.1%} / {implied_match_p2:.1%}")

with col_odds2:
    st.markdown("**🎯 Current Set Winner Odds**")
    new_p1_set_odds = st.number_input(f"{st.session_state.p1_name} ", 1.01, 50.0, st.session_state.live_p1_set_odds, 0.01, key="live_set_p1")
    new_p2_set_odds = st.number_input(f"{st.session_state.p2_name} ", 1.01, 50.0, st.session_state.live_p2_set_odds, 0.01, key="live_set_p2")
    implied_set_p1 = 1 / new_p1_set_odds
    implied_set_p2 = 1 / new_p2_set_odds
    st.caption(f"Implied: {implied_set_p1:.1%} / {implied_set_p2:.1%}")

with col_odds3:
    st.markdown("**🎾 Current Game Winner Odds**")
    new_p1_game_odds = st.number_input(f"{st.session_state.p1_name}  ", 1.01, 50.0, st.session_state.live_p1_game_odds, 0.01, key="live_game_p1")
    new_p2_game_odds = st.number_input(f"{st.session_state.p2_name}  ", 1.01, 50.0, st.session_state.live_p2_game_odds, 0.01, key="live_game_p2")
    implied_game_p1 = 1 / new_p1_game_odds
    implied_game_p2 = 1 / new_p2_game_odds
    st.caption(f"Implied: {implied_game_p1:.1%} / {implied_game_p2:.1%}")

if st.button("💰 Update Live Odds", type="secondary", use_container_width=True):
    st.session_state.live_p1_match_odds = new_p1_match_odds
    st.session_state.live_p2_match_odds = new_p2_match_odds
    st.session_state.live_p1_set_odds = new_p1_set_odds
    st.session_state.live_p2_set_odds = new_p2_set_odds
    st.session_state.live_p1_game_odds = new_p1_game_odds
    st.session_state.live_p2_game_odds = new_p2_game_odds
    st.rerun()

st.markdown("---")

st.markdown("### �📊 Manual Score & Stats Input")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.markdown(f"**{st.session_state.p1_name} Score**")
    new_sets1 = st.number_input("Sets", 0, 5, st.session_state.sets1, key="input_sets1")
    new_games1 = st.number_input("Games", 0, 15, st.session_state.games1, key="input_games1")
    new_points1 = st.number_input("Points", 0, 10, st.session_state.points1, key="input_points1")

with col_s2:
    st.markdown(f"**{st.session_state.p2_name} Score**")
    new_sets2 = st.number_input("Sets ", 0, 5, st.session_state.sets2, key="input_sets2")
    new_games2 = st.number_input("Games ", 0, 15, st.session_state.games2, key="input_games2")
    new_points2 = st.number_input("Points ", 0, 10, st.session_state.points2, key="input_points2")

with col_s3:
    st.markdown(f"**{st.session_state.p1_name} Live Stats**")
    new_bp1 = st.number_input("BP Faced", 0, 50, st.session_state.total_bp_p1, key="input_bp1")
    new_bp_conv2 = st.number_input("BP Converted (opp)", 0, 50, st.session_state.bp_converted_p2, key="input_bpc2")
    new_aces1 = st.number_input("Aces", 0, 50, st.session_state.aces_p1, key="input_aces1")
    new_dfs1 = st.number_input("DFs", 0, 50, st.session_state.dfs_p1, key="input_dfs1")

with col_s4:
    st.markdown(f"**{st.session_state.p2_name} Live Stats**")
    new_bp2 = st.number_input("BP Faced ", 0, 50, st.session_state.total_bp_p2, key="input_bp2")
    new_bp_conv1 = st.number_input("BP Converted (opp) ", 0, 50, st.session_state.bp_converted_p1, key="input_bpc1")
    new_aces2 = st.number_input("Aces ", 0, 50, st.session_state.aces_p2, key="input_aces2")
    new_dfs2 = st.number_input("DFs ", 0, 50, st.session_state.dfs_p2, key="input_dfs2")

col_update1, col_update2, col_update3 = st.columns(3)
with col_update1:
    if st.button("✅ Update Score & Stats", type="primary", use_container_width=True, key="update_score_button"):
        # Update all score values
        st.session_state.sets1 = new_sets1
        st.session_state.sets2 = new_sets2
        st.session_state.games1 = new_games1
        st.session_state.games2 = new_games2
        st.session_state.points1 = new_points1
        st.session_state.points2 = new_points2
        
        # Update stats
        st.session_state.total_bp_p1 = new_bp1
        st.session_state.total_bp_p2 = new_bp2
        st.session_state.bp_converted_p1 = new_bp_conv1
        st.session_state.bp_converted_p2 = new_bp_conv2
        st.session_state.aces_p1 = new_aces1
        st.session_state.aces_p2 = new_aces2
        st.session_state.dfs_p1 = new_dfs1
        st.session_state.dfs_p2 = new_dfs2
        
        # Show success message before rerun
        st.success(f"✅ Score updated to {new_sets1}-{new_sets2} | {new_games1}-{new_games2} | {new_points1}-{new_points2}")
        
        # Force page refresh
        st.rerun()

with col_update2:
    current_server = st.selectbox("Current Server", [st.session_state.p1_name, st.session_state.p2_name], 
                                  index=0 if st.session_state.server == 1 else 1)
    if st.button("Update Server", use_container_width=True):
        st.session_state.server = 1 if current_server == st.session_state.p1_name else 2
        st.rerun()

with col_update3:
    if st.button("🔄 Reset All", use_container_width=True):
        for key in ['sets1', 'sets2', 'games1', 'games2', 'points1', 'points2', 
                   'total_bp_p1', 'total_bp_p2', 'bp_converted_p1', 'bp_converted_p2',
                   'aces_p1', 'aces_p2', 'dfs_p1', 'dfs_p2', 'winners_p1', 'winners_p2',
                   'ues_p1', 'ues_p2', 'prob_history']:
            if key in st.session_state:
                if key == 'prob_history':
                    st.session_state[key] = []
                else:
                    st.session_state[key] = 0
        st.session_state.server = 1
        st.rerun()

st.markdown("---")

# ============================================================================
# POINT TRACKING CONTROLS (Quick Update Buttons)
# ============================================================================

def update_score(winner, is_ace=False, is_df=False, rally_count=3):
    """Update score after a point and process through paper's momentum pipeline"""
    # Process through EWMA momentum pipeline (Qian et al. 2025)
    process_point_result(winner, rally_count=rally_count, is_ace=is_ace, is_df=is_df)
    
    if winner == 1:
        st.session_state.points1 += 1
    else:
        st.session_state.points2 += 1
    
    # Check for game win
    if st.session_state.points1 >= 4 and st.session_state.points1 - st.session_state.points2 >= 2:
        st.session_state.games1 += 1
        st.session_state.points1, st.session_state.points2 = 0, 0
        # Track return game stats before switching
        if st.session_state.server == 1:
            # P2 was returning this game
            pass  # Return tracking handled elsewhere
        st.session_state.server = 2 if st.session_state.server == 1 else 1  # Switch server
        
        # Check for set win
        if st.session_state.games1 >= 6 and st.session_state.games1 - st.session_state.games2 >= 2:
            st.session_state.sets1 += 1
            st.session_state.games1, st.session_state.games2 = 0, 0
    
    elif st.session_state.points2 >= 4 and st.session_state.points2 - st.session_state.points1 >= 2:
        st.session_state.games2 += 1
        st.session_state.points1, st.session_state.points2 = 0, 0
        st.session_state.server = 2 if st.session_state.server == 1 else 1  # Switch server
        
        # Check for set win
        if st.session_state.games2 >= 6 and st.session_state.games2 - st.session_state.games1 >= 2:
            st.session_state.sets2 += 1
            st.session_state.games1, st.session_state.games2 = 0, 0

st.markdown("### 🎮 Quick Point Tracking (Alternative to Manual Input)")

col_pt1, col_pt2, col_pt3, col_pt4, col_pt5, col_pt6 = st.columns(6)

with col_pt1:
    if st.button(f"✅ {st.session_state.p1_name[:8]} Point", use_container_width=True, type="primary"):
        update_score(1, rally_count=3)
        st.rerun()

with col_pt2:
    if st.button(f"✅ {st.session_state.p2_name[:8]} Point", use_container_width=True, type="primary"):
        update_score(2, rally_count=3)
        st.rerun()

with col_pt3:
    if st.button("🎾 ACE", use_container_width=True):
        if st.session_state.server == 1:
            st.session_state.aces_p1 += 1
            update_score(1, is_ace=True, rally_count=1)
        else:
            st.session_state.aces_p2 += 1
            update_score(2, is_ace=True, rally_count=1)
        st.rerun()

with col_pt4:
    if st.button("❌ DF", use_container_width=True):
        if st.session_state.server == 1:
            st.session_state.dfs_p1 += 1
            update_score(2, is_df=True, rally_count=1)
        else:
            st.session_state.dfs_p2 += 1
            update_score(1, is_df=True, rally_count=1)
        st.rerun()

with col_pt5:
    if st.button("🔴 BP", help="Track break point", use_container_width=True):
        if st.session_state.server == 1:
            st.session_state.total_bp_p1 += 1
        else:
            st.session_state.total_bp_p2 += 1
        st.rerun()

with col_pt6:
    if st.button("🔄 Reset", help="Reset match", use_container_width=True):
        for key in ['sets1', 'sets2', 'games1', 'games2', 'points1', 'points2', 
                   'total_bp_p1', 'total_bp_p2', 'bp_converted_p1', 'bp_converted_p2',
                   'aces_p1', 'aces_p2', 'dfs_p1', 'dfs_p2', 'winners_p1', 'winners_p2',
                   'ues_p1', 'ues_p2', 'serve_faults_p1', 'serve_faults_p2',
                   'total_serves_p1', 'total_serves_p2',
                   'current_streak_p1', 'current_streak_p2']:
            if key in st.session_state:
                st.session_state[key] = 0
        for key in ['prob_history', 'point_history', 'performance_history_p1', 
                     'performance_history_p2', 'momentum_history', 'sim_history',
                     'return_games_p1', 'return_games_p2', 'recent_games_p1', 'recent_games_p2']:
            if key in st.session_state:
                st.session_state[key] = []
        st.session_state.server = 1
        st.session_state.ewma_momentum_p1 = 50.0
        st.session_state.ewma_momentum_p2 = 50.0
        st.session_state.sim_win_prob_p1 = 0.5
        st.rerun()

# Additional tracking buttons
col_t1, col_t2, col_t3, col_t4 = st.columns(4)
with col_t1:
    if st.button(f"🎯 {st.session_state.p1_name} Winner", use_container_width=True):
        st.session_state.winners_p1 += 1
with col_t2:
    if st.button(f"🎯 {st.session_state.p2_name} Winner", use_container_width=True):
        st.session_state.winners_p2 += 1
with col_t3:
    if st.button(f"💥 {st.session_state.p1_name} UE", use_container_width=True):
        st.session_state.ues_p1 += 1
with col_t4:
    if st.button(f"💥 {st.session_state.p2_name} UE", use_container_width=True):
        st.session_state.ues_p2 += 1

st.markdown("---")

# ============================================================================
# LIVE BETTING EDGES - MATCH/SET/GAME LEVEL
# ============================================================================

st.markdown("## 💰 LIVE BETTING EDGES (Match / Set / Game)")
st.caption("Real-time edges calculated from live bookmaker odds vs TRUE P")

# Calculate edges at all levels
col_edge1, col_edge2, col_edge3 = st.columns(3)

with col_edge1:
    st.markdown("### 🏆 Match Winner Edge")
    
    # Player 1
    match_implied_p1 = 1 / st.session_state.live_p1_match_odds
    match_edge_p1 = true_p1 - match_implied_p1
    
    st.markdown(f"**{st.session_state.p1_name}**")
    st.metric("Bookmaker Odds", f"{st.session_state.live_p1_match_odds:.2f}")
    st.metric("Implied Prob", f"{match_implied_p1:.1%}")
    st.metric("TRUE P", f"{true_p1:.1%}")
    
    if match_edge_p1 > 0.05:
        kelly = match_edge_p1 / (st.session_state.live_p1_match_odds - 1)
        stake = min(kelly * 100, 150)
        st.markdown(f"""
        <div class="value-bet">
            ✅ <strong>VALUE BET!</strong>
            <br/>Edge: {match_edge_p1:+.1%}
            <br/>Kelly: {kelly:.1%}
            <br/>💰 Bet ${stake:.0f}
        </div>
        """, unsafe_allow_html=True)
    elif match_edge_p1 > 0:
        st.success(f"✅ Small edge: {match_edge_p1:+.1%}")
    else:
        st.error(f"❌ No value: {match_edge_p1:+.1%}")
    
    st.markdown("---")
    
    # Player 2
    match_implied_p2 = 1 / st.session_state.live_p2_match_odds
    match_edge_p2 = true_p2 - match_implied_p2
    
    st.markdown(f"**{st.session_state.p2_name}**")
    st.metric("Bookmaker Odds", f"{st.session_state.live_p2_match_odds:.2f}")
    st.metric("Implied Prob", f"{match_implied_p2:.1%}")
    st.metric("TRUE P", f"{true_p2:.1%}")
    
    if match_edge_p2 > 0.05:
        kelly = match_edge_p2 / (st.session_state.live_p2_match_odds - 1)
        stake = min(kelly * 100, 150)
        st.markdown(f"""
        <div class="value-bet">
            ✅ <strong>VALUE BET!</strong>
            <br/>Edge: {match_edge_p2:+.1%}
            <br/>Kelly: {kelly:.1%}
            <br/>💰 Bet ${stake:.0f}
        </div>
        """, unsafe_allow_html=True)
    elif match_edge_p2 > 0:
        st.success(f"✅ Small edge: {match_edge_p2:+.1%}")
    else:
        st.error(f"❌ No value: {match_edge_p2:+.1%}")

with col_edge2:
    st.markdown("### 🎯 Current Set Edge")
    
    # Player 1
    set_implied_p1 = 1 / st.session_state.live_p1_set_odds
    set_edge_p1 = p1_set - set_implied_p1
    
    st.markdown(f"**{st.session_state.p1_name}**")
    st.metric("Bookmaker Odds", f"{st.session_state.live_p1_set_odds:.2f}")
    st.metric("Implied Prob", f"{set_implied_p1:.1%}")
    st.metric("TRUE P", f"{p1_set:.1%}")
    
    if set_edge_p1 > 0.05:
        kelly = set_edge_p1 / (st.session_state.live_p1_set_odds - 1)
        stake = min(kelly * 100, 150)
        st.markdown(f"""
        <div class="value-bet">
            ✅ <strong>VALUE BET!</strong>
            <br/>Edge: {set_edge_p1:+.1%}
            <br/>Kelly: {kelly:.1%}
            <br/>💰 Bet ${stake:.0f}
        </div>
        """, unsafe_allow_html=True)
    elif set_edge_p1 > 0:
        st.success(f"✅ Small edge: {set_edge_p1:+.1%}")
    else:
        st.error(f"❌ No value: {set_edge_p1:+.1%}")
    
    st.markdown("---")
    
    # Player 2
    set_implied_p2 = 1 / st.session_state.live_p2_set_odds
    set_edge_p2 = p2_set - set_implied_p2
    
    st.markdown(f"**{st.session_state.p2_name}**")
    st.metric("Bookmaker Odds", f"{st.session_state.live_p2_set_odds:.2f}")
    st.metric("Implied Prob", f"{set_implied_p2:.1%}")
    st.metric("TRUE P", f"{p2_set:.1%}")
    
    if set_edge_p2 > 0.05:
        kelly = set_edge_p2 / (st.session_state.live_p2_set_odds - 1)
        stake = min(kelly * 100, 150)
        st.markdown(f"""
        <div class="value-bet">
            ✅ <strong>VALUE BET!</strong>
            <br/>Edge: {set_edge_p2:+.1%}
            <br/>Kelly: {kelly:.1%}
            <br/>💰 Bet ${stake:.0f}
        </div>
        """, unsafe_allow_html=True)
    elif set_edge_p2 > 0:
        st.success(f"✅ Small edge: {set_edge_p2:+.1%}")
    else:
        st.error(f"❌ No value: {set_edge_p2:+.1%}")

with col_edge3:
    st.markdown("### 🎾 Current Game Edge")
    
    # Use dynamically calculated current game probabilities (already adjusted for point score)
    p1_game_prob = p1_current_game_prob
    p2_game_prob = p2_current_game_prob
    
    # Player 1
    game_implied_p1 = 1 / st.session_state.live_p1_game_odds
    game_edge_p1 = p1_game_prob - game_implied_p1
    
    server_indicator = "🎾 Serving" if st.session_state.server == 1 else "🔄 Returning"
    st.markdown(f"**{st.session_state.p1_name}** {server_indicator}")
    st.metric("Bookmaker Odds", f"{st.session_state.live_p1_game_odds:.2f}")
    st.metric("Implied Prob", f"{game_implied_p1:.1%}")
    st.metric("TRUE P", f"{p1_game_prob:.1%}")
    
    if game_edge_p1 > 0.05:
        kelly = game_edge_p1 / (st.session_state.live_p1_game_odds - 1)
        stake = min(kelly * 100, 150)
        st.markdown(f"""
        <div class="value-bet">
            ✅ <strong>VALUE BET!</strong>
            <br/>Edge: {game_edge_p1:+.1%}
            <br/>Kelly: {kelly:.1%}
            <br/>💰 Bet ${stake:.0f}
        </div>
        """, unsafe_allow_html=True)
    elif game_edge_p1 > 0:
        st.success(f"✅ Small edge: {game_edge_p1:+.1%}")
    else:
        st.error(f"❌ No value: {game_edge_p1:+.1%}")
    
    st.markdown("---")
    
    # Player 2
    game_implied_p2 = 1 / st.session_state.live_p2_game_odds
    game_edge_p2 = p2_game_prob - game_implied_p2
    
    server_indicator = "🎾 Serving" if st.session_state.server == 2 else "🔄 Returning"
    st.markdown(f"**{st.session_state.p2_name}** {server_indicator}")
    st.metric("Bookmaker Odds", f"{st.session_state.live_p2_game_odds:.2f}")
    st.metric("Implied Prob", f"{game_implied_p2:.1%}")
    st.metric("TRUE P", f"{p2_game_prob:.1%}")
    
    if game_edge_p2 > 0.05:
        kelly = game_edge_p2 / (st.session_state.live_p2_game_odds - 1)
        stake = min(kelly * 100, 150)
        st.markdown(f"""
        <div class="value-bet">
            ✅ <strong>VALUE BET!</strong>
            <br/>Edge: {game_edge_p2:+.1%}
            <br/>Kelly: {kelly:.1%}
            <br/>💰 Bet ${stake:.0f}
        </div>
        """, unsafe_allow_html=True)
    elif game_edge_p2 > 0:
        st.success(f"✅ Small edge: {game_edge_p2:+.1%}")
    else:
        st.error(f"❌ No value: {game_edge_p2:+.1%}")

# Summary of best current bet
st.markdown("---")
st.markdown("### 🎯 BEST CURRENT BET")

all_edges = [
    {'market': f'Match Winner - {st.session_state.p1_name}', 'edge': match_edge_p1, 'odds': st.session_state.live_p1_match_odds, 'prob': true_p1},
    {'market': f'Match Winner - {st.session_state.p2_name}', 'edge': match_edge_p2, 'odds': st.session_state.live_p2_match_odds, 'prob': true_p2},
    {'market': f'Set Winner - {st.session_state.p1_name}', 'edge': set_edge_p1, 'odds': st.session_state.live_p1_set_odds, 'prob': p1_set},
    {'market': f'Set Winner - {st.session_state.p2_name}', 'edge': set_edge_p2, 'odds': st.session_state.live_p2_set_odds, 'prob': p2_set},
    {'market': f'Game Winner - {st.session_state.p1_name}', 'edge': game_edge_p1, 'odds': st.session_state.live_p1_game_odds, 'prob': p1_game_prob},
    {'market': f'Game Winner - {st.session_state.p2_name}', 'edge': game_edge_p2, 'odds': st.session_state.live_p2_game_odds, 'prob': p2_game_prob},
]

best_edge = max(all_edges, key=lambda x: x['edge'])

if best_edge['edge'] > 0.05:
    kelly = best_edge['edge'] / (best_edge['odds'] - 1)
    stake = min(kelly * 100, 150)
    ev = (best_edge['prob'] * (best_edge['odds'] - 1)) - (1 - best_edge['prob'])
    
    st.markdown(f"""
    <div class="edge-critical">
        🚨 <strong>MAXIMUM EDGE DETECTED</strong>
        <br/>Market: <strong>{best_edge['market']}</strong>
        <br/>📊 TRUE P: {best_edge['prob']:.1%} vs Implied: {1/best_edge['odds']:.1%}
        <br/>💰 Edge: {best_edge['edge']:+.1%} | EV: {ev:+.1%}
        <br/>🎯 Odds: {best_edge['odds']:.2f}
        <br/>💵 Kelly Stake: {kelly:.1%} of bankroll = ${stake:.0f}
        <br/><br/>
        <strong>✅ BET ${stake:.0f} on {best_edge['market']} @ {best_edge['odds']:.2f}</strong>
    </div>
    """, unsafe_allow_html=True)
elif best_edge['edge'] > 0:
    st.info(f"💡 Best edge: {best_edge['market']} ({best_edge['edge']:+.1%}) - Consider small bet")
else:
    st.warning("⚠️ No positive edges detected at current odds")

st.markdown("---")

# ============================================================================
# BOOKMAKER EDGE DETECTION & BET TRACKING SYSTEM
# ============================================================================

st.markdown("## 📊 BOOKMAKER CONTROL PANEL")

# Odds Variance Tracking
col_var1, col_var2, col_var3 = st.columns(3)

with col_var1:
    st.markdown("### 📉 ODDS MOVEMENT")
    pre_p1_implied = 1 / st.session_state.pre_match_p1_odds
    live_p1_implied = 1 / st.session_state.live_p1_match_odds
    odds_drift_p1 = ((st.session_state.live_p1_match_odds - st.session_state.pre_match_p1_odds) / st.session_state.pre_match_p1_odds) * 100
    
    st.metric(f"{st.session_state.p1_name}", 
              f"{st.session_state.live_p1_match_odds:.2f}",
              f"{odds_drift_p1:+.1f}% from {st.session_state.pre_match_p1_odds:.2f}",
              delta_color="inverse")
    
    if abs(odds_drift_p1) > 20:
        st.error(f"🚨 MAJOR DRIFT: {odds_drift_p1:+.1f}%")
    elif abs(odds_drift_p1) > 10:
        st.warning(f"⚠️ Significant move")
    
with col_var2:
    pre_p2_implied = 1 / st.session_state.pre_match_p2_odds
    live_p2_implied = 1 / st.session_state.live_p2_match_odds
    odds_drift_p2 = ((st.session_state.live_p2_match_odds - st.session_state.pre_match_p2_odds) / st.session_state.pre_match_p2_odds) * 100
    
    st.metric(f"{st.session_state.p2_name}", 
              f"{st.session_state.live_p2_match_odds:.2f}",
              f"{odds_drift_p2:+.1f}% from {st.session_state.pre_match_p2_odds:.2f}",
              delta_color="inverse")
    
    if abs(odds_drift_p2) > 20:
        st.error(f"🚨 MAJOR DRIFT: {odds_drift_p2:+.1f}%")
    elif abs(odds_drift_p2) > 10:
        st.warning(f"⚠️ Significant move")

with col_var3:
    st.markdown("### 🎯 TRUE P vs MARKET")
    true_edge_p1 = true_p1 - live_p1_implied
    true_edge_p2 = true_p2 - live_p2_implied
    
    if abs(true_edge_p1) > abs(true_edge_p2):
        if true_edge_p1 > 0.05:
            st.success(f"✅ {st.session_state.p1_name}: {true_edge_p1:+.1%} VALUE")
        elif true_edge_p1 < -0.05:
            st.error(f"❌ {st.session_state.p1_name}: {true_edge_p1:+.1%} OVERPRICED")
    else:
        if true_edge_p2 > 0.05:
            st.success(f"✅ {st.session_state.p2_name}: {true_edge_p2:+.1%} VALUE")
        elif true_edge_p2 < -0.05:
            st.error(f"❌ {st.session_state.p2_name}: {true_edge_p2:+.1%} OVERPRICED")

st.markdown("---")

# Momentum & Match Intelligence
st.markdown("### 🧠 MATCH INTELLIGENCE (Live Learning)")

col_mom1, col_mom2, col_mom3, col_mom4 = st.columns(4)

with col_mom1:
    st.markdown("**🔥 Momentum**")
    if st.session_state.momentum_score > 3:
        st.success(f"{st.session_state.p1_name} +{st.session_state.momentum_score}")
    elif st.session_state.momentum_score < -3:
        st.success(f"{st.session_state.p2_name} {st.session_state.momentum_score}")
    else:
        st.info(f"Neutral ({st.session_state.momentum_score})")

with col_mom2:
    st.markdown("**💎 Clutch Performance**")
    p1_clutch = st.session_state.p1_clutch_points_won / max(1, st.session_state.p1_clutch_points_total)
    p2_clutch = st.session_state.p2_clutch_points_won / max(1, st.session_state.p2_clutch_points_total)
    if p1_clutch > p2_clutch + 0.1:
        st.success(f"{st.session_state.p1_name}: {p1_clutch:.0%}")
    elif p2_clutch > p1_clutch + 0.1:
        st.success(f"{st.session_state.p2_name}: {p2_clutch:.0%}")
    else:
        st.info(f"Even: {p1_clutch:.0%} / {p2_clutch:.0%}")

with col_mom3:
    st.markdown("**📈 Recent Form (L5)**")
    p1_recent = len([g for g in st.session_state.recent_games_p1[-5:] if g]) if st.session_state.recent_games_p1 else 0
    p2_recent = len([g for g in st.session_state.recent_games_p2[-5:] if g]) if st.session_state.recent_games_p2 else 0
    st.metric(f"{st.session_state.p1_name}", f"{p1_recent}/5")
    st.metric(f"{st.session_state.p2_name}", f"{p2_recent}/5")

with col_mom4:
    st.markdown("**⚡ Break Point Strength**")
    bp_strength_p1 = st.session_state.bp_converted_p1 / max(1, st.session_state.total_bp_p2) if st.session_state.total_bp_p2 > 0 else 0
    bp_strength_p2 = st.session_state.bp_converted_p2 / max(1, st.session_state.total_bp_p1) if st.session_state.total_bp_p1 > 0 else 0
    if bp_strength_p1 > bp_strength_p2 + 0.15:
        st.success(f"{st.session_state.p1_name}: {bp_strength_p1:.0%}")
    elif bp_strength_p2 > bp_strength_p1 + 0.15:
        st.success(f"{st.session_state.p2_name}: {bp_strength_p2:.0%}")
    else:
        st.info(f"Even")

st.markdown("---")

# ============================================================================
# BET PLACEMENT & TRACKING SYSTEM
# ============================================================================

st.markdown("## 💰 BET TRACKING SYSTEM")

col_bet1, col_bet2 = st.columns([2, 1])

with col_bet1:
    st.markdown("### 🎯 PLACE NEW BET")
    
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    
    with col_b1:
        bet_market = st.selectbox("Market", [
            "Match Winner", "Set Winner", "Game Winner",
            "Break Point", "Next Game", "Handicap"
        ], key="bet_market_select")
    
    with col_b2:
        bet_player = st.selectbox("Player", [
            st.session_state.p1_name, 
            st.session_state.p2_name
        ], key="bet_player_select")
    
    with col_b3:
        # Auto-fill odds based on selection
        if bet_market == "Match Winner":
            default_odds = st.session_state.live_p1_match_odds if bet_player == st.session_state.p1_name else st.session_state.live_p2_match_odds
        elif bet_market == "Set Winner":
            default_odds = st.session_state.live_p1_set_odds if bet_player == st.session_state.p1_name else st.session_state.live_p2_set_odds
        else:  # Game Winner
            default_odds = st.session_state.live_p1_game_odds if bet_player == st.session_state.p1_name else st.session_state.live_p2_game_odds
        
        bet_odds = st.number_input("Odds", 1.01, 50.0, float(default_odds), 0.01, key="bet_odds_input")
    
    with col_b4:
        bet_stake = st.number_input("Stake ($)", 1.0, 1000.0, 10.0, 1.0, key="bet_stake_input")
    
    potential_return = bet_stake * bet_odds
    potential_profit = potential_return - bet_stake
    
    col_take1, col_take2 = st.columns(2)
    with col_take1:
        st.metric("Potential Return", f"${potential_return:.2f}")
        st.metric("Potential Profit", f"${potential_profit:.2f}", f"{((potential_profit/bet_stake)*100):.1f}%")
    
    with col_take2:
        if st.button("✅ TAKE BET", type="primary", use_container_width=True):
            new_bet = {
                'id': len(st.session_state.active_bets) + len(st.session_state.settled_bets) + 1,
                'market': bet_market,
                'player': bet_player,
                'odds': bet_odds,
                'stake': bet_stake,
                'potential_return': potential_return,
                'time': datetime.now().strftime("%H:%M:%S"),
                'score': f"{st.session_state.sets1}-{st.session_state.sets2}, {st.session_state.games1}-{st.session_state.games2}",
                'status': 'ACTIVE'
            }
            st.session_state.active_bets.append(new_bet)
            st.session_state.total_staked += bet_stake
            st.success(f"✅ Bet #{new_bet['id']} placed: ${bet_stake:.2f} on {bet_player} @ {bet_odds:.2f}")
            st.rerun()

with col_bet2:
    st.markdown("### 📊 P&L SUMMARY")
    st.metric("Total Staked", f"${st.session_state.total_staked:.2f}")
    st.metric("Total Returns", f"${st.session_state.total_returns:.2f}")
    st.metric("Net Profit", f"${st.session_state.total_profit:.2f}", 
              f"{(st.session_state.total_profit/max(1,st.session_state.total_staked)*100):.1f}% ROI")
    
    settled_count = len(st.session_state.settled_bets)
    won_count = len([b for b in st.session_state.settled_bets if b.get('result') == 'WON'])
    if settled_count > 0:
        st.metric("Win Rate", f"{won_count}/{settled_count}", f"{(won_count/settled_count*100):.1f}%")

# Active Bets Display
if st.session_state.active_bets:
    st.markdown("### 🎫 ACTIVE BETS")
    for bet in st.session_state.active_bets:
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        
        with col1:
            st.write(f"**Bet #{bet['id']}**: {bet['market']} - {bet['player']}")
            st.caption(f"Placed: {bet['time']} | Score: {bet['score']}")
        
        with col2:
            st.write(f"💰 ${bet['stake']:.2f} @ {bet['odds']:.2f}")
            st.caption(f"Return: ${bet['potential_return']:.2f}")
        
        with col3:
            if st.button(f"✅ Won", key=f"won_{bet['id']}", use_container_width=True):
                bet['result'] = 'WON'
                bet['profit'] = bet['potential_return'] - bet['stake']
                st.session_state.total_returns += bet['potential_return']
                st.session_state.total_profit += bet['profit']
                st.session_state.settled_bets.append(bet)
                st.session_state.active_bets.remove(bet)
                st.rerun()
        
        with col4:
            if st.button(f"❌ Lost", key=f"lost_{bet['id']}", use_container_width=True):
                bet['result'] = 'LOST'
                bet['profit'] = -bet['stake']
                st.session_state.total_profit -= bet['stake']
                st.session_state.settled_bets.append(bet)
                st.session_state.active_bets.remove(bet)
                st.rerun()
        
        st.markdown("---")

# Settled Bets History
if st.session_state.settled_bets:
    with st.expander(f"📜 BETTING HISTORY ({len(st.session_state.settled_bets)} bets)"):
        for bet in reversed(st.session_state.settled_bets[-10:]):  # Last 10
            result_icon = "✅" if bet['result'] == 'WON' else "❌"
            profit_color = "green" if bet['profit'] > 0 else "red"
            st.markdown(f"""
            {result_icon} **Bet #{bet['id']}** {bet['market']} - {bet['player']} @ {bet['odds']:.2f}
            - Stake: ${bet['stake']:.2f} | Profit: <span style='color:{profit_color}'>${bet['profit']:+.2f}</span>
            - Time: {bet['time']} | Score: {bet['score']}
            """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# EDGE DETECTION DISPLAY - THE MAIN FEATURE
# ============================================================================

st.markdown("## 🎯 LIVE EDGE DETECTION")

# Detect all edges
detected_edges = detect_all_edges(p1_stats, p2_stats, live_stats, current_probs, match_conditions)

if detected_edges:
    st.markdown(f"**{len(detected_edges)} EDGES DETECTED** - Sorted by severity and value")
    
    for edge in detected_edges:
        if edge['severity'] == 'CRITICAL':
            st.markdown(f"""
            <div class="edge-critical">
                🚨 CRITICAL EDGE - {edge['type']}
                <br/>Player: <strong>{edge['player']}</strong>
                <br/>📊 {edge['message']}
                <br/>💰 Edge Value: {edge['edge_value']:.1f}%
                <br/>✅ Action: {edge['action']}
            </div>
            """, unsafe_allow_html=True)
        
        elif edge['severity'] == 'HIGH':
            st.markdown(f"""
            <div class="edge-high">
                ⚠️ HIGH EDGE - {edge['type']}
                <br/>Player: <strong>{edge['player']}</strong>
                <br/>📊 {edge['message']}
                <br/>💰 Edge Value: {edge['edge_value']:.1f}%
                <br/>✅ Action: {edge['action']}
            </div>
            """, unsafe_allow_html=True)
        
        else:  # MEDIUM
            st.markdown(f"""
            <div class="edge-medium">
                💡 {edge['type']} - {edge['player']}
                <br/>{edge['message']} (Edge: {edge['edge_value']:.1f}%)
                <br/><em>{edge['action']}</em>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("👀 No significant edges detected yet. Continue tracking match...")

st.markdown("---")

# ============================================================================
# MODEL PREDICTIONS & TRUE P
# ============================================================================

st.markdown("## 🤖 TRUE P - Model Predictions")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown("### 📈 Markov Chain")
    st.metric(f"{st.session_state.p1_name}", f"{p1_match_markov:.1%}")
    st.metric(f"{st.session_state.p2_name}", f"{p2_match_markov:.1%}")
    st.caption("Based on serve/return win %")

with col_m2:
    if ml_probs:
        st.markdown("### 🧠 ML Ensemble")
        st.caption(f"**Logistic Regression:** {ml_probs['logistic_regression']:.1%} / {1-ml_probs['logistic_regression']:.1%}")
        st.caption(f"**Random Forest:** {ml_probs['random_forest']:.1%} / {1-ml_probs['random_forest']:.1%}")
        st.metric("Ensemble (60/40)", f"{ml_probs['ensemble']:.1%}")
    else:
        st.markdown("### 🧠 ML Ensemble")
        st.warning("Models not loaded")

with col_m3:
    st.markdown("### 🔬 EWMA Simulation")
    st.caption("*Qian et al. (2025) method*")
    if has_momentum_data:
        st.metric(f"{st.session_state.p1_name}", f"{sim_p1:.1%}")
        st.metric(f"{st.session_state.p2_name}", f"{1 - sim_p1:.1%}")
        st.caption(f"λ={EWMA_LAMBDA} | {len(st.session_state.point_history)} pts tracked")
        st.caption(f"Momentum: {st.session_state.ewma_momentum_p1:.1f} vs {st.session_state.ewma_momentum_p2:.1f}")
    else:
        st.info("Track 3+ points to activate")
        st.caption("Uses EWMA momentum + Monte Carlo")

with col_m4:
    st.markdown("### ⭐ TRUE P (Final)")
    st.markdown(f'<div class="prob-display" style="background:#28a745;color:white">{st.session_state.p1_name}: {true_p1:.1%}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prob-display" style="background:#007bff;color:white">{st.session_state.p2_name}: {true_p2:.1%}</div>', unsafe_allow_html=True)
    if has_momentum_data and ml_probs:
        st.caption("Blend: ML 35% + Markov 25% + Simulation 40%")
    elif ml_probs:
        st.caption("Blend: ML 60% + Markov 40%")
    elif has_momentum_data:
        st.caption("Blend: Markov 50% + Simulation 50%")
    else:
        st.caption("Markov only (track points for full model)")

st.markdown("---")

# ============================================================================
# VALUE BETS BASED ON TRUE P
# ============================================================================

st.markdown("## 💰 Value Bet Analysis")

col_vb1, col_vb2 = st.columns(2)

with col_vb1:
    st.markdown(f"### {st.session_state.p1_name}")
    implied_p1 = 1 / st.session_state.p1_match_odds
    edge_p1 = true_p1 - implied_p1
    
    st.metric("Bookmaker Odds", f"{st.session_state.p1_match_odds:.2f}")
    st.metric("Implied Probability", f"{implied_p1:.1%}")
    st.metric("TRUE P (Our Model)", f"{true_p1:.1%}")
    st.metric("EDGE", f"{edge_p1:+.1%}", delta=f"{edge_p1:+.1%}")
    
    if edge_p1 > 0.05:  # 5% edge
        ev = (true_p1 * (st.session_state.p1_match_odds - 1)) - (1 - true_p1)
        kelly = edge_p1 / (st.session_state.p1_match_odds - 1)
        stake = min(kelly * 100, 150)  # Max 150 units
        
        st.markdown(f"""
        <div class="value-bet">
            ✅ <strong>VALUE BET DETECTED!</strong>
            <br/>Expected Value: {ev:+.1%}
            <br/>Kelly Criterion: {kelly:.1%} of bankroll
            <br/>💰 Recommended Stake: ${stake:.0f}
            <br/><br/>
            <strong>Bet ${stake:.0f} on {st.session_state.p1_name} @ {st.session_state.p1_match_odds:.2f}</strong>
        </div>
        """, unsafe_allow_html=True)
    elif edge_p1 > 0:
        st.info(f"Slight edge ({edge_p1:.1%}) - consider small bet")
    else:
        st.warning(f"No value - bookmaker price too low")

with col_vb2:
    st.markdown(f"### {st.session_state.p2_name}")
    implied_p2 = 1 / st.session_state.p2_match_odds
    edge_p2 = true_p2 - implied_p2
    
    st.metric("Bookmaker Odds", f"{st.session_state.p2_match_odds:.2f}")
    st.metric("Implied Probability", f"{implied_p2:.1%}")
    st.metric("TRUE P (Our Model)", f"{true_p2:.1%}")
    st.metric("EDGE", f"{edge_p2:+.1%}", delta=f"{edge_p2:+.1%}")
    
    if edge_p2 > 0.05:
        ev = (true_p2 * (st.session_state.p2_match_odds - 1)) - (1 - true_p2)
        kelly = edge_p2 / (st.session_state.p2_match_odds - 1)
        stake = min(kelly * 100, 150)
        
        st.markdown(f"""
        <div class="value-bet">
            ✅ <strong>VALUE BET DETECTED!</strong>
            <br/>Expected Value: {ev:+.1%}
            <br/>Kelly Criterion: {kelly:.1%} of bankroll
            <br/>💰 Recommended Stake: ${stake:.0f}
            <br/><br/>
            <strong>Bet ${stake:.0f} on {st.session_state.p2_name} @ {st.session_state.p2_match_odds:.2f}</strong>
        </div>
        """, unsafe_allow_html=True)
    elif edge_p2 > 0:
        st.info(f"Slight edge ({edge_p2:.1%}) - consider small bet")
    else:
        st.warning(f"No value - bookmaker price too low")

# ============================================================================
# PROBABILITY EVOLUTION CHART
# ============================================================================

if len(st.session_state.prob_history) > 1:
    st.markdown("---")
    st.markdown("## 📈 Probability Evolution")
    
    df_hist = pd.DataFrame(st.session_state.prob_history)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hist['point'],
        y=df_hist['p1_prob'] * 100,
        mode='lines+markers',
        name=st.session_state.p1_name,
        line=dict(color='#28a745', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=df_hist['point'],
        y=df_hist['p2_prob'] * 100,
        mode='lines+markers',
        name=st.session_state.p2_name,
        line=dict(color='#007bff', width=3)
    ))
    
    fig.update_layout(
        title="Match Win Probability Over Time",
        xaxis_title="Points Played",
        yaxis_title="Win Probability (%)",
        yaxis_range=[0, 100],
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# EWMA MOMENTUM CHART (Qian et al. 2025 - Fig 5)
# ============================================================================

if len(st.session_state.momentum_history) > 1:
    st.markdown("---")
    st.markdown("## 🔬 EWMA Momentum Analysis")
    st.caption("*Based on Qian et al. (2025) — Exponentially Weighted Moving Average, λ=0.6467*")
    
    from plotly.subplots import make_subplots
    
    mom_data = st.session_state.momentum_history
    points_x = list(range(1, len(mom_data) + 1))
    p1_mom = [m['momentum_p1'] for m in mom_data]
    p2_mom = [m['momentum_p2'] for m in mom_data]
    p1_perf = [m.get('perf_p1', 50) for m in mom_data]
    p2_perf = [m.get('perf_p2', 50) for m in mom_data]
    
    fig_mom = make_subplots(
        rows=2, cols=1,
        subplot_titles=("EWMA Momentum (λ=0.6467)", "4-Parameter Performance Score"),
        vertical_spacing=0.15,
        row_heights=[0.55, 0.45]
    )
    
    # EWMA Momentum traces
    fig_mom.add_trace(go.Scatter(
        x=points_x, y=p1_mom,
        mode='lines+markers', name=f"{st.session_state.p1_name} Momentum",
        line=dict(color='#28a745', width=3),
        marker=dict(size=6)
    ), row=1, col=1)
    fig_mom.add_trace(go.Scatter(
        x=points_x, y=p2_mom,
        mode='lines+markers', name=f"{st.session_state.p2_name} Momentum",
        line=dict(color='#007bff', width=3),
        marker=dict(size=6)
    ), row=1, col=1)
    # 50-line reference
    fig_mom.add_hline(y=50, line_dash="dash", line_color="gray",
                      annotation_text="Neutral", row=1, col=1)
    
    # Performance traces
    fig_mom.add_trace(go.Scatter(
        x=points_x, y=p1_perf,
        mode='lines', name=f"{st.session_state.p1_name} Performance",
        line=dict(color='#28a745', width=2, dash='dot')
    ), row=2, col=1)
    fig_mom.add_trace(go.Scatter(
        x=points_x, y=p2_perf,
        mode='lines', name=f"{st.session_state.p2_name} Performance",
        line=dict(color='#007bff', width=2, dash='dot')
    ), row=2, col=1)
    
    fig_mom.update_layout(
        height=550, hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5)
    )
    fig_mom.update_yaxes(title_text="Momentum", row=1, col=1)
    fig_mom.update_yaxes(title_text="Performance", row=2, col=1)
    fig_mom.update_xaxes(title_text="Points Played", row=2, col=1)
    
    st.plotly_chart(fig_mom, use_container_width=True)
    
    # Momentum summary metrics
    col_ms1, col_ms2, col_ms3 = st.columns(3)
    with col_ms1:
        st.metric(f"{st.session_state.p1_name} Current Momentum",
                  f"{st.session_state.ewma_momentum_p1:.1f}",
                  delta=f"{st.session_state.ewma_momentum_p1 - 50:+.1f} vs neutral")
    with col_ms2:
        st.metric(f"{st.session_state.p2_name} Current Momentum",
                  f"{st.session_state.ewma_momentum_p2:.1f}",
                  delta=f"{st.session_state.ewma_momentum_p2 - 50:+.1f} vs neutral")
    with col_ms3:
        mom_diff = st.session_state.ewma_momentum_p1 - st.session_state.ewma_momentum_p2
        if abs(mom_diff) > 15:
            st.metric("Momentum Gap", f"{abs(mom_diff):.1f}", delta="⚡ STRONG SHIFT")
        elif abs(mom_diff) > 8:
            st.metric("Momentum Gap", f"{abs(mom_diff):.1f}", delta="📈 Building")
        else:
            st.metric("Momentum Gap", f"{abs(mom_diff):.1f}", delta="≈ Even")

# ============================================================================
# SIMULATION WIN PROBABILITY EVOLUTION
# ============================================================================

if len(st.session_state.sim_history) > 1:
    st.markdown("---")
    st.markdown("## 🎲 Monte Carlo Simulation Probability")
    st.caption("*1000-iteration Monte Carlo match simulation from current score position*")
    
    sim_data = st.session_state.sim_history
    sim_x = list(range(1, len(sim_data) + 1))
    sim_p1_vals = [s['sim_p1'] for s in sim_data]
    
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=sim_x, y=[v * 100 for v in sim_p1_vals],
        mode='lines+markers', name=f"{st.session_state.p1_name} Sim Win %",
        line=dict(color='#ff6b35', width=3),
        fill='tozeroy', fillcolor='rgba(255,107,53,0.1)'
    ))
    fig_sim.add_trace(go.Scatter(
        x=sim_x, y=[(1 - v) * 100 for v in sim_p1_vals],
        mode='lines+markers', name=f"{st.session_state.p2_name} Sim Win %",
        line=dict(color='#6b35ff', width=3),
        fill='tozeroy', fillcolor='rgba(107,53,255,0.1)'
    ))
    fig_sim.add_hline(y=50, line_dash="dash", line_color="gray")
    fig_sim.update_layout(
        xaxis_title="Points Played", yaxis_title="Simulated Win Probability (%)",
        yaxis_range=[0, 100], height=350, hovermode='x unified'
    )
    st.plotly_chart(fig_sim, use_container_width=True)

st.markdown("---")
st.caption("🎾 Powered by 112,384 real matches | ML + Markov + EWMA Momentum + Monte Carlo Simulation | Qian et al. (2025)")

# ============================================================================
# API AUTO-REFRESH (must be at the very end)
# ============================================================================
if (_API_AVAILABLE
    and st.session_state.api_auto_refresh
    and st.session_state.api_match_key):
    import time as _time
    _time.sleep(30)
    # Auto-refresh: pull new score data from ESPN
    try:
        _svc = get_free_service()
        _key_parts = str(st.session_state.api_match_key).split("_")
        if len(_key_parts) == 2:
            _eid, _cid = _key_parts
            _match = None
            for _tour in ["atp", "wta"]:
                _match = _svc.get_match_detail(_eid, _cid, _tour)
                if _match:
                    break
            if _match:
                st.session_state.sets1 = _match["sets_p1"]
                st.session_state.sets2 = _match["sets_p2"]
                st.session_state.games1 = _match["games_p1"]
                st.session_state.games2 = _match["games_p2"]
                st.session_state.api_last_refresh = datetime.now().strftime("%H:%M:%S")
    except Exception:
        pass  # Silently skip on error
    st.rerun()
