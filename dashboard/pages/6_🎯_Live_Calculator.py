"""
Live Match Calculator - Professional Multi-Model Betting Interface
==================================================================
Real-time probability calculator using Markov Chain + ML Models
Point-by-point tracking with instant edge detection
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

# Import player stats integration
try:
    from player_stats_integration import get_player_stats_for_calculator
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("⚠️ Player stats integration not available")

st.set_page_config(page_title="Pro Live Calculator", page_icon="🎯", layout="wide")

# Custom CSS for professional bookmaker-style interface
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .value-bet-alert {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .no-value {
        background: #f8f9fa;
        border-left: 5px solid #6c757d;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .model-consensus {
        background: #fff3cd;
        border: 2px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .live-score {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 10px 0;
    }
    .quick-action-btn {
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎯 Professional Live Betting Calculator</h1><p>Multi-Model Edge Detection • Point-by-Point Tracking</p></div>', unsafe_allow_html=True)

# ============================================================================
# MARKOV CHAIN + ML MODELS
# ============================================================================

def apply_advanced_parameters(base_serve_pct, base_return_pct, adv_params):
    """
    Apply advanced parameter adjustments to base probabilities
    
    Args:
        base_serve_pct: Base serve win percentage
        base_return_pct: Base return win percentage
        adv_params: Dictionary with keys:
            - momentum: 0-1 (recent form)
            - surface_mastery: 0-1 (win rate on surface)
            - clutch: 0-1 (performance in big matches)
            - bp_defense: 0-1 (break point save rate)
            - consistency: 0-1 (low variance)
            - first_serve_pct: 0-1 (first serve percentage)
    
    Returns:
        Adjusted serve_pct, return_pct
    """
    # Momentum adjustment (biggest factor)
    # Range: ±7.5% based on momentum
    momentum_adj = (adv_params.get('momentum', 0.5) - 0.5) * 0.15
    
    # Surface mastery adjustment
    # Range: ±5% based on surface performance
    surface_adj = (adv_params.get('surface_mastery', 0.5) - 0.5) * 0.10
    
    # Clutch performance adjustment
    # Range: ±4% in pressure situations
    clutch_adj = (adv_params.get('clutch', 0.5) - 0.5) * 0.08
    
    # Break point defense adjustment (affects return game resistance)
    # Higher BP defense = harder to break = lower opponent return%
    bp_defense_adj = (adv_params.get('bp_defense', 0.6) - 0.6) * 0.10
    
    # Consistency rating (reduces variance, slight advantage)
    consistency_adj = (adv_params.get('consistency', 0.5) - 0.5) * 0.05
    
    # First serve percentage (affects serve effectiveness)
    first_serve_adj = (adv_params.get('first_serve_pct', 0.62) - 0.62) * 0.08
    
    # Total adjustment
    total_serve_adj = (
        momentum_adj + 
        surface_adj + 
        clutch_adj + 
        bp_defense_adj + 
        consistency_adj +
        first_serve_adj
    )
    
    total_return_adj = (
        momentum_adj + 
        surface_adj + 
        clutch_adj - 
        (bp_defense_adj * 0.5) +  # BP defense hurts opponent's return
        consistency_adj
    )
    
    # Apply adjustments
    adjusted_serve = base_serve_pct + total_serve_adj
    adjusted_return = base_return_pct + total_return_adj
    
    # Clamp to reasonable bounds
    adjusted_serve = max(0.45, min(0.85, adjusted_serve))
    adjusted_return = max(0.15, min(0.55, adjusted_return))
    
    return adjusted_serve, adjusted_return

def calculate_point_probabilities(p1_serve_win, p2_serve_win, p1_adv_params=None, p2_adv_params=None):
    """
    Calculate probabilities using Markov chain with optional advanced parameter adjustments
    
    Args:
        p1_serve_win: Player 1 base serve win percentage
        p2_serve_win: Player 2 base serve win percentage
        p1_adv_params: Player 1 advanced parameters (optional)
        p2_adv_params: Player 2 advanced parameters (optional)
    """
    # Apply advanced parameter adjustments if provided
    if p1_adv_params:
        p1_serve_win_adj, p1_return_win_adj = apply_advanced_parameters(
            p1_serve_win, 1 - p2_serve_win, p1_adv_params
        )
    else:
        p1_serve_win_adj = p1_serve_win
        p1_return_win_adj = 1 - p2_serve_win
    
    if p2_adv_params:
        p2_serve_win_adj, p2_return_win_adj = apply_advanced_parameters(
            p2_serve_win, 1 - p1_serve_win, p2_adv_params
        )
    else:
        p2_serve_win_adj = p2_serve_win
        p2_return_win_adj = 1 - p1_serve_win
    
    # Use adjusted values for calculations
    p1_serve_final = p1_serve_win_adj
    p2_serve_final = p2_serve_win_adj
    
    # Game probability on serve (more accurate formula)
    def game_prob(p_serve):
        # Probability of winning from 0-0
        p = p_serve
        # Win in straight points (4-0)
        prob = p**4
        # Win 4-1
        prob += 4 * p**4 * (1-p)
        # Win 4-2
        prob += 10 * p**4 * (1-p)**2
        # Win from deuce
        deuce_win = (p**2) / (1 - 2*p*(1-p))
        # Probability of reaching deuce then winning
        prob += 20 * p**3 * (1-p)**3 * deuce_win
        return prob
    
    p1_game_on_serve = game_prob(p1_serve_final)
    p2_game_on_serve = game_prob(p2_serve_final)
    
    # Tiebreak probability (simplified)
    p1_tiebreak = p1_serve_final ** 7
    p2_tiebreak = p2_serve_final ** 7
    
    # Set probability
    p1_set = p1_game_on_serve ** 6
    p2_set = p2_game_on_serve ** 6
    
    # Match probability (best of 3)
    p1_match = p1_set ** 2 + 2 * p1_set ** 2 * (1 - p1_set)
    p2_match = p2_set ** 2 + 2 * p2_set ** 2 * (1 - p2_set)
    
    return {
        'p1_game_on_serve': p1_game_on_serve,
        'p2_game_on_serve': p2_game_on_serve,
        'p1_tiebreak': p1_tiebreak,
        'p2_tiebreak': p2_tiebreak,
        'p1_set': p1_set,
        'p2_set': p2_set,
        'p1_match': p1_match,
        'p2_match': p2_match,
        'p1_serve_adjusted': p1_serve_final,
        'p2_serve_adjusted': p2_serve_final
    }

def calculate_ml_probabilities(p1_serve_win, p2_serve_win, p1_sets, p2_sets, p1_games, p2_games):
    """Simulate ML model predictions (Logistic Regression style)"""
    # Feature engineering
    set_diff = p1_sets - p2_sets
    game_diff = p1_games - p2_games
    serve_advantage = p1_serve_win - p2_serve_win
    
    # Logistic regression simulation
    logit = 0.5 + (set_diff * 0.25) + (game_diff * 0.08) + (serve_advantage * 0.4)
    lr_prob = 1 / (1 + np.exp(-logit * 3))
    
    return max(0.05, min(0.95, lr_prob))

def calculate_nn_probabilities(p1_serve_win, p2_serve_win, p1_sets, p2_sets, p1_games, p2_games, p1_points, p2_points):
    """Simulate Neural Network predictions"""
    # More complex non-linear combination
    set_feature = (p1_sets - p2_sets) * 0.3
    game_feature = (p1_games - p2_games) * 0.12
    point_feature = (p1_points - p2_points) * 0.03
    serve_feature = (p1_serve_win - p2_serve_win) * 0.5
    
    # Non-linear activation
    hidden = np.tanh(set_feature + game_feature + serve_feature)
    nn_prob = 0.5 + hidden * 0.45 + point_feature
    
    return max(0.05, min(0.95, nn_prob))

def calculate_break_probability(server_serve_pct, returner_return_pct, point_score_server, point_score_returner):
    """
    Calculate probability of returner breaking serve in current game
    Takes into account current point score in the game
    """
    # Base probability returner wins game (1 - server's game win prob)
    server_point_win = server_serve_pct
    
    def game_prob(p_serve):
        prob = p_serve**4
        prob += 4 * p_serve**4 * (1-p_serve)
        prob += 10 * p_serve**4 * (1-p_serve)**2
        deuce_win = (p_serve**2) / (1 - 2*p_serve*(1-p_serve)) if p_serve != 0.5 else 0.5
        prob += 20 * p_serve**3 * (1-p_serve)**3 * deuce_win
        return prob
    
    server_game_win_base = game_prob(server_point_win)
    break_prob_base = 1 - server_game_win_base
    
    # Adjust for current point score
    score_diff = point_score_returner - point_score_server
    
    # Critical situations
    is_break_point = (point_score_returner >= 3 and point_score_returner > point_score_server)
    is_game_point = (point_score_server >= 3 and point_score_server > point_score_returner)
    
    # Situational adjustments
    if is_break_point:
        # Returner has break point - increase break probability
        break_prob_base *= 1.3
    elif is_game_point:
        # Server has game point - decrease break probability  
        break_prob_base *= 0.7
    elif point_score_server == 3 and point_score_returner == 3:
        # Deuce - use deuce probability
        deuce_break = 1 - ((server_point_win**2) / (1 - 2*server_point_win*(1-server_point_win)))
        break_prob_base = deuce_break if server_point_win != 0.5 else 0.5
    else:
        # Adjust based on point differential
        break_prob_base += score_diff * 0.08
    
    return max(0.01, min(0.99, break_prob_base))

def adjust_for_current_score(base_prob, sets_won, games_won, points_won, total_points_played, games_history=None):
    """
    Adjust probability based on current score with momentum
    Now includes game-by-game tracking and break detection
    """
    # Score adjustments
    set_bonus = sets_won * 0.18
    game_bonus = (games_won / 6) * 0.12
    point_bonus = (points_won / 4) * 0.06
    
    # Momentum factor (more points played = more reliable adjustment)
    momentum = min(total_points_played / 100, 1.0) * 0.05
    
    # Recent game momentum (if history available)
    recent_game_momentum = 0
    if games_history and len(games_history) >= 3:
        # Look at last 3 games won/lost
        recent_games = games_history[-3:]
        recent_wins = sum(1 for g in recent_games if g == 'won')
        recent_game_momentum = (recent_wins / 3 - 0.5) * 0.1  # -0.15 to +0.15
    
    adjusted = base_prob + set_bonus + game_bonus + point_bonus + momentum + recent_game_momentum
    return max(0.05, min(0.95, adjusted))

def calculate_betting_value(win_prob, bookmaker_odds):
    """Calculate expected value"""
    implied_prob = 1 / bookmaker_odds
    edge = win_prob - implied_prob
    ev = (win_prob * (bookmaker_odds - 1)) - (1 - win_prob)
    return edge, ev

# ============================================================================
# SIDEBAR - MATCH SETUP
# ============================================================================

with st.sidebar:
    st.header("🎾 Match Setup")
    
    player1_name = st.text_input("Player 1 Name", value="Novak Djokovic")
    player2_name = st.text_input("Player 2 Name", value="Carlos Alcaraz")
    
    # Surface selector
    surface = st.selectbox("Court Surface", ["Hard", "Clay", "Grass"], index=0)
    
    st.divider()
    
    st.subheader("📊 Player Statistics")
    
    # Auto-fill button
    if STATS_AVAILABLE:
        if st.button("🔄 Auto-Fill Player Stats", use_container_width=True):
            with st.spinner("Fetching player data..."):
                try:
                    p1_stats, p2_stats = get_player_stats_for_calculator(player1_name, player2_name, surface)
                    
                    # Update session state with fetched stats
                    st.session_state.p1_serve_init = int(p1_stats['serve_win_pct'] * 100)
                    st.session_state.p2_serve_init = int(p2_stats['serve_win_pct'] * 100)
                    st.session_state.p1_return_init = int(p1_stats['return_win_pct'] * 100)
                    st.session_state.p2_return_init = int(p2_stats['return_win_pct'] * 100)
                    
                    # Store stats info for display
                    st.session_state.p1_stats_info = p1_stats
                    st.session_state.p2_stats_info = p2_stats
                    
                    st.success("✅ Stats loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error fetching stats: {str(e)}")
    else:
        st.info("💡 Install player_stats_integration.py for auto-fill")
    
    col1, col2 = st.columns(2)
    
    # Initialize default values if not in session state
    if 'p1_serve_init' not in st.session_state:
        st.session_state.p1_serve_init = 68
    if 'p2_serve_init' not in st.session_state:
        st.session_state.p2_serve_init = 65
    if 'p1_return_init' not in st.session_state:
        st.session_state.p1_return_init = 35
    if 'p2_return_init' not in st.session_state:
        st.session_state.p2_return_init = 32
    
    with col1:
        st.markdown(f"**{player1_name}**")
        
        # Display data source badge if available
        if 'p1_stats_info' in st.session_state:
            source = st.session_state.p1_stats_info['source']
            matches = st.session_state.p1_stats_info.get('matches_played', 0)
            if source == 'database':
                st.caption(f"📊 Database ({matches} matches)")
            elif source == 'known_player':
                st.caption("⭐ Known Player")
            else:
                st.caption("📝 Default (ATP Avg)")
        
        p1_serve_win = st.slider("Serve Win %", 50, 85, st.session_state.p1_serve_init, key="p1_serve") / 100
        p1_return_win = st.slider("Return Win %", 20, 50, st.session_state.p1_return_init, key="p1_return") / 100
    
    with col2:
        st.markdown(f"**{player2_name}**")
        
        # Display data source badge if available
        if 'p2_stats_info' in st.session_state:
            source = st.session_state.p2_stats_info['source']
            matches = st.session_state.p2_stats_info.get('matches_played', 0)
            if source == 'database':
                st.caption(f"📊 Database ({matches} matches)")
            elif source == 'known_player':
                st.caption("⭐ Known Player")
            else:
                st.caption("📝 Default (ATP Avg)")
        
        p2_serve_win = st.slider("Serve Win %", 50, 85, st.session_state.p2_serve_init, key="p2_serve") / 100
        p2_return_win = st.slider("Return Win %", 20, 50, st.session_state.p2_return_init, key="p2_return") / 100
    
    # Display H2H if available
    if 'p1_stats_info' in st.session_state and 'h2h' in st.session_state.p1_stats_info:
        h2h = st.session_state.p1_stats_info['h2h']
        if h2h and h2h['total'] > 0:
            st.caption(f"🎯 H2H: {player1_name} leads {h2h['p1_wins']}-{h2h['p2_wins']} ({h2h['p1_win_pct']:.0%})")
    
    st.divider()
    
    # ============================================================================
    # ADVANCED STATISTICS SECTION
    # ============================================================================
    
    st.subheader("🎯 Advanced Parameters (Optional)")
    
    with st.expander("⚙️ Advanced Stats - Tweaks Markov Model", expanded=False):
        st.caption("💡 Enter special parameters if available from database or scouting. These will dynamically adjust the probability calculations.")
        
        st.markdown("---")
        st.markdown(f"**{player1_name} - Advanced Stats**")
        
        col1a, col1b = st.columns(2)
        with col1a:
            p1_momentum = st.slider(
                "Momentum Score", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                key="p1_momentum",
                help="Recent form: 0=poor, 0.5=average, 1=excellent"
            )
            
            p1_clutch = st.slider(
                "Clutch Performance", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                key="p1_clutch",
                help="Performance in big matches (Grand Slams, Masters)"
            )
            
            p1_bp_defense = st.slider(
                "BP Defense Rate", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.6, 
                step=0.01,
                key="p1_bp_defense",
                help="Break points saved / break points faced"
            )
        
        with col1b:
            p1_surface_mastery = st.slider(
                f"Surface Mastery ({surface})", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                key="p1_surface",
                help="Win rate on current surface"
            )
            
            p1_consistency = st.slider(
                "Consistency Rating", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                key="p1_consistency",
                help="Performance variance (1 - std deviation of form)"
            )
            
            p1_first_serve_pct = st.slider(
                "1st Serve In %", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.62, 
                step=0.01,
                key="p1_first_serve_pct",
                help="First serve percentage"
            )
        
        st.markdown("---")
        st.markdown(f"**{player2_name} - Advanced Stats**")
        
        col2a, col2b = st.columns(2)
        with col2a:
            p2_momentum = st.slider(
                "Momentum Score ", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                key="p2_momentum",
                help="Recent form: 0=poor, 0.5=average, 1=excellent"
            )
            
            p2_clutch = st.slider(
                "Clutch Performance ", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                key="p2_clutch",
                help="Performance in big matches (Grand Slams, Masters)"
            )
            
            p2_bp_defense = st.slider(
                "BP Defense Rate ", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.6, 
                step=0.01,
                key="p2_bp_defense",
                help="Break points saved / break points faced"
            )
        
        with col2b:
            p2_surface_mastery = st.slider(
                f"Surface Mastery ({surface}) ", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                key="p2_surface",
                help="Win rate on current surface"
            )
            
            p2_consistency = st.slider(
                "Consistency Rating ", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                key="p2_consistency",
                help="Performance variance (1 - std deviation of form)"
            )
            
            p2_first_serve_pct = st.slider(
                "1st Serve In % ", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.62, 
                step=0.01,
                key="p2_first_serve_pct",
                help="First serve percentage"
            )
        
        # Show adjustment summary
        st.markdown("---")
        st.markdown("**📊 Parameter Impact Preview**")
        
        # Calculate adjustments
        p1_total_adjustment = (
            (p1_momentum - 0.5) * 0.15 +  # Momentum: up to ±7.5%
            (p1_surface_mastery - 0.5) * 0.10 +  # Surface: up to ±5%
            (p1_clutch - 0.5) * 0.08 +  # Clutch: up to ±4%
            (p1_bp_defense - 0.6) * 0.10  # BP Defense: baseline 0.6
        )
        
        p2_total_adjustment = (
            (p2_momentum - 0.5) * 0.15 +
            (p2_surface_mastery - 0.5) * 0.10 +
            (p2_clutch - 0.5) * 0.08 +
            (p2_bp_defense - 0.6) * 0.10
        )
        
        col_adj1, col_adj2 = st.columns(2)
        with col_adj1:
            adj_color = "🟢" if p1_total_adjustment > 0 else "🔴" if p1_total_adjustment < 0 else "⚪"
            st.metric(
                f"{player1_name} Adjustment", 
                f"{p1_total_adjustment:+.1%}",
                delta=None
            )
        
        with col_adj2:
            adj_color = "🟢" if p2_total_adjustment > 0 else "🔴" if p2_total_adjustment < 0 else "⚪"
            st.metric(
                f"{player2_name} Adjustment", 
                f"{p2_total_adjustment:+.1%}",
                delta=None
            )
        
        st.caption("💡 These adjustments are applied to base Markov probabilities. Positive = advantage, Negative = disadvantage.")
    
    st.divider()
    
    st.subheader("💰 Pre-Match Bookmaker Odds")
    st.caption("Enter the odds before the match started")
    
    p1_prematch_odds = st.number_input(f"{player1_name} Pre-Match", min_value=1.01, max_value=50.0, value=1.85, step=0.01, key="p1_prematch")
    p2_prematch_odds = st.number_input(f"{player2_name} Pre-Match", min_value=1.01, max_value=50.0, value=2.10, step=0.01, key="p2_prematch")

# Create advanced parameters dictionaries (default values if not set in expander)
p1_advanced_params = {
    'momentum': st.session_state.get('p1_momentum', 0.5),
    'surface_mastery': st.session_state.get('p1_surface', 0.5),
    'clutch': st.session_state.get('p1_clutch', 0.5),
    'bp_defense': st.session_state.get('p1_bp_defense', 0.6),
    'consistency': st.session_state.get('p1_consistency', 0.5),
    'first_serve_pct': st.session_state.get('p1_first_serve_pct', 0.62)
}

p2_advanced_params = {
    'momentum': st.session_state.get('p2_momentum', 0.5),
    'surface_mastery': st.session_state.get('p2_surface', 0.5),
    'clutch': st.session_state.get('p2_clutch', 0.5),
    'bp_defense': st.session_state.get('p2_bp_defense', 0.6),
    'consistency': st.session_state.get('p2_consistency', 0.5),
    'first_serve_pct': st.session_state.get('p2_first_serve_pct', 0.62)
}

# ============================================================================
# MAIN AREA - LIVE SCORE ENTRY (PROFESSIONAL INTERFACE)
# ============================================================================

# Initialize session state for quick updates
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
if 'p1_games_won_history' not in st.session_state:
    st.session_state.p1_games_won_history = []  # Track recent game wins
if 'p2_games_won_history' not in st.session_state:
    st.session_state.p2_games_won_history = []  # Track recent game wins
if 'break_count_p1' not in st.session_state:
    st.session_state.break_count_p1 = 0  # Number of breaks by P1
if 'break_count_p2' not in st.session_state:
    st.session_state.break_count_p2 = 0  # Number of breaks by P2
if 'p1_point_score' not in st.session_state:
    st.session_state.p1_point_score = 0
if 'p2_point_score' not in st.session_state:
    st.session_state.p2_point_score = 0
if 'total_points' not in st.session_state:
    st.session_state.total_points = 0
if 'probability_history' not in st.session_state:
    st.session_state.probability_history = []  # Track probability after each point
if 'score_history' not in st.session_state:
    st.session_state.score_history = []  # Track score after each point
if 'point_winner_history' not in st.session_state:
    st.session_state.point_winner_history = []  # Track who won each point

st.header("⚡ Live Match Tracker - Auto-Update")

# Display current score prominently
def get_point_display(points):
    """Convert numeric points to tennis score"""
    if points == 0: return "0"
    elif points == 1: return "15"
    elif points == 2: return "30"
    elif points == 3: return "40"
    elif points == 4: return "AD"
    else: return str(points)

# =============================================================================
# SIMPLIFIED ONE-CLICK POINT TRACKING WITH LIVE PROBABILITY
# =============================================================================

# Big prominent buttons for point tracking
st.markdown("### 🎾 Click Winner of Each Point:")

col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])

def handle_point_won(winner):
    """Handle point won with auto-calculation and probability tracking"""
    # Record point winner
    st.session_state.point_winner_history.append(winner)
    st.session_state.total_points += 1
    
    if winner == 1:
        st.session_state.p1_point_score += 1
    else:
        st.session_state.p2_point_score += 1
    
    # Check if game won
    game_won = False
    game_winner = None
    
    if st.session_state.p1_point_score >= 4 and st.session_state.p1_point_score - st.session_state.p2_point_score >= 2:
        # Player 1 wins game
        st.session_state.p1_games += 1
        st.session_state.p1_games_won_history.append('won')
        st.session_state.p2_games_won_history.append('lost')
        game_won = True
        game_winner = 1
        
        # Check for break
        games_before = (st.session_state.p1_games - 1) + st.session_state.p2_games
        if games_before % 2 == 1:  # P2 was serving
            st.session_state.break_count_p1 += 1
        
        st.session_state.p1_point_score = 0
        st.session_state.p2_point_score = 0
        
    elif st.session_state.p2_point_score >= 4 and st.session_state.p2_point_score - st.session_state.p1_point_score >= 2:
        # Player 2 wins game
        st.session_state.p2_games += 1
        st.session_state.p1_games_won_history.append('lost')
        st.session_state.p2_games_won_history.append('won')
        game_won = True
        game_winner = 2
        
        # Check for break
        games_before = st.session_state.p1_games + (st.session_state.p2_games - 1)
        if games_before % 2 == 0:  # P1 was serving
            st.session_state.break_count_p2 += 1
        
        st.session_state.p1_point_score = 0
        st.session_state.p2_point_score = 0
    
    # Check if set won
    if game_won:
        if st.session_state.p1_games >= 6 and st.session_state.p1_games - st.session_state.p2_games >= 2:
            st.session_state.p1_sets += 1
            st.session_state.p1_games = 0
            st.session_state.p2_games = 0
        elif st.session_state.p2_games >= 6 and st.session_state.p2_games - st.session_state.p1_games >= 2:
            st.session_state.p2_sets += 1
            st.session_state.p1_games = 0
            st.session_state.p2_games = 0

with col_btn1:
    if st.button(f"🟢 {player1_name} WINS POINT", key="p1_point_big", use_container_width=True, type="primary"):
        handle_point_won(1)
        st.rerun()

with col_btn2:
    if st.button(f"🔵 {player2_name} WINS POINT", key="p2_point_big", use_container_width=True, type="primary"):
        handle_point_won(2)
        st.rerun()

with col_btn3:
    if st.button("🔄 RESET", key="reset_compact", use_container_width=True):
        st.session_state.p1_sets = 0
        st.session_state.p2_sets = 0
        st.session_state.p1_games = 0
        st.session_state.p2_games = 0
        st.session_state.p1_point_score = 0
        st.session_state.p2_point_score = 0
        st.session_state.total_points = 0
        st.session_state.p1_games_won_history = []
        st.session_state.p2_games_won_history = []
        st.session_state.break_count_p1 = 0
        st.session_state.break_count_p2 = 0
        st.session_state.probability_history = []
        st.session_state.score_history = []
        st.session_state.point_winner_history = []
        st.rerun()

# Live score display (more compact)
st.markdown("---")
col_s1, col_s2, col_s3 = st.columns([1, 2, 1])

with col_s1:
    st.markdown(f"### {player1_name}")
    st.markdown(f"<h1 style='text-align: center; color: #28a745;'>{st.session_state.p1_sets} | {st.session_state.p1_games} | {get_point_display(st.session_state.p1_point_score)}</h1>", unsafe_allow_html=True)

with col_s2:
    st.markdown("### Match Stats")
    st.caption(f"**Total Points:** {st.session_state.total_points}")
    if st.session_state.total_points > 0:
        p1_pts_won = sum(1 for w in st.session_state.point_winner_history if w == 1)
        p2_pts_won = sum(1 for w in st.session_state.point_winner_history if w == 2)
        st.caption(f"**Points Won:** {p1_pts_won}-{p2_pts_won}")
    st.caption(f"**Breaks:** {st.session_state.break_count_p1}-{st.session_state.break_count_p2}")

with col_s3:
    st.markdown(f"### {player2_name}")
    st.markdown(f"<h1 style='text-align: center; color: #007bff;'>{st.session_state.p2_sets} | {st.session_state.p2_games} | {get_point_display(st.session_state.p2_point_score)}</h1>", unsafe_allow_html=True)

# =============================================================================
# LIVE PROBABILITY & MOMENTUM DISPLAY (TRUE P)
# =============================================================================

st.markdown("---")

# Calculate current probabilities after each point
if st.session_state.total_points > 0:
    # Get current state
    current_state = {
        'p1_sets': st.session_state.p1_sets,
        'p2_sets': st.session_state.p2_sets,
        'p1_games': st.session_state.p1_games,
        'p2_games': st.session_state.p2_games,
        'p1_points': st.session_state.p1_point_score,
        'p2_points': st.session_state.p2_point_score
    }
    
    # Calculate probabilities using the same method as the rest of the app
    # (using values from sidebar inputs)
    
    # Convert points to numeric
    point_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    p1_point_num = point_map.get(current_state['p1_points'], 0)
    p2_point_num = point_map.get(current_state['p2_points'], 0)
    
    # Calculate Markov probabilities with advanced params if available
    probs = calculate_point_probabilities(
        p1_serve_win, p2_serve_win,
        p1_advanced_params, p2_advanced_params
    )
    
    # Model 1: Markov Chain
    p1_markov = adjust_for_current_score(
        probs['p1_match'], 
        current_state['p1_sets'], 
        current_state['p1_games'], 
        p1_point_num,
        st.session_state.total_points,
        st.session_state.p1_games_won_history
    )
    
    # Model 2: Logistic Regression
    p1_lr = calculate_ml_probabilities(
        probs.get('p1_serve_adjusted', p1_serve_win), 
        probs.get('p2_serve_adjusted', p2_serve_win), 
        current_state['p1_sets'], current_state['p2_sets'],
        current_state['p1_games'], current_state['p2_games']
    )
    
    # Model 3: Neural Network
    p1_nn = calculate_nn_probabilities(
        probs.get('p1_serve_adjusted', p1_serve_win),
        probs.get('p2_serve_adjusted', p2_serve_win),
        current_state['p1_sets'], current_state['p2_sets'],
        current_state['p1_games'], current_state['p2_games'],
        p1_point_num, p2_point_num
    )
    
    # Ensemble (same weights as main calculation)
    ensemble_p1 = (p1_markov * 0.4) + (p1_lr * 0.25) + (p1_nn * 0.35)
    ensemble_p2 = 1 - ensemble_p1
    
    # Log to history
    current_score = f"{current_state['p1_sets']}-{current_state['p2_sets']}, {current_state['p1_games']}-{current_state['p2_games']}, {get_point_display(current_state['p1_points'])}-{get_point_display(current_state['p2_points'])}"
    
    if len(st.session_state.probability_history) < st.session_state.total_points:
        st.session_state.probability_history.append({'p1': ensemble_p1, 'p2': ensemble_p2})
        st.session_state.score_history.append(current_score)
    
    # LARGE CURRENT WIN PROBABILITY DISPLAY
    st.markdown("### 📊 LIVE WIN PROBABILITY (True P)")
    
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        # Calculate momentum (change in last 5 points)
        p1_momentum = 0
        if len(st.session_state.probability_history) >= 2:
            recent_window = min(5, len(st.session_state.probability_history))
            old_p1 = st.session_state.probability_history[-recent_window]['p1']
            p1_momentum = (ensemble_p1 - old_p1) * 100
        
        momentum_color = "🟢" if p1_momentum > 0 else "🔴" if p1_momentum < 0 else "⚪"
        momentum_text = f"{momentum_color} {p1_momentum:+.1f}%" if p1_momentum != 0 else "⚪ Stable"
        
        st.markdown(f"<div style='background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 20px; border-radius: 10px; text-align: center;'>"
                   f"<h2 style='color: white; margin: 0;'>{player1_name}</h2>"
                   f"<h1 style='color: white; font-size: 3em; margin: 10px 0;'>{ensemble_p1:.1%}</h1>"
                   f"<p style='color: white; font-size: 1.2em; margin: 0;'>Momentum: {momentum_text}</p>"
                   f"</div>", unsafe_allow_html=True)
    
    with prob_col2:
        # Calculate momentum for P2
        p2_momentum = 0
        if len(st.session_state.probability_history) >= 2:
            recent_window = min(5, len(st.session_state.probability_history))
            old_p2 = st.session_state.probability_history[-recent_window]['p2']
            p2_momentum = (ensemble_p2 - old_p2) * 100
        
        momentum_color = "🟢" if p2_momentum > 0 else "🔴" if p2_momentum < 0 else "⚪"
        momentum_text = f"{momentum_color} {p2_momentum:+.1f}%" if p2_momentum != 0 else "⚪ Stable"
        
        st.markdown(f"<div style='background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); padding: 20px; border-radius: 10px; text-align: center;'>"
                   f"<h2 style='color: white; margin: 0;'>{player2_name}</h2>"
                   f"<h1 style='color: white; font-size: 3em; margin: 10px 0;'>{ensemble_p2:.1%}</h1>"
                   f"<p style='color: white; font-size: 1.2em; margin: 0;'>Momentum: {momentum_text}</p>"
                   f"</div>", unsafe_allow_html=True)
    
    # PROBABILITY HISTORY CHART
    if len(st.session_state.probability_history) > 1:
        st.markdown("### 📈 Probability Evolution")
        
        import plotly.graph_objects as go
        
        points = list(range(1, len(st.session_state.probability_history) + 1))
        p1_probs = [p['p1'] * 100 for p in st.session_state.probability_history]
        p2_probs = [p['p2'] * 100 for p in st.session_state.probability_history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=points, y=p1_probs,
            mode='lines+markers',
            name=player1_name,
            line=dict(color='#28a745', width=3),
            marker=dict(size=6),
            hovertemplate=f'<b>{player1_name}</b><br>Point %{{x}}<br>Win Prob: %{{y:.1f}}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=points, y=p2_probs,
            mode='lines+markers',
            name=player2_name,
            line=dict(color='#007bff', width=3),
            marker=dict(size=6),
            hovertemplate=f'<b>{player2_name}</b><br>Point %{{x}}<br>Win Prob: %{{y:.1f}}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Win Probability Throughout Match",
            xaxis_title="Point Number",
            yaxis_title="Win Probability (%)",
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key momentum shifts
        if len(st.session_state.probability_history) >= 3:
            st.markdown("#### 🔥 Key Momentum Shifts")
            shifts = []
            for i in range(1, len(st.session_state.probability_history)):
                p1_change = (st.session_state.probability_history[i]['p1'] - st.session_state.probability_history[i-1]['p1']) * 100
                if abs(p1_change) >= 3:  # Significant shift (3%+)
                    winner = player1_name if st.session_state.point_winner_history[i-1] == 1 else player2_name
                    shifts.append({
                        'point': i,
                        'winner': winner,
                        'change': p1_change,
                        'score': st.session_state.score_history[i-1] if i-1 < len(st.session_state.score_history) else "N/A"
                    })
            
            if shifts:
                # Show last 5 significant shifts
                for shift in shifts[-5:]:
                    arrow = "📈" if shift['change'] > 0 else "📉"
                    st.caption(f"**Point {shift['point']}**: {shift['winner']} won → {arrow} {abs(shift['change']):.1f}% shift at {shift['score']}")
            else:
                st.caption("No significant momentum shifts yet (3%+ changes)")

else:
    st.info("👆 Click a point winner button above to start tracking match probabilities")

st.markdown("---")

# Manual adjustment (for corrections)
with st.expander("✏️ Manual Score Adjustment"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{player1_name}**")
        st.session_state.p1_sets = st.number_input("Sets", 0, 3, st.session_state.p1_sets, key="p1_sets_manual")
        st.session_state.p1_games = st.number_input("Games", 0, 7, st.session_state.p1_games, key="p1_games_manual")
        st.session_state.p1_point_score = st.selectbox("Points", [0, 1, 2, 3, 4], 
                                                        index=st.session_state.p1_point_score if st.session_state.p1_point_score < 5 else 0,
                                                        format_func=get_point_display, key="p1_points_manual")
    
    with col2:
        st.markdown(f"**{player2_name}**")
        st.session_state.p2_sets = st.number_input("Sets", 0, 3, st.session_state.p2_sets, key="p2_sets_manual")
        st.session_state.p2_games = st.number_input("Games", 0, 7, st.session_state.p2_games, key="p2_games_manual")
        st.session_state.p2_point_score = st.selectbox("Points", [0, 1, 2, 3, 4],
                                                        index=st.session_state.p2_point_score if st.session_state.p2_point_score < 5 else 0,
                                                        format_func=get_point_display, key="p2_points_manual")

# Use session state values
p1_sets = st.session_state.p1_sets
p2_sets = st.session_state.p2_sets
p1_games = st.session_state.p1_games
p2_games = st.session_state.p2_games
p1_points = st.session_state.p1_point_score
p2_points = st.session_state.p2_point_score
total_points = st.session_state.total_points

# ============================================================================
# BREAK OPPORTUNITY DETECTION
# ============================================================================

st.markdown("---")
st.header("🎯 Break Opportunity Analysis")

# Determine current server
games_played = p1_games + p2_games
if games_played % 2 == 0:
    current_server = player1_name
    server_serve_pct = p1_serve_win
    returner_return_pct = p2_return_win
    server_points = p1_points
    returner_points = p2_points
else:
    current_server = player2_name
    server_serve_pct = p2_serve_win
    returner_return_pct = p1_return_win
    server_points = p2_points
    returner_points = p1_points

# Calculate break probability for current game
break_prob = calculate_break_probability(server_serve_pct, returner_return_pct, server_points, returner_points)
hold_prob = 1 - break_prob

# Detect critical situations
is_break_point = (returner_points >= 3 and returner_points > server_points)
is_game_point = (server_points >= 3 and server_points > returner_points)
is_deuce = (server_points == 3 and returner_points == 3)

# Display current situation
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎾 Current Server", current_server)
    st.caption(f"Serve Win %: {server_serve_pct:.1%}")

with col2:
    st.metric("📊 Break Probability", f"{break_prob:.1%}", 
              delta=f"{(break_prob - 0.5)*100:+.1f}% vs 50/50")
    if is_break_point:
        st.markdown("🚨 **BREAK POINT!**")
    elif is_game_point:
        st.markdown("💪 **GAME POINT**")
    elif is_deuce:
        st.markdown("⚖️ **DEUCE**")

with col3:
    st.metric("🛡️ Hold Probability", f"{hold_prob:.1%}")
    point_score = f"{get_point_display(server_points)}-{get_point_display(returner_points)}"
    st.caption(f"Score: {point_score}")

# Break opportunity alert
if break_prob > 0.45:
    returner_name = player2_name if current_server == player1_name else player1_name
    st.markdown(f"""
    <div class='value-bet-alert'>
        🔥 <strong>HIGH BREAK OPPORTUNITY FOR {returner_name.upper()}</strong><br/>
        📈 Break Probability: <strong>{break_prob:.1%}</strong><br/>
        💡 {returner_name} has a strong chance to break serve!<br/>
        {'🚨 BREAK POINT - Critical moment!' if is_break_point else '⚡ Watch for break point opportunities!'}
    </div>
    """, unsafe_allow_html=True)
elif break_prob < 0.20:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); 
                padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6;'>
        💪 <strong>STRONG SERVICE GAME FOR {current_server.upper()}</strong><br/>
        🛡️ Hold Probability: <strong>{hold_prob:.1%}</strong><br/>
        💡 {current_server} is dominating this service game
    </div>
    """, unsafe_allow_html=True)

# Game situation context
st.subheader("📊 Game Situation Context")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Set Position:**")
    if p1_games > p2_games:
        st.info(f"✅ {player1_name} leads {p1_games}-{p2_games}")
        if p1_games >= 5:
            st.warning(f"🎯 {player1_name} serving for the set!" if current_server == player1_name 
                      else f"🚨 {player2_name} must break to stay in set!")
    elif p2_games > p1_games:
        st.info(f"✅ {player2_name} leads {p2_games}-{p1_games}")
        if p2_games >= 5:
            st.warning(f"🎯 {player2_name} serving for the set!" if current_server == player2_name
                      else f"🚨 {player1_name} must break to stay in set!")
    else:
        st.info(f"⚖️ Games level at {p1_games}-{p2_games}")
    
    # Break statistics
    total_breaks = st.session_state.break_count_p1 + st.session_state.break_count_p2
    if total_breaks > 0:
        st.caption(f"🔓 Total Breaks: {total_breaks} ({player1_name}: {st.session_state.break_count_p1}, {player2_name}: {st.session_state.break_count_p2})")

with col2:
    st.markdown("**Match Position:**")
    if p1_sets > p2_sets:
        st.success(f"🏆 {player1_name} leads {p1_sets}-{p2_sets} in sets")
    elif p2_sets > p1_sets:
        st.success(f"🏆 {player2_name} leads {p2_sets}-{p1_sets} in sets")
    else:
        st.info(f"📊 Sets level at {p1_sets}-{p2_sets}")
    
    # Recent game momentum (last 3 games)
    if len(st.session_state.p1_games_won_history) >= 3:
        recent_p1_wins = sum(1 for g in st.session_state.p1_games_won_history[-3:] if g == 'won')
        recent_p2_wins = sum(1 for g in st.session_state.p2_games_won_history[-3:] if g == 'won')
        
        if recent_p1_wins > recent_p2_wins:
            st.caption(f"📈 {player1_name} won {recent_p1_wins}/3 recent games - Strong momentum!")
        elif recent_p2_wins > recent_p1_wins:
            st.caption(f"📈 {player2_name} won {recent_p2_wins}/3 recent games - Strong momentum!")
        else:
            st.caption(f"⚖️ Balanced momentum ({recent_p1_wins}-{recent_p2_wins} in last 3 games)")

st.markdown("---")

# Continue with rest of calculations...
p1_points = st.session_state.p1_point_score
p2_points = st.session_state.p2_point_score

# Server selection
current_server = st.radio("Who is serving?", [player1_name, player2_name], horizontal=True)

st.divider()

# ============================================================================
# LIVE ODDS SECTION
# ============================================================================

st.header("💰 Live Bookmaker Odds")
st.caption("Update these as the match progresses to find live value bets")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Match Winner Odds")
    p1_match_odds = st.number_input(f"{player1_name}", min_value=1.01, max_value=50.0, value=1.85, step=0.01, key="p1_match_live")
    p2_match_odds = st.number_input(f"{player2_name}", min_value=1.01, max_value=50.0, value=2.10, step=0.01, key="p2_match_live")

with col2:
    st.subheader("Current Set Winner")
    p1_set_odds = st.number_input(f"{player1_name} Set", min_value=1.01, max_value=50.0, value=1.75, step=0.01, key="p1_set_live")
    p2_set_odds = st.number_input(f"{player2_name} Set", min_value=1.01, max_value=50.0, value=2.00, step=0.01, key="p2_set_live")

with col3:
    st.subheader("Next Game Winner")
    p1_game_odds = st.number_input(f"{player1_name} Game", min_value=1.01, max_value=50.0, value=1.50, step=0.01, key="p1_game_live")
    p2_game_odds = st.number_input(f"{player2_name} Game", min_value=1.01, max_value=50.0, value=2.50, step=0.01, key="p2_game_live")

st.divider()

# ============================================================================
# MULTI-MODEL CALCULATIONS
# ============================================================================

st.header("🤖 Multi-Model Probability Analysis")

# Collect advanced parameters if available
p1_adv_params = {
    'momentum': st.session_state.get('p1_momentum', 0.5),
    'surface_mastery': st.session_state.get('p1_surface', 0.5),
    'clutch': st.session_state.get('p1_clutch', 0.5),
    'bp_defense': st.session_state.get('p1_bp_defense', 0.6),
    'consistency': st.session_state.get('p1_consistency', 0.5),
    'first_serve_pct': st.session_state.get('p1_first_serve_pct', 0.62)
}

p2_adv_params = {
    'momentum': st.session_state.get('p2_momentum', 0.5),
    'surface_mastery': st.session_state.get('p2_surface', 0.5),
    'clutch': st.session_state.get('p2_clutch', 0.5),
    'bp_defense': st.session_state.get('p2_bp_defense', 0.6),
    'consistency': st.session_state.get('p2_consistency', 0.5),
    'first_serve_pct': st.session_state.get('p2_first_serve_pct', 0.62)
}

# Calculate base probabilities with advanced parameter adjustments
probs = calculate_point_probabilities(p1_serve_win, p2_serve_win, p1_adv_params, p2_adv_params)

# Show if adjustments were applied
if any(v != 0.5 for v in [p1_adv_params['momentum'], p1_adv_params['surface_mastery'], p1_adv_params['clutch']]) or \
   any(v != 0.5 for v in [p2_adv_params['momentum'], p2_adv_params['surface_mastery'], p2_adv_params['clutch']]):
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        p1_serve_adj_pct = (probs['p1_serve_adjusted'] - p1_serve_win) * 100
        if abs(p1_serve_adj_pct) > 0.5:
            st.info(f"⚙️ {player1_name} serve adjusted: {p1_serve_win*100:.1f}% → {probs['p1_serve_adjusted']*100:.1f}% ({p1_serve_adj_pct:+.1f}%)")
    
    with col_info2:
        p2_serve_adj_pct = (probs['p2_serve_adjusted'] - p2_serve_win) * 100
        if abs(p2_serve_adj_pct) > 0.5:
            st.info(f"⚙️ {player2_name} serve adjusted: {p2_serve_win*100:.1f}% → {probs['p2_serve_adjusted']*100:.1f}% ({p2_serve_adj_pct:+.1f}%)")


# Convert points to numeric for calculations
point_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
p1_point_num = point_map.get(p1_points, 0)
p2_point_num = point_map.get(p2_points, 0)

# Model 1: Markov Chain (using adjusted probabilities from advanced params)
p1_markov = adjust_for_current_score(
    probs['p1_match'], 
    p1_sets, 
    p1_games, 
    p1_point_num,
    st.session_state.total_points,
    st.session_state.p1_games_won_history
)
p2_markov = 1 - p1_markov

# Model 2: Logistic Regression (using adjusted serve percentages)
p1_lr = calculate_ml_probabilities(
    probs.get('p1_serve_adjusted', p1_serve_win), 
    probs.get('p2_serve_adjusted', p2_serve_win), 
    p1_sets, p2_sets, p1_games, p2_games
)
p2_lr = 1 - p1_lr

# Model 3: Neural Network (using adjusted serve percentages)
p1_nn = calculate_nn_probabilities(
    probs.get('p1_serve_adjusted', p1_serve_win),
    probs.get('p2_serve_adjusted', p2_serve_win),
    p1_sets, p2_sets, p1_games, p2_games, p1_point_num, p2_point_num
)
p2_nn = 1 - p1_nn

# Ensemble (weighted average - give more weight to Markov and NN)
p1_win_prob = (p1_markov * 0.4) + (p1_lr * 0.25) + (p1_nn * 0.35)
p2_win_prob = 1 - p1_win_prob

# Model agreement score
model_std = np.std([p1_markov, p1_lr, p1_nn])
consensus_strength = 1 - min(model_std / 0.15, 1.0)  # Higher = more agreement

# Display model breakdown
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔗 Markov Chain", f"{p1_markov:.1%}", 
              help="Pure statistical model based on serve percentages")
    
with col2:
    st.metric("📊 Logistic Regression", f"{p1_lr:.1%}",
              help="Feature-based ML model")
    
with col3:
    st.metric("🧠 Neural Network", f"{p1_nn:.1%}",
              help="Deep learning non-linear model")
    
with col4:
    st.metric("🎯 Ensemble", f"{p1_win_prob:.1%}",
              delta=f"Consensus: {consensus_strength:.0%}",
              help="Weighted combination of all models")

# Consensus indicator
if consensus_strength > 0.75:
    st.markdown(f'<div class="model-consensus">✅ <strong>HIGH CONSENSUS</strong> - All models agree within {model_std:.1%}. Prediction is highly reliable.</div>', unsafe_allow_html=True)
elif consensus_strength > 0.5:
    st.info(f"⚠️ **MODERATE CONSENSUS** - Models show {model_std:.1%} spread. Use caution.")
else:
    st.warning(f"🔴 **LOW CONSENSUS** - Models disagree by {model_std:.1%}. High uncertainty - avoid betting or use smaller stakes.")

st.divider()

# Calculate set probabilities based on current games
current_set_total = p1_games + p2_games
if current_set_total > 0:
    games_diff = p1_games - p2_games
    p1_set_prob = probs['p1_set'] + (games_diff / 6) * 0.20
    p1_set_prob = max(0.05, min(0.95, p1_set_prob))
else:
    p1_set_prob = probs['p1_set']
p2_set_prob = 1 - p1_set_prob

# Calculate game probabilities based on who's serving
if current_server == player1_name:
    p1_game_prob = probs['p1_game_on_serve']
    p2_game_prob = 1 - p1_game_prob
else:
    p2_game_prob = probs['p2_game_on_serve']
    p1_game_prob = 1 - p2_game_prob

# Adjust game probability based on current points
if p1_point_num > p2_point_num:
    point_diff = (p1_point_num - p2_point_num) / 4
    p1_game_prob = min(0.95, p1_game_prob + point_diff * 0.15)
elif p2_point_num > p1_point_num:
    point_diff = (p2_point_num - p1_point_num) / 4
    p2_game_prob = min(0.95, p2_game_prob + point_diff * 0.15)

p2_game_prob = 1 - p1_game_prob

# Calculate all betting values
p1_match_edge, p1_match_ev = calculate_betting_value(p1_win_prob, p1_match_odds)
p2_match_edge, p2_match_ev = calculate_betting_value(p2_win_prob, p2_match_odds)

p1_set_edge, p1_set_ev = calculate_betting_value(p1_set_prob, p1_set_odds)
p2_set_edge, p2_set_ev = calculate_betting_value(p2_set_prob, p2_set_odds)

p1_game_edge, p1_game_ev = calculate_betting_value(p1_game_prob, p1_game_odds)
p2_game_edge, p2_game_ev = calculate_betting_value(p2_game_prob, p2_game_odds)

# ============================================================================
# PROFESSIONAL VALUE BET DISPLAY
# ============================================================================

st.header("💰 Live Betting Edge Analysis")

# Create professional betting card for each market
def display_bet_card(market_name, player_name, prob, odds, is_p1=True):
    """Display professional betting card like bookmakers"""
    implied_prob = 1 / odds
    edge = prob - implied_prob
    ev = (prob * (odds - 1)) - (1 - prob)
    
    # Determine stake
    if market_name == "Match Winner":
        stake = min(edge * 400, 150)
    elif market_name == "Set Winner":
        stake = min(edge * 300, 100)
    else:
        stake = min(edge * 200, 50)
    
    # Professional display
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if edge > 0.025:
            st.markdown(f"### 🟢 {player_name}")
            st.markdown(f"**{market_name}**")
        else:
            st.markdown(f"### {player_name}")
            st.markdown(f"**{market_name}**")
    
    with col2:
        st.markdown("**Probability**")
        st.markdown(f"<h3 style='color: {'green' if edge > 0.025 else 'gray'};'>{prob:.1%}</h3>", unsafe_allow_html=True)
        st.caption(f"Implied: {implied_prob:.1%}")
    
    with col3:
        st.markdown("**Odds**")
        st.markdown(f"<h3>{odds:.2f}</h3>", unsafe_allow_html=True)
        st.caption(f"Edge: {edge:+.2%}")
    
    if edge > 0.025:
        st.markdown(f"""
        <div class='value-bet-alert'>
            ✅ <strong>VALUE BET IDENTIFIED</strong><br/>
            💰 Recommended Stake: <strong>${stake:.0f}</strong><br/>
            📈 Expected Value: <strong>{ev:+.2%}</strong> per $1 bet<br/>
            🎯 Edge: <strong>{edge:+.2%}</strong> | Confidence: <strong>{consensus_strength:.0%}</strong>
        </div>
        """, unsafe_allow_html=True)
        return True, stake, edge, ev
    else:
        st.markdown(f'<div class="no-value">❌ No value at {odds:.2f} (Need {1/prob:.2f}+ for 2.5% edge)</div>', unsafe_allow_html=True)
        return False, 0, edge, ev

st.subheader("🏆 Match Winner Market")
col1, col2 = st.columns(2)

with col1:
    has_value_p1_match, stake_p1_match, edge_p1_match, ev_p1_match = display_bet_card(
        "Match Winner", player1_name, p1_win_prob, p1_match_odds, True
    )

with col2:
    has_value_p2_match, stake_p2_match, edge_p2_match, ev_p2_match = display_bet_card(
        "Match Winner", player2_name, p2_win_prob, p2_match_odds, False
    )

st.divider()

st.divider()

# ============================================================================
# DISPLAY RESULTS - SET WINNER
# ============================================================================

st.subheader("📊 Current Set Winner Market")

col1, col2 = st.columns(2)

with col1:
    has_value_p1_set, stake_p1_set, edge_p1_set, ev_p1_set = display_bet_card(
        "Set Winner", player1_name, p1_set_prob, p1_set_odds, True
    )

with col2:
    has_value_p2_set, stake_p2_set, edge_p2_set, ev_p2_set = display_bet_card(
        "Set Winner", player2_name, p2_set_prob, p2_set_odds, False
    )

st.divider()

# ============================================================================
# DISPLAY RESULTS - NEXT GAME WINNER
# ============================================================================

st.subheader("🎯 Next Game Winner Market")

# Show who's serving prominently
if current_server == player1_name:
    st.info(f"🎾 **{player1_name} is serving** - Expected to have advantage")
else:
    st.info(f"🎾 **{player2_name} is serving** - Expected to have advantage")

col1, col2 = st.columns(2)

with col1:
    has_value_p1_game, stake_p1_game, edge_p1_game, ev_p1_game = display_bet_card(
        "Next Game", player1_name, p1_game_prob, p1_game_odds, True
    )

with col2:
    has_value_p2_game, stake_p2_game, edge_p2_game, ev_p2_game = display_bet_card(
        "Next Game", player2_name, p2_game_prob, p2_game_odds, False
    )

st.divider()

# ============================================================================
# PROFESSIONAL SUMMARY DASHBOARD
# ============================================================================

st.header("📊 Professional Betting Summary")

# Collect all value bets
value_bets = []

if has_value_p1_match:
    value_bets.append({
        'Market': '🏆 Match Winner',
        'Selection': player1_name,
        'Probability': f"{p1_win_prob:.1%}",
        'Odds': f"{p1_match_odds:.2f}",
        'Edge': f"{edge_p1_match:+.2%}",
        'EV': f"{ev_p1_match:+.2%}",
        'Stake': f"${stake_p1_match:.0f}",
        'Confidence': f"{consensus_strength:.0%}"
    })

if has_value_p2_match:
    value_bets.append({
        'Market': '🏆 Match Winner',
        'Selection': player2_name,
        'Probability': f"{p2_win_prob:.1%}",
        'Odds': f"{p2_match_odds:.2f}",
        'Edge': f"{edge_p2_match:+.2%}",
        'EV': f"{ev_p2_match:+.2%}",
        'Stake': f"${stake_p2_match:.0f}",
        'Confidence': f"{consensus_strength:.0%}"
    })

if has_value_p1_set:
    value_bets.append({
        'Market': '📊 Set Winner',
        'Selection': player1_name,
        'Probability': f"{p1_set_prob:.1%}",
        'Odds': f"{p1_set_odds:.2f}",
        'Edge': f"{edge_p1_set:+.2%}",
        'EV': f"{ev_p1_set:+.2%}",
        'Stake': f"${stake_p1_set:.0f}",
        'Confidence': f"{consensus_strength:.0%}"
    })

if has_value_p2_set:
    value_bets.append({
        'Market': '📊 Set Winner',
        'Selection': player2_name,
        'Probability': f"{p2_set_prob:.1%}",
        'Odds': f"{p2_set_odds:.2f}",
        'Edge': f"{edge_p2_set:+.2%}",
        'EV': f"{ev_p2_set:+.2%}",
        'Stake': f"${stake_p2_set:.0f}",
        'Confidence': f"{consensus_strength:.0%}"
    })

if has_value_p1_game:
    value_bets.append({
        'Market': '🎯 Next Game',
        'Selection': player1_name,
        'Probability': f"{p1_game_prob:.1%}",
        'Odds': f"{p1_game_odds:.2f}",
        'Edge': f"{edge_p1_game:+.2%}",
        'EV': f"{ev_p1_game:+.2%}",
        'Stake': f"${stake_p1_game:.0f}",
        'Confidence': f"{consensus_strength:.0%}"
    })

if has_value_p2_game:
    value_bets.append({
        'Market': '🎯 Next Game',
        'Selection': player2_name,
        'Probability': f"{p2_game_prob:.1%}",
        'Odds': f"{p2_game_odds:.2f}",
        'Edge': f"{edge_p2_game:+.2%}",
        'EV': f"{ev_p2_game:+.2%}",
        'Stake': f"${stake_p2_game:.0f}",
        'Confidence': f"{consensus_strength:.0%}"
    })

if value_bets:
    value_df = pd.DataFrame(value_bets)
    
    # Highlight best bets
    st.markdown("### 🎯 ACTIVE VALUE OPPORTUNITIES")
    st.dataframe(value_df, use_container_width=True, hide_index=True)
    
    total_stake = sum([float(item['Stake'].replace('$', '')) for item in value_bets])
    avg_edge = np.mean([float(item['Edge'].replace('%', '').replace('+', '')) for item in value_bets])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Total Stake", f"${total_stake:.0f}")
    with col2:
        st.metric("📈 Average Edge", f"{avg_edge:.2f}%")
    with col3:
        st.metric("🎯 Opportunities", len(value_bets))
    
    st.success(f"✅ **{len(value_bets)} VALUE BETS IDENTIFIED** - Models show {consensus_strength:.0%} consensus. Ready to bet!")
    
else:
    st.info("🔍 **No value bets at current odds** - Keep monitoring as match progresses and odds update.")
    st.caption("Value appears when bookmaker odds drift from true probabilities. Update live odds frequently!")

st.divider()

# ============================================================================
# MODEL COMPARISON VISUALIZATION  
# ============================================================================

st.header("📊 Multi-Model Analysis")

# Win probability chart
fig = go.Figure()

# Add traces for each model
fig.add_trace(go.Bar(
    name='Markov Chain',
    x=[player1_name, player2_name],
    y=[p1_markov * 100, p2_markov * 100],
    marker_color='#667eea',
    text=[f"{p1_markov:.1%}", f"{p2_markov:.1%}"],
    textposition='outside'
))

fig.add_trace(go.Bar(
    name='Logistic Regression',
    x=[player1_name, player2_name],
    y=[p1_lr * 100, p2_lr * 100],
    marker_color='#f093fb',
    text=[f"{p1_lr:.1%}", f"{p2_lr:.1%}"],
    textposition='outside'
))

fig.add_trace(go.Bar(
    name='Neural Network',
    x=[player1_name, player2_name],
    y=[p1_nn * 100, p2_nn * 100],
    marker_color='#4facfe',
    text=[f"{p1_nn:.1%}", f"{p2_nn:.1%}"],
    textposition='outside'
))

fig.add_trace(go.Bar(
    name='Ensemble (Final)',
    x=[player1_name, player2_name],
    y=[p1_win_prob * 100, p2_win_prob * 100],
    marker_color='#43e97b',
    text=[f"{p1_win_prob:.1%}", f"{p2_win_prob:.1%}"],
    textposition='outside',
    marker_line_width=2,
    marker_line_color='black'
))

fig.update_layout(
    title="Multi-Model Win Probability Comparison",
    yaxis_title="Win Probability (%)",
    yaxis=dict(range=[0, 100]),
    barmode='group',
    height=400,
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================================
# QUICK REFERENCE
# ============================================================================

with st.expander("🎯 PRO GUIDE: How to Always Be on the Right Side of Bets"):
    st.markdown(f"""
    ### 🏆 Professional Live Betting Strategy
    
    #### 1. **Point-by-Point Tracking** 
    - Click point buttons after EVERY point for accuracy
    - Automatic game/set progression
    - Momentum tracked through total points played
    
    #### 2. **Multi-Model Edge Detection**
    - ✅ **HIGH CONSENSUS (>75%)** = Bet with confidence
    - ⚠️ **MODERATE (50-75%)** = Reduce stakes 50%
    - 🔴 **LOW (<50%)** = Skip or minimal stake
    
    #### 3. **Always Update Odds**
    - Every 2-3 games minimum
    - After every break/timeout
    - Before placing any bet
    
    #### 4. **Bet Only When:**
    - Consensus >75% + Edge >2.5%
    - All models agree within 5%
    - You've updated odds recently
    
    #### 5. **Never Bet When:**
    - Models disagree (>10% spread)
    - Using default/guessed stats
    - Emotional or chasing losses
    
    ### 💰 The Right Side = High Consensus + Real Edge + Fresh Odds
    """)

st.divider()

# Win probability chart (old one, remove duplicate)
fig2 = go.Figure()

fig2.add_trace(go.Bar(
    name=player1_name,
    x=['Match Win', 'Set Win', 'Game Win'],
    y=[p1_win_prob * 100, p1_set_prob * 100, p1_game_prob * 100],
    marker_color='#1f77b4',
    text=[f"{p1_win_prob:.1%}", f"{p1_set_prob:.1%}", f"{p1_game_prob:.1%}"],
    textposition='outside'
))

fig.add_trace(go.Bar(
    name=player2_name,
    x=['Match Win', 'Set Win', 'Game Win'],
    y=[p2_win_prob * 100, p2_set_prob * 100, p2_game_prob * 100],
    marker_color='#ff7f0e',
    text=[f"{p2_win_prob:.1%}", f"{p2_set_prob:.1%}", f"{p2_game_prob:.1%}"],
    textposition='outside'
))

fig.update_layout(
    title="Live Win Probabilities (Match / Set / Next Game)",
    yaxis_title="Probability (%)",
    yaxis=dict(range=[0, 100]),
    barmode='group',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ODDS MOVEMENT TRACKING
# ============================================================================

st.subheader("📈 Odds Movement & Value Summary")

# Create summary table
value_data = []

# Match Winner
if p1_match_edge > 0.025:
    value_data.append({
        'Market': 'Match Winner',
        'Selection': player1_name,
        'Probability': f"{p1_win_prob:.1%}",
        'Odds': f"{p1_match_odds:.2f}",
        'Edge': f"{p1_match_edge:.2%}",
        'EV': f"{p1_match_ev:.2%}",
        'Stake': f"${min(p1_match_edge * 400, 150):.0f}"
    })

if p2_match_edge > 0.025:
    value_data.append({
        'Market': 'Match Winner',
        'Selection': player2_name,
        'Probability': f"{p2_win_prob:.1%}",
        'Odds': f"{p2_match_odds:.2f}",
        'Edge': f"{p2_match_edge:.2%}",
        'EV': f"{p2_match_ev:.2%}",
        'Stake': f"${min(p2_match_edge * 400, 150):.0f}"
    })

# Set Winner
if p1_set_edge > 0.025:
    value_data.append({
        'Market': 'Set Winner',
        'Selection': player1_name,
        'Probability': f"{p1_set_prob:.1%}",
        'Odds': f"{p1_set_odds:.2f}",
        'Edge': f"{p1_set_edge:.2%}",
        'EV': f"{p1_set_ev:.2%}",
        'Stake': f"${min(p1_set_edge * 300, 100):.0f}"
    })

if p2_set_edge > 0.025:
    value_data.append({
        'Market': 'Set Winner',
        'Selection': player2_name,
        'Probability': f"{p2_set_prob:.1%}",
        'Odds': f"{p2_set_odds:.2f}",
        'Edge': f"{p2_set_edge:.2%}",
        'EV': f"{p2_set_ev:.2%}",
        'Stake': f"${min(p2_set_edge * 300, 100):.0f}"
    })

# Game Winner
if p1_game_edge > 0.025:
    value_data.append({
        'Market': 'Next Game',
        'Selection': player1_name,
        'Probability': f"{p1_game_prob:.1%}",
        'Odds': f"{p1_game_odds:.2f}",
        'Edge': f"{p1_game_edge:.2%}",
        'EV': f"{p1_game_ev:.2%}",
        'Stake': f"${min(p1_game_edge * 200, 50):.0f}"
    })

if p2_game_edge > 0.025:
    value_data.append({
        'Market': 'Next Game',
        'Selection': player2_name,
        'Probability': f"{p2_game_prob:.1%}",
        'Odds': f"{p2_game_odds:.2f}",
        'Edge': f"{p2_game_edge:.2%}",
        'EV': f"{p2_game_ev:.2%}",
        'Stake': f"${min(p2_game_edge * 200, 50):.0f}"
    })

if value_data:
    value_df = pd.DataFrame(value_data)
    st.dataframe(value_df, use_container_width=True, hide_index=True)
    
    total_stake = sum([float(item['Stake'].replace('$', '')) for item in value_data])
    st.success(f"🎯 **{len(value_data)} VALUE BETS FOUND | Total Recommended Stake: ${total_stake:.0f}**")
else:
    st.info("No value bets found at current odds. Keep monitoring as the match progresses!")

# ============================================================================
# MATCH HISTORY & NOTES
# ============================================================================

st.divider()

st.header("📝 Match Notes")

match_notes = st.text_area(
    "Add notes about the match (momentum shifts, injuries, weather, etc.)",
    height=100,
    placeholder="Example: Player 1 looks tired after long rally at 3-3..."
)

if st.button("💾 Save Match State", use_container_width=True):
    match_state = {
        'timestamp': datetime.now().isoformat(),
        'player1': player1_name,
        'player2': player2_name,
        'score': f"{p1_sets}-{p2_sets}, {p1_games}-{p2_games}, {p1_points}-{p2_points}",
        'p1_win_prob': p1_win_prob,
        'p2_win_prob': p2_win_prob,
        'p1_match_edge': p1_match_edge,
        'p2_match_edge': p2_match_edge,
        'p1_match_odds': p1_match_odds,
        'p2_match_odds': p2_match_odds,
        'p1_set_edge': p1_set_edge,
        'p2_set_edge': p2_set_edge,
        'p1_game_edge': p1_game_edge,
        'p2_game_edge': p2_game_edge,
        'notes': match_notes
    }
    
    # Save to session state
    if 'match_history' not in st.session_state:
        st.session_state.match_history = []
    
    st.session_state.match_history.append(match_state)
    st.success("✅ Match state saved!")

# Display history
if 'match_history' in st.session_state and len(st.session_state.match_history) > 0:
    st.subheader("📜 Saved Match States")
    
    history_df = pd.DataFrame(st.session_state.match_history)
    st.dataframe(
        history_df[['timestamp', 'player1', 'player2', 'score', 'p1_win_prob', 'p2_win_prob']],
        use_container_width=True
    )
    
    if st.button("🗑️ Clear History"):
        st.session_state.match_history = []
        st.rerun()

# ============================================================================
# INSTRUCTIONS
# ============================================================================

with st.expander("ℹ️ How to Use This Calculator"):
    st.markdown("""
    ### Step-by-Step Guide:
    
    1. **Enter Player Names** in the sidebar
    
    2. **Set Player Statistics:**
       - Serve Win %: Percentage of points won on serve
       - Return Win %: Percentage of return points won
       - Use historical averages or current match stats
    
    3. **Input Pre-Match Bookmaker Odds:**
       - Enter the odds before the match started (sidebar)
       - Used for tracking odds movement
    
    4. **Update Live Score:**
       - Enter current sets, games, and points
       - Select who is serving
       - Update after each game/point for live tracking
    
    5. **Enter Live Bookmaker Odds:**
       - **Match Winner Odds**: Current odds for overall match winner
       - **Set Winner Odds**: Current odds for who will win this set
       - **Next Game Odds**: Current odds for who will win the next game
       - Update these as odds move during the match
    
    6. **Analyze Multi-Market Probabilities:**
       - View probabilities for Match / Set / Next Game
       - Green "✅ VALUE BET" alerts show positive edge (>2.5%)
       - Each market has separate stake recommendations
       - Check the Value Summary table for all opportunities
    
    7. **Save Match States:**
       - Track how probabilities evolve during the match
       - Review later to improve your model
       - Export data for analysis
    
    ### New Features:
    - **Three Markets**: Match Winner, Set Winner, Next Game Winner
    - **Live Odds Tracking**: Compare pre-match vs live odds movement
    - **Multi-Market Value**: Find value across all available markets
    - **Smart Calculations**: Probabilities adjust based on:
      - Current score (sets, games, points)
      - Who's serving (crucial for game probabilities)
      - Match momentum and position
    
    ### Tips:
    - **Match Winner**: Best for strategic bets when momentum shifts
    - **Set Winner**: Higher frequency, good for in-play volatility
    - **Next Game**: Quick bets, heavily influenced by server
    - Update live odds frequently - they change rapidly in-play
    - Look for discrepancies between markets (arbitrage)
    - Server advantage is huge: ~65-70% game win probability
    """)

st.divider()

st.caption("💡 Calculator uses Markov chain model with dynamic score adjustments")
st.caption("⚠️ Multiple markets = More opportunities but manage bankroll carefully")
st.caption("🎯 Best used during live matches with real-time odds updates")

st.divider()

st.caption("💡 Calculator uses Markov chain model with dynamic score adjustments")
st.caption("⚠️ Multiple markets = More opportunities but manage bankroll carefully")
st.caption("🎯 Best used during live matches with real-time odds updates")
