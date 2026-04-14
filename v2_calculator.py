"""
🎾 V2 TENNIS CALCULATOR - Complete Parameter Input & Analysis
===========================================================
Full odds analysis with true probability calculations

Run: streamlit run v2_calculator.py --server.port 8503
"""

import streamlit as st
import numpy as np
from scipy.special import comb
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🎾 V2 Tennis Calculator",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0a0e27; color: #e0e0e0; }
    
    /* Input section */
    .input-section {
        background: #151932;
        border: 2px solid #2d3561;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    /* Results panel */
    .results-panel {
        background: #1a1f3a;
        border: 2px solid #00ff88;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    /* True probability display */
    .true-prob {
        font-size: 1.3rem;
        font-weight: bold;
        color: #00ff88;
        font-family: monospace;
    }
    
    /* Implied probability */
    .implied-prob {
        font-size: 1.1rem;
        color: #ffc107;
        font-family: monospace;
    }
    
    /* Edge display */
    .edge-positive {
        color: #00ff88;
        font-weight: bold;
    }
    
    .edge-negative {
        color: #ff6b6b;
        font-weight: bold;
    }
    
    /* Probability bar */
    .prob-bar-container {
        background: #2d3561;
        height: 30px;
        border-radius: 5px;
        position: relative;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .prob-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ff88 0%, #00ff88 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #000;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* Table styling */
    .stat-table {
        width: 100%;
        border-collapse: collapse;
        font-family: monospace;
    }
    
    .stat-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #2d3561;
    }
    
    .stat-table tr:hover {
        background: rgba(0, 255, 136, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'params' not in st.session_state:
    st.session_state.params = {
        'p1_name': 'Player 1',
        'p2_name': 'Player 2',
        'p1_rank': 50,
        'p2_rank': 100,
        'surface': 'Hard',
        'best_of': 3,
        'p1_serve_pct': 65.0,
        'p2_serve_pct': 62.0,
        'p1_return_pct': 35.0,
        'p2_return_pct': 38.0,
        'p1_1st_serve': 60.0,
        'p2_1st_serve': 62.0,
        'p1_bp_save': 65.0,
        'p2_bp_save': 68.0,
        'p1_bp_convert': 40.0,
        'p2_bp_convert': 42.0,
        'p1_odds': 2.0,
        'p2_odds': 1.90,
        'momentum_p1': 0,
        'momentum_p2': 0,
        'surface_mastery_p1': 0,
        'surface_mastery_p2': 0,
        'clutch_p1': 0,
        'clutch_p2': 0,
    }

# ==================== PROBABILITY FUNCTIONS ====================

def p_game_from_points(server_pts: int, returner_pts: int, p_point: float) -> float:
    """Calculate probability that server wins current game from given points."""
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
        result = p_point * prob_from(min(s+1, 4), r) + (1-p_point) * prob_from(s, min(r+1, 4))
        cache[(s, r)] = result
        return result
    return prob_from(server_pts, returner_pts)

def p_win_game(p_point: float) -> float:
    """Win probability for a single game."""
    p, q = p_point, 1 - p_point
    p_before_deuce = p**4 * (1 + 4*q + 10*q**2)
    p_deuce = comb(6, 3) * p**3 * q**3 * (p**2 / (p**2 + q**2))
    return p_before_deuce + p_deuce

def p_set_from_games(g1: int, g2: int, server: int, p1_serve: float, p2_serve: float) -> float:
    """Calculate probability that player 1 wins set from given games."""
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
    """Calculate probability that player 1 wins match from given sets."""
    sets_to_win = (best_of + 1) // 2
    cache = {}
    def prob_from(s1, s2):
        if (s1, s2) in cache: 
            return cache[(s1, s2)]
        if s1 >= sets_to_win: 
            return 1.0
        if s2 >= sets_to_win: 
            return 0.0
        result = p_set * prob_from(s1+1, s2) + (1-p_set) * prob_from(s1, s2+1)
        cache[(s1, s2)] = result
        return result
    return prob_from(s1, s2)

def calculate_ranking_probability(p1_rank, p2_rank):
    """Estimate win probability from ranking difference."""
    rank_ratio = p2_rank / p1_rank
    p1_prob = 1 / (1 + rank_ratio ** 0.8)
    return p1_prob

def adjust_for_modifiers(prob, momentum_p1, momentum_p2, surface_mastery_p1, surface_mastery_p2, clutch_p1, clutch_p2):
    """Adjust probability based on player modifiers."""
    momentum_adjustment = (momentum_p1 - momentum_p2) * 0.02  # ±2% per point
    mastery_adjustment = (surface_mastery_p1 - surface_mastery_p2) * 0.01  # ±1% per point
    clutch_adjustment = (clutch_p1 - clutch_p2) * 0.015  # ±1.5% per point
    
    total_adjustment = momentum_adjustment + mastery_adjustment + clutch_adjustment
    adjusted_prob = np.clip(prob + total_adjustment, 0.05, 0.95)
    return adjusted_prob

# ==================== MAIN APP ====================

st.title("🎾 V2 Tennis Calculator")
st.markdown("**Complete parameter input with true probability & odds analysis**")
st.markdown("---")

# ==================== SECTION 1: PLAYER & MATCH INFO ====================
with st.expander("⚙️ MATCH SETUP", expanded=True):
    st.subheader("Match Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.params['p1_name'] = st.text_input(
            "Player 1 Name",
            st.session_state.params['p1_name']
        )
        st.session_state.params['p1_rank'] = st.slider(
            "Player 1 Ranking",
            1, 500, st.session_state.params['p1_rank'],
            help="ATP/WTA ranking (lower is better)"
        )
    
    with col2:
        st.session_state.params['p2_name'] = st.text_input(
            "Player 2 Name",
            st.session_state.params['p2_name']
        )
        st.session_state.params['p2_rank'] = st.slider(
            "Player 2 Ranking",
            1, 500, st.session_state.params['p2_rank'],
            help="ATP/WTA ranking (lower is better)"
        )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.params['surface'] = st.selectbox(
            "Surface",
            ["Hard", "Clay", "Grass"],
            index=["Hard", "Clay", "Grass"].index(st.session_state.params['surface'])
        )
    
    with col2:
        st.session_state.params['best_of'] = st.selectbox(
            "Best Of",
            [3, 5],
            index=[3, 5].index(st.session_state.params['best_of'])
        )
    
    with col3:
        pass  # Spacer

# ==================== SECTION 2: SERVE & RETURN STATS ====================
with st.expander("📊 SERVE & RETURN STATISTICS", expanded=True):
    st.subheader("Player 1 - Serve Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state.params['p1_serve_pct'] = st.number_input(
            "P1 Serve Win %",
            0.0, 100.0, st.session_state.params['p1_serve_pct'], 1.0,
            help="% of points won on own serve"
        )
    with col2:
        st.session_state.params['p1_return_pct'] = st.number_input(
            "P1 Return Win %",
            0.0, 100.0, st.session_state.params['p1_return_pct'], 1.0,
            help="% of points won on opponent's serve"
        )
    with col3:
        st.session_state.params['p1_1st_serve'] = st.number_input(
            "P1 1st Serve %",
            0.0, 100.0, st.session_state.params['p1_1st_serve'], 1.0,
            help="% of serves that are first serves"
        )
    with col4:
        pass
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state.params['p1_bp_save'] = st.number_input(
            "P1 BP Save %",
            0.0, 100.0, st.session_state.params['p1_bp_save'], 1.0,
            help="% of break points saved"
        )
    with col2:
        st.session_state.params['p1_bp_convert'] = st.number_input(
            "P1 BP Convert %",
            0.0, 100.0, st.session_state.params['p1_bp_convert'], 1.0,
            help="% of break points converted"
        )
    with col3, col4:
        pass
    
    st.markdown("---")
    st.subheader("Player 2 - Serve Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state.params['p2_serve_pct'] = st.number_input(
            "P2 Serve Win %",
            0.0, 100.0, st.session_state.params['p2_serve_pct'], 1.0
        )
    with col2:
        st.session_state.params['p2_return_pct'] = st.number_input(
            "P2 Return Win %",
            0.0, 100.0, st.session_state.params['p2_return_pct'], 1.0
        )
    with col3:
        st.session_state.params['p2_1st_serve'] = st.number_input(
            "P2 1st Serve %",
            0.0, 100.0, st.session_state.params['p2_1st_serve'], 1.0
        )
    with col4:
        pass
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state.params['p2_bp_save'] = st.number_input(
            "P2 BP Save %",
            0.0, 100.0, st.session_state.params['p2_bp_save'], 1.0
        )
    with col2:
        st.session_state.params['p2_bp_convert'] = st.number_input(
            "P2 BP Convert %",
            0.0, 100.0, st.session_state.params['p2_bp_convert'], 1.0
        )
    with col3, col4:
        pass

# ==================== SECTION 3: PRE-MATCH ODDS ====================
with st.expander("💰 PRE-MATCH BOOKMAKER ODDS", expanded=True):
    st.subheader("Match Odds")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.params['p1_odds'] = st.number_input(
            f"Player 1 Odds ({st.session_state.params['p1_name'][:15]})",
            1.01, 50.0, st.session_state.params['p1_odds'], 0.01,
            help="Decimal odds for Player 1 to win match"
        )
    
    with col2:
        st.session_state.params['p2_odds'] = st.number_input(
            f"Player 2 Odds ({st.session_state.params['p2_name'][:15]})",
            1.01, 50.0, st.session_state.params['p2_odds'], 0.01,
            help="Decimal odds for Player 2 to win match"
        )

# ==================== SECTION 4: PLAYER MODIFIERS ====================
with st.expander("⚡ PLAYER MODIFIERS (-10 to +10)", expanded=True):
    st.subheader("Momentum & Confidence Adjustments")
    st.caption("Positive = advantage | Negative = disadvantage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Player 1 Modifiers**")
        st.session_state.params['momentum_p1'] = st.slider(
            "P1 Momentum",
            -10, 10, st.session_state.params['momentum_p1'],
            help="Recent form: -10 (terrible) to +10 (hot streak)"
        )
        st.session_state.params['surface_mastery_p1'] = st.slider(
            "P1 Surface Mastery",
            -10, 10, st.session_state.params['surface_mastery_p1'],
            help="Performance on this surface type"
        )
        st.session_state.params['clutch_p1'] = st.slider(
            "P1 Clutch Factor",
            -10, 10, st.session_state.params['clutch_p1'],
            help="Big point performance"
        )
    
    with col2:
        st.write("**Player 2 Modifiers**")
        st.session_state.params['momentum_p2'] = st.slider(
            "P2 Momentum",
            -10, 10, st.session_state.params['momentum_p2']
        )
        st.session_state.params['surface_mastery_p2'] = st.slider(
            "P2 Surface Mastery",
            -10, 10, st.session_state.params['surface_mastery_p2']
        )
        st.session_state.params['clutch_p2'] = st.slider(
            "P2 Clutch Factor",
            -10, 10, st.session_state.params['clutch_p2']
        )

# ==================== ANALYSIS SECTION ====================
st.markdown("---")
st.header("📈 ANALYSIS & CALCULATIONS")

# Extract parameters
p1_name = st.session_state.params['p1_name']
p2_name = st.session_state.params['p2_name']
p1_rank = st.session_state.params['p1_rank']
p2_rank = st.session_state.params['p2_rank']
surface = st.session_state.params['surface']
best_of = st.session_state.params['best_of']

p1_serve_pct = st.session_state.params['p1_serve_pct'] / 100.0
p2_serve_pct = st.session_state.params['p2_serve_pct'] / 100.0
p1_return_pct = st.session_state.params['p1_return_pct'] / 100.0
p2_return_pct = st.session_state.params['p2_return_pct'] / 100.0

p1_odds = st.session_state.params['p1_odds']
p2_odds = st.session_state.params['p2_odds']

# ==================== CALCULATE TRUE PROBABILITIES ====================

# 1. Ranking-based probability
p1_rank_prob = calculate_ranking_probability(p1_rank, p2_rank)

# 2. Serve/Return based probability
p1_hold = p_win_game(p1_serve_pct)
p2_hold = p_win_game(p2_serve_pct)

p1_set_prob = p_set_from_games(0, 0, 1, p1_serve_pct, p2_serve_pct)
p1_match_prob = p_match_from_sets(0, 0, p1_set_prob, best_of)

# 3. Adjustment for modifiers
p1_match_prob_adjusted = adjust_for_modifiers(
    p1_match_prob,
    st.session_state.params['momentum_p1'],
    st.session_state.params['momentum_p2'],
    st.session_state.params['surface_mastery_p1'],
    st.session_state.params['surface_mastery_p2'],
    st.session_state.params['clutch_p1'],
    st.session_state.params['clutch_p2']
)

# Weighted ensemble (70% serve/return, 30% ranking)
p1_true_prob = 0.70 * p1_match_prob_adjusted + 0.30 * p1_rank_prob
p2_true_prob = 1.0 - p1_true_prob

# ==================== CALCULATE IMPLIED PROBABILITIES ====================
p1_implied = 1.0 / p1_odds
p2_implied = 1.0 / p2_odds
total_implied = p1_implied + p2_implied
overround = total_implied - 1.0

# Normalize implied probabilities
p1_implied_norm = p1_implied / total_implied
p2_implied_norm = p2_implied / total_implied

# ==================== CALCULATE EDGES ====================
p1_edge = p1_true_prob - p1_implied_norm
p2_edge = p2_true_prob - p2_implied_norm

p1_edge_pct = (p1_true_prob / p1_implied_norm - 1) * 100 if p1_implied_norm > 0 else 0
p2_edge_pct = (p2_true_prob / p2_implied_norm - 1) * 100 if p2_implied_norm > 0 else 0

# ==================== DISPLAY RESULTS ====================

# Header with match info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Match", value=f"{p1_name[:20]} vs {p2_name[:20]}")
with col2:
    st.metric(label="Surface", value=surface)
with col3:
    st.metric(label="Best of", value=best_of)

st.markdown("")

# Main results grid
col1, col2 = st.columns(2)

# ==================== PLAYER 1 PANEL ====================
with col1:
    st.markdown(f"""
    <div class="results-panel">
        <h3>{p1_name}</h3>
        <table class="stat-table">
            <tr>
                <td><strong>Ranking:</strong></td>
                <td>{p1_rank}</td>
            </tr>
            <tr>
                <td><strong>Serve Win %:</strong></td>
                <td>{st.session_state.params['p1_serve_pct']:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Return Win %:</strong></td>
                <td>{st.session_state.params['p1_return_pct']:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Game Hold Prob:</strong></td>
                <td>{p1_hold*100:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Set Win Prob:</strong></td>
                <td>{p1_set_prob*100:.1f}%</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="results-panel">
        <h4>TRUE PROBABILITY</h4>
        <div class="true-prob">{p1_true_prob*100:.1f}%</div>
        <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 5px;">
        Based on serve/return stats, ranking, and modifiers
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="results-panel">
        <h4>MARKET PROBABILITY</h4>
        <div class="implied-prob">{p1_odds:.2f} odds → {p1_implied_norm*100:.1f}%</div>
        <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 5px;">
        Implied from bookmaker odds
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== PLAYER 2 PANEL ====================
with col2:
    st.markdown(f"""
    <div class="results-panel">
        <h3>{p2_name}</h3>
        <table class="stat-table">
            <tr>
                <td><strong>Ranking:</strong></td>
                <td>{p2_rank}</td>
            </tr>
            <tr>
                <td><strong>Serve Win %:</strong></td>
                <td>{st.session_state.params['p2_serve_pct']:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Return Win %:</strong></td>
                <td>{st.session_state.params['p2_return_pct']:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Game Hold Prob:</strong></td>
                <td>{p2_hold*100:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Set Win Prob:</strong></td>
                <td>{(1-p1_set_prob)*100:.1f}%</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="results-panel">
        <h4>TRUE PROBABILITY</h4>
        <div class="true-prob">{p2_true_prob*100:.1f}%</div>
        <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 5px;">
        Based on serve/return stats, ranking, and modifiers
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="results-panel">
        <h4>MARKET PROBABILITY</h4>
        <div class="implied-prob">{p2_odds:.2f} odds → {p2_implied_norm*100:.1f}%</div>
        <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 5px;">
        Implied from bookmaker odds
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ==================== EDGE ANALYSIS ====================
st.subheader("💡 VALUE & EDGE ANALYSIS")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="results-panel">
        <h4>{p1_name} ({p1_odds:.2f})</h4>
        <table class="stat-table">
            <tr>
                <td><strong>True Probability:</strong></td>
                <td class="true-prob">{p1_true_prob*100:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Implied Probability:</strong></td>
                <td class="implied-prob">{p1_implied_norm*100:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Edge:</strong></td>
                <td class="{'edge-positive' if p1_edge > 0 else 'edge-negative'}">{p1_edge*100:+.1f}%</td>
            </tr>
            <tr>
                <td><strong>ROI:</strong></td>
                <td class="{'edge-positive' if p1_edge_pct > 0 else 'edge-negative'}">{p1_edge_pct:+.1f}%</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
    
    if p1_edge > 0:
        st.success(f"✅ VALUE BET: {p1_edge*100:.1f}% edge found!")
    else:
        st.warning(f"❌ FADE: {abs(p1_edge)*100:.1f}% against the play")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="results-panel">
        <h4>{p2_name} ({p2_odds:.2f})</h4>
        <table class="stat-table">
            <tr>
                <td><strong>True Probability:</strong></td>
                <td class="true-prob">{p2_true_prob*100:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Implied Probability:</strong></td>
                <td class="implied-prob">{p2_implied_norm*100:.1f}%</td>
            </tr>
            <tr>
                <td><strong>Edge:</strong></td>
                <td class="{'edge-positive' if p2_edge > 0 else 'edge-negative'}">{p2_edge*100:+.1f}%</td>
            </tr>
            <tr>
                <td><strong>ROI:</strong></td>
                <td class="{'edge-positive' if p2_edge_pct > 0 else 'edge-negative'}">{p2_edge_pct:+.1f}%</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)
    
    if p2_edge > 0:
        st.success(f"✅ VALUE BET: {p2_edge*100:.1f}% edge found!")
    else:
        st.warning(f"❌ FADE: {abs(p2_edge)*100:.1f}% against the play")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== MARKET INFO ====================
st.markdown("")
st.subheader("📊 Market Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Bookmaker Margin", f"{overround*100:.2f}%")
with col2:
    if abs(p1_edge) > abs(p2_edge):
        recommendation = f"Play {p1_name}" if p1_edge > 0 else f"Fade {p1_name}"
    else:
        recommendation = f"Play {p2_name}" if p2_edge > 0 else f"Fade {p2_name}"
    st.metric("Recommendation", recommendation)
with col3:
    stronger_edge = max(abs(p1_edge), abs(p2_edge))
    confidence = "Strong" if stronger_edge > 0.10 else ("Moderate" if stronger_edge > 0.05 else "Weak")
    st.metric("Edge Strength", confidence)

# ==================== PROBABILITY COMPARISON CHART ====================
st.markdown("")
st.subheader("📈 Probability Comparison")

col1, col2 = st.columns(2)

with col1:
    p1_pct = p1_true_prob * 100
    st.markdown(f"""
    <div class="prob-bar-container">
        <div class="prob-bar-fill" style="width: {p1_pct}%;">
            {p1_name[:15]}: {p1_pct:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    p2_pct = p2_true_prob * 100
    st.markdown(f"""
    <div class="prob-bar-container">
        <div class="prob-bar-fill" style="width: {p2_pct}%;">
            {p2_name[:15]}: {p2_pct:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("🎾 V2 Tennis Calculator | True Probability Analysis | Decimal Odds Input | Edge Detection")
