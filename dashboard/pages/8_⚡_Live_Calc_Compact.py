"""
Live Match Calculator - COMPACT VERSION
========================================
All V1 features in a single-screen compressed layout
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

st.set_page_config(page_title="Compact Live Calc", page_icon="⚡", layout="wide")

# Minimal CSS
st.markdown("""
<style>
    .compact-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 10px;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    .value-bet {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 8px;
        margin: 5px 0;
        border-radius: 4px;
        font-size: 13px;
    }
    .score-display {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 8px;
        background: #f8f9fa;
        border-radius: 6px;
    }
    .stButton button {
        padding: 4px 8px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="compact-header"><h2>⚡ Live Calculator - Compact</h2></div>', unsafe_allow_html=True)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def apply_advanced_parameters(base_serve, base_return, params):
    """Apply advanced parameter adjustments"""
    serve_adj = 1.0
    return_adj = 1.0
    
    # Momentum (±15%)
    serve_adj *= (0.85 + params.get('momentum', 0.5) * 0.30)
    return_adj *= (0.85 + params.get('momentum', 0.5) * 0.30)
    
    # Surface mastery (±10%)
    serve_adj *= (0.90 + params.get('surface_mastery', 0.5) * 0.20)
    return_adj *= (0.90 + params.get('surface_mastery', 0.5) * 0.20)
    
    # Clutch (±8%)
    serve_adj *= (0.92 + params.get('clutch', 0.5) * 0.16)
    
    # BP Defense (±5%)
    serve_adj *= (0.95 + params.get('bp_defense', 0.6) * 0.10)
    
    # Consistency (±4%)
    serve_adj *= (0.96 + params.get('consistency', 0.5) * 0.08)
    return_adj *= (0.96 + params.get('consistency', 0.5) * 0.08)
    
    # First serve % (±3%)
    first_srv = params.get('first_serve_pct', 0.62)
    serve_adj *= (0.97 + (first_srv - 0.62) * 0.15)
    
    return min(base_serve * serve_adj, 0.95), min(base_return * return_adj, 0.95)

def calculate_game_prob(p_serve_pt, p_return_pt):
    """Calculate probability of winning a game on serve"""
    p = p_serve_pt
    q = 1 - p
    
    prob_40_0 = p**3
    prob_40_15 = 3 * (p**3) * q
    prob_40_30 = 6 * (p**3) * (q**2)
    prob_deuce = 20 * (p**3) * (q**3)
    prob_win_deuce = (p**2) / (1 - 2*p*q)
    
    return prob_40_0 + prob_40_15 + prob_40_30 + (prob_deuce * prob_win_deuce)

def calculate_set_prob(p_game_serve, p_game_return):
    """Calculate probability of winning a set"""
    p_hold = p_game_serve
    p_break = 1 - p_game_return
    
    # Simplified calculation
    games_advantage = (p_hold - 0.5) + (p_break - 0.5)
    set_prob = 0.5 + (games_advantage * 2.5)
    
    return max(0.05, min(0.95, set_prob))

def calculate_match_prob(p_set, best_of=3):
    """Calculate match win probability"""
    if best_of == 3:
        return (p_set**2) * (1 + 2*(1-p_set))
    else:  # best of 5
        return (p_set**3) * (1 + 3*(1-p_set) + 6*((1-p_set)**2))

# ============================================================================
# SESSION STATE
# ============================================================================

if 'p1_name' not in st.session_state:
    st.session_state.p1_name = "Player 1"
if 'p2_name' not in st.session_state:
    st.session_state.p2_name = "Player 2"
if 'surface' not in st.session_state:
    st.session_state.surface = "Hard"
if 'p1_serve' not in st.session_state:
    st.session_state.p1_serve = 0.65
if 'p2_serve' not in st.session_state:
    st.session_state.p2_serve = 0.65
if 'p1_return' not in st.session_state:
    st.session_state.p1_return = 0.35
if 'p2_return' not in st.session_state:
    st.session_state.p2_return = 0.35

# Advanced params
if 'p1_params' not in st.session_state:
    st.session_state.p1_params = {'momentum': 0.5, 'surface_mastery': 0.5, 'clutch': 0.5, 
                                    'bp_defense': 0.6, 'consistency': 0.5, 'first_serve_pct': 0.62}
if 'p2_params' not in st.session_state:
    st.session_state.p2_params = {'momentum': 0.5, 'surface_mastery': 0.5, 'clutch': 0.5, 
                                    'bp_defense': 0.6, 'consistency': 0.5, 'first_serve_pct': 0.62}

# Score tracking
if 'sets1' not in st.session_state:
    st.session_state.sets1 = 0
if 'sets2' not in st.session_state:
    st.session_state.sets2 = 0
if 'games1' not in st.session_state:
    st.session_state.games1 = 0
if 'games2' not in st.session_state:
    st.session_state.games2 = 0
if 'points1' not in st.session_state:
    st.session_state.points1 = 0
if 'points2' not in st.session_state:
    st.session_state.points2 = 0
if 'prob_history' not in st.session_state:
    st.session_state.prob_history = []

# ============================================================================
# COMPACT INPUT LAYOUT (2 rows)
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.session_state.p1_name = st.text_input("Player 1", st.session_state.p1_name, key="p1n")
    
with col2:
    st.session_state.p2_name = st.text_input("Player 2", st.session_state.p2_name, key="p2n")

# Surface and pre-match odds in one row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.session_state.surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], key="surf")
with c2:
    p1_prematch_odds = st.number_input(f"{st.session_state.p1_name[:10]} Pre-Odds", 1.01, 20.0, 1.85, 0.05, key="pmo1")
with c3:
    p2_prematch_odds = st.number_input(f"{st.session_state.p2_name[:10]} Pre-Odds", 1.01, 20.0, 2.10, 0.05, key="pmo2")
with c4:
    best_of = st.selectbox("Format", [3, 5], key="bo")

st.markdown("---")

# Player stats in expandable section (compact)
with st.expander("📊 Player Stats & Advanced Parameters", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{st.session_state.p1_name}**")
        c1a, c1b = st.columns(2)
        with c1a:
            st.session_state.p1_serve = st.slider("Serve Win", 50, 85, 65, key="p1s") / 100
            st.session_state.p1_return = st.slider("Return Win", 20, 50, 35, key="p1r") / 100
        with c1b:
            st.session_state.p1_params['momentum'] = st.slider("Momentum", 0.0, 1.0, 0.5, 0.05, key="p1mom")
            st.session_state.p1_params['surface_mastery'] = st.slider("Surface", 0.0, 1.0, 0.5, 0.05, key="p1surf")
        
        c1c, c1d = st.columns(2)
        with c1c:
            st.session_state.p1_params['clutch'] = st.slider("Clutch", 0.0, 1.0, 0.5, 0.05, key="p1cl")
            st.session_state.p1_params['bp_defense'] = st.slider("BP Def", 0.0, 1.0, 0.6, 0.05, key="p1bp")
        with c1d:
            st.session_state.p1_params['consistency'] = st.slider("Consistency", 0.0, 1.0, 0.5, 0.05, key="p1con")
            st.session_state.p1_params['first_serve_pct'] = st.slider("1st Srv %", 0.0, 1.0, 0.62, 0.01, key="p1fs")
    
    with col2:
        st.markdown(f"**{st.session_state.p2_name}**")
        c2a, c2b = st.columns(2)
        with c2a:
            st.session_state.p2_serve = st.slider("Serve Win ", 50, 85, 65, key="p2s") / 100
            st.session_state.p2_return = st.slider("Return Win ", 20, 50, 35, key="p2r") / 100
        with c2b:
            st.session_state.p2_params['momentum'] = st.slider("Momentum ", 0.0, 1.0, 0.5, 0.05, key="p2mom")
            st.session_state.p2_params['surface_mastery'] = st.slider("Surface ", 0.0, 1.0, 0.5, 0.05, key="p2surf")
        
        c2c, c2d = st.columns(2)
        with c2c:
            st.session_state.p2_params['clutch'] = st.slider("Clutch ", 0.0, 1.0, 0.5, 0.05, key="p2cl")
            st.session_state.p2_params['bp_defense'] = st.slider("BP Def ", 0.0, 1.0, 0.6, 0.05, key="p2bp")
        with c2d:
            st.session_state.p2_params['consistency'] = st.slider("Consistency ", 0.0, 1.0, 0.5, 0.05, key="p2con")
            st.session_state.p2_params['first_serve_pct'] = st.slider("1st Srv % ", 0.0, 1.0, 0.62, 0.01, key="p2fs")

st.markdown("---")

# ============================================================================
# LIVE PROBABILITIES (COMPACT DISPLAY)
# ============================================================================

# Apply advanced parameters
p1_s_adj, p1_r_adj = apply_advanced_parameters(st.session_state.p1_serve, st.session_state.p1_return, st.session_state.p1_params)
p2_s_adj, p2_r_adj = apply_advanced_parameters(st.session_state.p2_serve, st.session_state.p2_return, st.session_state.p2_params)

# Calculate probabilities
p1_game_on_serve = calculate_game_prob(p1_s_adj, p2_r_adj)
p2_game_on_serve = calculate_game_prob(p2_s_adj, p1_r_adj)
p1_game_on_return = 1 - p2_game_on_serve
p2_game_on_return = 1 - p1_game_on_serve

p1_set = calculate_set_prob(p1_game_on_serve, p2_game_on_serve)
p2_set = 1 - p1_set

p1_match = calculate_match_prob(p1_set, best_of)
p2_match = 1 - p1_match

# Store in history
st.session_state.prob_history.append({'p1': p1_match, 'p2': p2_match})

# Compact score display
col_score1, col_score2 = st.columns(2)
with col_score1:
    st.markdown(f'<div class="score-display">{st.session_state.p1_name}<br/>{st.session_state.sets1} - {st.session_state.games1} - {st.session_state.points1}<br/><span style="color:#28a745">{p1_match:.1%}</span></div>', unsafe_allow_html=True)
with col_score2:
    st.markdown(f'<div class="score-display">{st.session_state.p2_name}<br/>{st.session_state.sets2} - {st.session_state.games2} - {st.session_state.points2}<br/><span style="color:#007bff">{p2_match:.1%}</span></div>', unsafe_allow_html=True)

# Point tracking buttons (compact)
col_btn1, col_btn2, col_reset = st.columns([2, 2, 1])

def point_won(winner):
    if winner == 1:
        st.session_state.points1 += 1
    else:
        st.session_state.points2 += 1
    
    # Game logic
    if st.session_state.points1 >= 4 and st.session_state.points1 - st.session_state.points2 >= 2:
        st.session_state.games1 += 1
        st.session_state.points1 = 0
        st.session_state.points2 = 0
        if st.session_state.games1 >= 6 and st.session_state.games1 - st.session_state.games2 >= 2:
            st.session_state.sets1 += 1
            st.session_state.games1 = 0
            st.session_state.games2 = 0
    elif st.session_state.points2 >= 4 and st.session_state.points2 - st.session_state.points1 >= 2:
        st.session_state.games2 += 1
        st.session_state.points1 = 0
        st.session_state.points2 = 0
        if st.session_state.games2 >= 6 and st.session_state.games2 - st.session_state.games1 >= 2:
            st.session_state.sets2 += 1
            st.session_state.games1 = 0
            st.session_state.games2 = 0

with col_btn1:
    if st.button(f"✅ {st.session_state.p1_name} WINS POINT", key="p1win", use_container_width=True):
        point_won(1)
        st.rerun()

with col_btn2:
    if st.button(f"✅ {st.session_state.p2_name} WINS POINT", key="p2win", use_container_width=True):
        point_won(2)
        st.rerun()

with col_reset:
    if st.button("🔄 Reset", key="reset"):
        st.session_state.sets1 = 0
        st.session_state.sets2 = 0
        st.session_state.games1 = 0
        st.session_state.games2 = 0
        st.session_state.points1 = 0
        st.session_state.points2 = 0
        st.session_state.prob_history = []
        st.rerun()

st.markdown("---")

# ============================================================================
# VALUE BETS (COMPACT - 3 MARKETS)
# ============================================================================

st.subheader("💰 Value Bets")

# Live odds inputs (compact)
col_o1, col_o2, col_o3, col_o4, col_o5, col_o6 = st.columns(6)
with col_o1:
    p1_match_odds = st.number_input("P1 Match", 1.01, 50.0, 1.85, 0.05, key="p1mo")
with col_o2:
    p2_match_odds = st.number_input("P2 Match", 1.01, 50.0, 2.10, 0.05, key="p2mo")
with col_o3:
    p1_set_odds = st.number_input("P1 Set", 1.01, 10.0, 1.75, 0.05, key="p1so")
with col_o4:
    p2_set_odds = st.number_input("P2 Set", 1.01, 10.0, 2.20, 0.05, key="p2so")
with col_o5:
    p1_game_odds = st.number_input("P1 Game", 1.01, 10.0, 1.50, 0.05, key="p1go")
with col_o6:
    p2_game_odds = st.number_input("P2 Game", 1.01, 10.0, 2.75, 0.05, key="p2go")

# Value calculations
col_v1, col_v2 = st.columns(2)

with col_v1:
    st.markdown(f"**{st.session_state.p1_name} Opportunities:**")
    
    # Match
    edge_m = p1_match - (1/p1_match_odds)
    if edge_m > 0.025:
        ev_m = (p1_match * (p1_match_odds - 1)) - (1 - p1_match)
        stake_m = min(edge_m * 400, 150)
        st.markdown(f'<div class="value-bet">✅ <b>MATCH</b> @ {p1_match_odds:.2f}<br/>Stake: ${stake_m:.0f} | EV: {ev_m:+.1%} | Edge: {edge_m:+.1%}</div>', unsafe_allow_html=True)
    
    # Set
    edge_s = p1_set - (1/p1_set_odds)
    if edge_s > 0.025:
        ev_s = (p1_set * (p1_set_odds - 1)) - (1 - p1_set)
        stake_s = min(edge_s * 300, 100)
        st.markdown(f'<div class="value-bet">✅ <b>SET</b> @ {p1_set_odds:.2f}<br/>Stake: ${stake_s:.0f} | EV: {ev_s:+.1%} | Edge: {edge_s:+.1%}</div>', unsafe_allow_html=True)
    
    # Game
    edge_g = p1_game_on_serve - (1/p1_game_odds)
    if edge_g > 0.025:
        ev_g = (p1_game_on_serve * (p1_game_odds - 1)) - (1 - p1_game_on_serve)
        stake_g = min(edge_g * 200, 50)
        st.markdown(f'<div class="value-bet">✅ <b>GAME</b> @ {p1_game_odds:.2f}<br/>Stake: ${stake_g:.0f} | EV: {ev_g:+.1%} | Edge: {edge_g:+.1%}</div>', unsafe_allow_html=True)

with col_v2:
    st.markdown(f"**{st.session_state.p2_name} Opportunities:**")
    
    # Match
    edge_m = p2_match - (1/p2_match_odds)
    if edge_m > 0.025:
        ev_m = (p2_match * (p2_match_odds - 1)) - (1 - p2_match)
        stake_m = min(edge_m * 400, 150)
        st.markdown(f'<div class="value-bet">✅ <b>MATCH</b> @ {p2_match_odds:.2f}<br/>Stake: ${stake_m:.0f} | EV: {ev_m:+.1%} | Edge: {edge_m:+.1%}</div>', unsafe_allow_html=True)
    
    # Set
    edge_s = p2_set - (1/p2_set_odds)
    if edge_s > 0.025:
        ev_s = (p2_set * (p2_set_odds - 1)) - (1 - p2_set)
        stake_s = min(edge_s * 300, 100)
        st.markdown(f'<div class="value-bet">✅ <b>SET</b> @ {p2_set_odds:.2f}<br/>Stake: ${stake_s:.0f} | EV: {ev_s:+.1%} | Edge: {edge_s:+.1%}</div>', unsafe_allow_html=True)
    
    # Game
    edge_g = p2_game_on_serve - (1/p2_game_odds)
    if edge_g > 0.025:
        ev_g = (p2_game_on_serve * (p2_game_odds - 1)) - (1 - p2_game_on_serve)
        stake_g = min(edge_g * 200, 50)
        st.markdown(f'<div class="value-bet">✅ <b>GAME</b> @ {p2_game_odds:.2f}<br/>Stake: ${stake_g:.0f} | EV: {ev_g:+.1%} | Edge: {edge_g:+.1%}</div>', unsafe_allow_html=True)

# Probability chart (compact)
if len(st.session_state.prob_history) > 1:
    st.markdown("---")
    with st.expander("📈 Probability Evolution", expanded=False):
        points = list(range(1, len(st.session_state.prob_history) + 1))
        p1_probs = [p['p1'] * 100 for p in st.session_state.prob_history]
        p2_probs = [p['p2'] * 100 for p in st.session_state.prob_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=points, y=p1_probs, mode='lines+markers', name=st.session_state.p1_name, line=dict(color='#28a745', width=2)))
        fig.add_trace(go.Scatter(x=points, y=p2_probs, mode='lines+markers', name=st.session_state.p2_name, line=dict(color='#007bff', width=2)))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

st.caption("⚡ Compact calculator with all V1 features | No data loss | Single-screen layout")
