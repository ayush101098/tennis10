"""
Live Match Calculator - Markov Chain with Manual Score Entry
=============================================================
Real-time match probability calculator using Markov chain model
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

st.set_page_config(page_title="Live Match Calculator", page_icon="🎯", layout="wide")

st.title("🎯 Live Match Calculator")
st.markdown("**Real-time Markov Chain probability calculator with manual score entry**")

# ============================================================================
# MARKOV CHAIN CALCULATOR
# ============================================================================

def calculate_point_probabilities(p1_serve_win, p2_serve_win):
    """Calculate probabilities using Markov chain"""
    # Game probability on serve
    p1_game_on_serve = p1_serve_win ** 4 + 4 * p1_serve_win ** 4 * (1 - p1_serve_win)
    p2_game_on_serve = p2_serve_win ** 4 + 4 * p2_serve_win ** 4 * (1 - p2_serve_win)
    
    # Tiebreak probability
    p1_tiebreak = p1_serve_win ** 7
    p2_tiebreak = p2_serve_win ** 7
    
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
        'p2_match': p2_match
    }

def adjust_for_current_score(base_prob, sets_won, games_won, points_won):
    """Adjust probability based on current score"""
    # Simple adjustment: increase probability if ahead
    set_bonus = sets_won * 0.15
    game_bonus = (games_won / 6) * 0.10
    point_bonus = (points_won / 4) * 0.05
    
    adjusted = min(base_prob + set_bonus + game_bonus + point_bonus, 0.95)
    return max(adjusted, 0.05)

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
    
    st.divider()
    
    st.subheader("📊 Player Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{player1_name}**")
        p1_serve_win = st.slider("Serve Win %", 50, 85, 68, key="p1_serve") / 100
        p1_return_win = st.slider("Return Win %", 20, 50, 35, key="p1_return") / 100
    
    with col2:
        st.markdown(f"**{player2_name}**")
        p2_serve_win = st.slider("Serve Win %", 50, 85, 65, key="p2_serve") / 100
        p2_return_win = st.slider("Return Win %", 20, 50, 32, key="p2_return") / 100
    
    st.divider()
    
    st.subheader("💰 Bookmaker Odds")
    
    p1_odds = st.number_input(f"{player1_name} Odds", min_value=1.01, max_value=50.0, value=1.85, step=0.01)
    p2_odds = st.number_input(f"{player2_name} Odds", min_value=1.01, max_value=50.0, value=2.10, step=0.01)

# ============================================================================
# MAIN AREA - LIVE SCORE ENTRY
# ============================================================================

st.header("📝 Current Match Score")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Sets")
    p1_sets = st.number_input(f"{player1_name} Sets", 0, 3, 0, key="p1_sets")
    p2_sets = st.number_input(f"{player2_name} Sets", 0, 3, 0, key="p2_sets")

with col2:
    st.subheader("Games (Current Set)")
    p1_games = st.number_input(f"{player1_name} Games", 0, 7, 0, key="p1_games")
    p2_games = st.number_input(f"{player2_name} Games", 0, 7, 0, key="p2_games")

with col3:
    st.subheader("Points (Current Game)")
    p1_points = st.selectbox(f"{player1_name} Points", [0, 15, 30, 40, "AD"], key="p1_points")
    p2_points = st.selectbox(f"{player2_name} Points", [0, 15, 30, 40, "AD"], key="p2_points")

# Server selection
current_server = st.radio("Who is serving?", [player1_name, player2_name], horizontal=True)

st.divider()

# ============================================================================
# CALCULATIONS
# ============================================================================

st.header("🧮 Live Probability Calculations")

# Calculate base probabilities
probs = calculate_point_probabilities(p1_serve_win, p2_serve_win)

# Convert points to numeric
point_map = {0: 0, 15: 1, 30: 2, 40: 3, "AD": 4}
p1_point_num = point_map.get(p1_points, 0)
p2_point_num = point_map.get(p2_points, 0)

# Adjust for current score
p1_win_prob = adjust_for_current_score(
    probs['p1_match'], 
    p1_sets, 
    p1_games, 
    p1_point_num
)

p2_win_prob = 1 - p1_win_prob

# Calculate betting value
p1_edge, p1_ev = calculate_betting_value(p1_win_prob, p1_odds)
p2_edge, p2_ev = calculate_betting_value(p2_win_prob, p2_odds)

# Display results
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### {player1_name}")
    st.metric("Win Probability", f"{p1_win_prob:.1%}", delta=f"{p1_win_prob - 0.5:.1%}")
    st.metric("Bookmaker Odds", f"{p1_odds:.2f}")
    st.metric("Edge", f"{p1_edge:.2%}", delta=f"{p1_edge:.2%}")
    st.metric("Expected Value", f"{p1_ev:.2%}")
    
    if p1_edge > 0.025:
        st.success(f"✅ **VALUE BET on {player1_name}**")
        recommended_stake = min(p1_edge * 400, 150)  # Simple Kelly
        st.write(f"Recommended Stake: ${recommended_stake:.0f}")
    else:
        st.info("No value at current odds")

with col2:
    st.markdown(f"### {player2_name}")
    st.metric("Win Probability", f"{p2_win_prob:.1%}", delta=f"{p2_win_prob - 0.5:.1%}")
    st.metric("Bookmaker Odds", f"{p2_odds:.2f}")
    st.metric("Edge", f"{p2_edge:.2%}", delta=f"{p2_edge:.2%}")
    st.metric("Expected Value", f"{p2_ev:.2%}")
    
    if p2_edge > 0.025:
        st.success(f"✅ **VALUE BET on {player2_name}**")
        recommended_stake = min(p2_edge * 400, 150)
        st.write(f"Recommended Stake: ${recommended_stake:.0f}")
    else:
        st.info("No value at current odds")

st.divider()

# ============================================================================
# VISUALIZATION
# ============================================================================

st.header("📊 Probability Breakdown")

# Win probability chart
fig = go.Figure()

fig.add_trace(go.Bar(
    name=player1_name,
    x=['Match Win', 'Set Win', 'Game Win (on serve)'],
    y=[p1_win_prob * 100, probs['p1_set'] * 100, probs['p1_game_on_serve'] * 100],
    marker_color='#1f77b4',
    text=[f"{p1_win_prob:.1%}", f"{probs['p1_set']:.1%}", f"{probs['p1_game_on_serve']:.1%}"],
    textposition='outside'
))

fig.add_trace(go.Bar(
    name=player2_name,
    x=['Match Win', 'Set Win', 'Game Win (on serve)'],
    y=[p2_win_prob * 100, probs['p2_set'] * 100, probs['p2_game_on_serve'] * 100],
    marker_color='#ff7f0e',
    text=[f"{p2_win_prob:.1%}", f"{probs['p2_set']:.1%}", f"{probs['p2_game_on_serve']:.1%}"],
    textposition='outside'
))

fig.update_layout(
    title="Win Probabilities at Different Stages",
    yaxis_title="Probability (%)",
    yaxis=dict(range=[0, 100]),
    barmode='group',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

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
        'p1_edge': p1_edge,
        'p2_edge': p2_edge,
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
    
    3. **Input Bookmaker Odds:**
       - Enter the current live betting odds for each player
       - These are used to calculate value bets
    
    4. **Update Live Score:**
       - Enter current sets, games, and points
       - Select who is serving
       - Update after each game/point for live tracking
    
    5. **Analyze Probabilities:**
       - Green "VALUE BET" alerts indicate positive edge
       - Recommended stakes based on Kelly Criterion
       - Watch probability changes as match progresses
    
    6. **Save Match States:**
       - Track how probabilities evolve during the match
       - Review later to improve your model
    
    ### Tips:
    - Update stats based on current match conditions (fatigue, injuries)
    - Compare live odds to calculated probabilities
    - Look for odds discrepancies of >2.5% for value
    - Be cautious with momentum shifts and psychological factors
    """)

st.divider()

st.caption("💡 Calculator uses Markov chain model for probability estimation")
st.caption("⚠️ Always bet responsibly and within your bankroll limits")
