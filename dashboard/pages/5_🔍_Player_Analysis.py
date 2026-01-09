"""
Page 5: Player Analysis
=======================
Deep dive into individual player statistics and predictions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

st.set_page_config(page_title="Player Analysis", page_icon="🔍", layout="wide")

st.title("🔍 Player Analysis")

st.info("""
**Player analysis module coming soon!**

This page will feature:
- Individual player statistics and trends
- Head-to-head matchup analysis
- Surface-specific performance metrics
- Recent form and momentum indicators
- Serve/return statistics breakdown
- Historical betting performance on specific players
""")

# ============================================================================
# PLAYER SEARCH
# ============================================================================

st.subheader("🔎 Search Player")

player_search = st.text_input(
    "Enter player name",
    placeholder="e.g., Novak Djokovic, Rafael Nadal, Carlos Alcaraz"
)

if player_search:
    st.info(f"Searching for: **{player_search}**")
    
    # Mock player data (replace with actual database query)
    st.markdown("### Player Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ATP Ranking", "#1")
        st.metric("Age", "36")
    
    with col2:
        st.metric("Win Rate (2024)", "82.5%")
        st.metric("Titles (2024)", "7")
    
    with col3:
        st.metric("Prize Money", "$15.2M")
        st.metric("Matches Played", "65")
    
    st.divider()
    
    # ========================================================================
    # SURFACE BREAKDOWN
    # ========================================================================
    
    st.subheader("📊 Performance by Surface")
    
    # Mock surface data
    surface_data = pd.DataFrame({
        'Surface': ['Hard', 'Clay', 'Grass', 'Indoor'],
        'Matches': [40, 15, 8, 12],
        'Wins': [34, 12, 7, 10],
        'Win Rate': [85.0, 80.0, 87.5, 83.3]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=surface_data['Surface'],
        y=surface_data['Win Rate'],
        marker=dict(color='#1f77b4'),
        text=[f"{x:.1f}%" for x in surface_data['Win Rate']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Win Rate by Surface",
        xaxis_title="Surface",
        yaxis_title="Win Rate (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # RECENT FORM
    # ========================================================================
    
    st.divider()
    st.subheader("📈 Recent Form (Last 10 Matches)")
    
    # Mock recent matches
    recent_matches = pd.DataFrame({
        'Date': ['2024-01-08', '2024-01-06', '2024-01-04', '2024-01-02', '2023-12-30'],
        'Opponent': ['Carlos Alcaraz', 'Jannik Sinner', 'Daniil Medvedev', 'Stefanos Tsitsipas', 'Alexander Zverev'],
        'Tournament': ['ATP Finals', 'ATP Finals', 'Paris Masters', 'Paris Masters', 'Turin'],
        'Surface': ['Indoor', 'Indoor', 'Hard', 'Hard', 'Hard'],
        'Result': ['W', 'W', 'L', 'W', 'W'],
        'Score': ['6-4, 7-5', '7-6, 6-3', '6-7, 4-6', '6-3, 6-4', '6-2, 6-3']
    })
    
    # Color code results
    def color_result(val):
        if val == 'W':
            return 'background-color: #c8e6c9; color: #2e7d32; font-weight: bold'
        else:
            return 'background-color: #ffcdd2; color: #c62828; font-weight: bold'
    
    styled_df = recent_matches.style.map(color_result, subset=['Result'])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Form summary
    wins = (recent_matches['Result'] == 'W').sum()
    total = len(recent_matches)
    form_str = ' - '.join(recent_matches['Result'].tolist())
    
    st.info(f"**Current Form:** {form_str} ({wins}W-{total-wins}L, {wins/total*100:.0f}%)")
    
    # ========================================================================
    # SERVE & RETURN STATS
    # ========================================================================
    
    st.divider()
    st.subheader("🎾 Serve & Return Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Serve Stats")
        st.metric("1st Serve %", "68%")
        st.metric("1st Serve Won", "75%")
        st.metric("2nd Serve Won", "58%")
        st.metric("Aces/Match", "8.5")
        st.metric("Double Faults/Match", "2.1")
    
    with col2:
        st.markdown("#### Return Stats")
        st.metric("Return Points Won", "42%")
        st.metric("1st Return Won", "35%")
        st.metric("2nd Return Won", "54%")
        st.metric("Break Points Won", "48%")
        st.metric("Return Games Won", "28%")
    
    # ========================================================================
    # HEAD-TO-HEAD DATABASE
    # ========================================================================
    
    st.divider()
    st.subheader("⚔️ Head-to-Head Records")
    
    h2h_opponent = st.selectbox(
        "Select opponent for H2H",
        ['Carlos Alcaraz', 'Jannik Sinner', 'Daniil Medvedev', 'Rafael Nadal', 'Roger Federer']
    )
    
    st.markdown(f"### vs {h2h_opponent}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Record", "15-8")
    
    with col2:
        st.metric("Last Meeting", "W 6-4, 7-5")
        st.caption("2024-01-08 | ATP Finals")
    
    with col3:
        st.metric("Winning %", "65.2%")
    
    # By surface
    st.markdown("#### By Surface")
    
    h2h_surface = pd.DataFrame({
        'Surface': ['Hard', 'Clay', 'Grass'],
        'Wins': [10, 3, 2],
        'Losses': [5, 2, 1],
        'Win Rate': [66.7, 60.0, 66.7]
    })
    
    st.dataframe(h2h_surface, use_container_width=True, hide_index=True)

else:
    # ========================================================================
    # TOP PLAYERS OVERVIEW
    # ========================================================================
    
    st.subheader("🏆 Top Players Overview")
    
    st.info("Enter a player name above to view detailed statistics")
    
    # Mock top players data
    top_players = pd.DataFrame({
        'Rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Player': [
            'Novak Djokovic', 'Carlos Alcaraz', 'Jannik Sinner',
            'Daniil Medvedev', 'Andrey Rublev', 'Stefanos Tsitsipas',
            'Holger Rune', 'Casper Ruud', 'Hubert Hurkacz', 'Alex de Minaur'
        ],
        'Points': [9855, 8855, 7760, 7350, 5015, 4960, 3795, 3600, 3385, 3250],
        'Win Rate': [82.5, 79.2, 76.8, 73.1, 68.5, 67.2, 65.8, 64.3, 62.7, 61.5],
        'Titles': [7, 6, 3, 2, 4, 2, 3, 2, 1, 2]
    })
    
    st.dataframe(
        top_players.style.format({
            'Win Rate': '{:.1f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

# ============================================================================
# BETTING INSIGHTS
# ============================================================================

st.divider()
st.subheader("💰 Betting Insights")

if player_search:
    st.markdown(f"### Betting Performance on {player_search}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Bets Placed", "12")
    
    with col2:
        st.metric("Bets Won", "9")
    
    with col3:
        st.metric("Win Rate", "75.0%")
    
    with col4:
        st.metric("ROI", "+12.5%")
    
    st.success("""
    **Recommendation:** This player has historically provided positive ROI.
    Consider betting when:
    - Playing on hard courts (highest win rate)
    - Odds are above 1.60 (good value)
    - Against lower-ranked opponents
    """)

else:
    st.info("Select a player to view betting insights and recommendations")

# Footer
st.divider()
st.caption("💡 Player statistics update daily after matches are completed")
st.caption("📊 Historical data sourced from ATP Official, Sofascore, and Flashscore")
