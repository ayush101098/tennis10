"""
Page 3: Betting History
=======================
Track active and settled bets, confirm new bets
"""

import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dashboard.data_loader import (
    get_active_bets,
    get_settled_bets,
    record_bet,
    get_bankroll_status
)
from dashboard.components.tables import (
    render_bets_table,
    render_performance_summary
)

st.set_page_config(page_title="Betting History", page_icon="💰", layout="wide")

st.title("💰 Betting History & Portfolio")

# ============================================================================
# PENDING BET CONFIRMATION
# ============================================================================

if 'pending_bet' in st.session_state:
    bet = st.session_state.pending_bet
    
    st.warning("⚠️ **Confirm Bet Placement**")
    
    with st.form("confirm_bet"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Match Details")
            st.write(f"**Match:** {bet['player1_name']} vs {bet['player2_name']}")
            st.write(f"**Tournament:** {bet.get('tournament', 'N/A')}")
            st.write(f"**Surface:** {bet.get('surface', 'N/A')}")
            
            # Determine which player to bet on
            if 'player1' in str(bet.get('action', '')).lower():
                bet_player = bet['player1_name']
                bet_odds = bet.get('best_p1_odds', 0)
            else:
                bet_player = bet['player2_name']
                bet_odds = bet.get('best_p2_odds', bet.get('best_p1_odds', 0))
            
            st.write(f"**Betting on:** {bet_player}")
            st.write(f"**Odds:** {bet_odds:.2f}")
        
        with col2:
            st.subheader("Bet Analysis")
            st.write(f"**Win Probability:** {bet.get('ensemble_p1_win', 0):.1%}")
            st.write(f"**Edge:** {bet.get('edge', 0):.2%}")
            st.write(f"**Expected Value:** {bet.get('expected_value', 0):.2%}")
            st.write(f"**Confidence:** {bet.get('confidence', 'medium').upper()}")
            st.write(f"**Model Agreement:** {bet.get('model_agreement', 0) * 100:.1f}%")
        
        st.divider()
        
        # Stake input
        recommended_stake = bet.get('recommended_stake', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Recommended Stake", f"${recommended_stake:.2f}")
        
        with col2:
            potential_profit = recommended_stake * (bet_odds - 1)
            st.metric("Potential Profit", f"${potential_profit:.2f}")
        
        actual_stake = st.number_input(
            "Actual Stake Amount ($)",
            min_value=0.0,
            max_value=1000.0,
            value=float(recommended_stake),
            step=5.0,
            help="Adjust the stake amount if needed"
        )
        
        # Calculate updated potential profit
        updated_profit = actual_stake * (bet_odds - 1)
        st.info(f"💰 Potential profit with ${actual_stake:.2f} stake: **${updated_profit:.2f}**")
        
        # Confirmation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("✅ Confirm Bet", use_container_width=True, type="primary"):
                # Record the bet
                bet_data = {
                    'match_id': bet.get('match_id', f"{bet['player1_name']}_{bet['player2_name']}"),
                    'player_bet_on': bet_player,
                    'odds': bet_odds,
                    'edge': bet.get('edge', 0),
                    'expected_value': bet.get('expected_value', 0),
                    'confidence': bet.get('confidence', 'medium')
                }
                
                try:
                    record_bet(bet_data, actual_stake)
                    st.success(f"✅ Bet recorded! ${actual_stake:.2f} on {bet_player}")
                    del st.session_state.pending_bet
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error recording bet: {str(e)}")
        
        with col2:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                del st.session_state.pending_bet
                st.rerun()

# ============================================================================
# ACTIVE BETS
# ============================================================================

st.subheader("🎯 Active Bets")

active_bets = get_active_bets()

if len(active_bets) > 0:
    for idx, bet in active_bets.iterrows():
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**{bet['match']}**")
                st.caption(f"Bet on: {bet['selection']} @ {bet['odds']:.2f}")
                
                if 'start_time' in bet:
                    start_time = pd.to_datetime(bet['start_time'])
                    time_until = start_time - datetime.now()
                    
                    if time_until.total_seconds() > 0:
                        hours = time_until.total_seconds() / 3600
                        st.caption(f"⏰ Starts in {hours:.1f} hours")
                    else:
                        st.caption("🔴 Match in progress")
            
            with col2:
                st.metric("Stake", f"${bet['stake']:.0f}")
            
            with col3:
                potential = bet.get('potential_profit', bet['stake'] * (bet['odds'] - 1))
                st.metric("Potential", f"${potential:.0f}")
            
            with col4:
                confidence = bet.get('confidence', 'medium')
                if confidence == 'high':
                    st.success("HIGH")
                elif confidence == 'medium':
                    st.info("MEDIUM")
                else:
                    st.warning("LOW")
    
    # Summary
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_staked = active_bets['stake'].sum()
        st.metric("Total Staked", f"${total_staked:.2f}")
    
    with col2:
        total_potential = active_bets.get('potential_profit', active_bets['stake'] * (active_bets['odds'] - 1)).sum()
        st.metric("Total Potential", f"${total_potential:.2f}")
    
    with col3:
        avg_odds = active_bets['odds'].mean()
        st.metric("Avg Odds", f"{avg_odds:.2f}")

else:
    st.info("""
    **No active bets**
    
    Visit the **Live Predictions** page to find betting opportunities.
    """)

st.divider()

# ============================================================================
# SETTLED BETS HISTORY
# ============================================================================

st.subheader("📊 Betting History")

# Filters
col1, col2, col3 = st.columns(3)

with col1:
    result_filter = st.multiselect(
        "Result",
        ['won', 'lost'],
        default=['won', 'lost'],
        format_func=lambda x: x.capitalize()
    )

with col2:
    days_back = st.selectbox(
        "Time Period",
        [7, 30, 90, 365],
        format_func=lambda x: f"Last {x} days",
        index=1
    )

with col3:
    confidence_filter = st.multiselect(
        "Confidence",
        ['high', 'medium', 'low'],
        default=['high', 'medium', 'low'],
        format_func=lambda x: x.capitalize()
    )

# Load settled bets
settled_bets = get_settled_bets(days=days_back)

# Apply filters
if len(settled_bets) > 0:
    filtered_bets = settled_bets[
        (settled_bets['result'].isin(result_filter)) &
        (settled_bets['confidence'].isin(confidence_filter))
    ]
    
    if len(filtered_bets) > 0:
        # Display table
        render_bets_table(filtered_bets, bet_type='settled')
        
        st.divider()
        
        # Summary stats
        st.subheader("📈 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_profit = filtered_bets['profit'].sum()
            profit_color = "normal" if total_profit >= 0 else "inverse"
            st.metric(
                "Total Profit",
                f"${total_profit:.0f}",
                delta=f"${total_profit:.0f}",
                delta_color=profit_color
            )
        
        with col2:
            total_staked = filtered_bets['stake'].sum()
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            st.metric("ROI", f"{roi:.1f}%")
        
        with col3:
            wins = (filtered_bets['result'] == 'won').sum()
            total = len(filtered_bets)
            win_rate = (wins / total * 100) if total > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%", f"{wins}/{total}")
        
        with col4:
            avg_odds = filtered_bets[filtered_bets['result'] == 'won']['odds'].mean()
            st.metric("Avg Winning Odds", f"{avg_odds:.2f}" if not pd.isna(avg_odds) else "N/A")
        
        st.divider()
        
        # Detailed performance summary
        render_performance_summary(filtered_bets)
        
        # Export data
        st.download_button(
            label="📥 Export to CSV",
            data=filtered_bets.to_csv(index=False),
            file_name=f"betting_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    else:
        st.info("No bets match the selected filters")

else:
    st.info("""
    **No betting history available yet**
    
    Start placing bets to build your history. The system will automatically track:
    - Win/loss record
    - ROI and profit
    - Performance by confidence level
    - Historical trends
    """)

# ============================================================================
# BANKROLL MANAGEMENT
# ============================================================================

st.divider()

st.subheader("💵 Bankroll Status")

try:
    bankroll_status = get_bankroll_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Bankroll",
            f"${bankroll_status['current']:.2f}",
            delta=f"${bankroll_status['change']:.2f}"
        )
    
    with col2:
        change_pct = bankroll_status['change_pct']
        st.metric(
            "Change",
            f"{change_pct:.1f}%",
            delta=f"{change_pct:.1f}%"
        )
    
    with col3:
        # Calculate available for betting (current - active bets)
        active_total = active_bets['stake'].sum() if len(active_bets) > 0 else 0
        available = bankroll_status['current'] - active_total
        st.metric("Available", f"${available:.2f}")
    
    # Warning if bankroll is low
    if bankroll_status['current'] < 500:
        st.warning("⚠️ Bankroll is below $500. Consider reducing bet sizes or adding funds.")

except Exception as e:
    st.error(f"Could not load bankroll status: {str(e)}")

# Footer
st.divider()
st.caption("💡 Active bets update in real-time. Settled bets are recorded when matches complete.")
