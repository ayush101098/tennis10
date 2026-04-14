"""
🎾 DELTA-NEUTRAL BETTING SYSTEM - STREAMLIT DASHBOARD
=====================================================
Live implementation of Tennis Delta-Neutral Strategy

Run: streamlit run delta_neutral_dashboard.py --server.port 8504
"""

import streamlit as st
import pandas as pd
import numpy as np
from delta_neutral_system import (
    DeltaNeutralBettingSystem, GameState, PositionStatus, Signal
)
import json
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🎾 Delta-Neutral Betting",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS ====================
st.markdown("""
<style>
    .main { background-color: #0a0e27; color: #e0e0e0; }
    .metric-box {
        background: #151932;
        border: 2px solid #00ff88;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .signal-entry { color: #00ff88; font-weight: bold; }
    .signal-hedge { color: #ffc107; font-weight: bold; }
    .signal-exit { color: #ff6b6b; font-weight: bold; }
    .delta-neutral { color: #00ff88; }
    .delta-aggressive { color: #ff6b6b; }
    .code-block {
        background: #1a1f3a;
        border-left: 3px solid #00ff88;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'system' not in st.session_state:
    st.session_state.system = DeltaNeutralBettingSystem()
    st.session_state.signals_log = []

# ==================== SIDEBAR ====================
st.sidebar.title("🎯 System Control")

st.sidebar.markdown("### Match Setup")
server_pts = st.sidebar.slider("Server Points", 0, 5, 0)
returner_pts = st.sidebar.slider("Returner Points", 0, 5, 0)

server_games = st.sidebar.slider("Server Games", 0, 6, 0, help="Games in current set")
returner_games = st.sidebar.slider("Returner Games", 0, 6, 0, help="Games in current set")

st.sidebar.markdown("### Live Odds")
col1, col2 = st.sidebar.columns(2)
with col1:
    break_odds = st.number_input("Break Odds", 1.01, 10.0, 3.20, 0.01)
with col2:
    hold_odds = st.number_input("Hold Odds", 1.01, 10.0, 1.80, 0.01)

# Update button
if st.sidebar.button("📊 Update Score & Odds", use_container_width=True):
    # Update score
    st.session_state.system.update_score(
        server_pts, returner_pts, server_games, returner_games
    )
    
    # Process odds
    signal = st.session_state.system.process_odds(break_odds, hold_odds)
    
    # Log signal
    st.session_state.signals_log.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "score": f"{server_pts}-{returner_pts}",
        "break_odds": break_odds,
        "hold_odds": hold_odds,
        "signal": signal.name if signal else "HOLD",
        "delta": st.session_state.system.betting_state.delta,
        "position": st.session_state.system.betting_state.position_status.value
    })

# Settle game button
st.sidebar.markdown("### Settle Game")
settle_col1, settle_col2 = st.sidebar.columns(2)
with settle_col1:
    if st.button("✅ Break Occurred", use_container_width=True):
        pnl = st.session_state.system.settle_game("BREAK")
        st.session_state.signals_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "event": "SETTLEMENT",
            "outcome": "BREAK",
            "pnl": pnl["total_pnl"],
            "roi": pnl["roi"]
        })

with settle_col2:
    if st.button("✅ Hold Occurred", use_container_width=True):
        pnl = st.session_state.system.settle_game("HOLD")
        st.session_state.signals_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "event": "SETTLEMENT",
            "outcome": "HOLD",
            "pnl": pnl["total_pnl"],
            "roi": pnl["roi"]
        })

# ==================== MAIN DISPLAY ====================

st.title("🎾 Delta-Neutral Betting System")
st.markdown("**Exact IF-THEN Rulebook Implementation**")
st.markdown("---")

# Top metrics
col1, col2, col3, col4 = st.columns(4)

status = st.session_state.system.get_status_report()

with col1:
    st.metric(label="Game State", value=status["current_state"])

with col2:
    st.metric(label="Score", value=status["game_score"])

with col3:
    delta_color = "🟢" if status["delta"] == 0.0 else "🔴"
    st.metric(
        label="Delta",
        value=f"{status['delta']:.1f}",
        delta=f"{status['delta_explanation']}"
    )

with col4:
    pos_status = status["position_status"]
    st.metric(label="Position", value=pos_status)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Live Positions", "🛡️ Hedging Logic", "📈 P&L", "📋 Signals Log"]
)

# ==================== TAB 1: LIVE POSITIONS ====================
with tab1:
    st.subheader("Current Positions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Account A: AGGRESSOR (Break Bet)")
        if status["position_a"]:
            st.markdown(f"""
            <div class="metric-box">
                <b>Status:</b> {'🟢 ACTIVE' if status['position_a']['active'] else '⚫ CLOSED'}<br>
                <b>Stake:</b> ${status['position_a']['stake']:.2f}<br>
                <b>Odds:</b> {status['position_a']['odds']:.2f}<br>
                <b>Exposure:</b> ${(status['position_a']['stake'] * (status['position_a']['odds'] - 1)):.2f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No active break position")
    
    with col2:
        st.markdown("#### Account B: STABILIZER (Hold Hedge)")
        if status["position_b"]:
            st.markdown(f"""
            <div class="metric-box">
                <b>Status:</b> {'🟢 ACTIVE' if status['position_b']['active'] else '⚫ CLOSED'}<br>
                <b>Stake:</b> ${status['position_b']['stake']:.2f}<br>
                <b>Odds:</b> {status['position_b']['odds']:.2f}<br>
                <b>Type:</b> {status['position_b']['type']} HEDGE<br>
                <b>Offset:</b> ${(status['position_b']['stake'] * (status['position_b']['odds'] - 1)):.2f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No active hedge position")
    
    # Delta explanation
    st.markdown("---")
    st.markdown("### Delta Status")
    
    delta_meanings = {
        1.0: "🔴 **FULL AGGRESSION** - Only Account A active (break exposure)",
        0.5: "🟡 **BALANCED** - 50% hedged (retains convexity)",
        0.0: "🟢 **NEUTRAL** - Fully hedged (both positions active)",
        -0.5: "🔵 **RARE** - Reverse hedge (exotic strategy)"
    }
    
    st.markdown(delta_meanings.get(status['delta'], "Unknown state"))
    
    # Net exposure
    if status["position_a"] and status["position_b"]:
        exposure_a = status['position_a']['stake'] * status['position_a']['odds']
        exposure_b = status['position_b']['stake']
        net_exposure = exposure_a - exposure_b
        
        st.markdown(f"""
        **Net Exposure Calculation:**
        - Break exposure (A): ${exposure_a:.2f}
        - Hedge offset (B): ${exposure_b:.2f}
        - **Net: ${net_exposure:+.2f}**
        """)

# ==================== TAB 2: HEDGING LOGIC ====================
with tab2:
    st.subheader("Hedge Trigger Rulebook")
    
    st.markdown("""
    ### H1: Server Dominance Hedge
    **Trigger:** State = S6 (server wins 3 straight points) AND hold odds ≤ 1.25
    
    **Action:** Full 100% hedge on hold bet
    
    **Formula:** $S_2 = S_1 × O_1 / O_2$
    """)
    
    st.markdown("""
    ### H2: Deuce Neutralization
    **Trigger:** State = S5 (Deuce reached)
    
    **Action:** Full 100% hedge on hold bet
    
    **Why:** Highest EV hedge state - deuce is natural breakpoint
    """)
    
    st.markdown("""
    ### H3: Missed Break Point Partial Hedge
    **Trigger:** State = S7 (break point saved) AND break odds jump ≥40%
    
    **Action:** 50% partial hedge
    
    **Formula:** $S_2 = 0.5 × S_1 × O_1 / O_2$
    
    **Benefit:** Retains convexity while reducing risk
    """)
    
    st.markdown("---")
    
    # Current state info
    st.markdown("### Current Game State")
    state = status["current_state"]
    
    state_descriptions = {
        "0-0 (start)": "Game starting - ready for E1 entry",
        "15-0 or 0-15": "Early points - no action",
        "30-15 or 15-30": "Mid-game points - monitor",
        "30-30 (deuce area)": "Approaching deuce - H triggers likely",
        "Deuce (40-40, Ad-In, Ad-Out)": "🎯 H2 HIGHEST EV STATE - ideal for hedge",
        "Server wins 3 straight": "🚨 H1 TRIGGER - server dominance",
        "Break Point (30-40, 15-40, Ad-Out)": "Critical state - H3 possible",
        "Break Point Saved": "H3 evaluation window",
        "Break Occurs": "Position settled"
    }
    
    st.info(f"📍 **{state}**\n\n{state_descriptions.get(state, 'Monitoring...')}")

# ==================== TAB 3: P&L TRACKING ====================
with tab3:
    st.subheader("Profit & Loss Summary")
    
    pnl_data = status["pnl_history"]
    cumulative_pnl = status["cumulative_pnl"]
    
    if pnl_data:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Games Settled", len(pnl_data))
        with col2:
            wins = len([p for p in pnl_data if p["total_pnl"] > 0])
            st.metric("Winning Games", wins)
        with col3:
            st.metric("Cumulative P&L", f"${cumulative_pnl:+.2f}")
        
        # P&L table
        st.markdown("### Game-by-Game P&L")
        df_pnl = pd.DataFrame(pnl_data)
        
        # Format for display
        display_cols = {
            "outcome": "Outcome",
            "pnl_a": "Account A (Break)",
            "pnl_b": "Account B (Hold)",
            "total_pnl": "Total P&L",
            "roi": "ROI %"
        }
        
        df_display = df_pnl[list(display_cols.keys())].rename(columns=display_cols)
        
        # Color formatting
        def color_pnl(val):
            if isinstance(val, (int, float)):
                color = "green" if val > 0 else "red" if val < 0 else "gray"
                return f"color: {color}"
            return ""
        
        st.dataframe(
            df_display.style.applymap(color_pnl, subset=["Account A (Break)", "Account B (Hold)", "Total P&L", "ROI %"]),
            use_container_width=True
        )
        
        # Chart
        df_pnl["cumulative"] = df_pnl["total_pnl"].cumsum()
        st.line_chart(df_pnl["cumulative"], use_container_width=True, height=300)
    else:
        st.info("📊 No games settled yet - P&L will appear here")

# ==================== TAB 4: SIGNALS LOG ====================
with tab4:
    st.subheader("System Signals History")
    
    if st.session_state.signals_log:
        df_signals = pd.DataFrame(st.session_state.signals_log)
        st.dataframe(df_signals, use_container_width=True, hide_index=True)
    else:
        st.info("📋 Signal log empty - signals will appear as system processes odds")

# ==================== BOTTOM: RULES REFERENCE ====================
st.markdown("---")

with st.expander("📖 **System Reference Guide**"):
    st.markdown("""
    ### ENTRY RULES (Account A)
    **Rule E1 — Initial Long Break Entry**
    
    IF Server starts a game AND Break odds ∈ [2.8, 3.4] AND No break in set
    THEN Place S₁ on Break, Delta = +1
    
    ---
    
    ### HEDGE RULES (Account B)
    **Rule H1 — Server Dominance Hedge**
    - IF State = S6 AND Hold odds ≤ 1.25 → Full hedge
    
    **Rule H2 — Deuce Neutralization**
    - IF State = S5 → Full hedge (highest EV)
    
    **Rule H3 — Missed Break Point Partial Hedge**
    - IF State = S7 AND Break odds +40% → 50% hedge
    
    ---
    
    ### RISK CONTROL
    **Rule R1 — Odds Explosion Emergency Exit**
    - IF Break odds > 8.0 AND no BP → Hedge immediately
    
    ---
    
    ### POSITION SIZING
    - **Full Hedge:** $S_2 = S_1 × O_1 / O_2$
    - **Partial Hedge:** $S_2 = 0.5 × S_1 × O_1 / O_2$
    
    ---
    
    ### DELTA TRACKING
    | State | Delta | Meaning |
    |-------|-------|---------|
    | Entry | +1.0 | Full break exposure |
    | Partial Hedge | +0.5 | 50% hedged, retains convexity |
    | Full Hedge | 0.0 | Delta-neutral, balanced |
    | Reverse | -0.5 | Exotic (rare) |
    """)

st.markdown("---")
st.caption("🎾 Delta-Neutral Betting System | Deterministic IF-THEN Rulebook | Real-time Implementation")
