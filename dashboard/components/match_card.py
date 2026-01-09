"""
Match Card Component - Beautiful match prediction cards
========================================================
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime


def render_match_card(match):
    """
    Render a beautiful match prediction card
    
    Args:
        match: Dict or Series with match data containing:
            - player1_name, player2_name
            - tournament, surface, scheduled_time
            - ensemble_p1_win, model_agreement
            - best_p1_odds, edge
            - action, confidence
            - recommended_stake (optional)
    """
    
    # Determine card color based on action
    action = match.get('action', 'watch')
    if action and ('bet' in str(action).lower()):
        border_color = "green"
        emoji = "💚"
    else:
        border_color = "gray"
        emoji = "⚪"
    
    with st.container(border=True):
        # Match header
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {match['player1_name']} vs {match['player2_name']}")
            
            tournament = match.get('tournament', 'Unknown Tournament')
            surface = match.get('surface', 'Unknown')
            
            if 'scheduled_time' in match:
                if isinstance(match['scheduled_time'], str):
                    time_str = match['scheduled_time']
                else:
                    time_str = match['scheduled_time'].strftime('%b %d, %H:%M')
            else:
                time_str = "TBD"
            
            st.caption(f"{tournament} | {surface} | {time_str}")
        
        with col2:
            if action and ('bet' in str(action).lower()):
                st.markdown(f"## {emoji}")
                confidence = match.get('confidence', 'medium').upper()
                st.markdown(f"**{confidence}**")
        
        # Probability visualization
        st.markdown("#### Win Probability")
        
        prob_p1 = match.get('ensemble_p1_win', 0.5)
        prob_p2 = 1 - prob_p1
        
        # Create probability bar
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=['Probability'],
            x=[prob_p1 * 100],
            name=match['player1_name'],
            orientation='h',
            marker=dict(color='#1f77b4'),
            text=f"{prob_p1:.1%}",
            textposition='inside',
            textfont=dict(color='white', size=14)
        ))
        
        fig.add_trace(go.Bar(
            y=['Probability'],
            x=[prob_p2 * 100],
            name=match['player2_name'],
            orientation='h',
            marker=dict(color='#ff7f0e'),
            text=f"{prob_p2:.1%}",
            textposition='inside',
            textfont=dict(color='white', size=14)
        ))
        
        fig.update_layout(
            barmode='stack',
            height=80,
            showlegend=False,
            xaxis={'visible': False},
            yaxis={'visible': False},
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model agreement
        agreement = match.get('model_agreement', 0)
        agreement_pct = agreement * 100 if agreement <= 1 else agreement
        
        st.progress(
            min(agreement, 1.0), 
            text=f"Model Agreement: {agreement_pct:.0f}%"
        )
        
        # Odds and edge
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_odds = match.get('best_p1_odds', 0)
            st.metric("Best Odds", f"{best_odds:.2f}" if best_odds > 0 else "N/A")
        
        with col2:
            if best_odds > 0:
                implied_prob = 1 / best_odds
                st.metric("Implied Prob", f"{implied_prob:.1%}")
            else:
                st.metric("Implied Prob", "N/A")
        
        with col3:
            edge = match.get('edge', 0)
            edge_color = "normal" if edge < 0.05 else "inverse"
            st.metric(
                "Edge", 
                f"{edge:.1%}",
                delta=f"{edge:.1%}",
                delta_color=edge_color
            )
        
        # Bet recommendation
        if action and ('bet' in str(action).lower()):
            recommended_stake = match.get('recommended_stake', 0)
            
            # Determine which player to bet on
            if 'player1' in str(action).lower():
                bet_player = match['player1_name']
            else:
                bet_player = match['player2_name']
            
            st.success(
                f"**Recommended:** Bet ${recommended_stake:.0f} on {bet_player}"
            )
            
            # One-click bet button
            match_id = match.get('match_id', f"{match['player1_name']}_{match['player2_name']}")
            
            if st.button("Place Bet", key=f"bet_{match_id}", use_container_width=True):
                st.session_state.pending_bet = match
                st.switch_page("pages/3_💰_Betting_History.py")
        else:
            reason = match.get('reason', 'Watching - No bet recommended')
            st.info(f"**Status:** {reason}")


def render_detailed_match_view(match):
    """
    Expanded view with all model predictions and historical H2H
    
    Args:
        match: Dict or Series with comprehensive match data
    """
    st.markdown(f"### {match['player1_name']} vs {match['player2_name']}")
    
    # Create tabs for different information
    tab1, tab2, tab3, tab4 = st.tabs([
        "Predictions", 
        "Head-to-Head", 
        "Recent Form",
        "Odds Movement"
    ])
    
    with tab1:
        # Show predictions from all models
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            markov = match.get('markov_p1_win', match.get('ensemble_p1_win', 0.5))
            st.metric("Markov Model", f"{markov:.1%}")
        
        with col2:
            lr = match.get('lr_p1_win', match.get('ensemble_p1_win', 0.5))
            st.metric("Logistic Reg", f"{lr:.1%}")
        
        with col3:
            nn = match.get('nn_p1_win', match.get('ensemble_p1_win', 0.5))
            st.metric("Neural Net", f"{nn:.1%}")
        
        with col4:
            ensemble = match.get('ensemble_p1_win', 0.5)
            st.metric("Ensemble", f"{ensemble:.1%}", 
                     delta="Final", delta_color="off")
        
        # Radar chart comparing models
        create_model_comparison_chart(match)
    
    with tab2:
        # Head-to-head history (mock data for now)
        st.info("Head-to-head data will be loaded from historical match database")
        
        st.write("**Overall:** Data loading...")
        
        # Mock recent matches
        st.subheader("Recent Encounters")
        st.caption("- 2025-12-15: Player 1 won 6-4, 7-5")
        st.caption("- 2025-10-03: Player 2 won 6-3, 3-6, 7-6")
        st.caption("- 2025-08-20: Player 1 won 7-6, 6-4")
    
    with tab3:
        # Recent form for both players
        st.info("Recent form data will be loaded from historical database")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(match['player1_name'])
            st.caption("Last 5 matches: W-W-L-W-W (80% win rate)")
        
        with col2:
            st.subheader(match['player2_name'])
            st.caption("Last 5 matches: W-L-W-W-L (60% win rate)")
    
    with tab4:
        # Odds movement chart
        st.info("Odds movement tracking coming soon")
        st.caption("Will display odds changes over time from multiple bookmakers")


def create_model_comparison_chart(match):
    """Create radar chart comparing model predictions"""
    
    markov = match.get('markov_p1_win', match.get('ensemble_p1_win', 0.5))
    lr = match.get('lr_p1_win', match.get('ensemble_p1_win', 0.5))
    nn = match.get('nn_p1_win', match.get('ensemble_p1_win', 0.5))
    
    categories = ['Markov', 'Logistic Reg', 'Neural Net']
    values = [markov * 100, lr * 100, nn * 100]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name='P1 Win Probability',
        marker=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_compact_match_row(match, index):
    """
    Render a compact single-row match view for tables
    
    Args:
        match: Match data
        index: Row index for unique keys
    """
    
    col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
    
    with col1:
        st.write(f"**{match['player1_name']}** vs **{match['player2_name']}**")
        st.caption(f"{match.get('tournament', 'Unknown')} | {match.get('surface', 'Unknown')}")
    
    with col2:
        prob = match.get('ensemble_p1_win', 0.5)
        st.metric("Win Prob", f"{prob:.1%}", label_visibility="collapsed")
    
    with col3:
        odds = match.get('best_p1_odds', 0)
        st.metric("Odds", f"{odds:.2f}" if odds > 0 else "N/A", label_visibility="collapsed")
    
    with col4:
        edge = match.get('edge', 0)
        st.metric("Edge", f"{edge:.1%}", label_visibility="collapsed")
    
    with col5:
        if match.get('action', '').startswith('bet'):
            if st.button("Bet", key=f"quick_bet_{index}", use_container_width=True):
                st.session_state.pending_bet = match
                st.switch_page("pages/3_💰_Betting_History.py")
