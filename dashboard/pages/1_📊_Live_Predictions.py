"""
Page 1: Live Predictions
========================
Display upcoming matches with predictions and betting recommendations
"""

import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dashboard.data_loader import get_upcoming_matches, get_predictions
from dashboard.components.match_card import render_match_card, render_detailed_match_view, render_compact_match_row
from dashboard.components.charts import create_edge_distribution_chart
from src.live_predictions.predictor import LivePredictor

st.set_page_config(page_title="Live Predictions", page_icon="📊", layout="wide")

st.title("🎾 Upcoming Matches & Predictions")

# Time filter tabs
time_tab1, time_tab2, time_tab3 = st.tabs([
    "Next 24 Hours",
    "Next 48 Hours",
    "Next Week"
])

with time_tab1:
    hours = 24
    st.subheader(f"Matches in Next {hours} Hours")
    
    # Get matches and predictions
    matches = get_upcoming_matches(hours=hours)
    
    if len(matches) == 0:
        st.info("""
        **No matches scheduled in the next 24 hours**
        
        This is normal during off-season periods. The Australian Open typically starts around January 12-15.
        
        **What to do:**
        - Check back when tennis season resumes
        - Run the automated scheduler to monitor for new matches:
          ```bash
          source setup_env.sh && python src/live_data/scheduler.py
          ```
        - Or manually check for matches:
          ```bash
          source setup_env.sh && python src/live_predictions/predictor.py
          ```
        """)
    else:
        # Get predictions for these matches
        match_ids = matches['match_id'].tolist()
        predictions = get_predictions(match_ids=match_ids)
        
        # If no predictions exist, generate them
        if len(predictions) == 0:
            with st.spinner("Generating predictions..."):
                try:
                    predictor = LivePredictor(bankroll=1000)
                    pred_df, bets_df = predictor.predict_upcoming_matches(days_ahead=1)
                    
                    if len(pred_df) > 0:
                        predictions = pred_df
                        st.success(f"✅ Generated predictions for {len(predictions)} matches")
                    else:
                        st.warning("Could not generate predictions")
                        predictions = matches  # Use matches without predictions
                
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                    predictions = matches
        
        # Apply filters from sidebar
        if 'filters' in st.session_state:
            filters = st.session_state.filters
            
            if 'confidence' in predictions.columns and filters['confidence']:
                predictions = predictions[predictions['confidence'].isin(
                    [c.lower() for c in filters['confidence']]
                )]
            
            if 'surface' in predictions.columns and filters['surface']:
                predictions = predictions[predictions['surface'].isin(filters['surface'])]
            
            if 'edge' in predictions.columns:
                predictions = predictions[predictions['edge'] >= filters['min_edge']]
        
        # Action buttons at top
        st.subheader("Recommended Actions")
        
        high_confidence_bets = predictions[
            (predictions.get('confidence', '') == 'high') &
            (predictions.get('action', '').str.startswith('bet_', na=False))
        ] if 'confidence' in predictions.columns and 'action' in predictions.columns else pd.DataFrame()
        
        if len(high_confidence_bets) > 0:
            st.success(f"✅ {len(high_confidence_bets)} high-confidence bets available")
            
            for idx, bet in high_confidence_bets.iterrows():
                with st.expander(
                    f"🎯 {bet['player1_name']} vs {bet['player2_name']} - "
                    f"Edge: {bet.get('edge', 0):.1%} | Stake: ${bet.get('recommended_stake', 0):.0f}",
                    expanded=True
                ):
                    render_match_card(bet)
        else:
            st.info("No high-confidence bets available right now. Check settings or wait for better opportunities.")
        
        st.divider()
        
        # All matches section
        st.subheader("All Upcoming Matches")
        
        # Display mode toggle
        view_mode = st.radio(
            "Display Mode",
            ['Cards', 'Table', 'Detailed'],
            horizontal=True
        )
        
        if view_mode == 'Cards':
            # Grid of match cards
            cols = st.columns(2)
            for idx, (_, match) in enumerate(predictions.iterrows()):
                with cols[idx % 2]:
                    render_match_card(match)
        
        elif view_mode == 'Table':
            # Compact table view
            from dashboard.components.tables import render_predictions_table
            render_predictions_table(predictions)
        
        else:  # Detailed view
            for _, match in predictions.iterrows():
                render_detailed_match_view(match)
                st.divider()
        
        # Edge distribution chart
        if 'edge' in predictions.columns:
            st.subheader("📊 Edge Distribution")
            create_edge_distribution_chart(predictions)

with time_tab2:
    hours = 48
    st.subheader(f"Matches in Next {hours} Hours")
    
    matches = get_upcoming_matches(hours=hours)
    
    if len(matches) == 0:
        st.info("No matches scheduled in the next 48 hours")
    else:
        from dashboard.components.tables import render_upcoming_matches_table
        render_upcoming_matches_table(matches)
        
        st.info(f"📅 Found {len(matches)} matches. Switch to 'Next 24 Hours' for detailed predictions.")

with time_tab3:
    hours = 168  # 7 days
    st.subheader(f"Matches in Next Week")
    
    matches = get_upcoming_matches(hours=hours)
    
    if len(matches) == 0:
        st.info("No matches scheduled in the next week")
    else:
        from dashboard.components.tables import render_upcoming_matches_table
        render_upcoming_matches_table(matches)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Matches", len(matches))
        
        with col2:
            if 'surface' in matches.columns:
                most_common_surface = matches['surface'].mode()[0] if len(matches) > 0 else "N/A"
                st.metric("Most Common Surface", most_common_surface)
        
        with col3:
            if 'tournament' in matches.columns:
                tournaments = matches['tournament'].nunique()
                st.metric("Tournaments", tournaments)

# Footer with refresh info
st.divider()
st.caption("💡 Predictions are updated every 30 minutes. Odds are updated every 15 minutes.")
st.caption("⚙️ Adjust filters in the sidebar to refine recommendations.")
