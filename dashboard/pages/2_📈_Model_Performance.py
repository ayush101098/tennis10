"""
Page 2: Model Performance
=========================
Analytics dashboard for model performance and calibration
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dashboard.data_loader import (
    get_performance_metrics,
    get_pnl_history,
    get_calibration_data,
    get_settled_bets
)
from dashboard.components.charts import (
    create_pnl_chart,
    create_drawdown_chart,
    create_model_comparison_radar,
    create_calibration_plot,
    create_roi_by_confidence_chart
)
from dashboard.components.tables import render_model_performance_table

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.title("📈 Model Performance Analytics")

# Time period selector
period = st.selectbox(
    "Analysis Period",
    ['Last 7 Days', 'Last 30 Days', 'Last 3 Months', 'All Time'],
    index=1
)

# Load performance metrics
metrics = get_performance_metrics(period)

# Top-level metrics
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Overall ROI",
        f"{metrics['roi']:.2%}",
        delta=f"{metrics['roi_change']:.2%}" if metrics['roi_change'] != 0 else None
    )

with col2:
    st.metric(
        "Win Rate",
        f"{metrics['win_rate']:.1%}",
        delta=f"{metrics['win_rate_change']:.1%}" if metrics['win_rate_change'] != 0 else None
    )

with col3:
    st.metric(
        "Log Loss",
        f"{metrics['log_loss']:.4f}",
        delta=f"{metrics['log_loss_change']:.4f}" if metrics['log_loss_change'] != 0 else None,
        delta_color="inverse"
    )

with col4:
    st.metric(
        "Sharpe Ratio",
        f"{metrics['sharpe']:.2f}",
        delta=f"{metrics['sharpe_change']:.2f}" if metrics['sharpe_change'] != 0 else None
    )

st.divider()

# Charts tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Cumulative PnL",
    "Model Comparison",
    "Calibration",
    "ROI by Confidence"
])

with tab1:
    st.subheader("Bankroll Growth Over Time")
    
    # Get PnL history
    days_map = {
        'Last 7 Days': 7,
        'Last 30 Days': 30,
        'Last 3 Months': 90,
        'All Time': 36500
    }
    days = days_map.get(period, 30)
    
    pnl_data = get_pnl_history(days=days)
    
    if len(pnl_data) > 0:
        create_pnl_chart(pnl_data)
        
        st.divider()
        
        st.subheader("Drawdown Analysis")
        create_drawdown_chart(pnl_data)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_return = ((pnl_data.iloc[-1]['bankroll'] / pnl_data.iloc[0]['bankroll']) - 1) * 100
            st.metric("Total Return", f"{total_return:.2%}")
        
        with col2:
            max_bankroll = pnl_data['bankroll'].max()
            st.metric("Peak Bankroll", f"${max_bankroll:.2f}")
        
        with col3:
            # Calculate max drawdown
            pnl_data['running_max'] = pnl_data['bankroll'].cummax()
            pnl_data['drawdown'] = (pnl_data['bankroll'] - pnl_data['running_max']) / pnl_data['running_max']
            max_dd = pnl_data['drawdown'].min() * 100
            st.metric("Max Drawdown", f"{max_dd:.2%}")
    
    else:
        st.info("""
        **No PnL history available yet**
        
        Start placing bets to build your performance history. The system will automatically track:
        - Daily bankroll changes
        - Cumulative returns
        - Drawdown periods
        - ROI trends
        """)

with tab2:
    st.subheader("Model Performance Comparison")
    
    # Mock comparison data (replace with actual model performance)
    comparison_df = pd.DataFrame({
        'Model': ['Markov', 'Logistic Reg', 'Neural Net', 'Ensemble'],
        'ROI': [2.4, 4.2, 4.4, 4.8],
        'Win Rate': [66.5, 68.1, 68.9, 69.2],
        'Log Loss': [0.622, 0.613, 0.611, 0.609],
        'Sharpe': [0.45, 0.68, 0.71, 0.75]
    })
    
    render_model_performance_table(comparison_df)
    
    st.divider()
    
    # Radar chart visualization
    create_model_comparison_radar(comparison_df)
    
    # Insights
    st.info("""
    **Model Insights:**
    - **Ensemble** outperforms individual models across all metrics
    - **Neural Network** shows strong predictive accuracy (lowest log loss)
    - **Markov Model** provides baseline but underperforms on ROI
    - Combined approach reduces variance and improves Sharpe ratio
    """)

with tab3:
    st.subheader("Calibration Curve")
    
    st.write("**How well do our predicted probabilities match reality?**")
    
    calibration_data = get_calibration_data()
    
    if len(calibration_data) > 0:
        create_calibration_plot(calibration_data)
        
        # Calculate Brier score
        if 'predicted_prob' in calibration_data.columns and 'actual_freq' in calibration_data.columns:
            brier_score = ((calibration_data['predicted_prob'] - calibration_data['actual_freq']) ** 2).mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Brier Score", f"{brier_score:.4f}")
                st.caption("Lower is better (perfect = 0)")
            
            with col2:
                # Calibration assessment
                if brier_score < 0.05:
                    calibration_quality = "Excellent"
                    color = "🟢"
                elif brier_score < 0.10:
                    calibration_quality = "Good"
                    color = "🟡"
                else:
                    calibration_quality = "Needs Improvement"
                    color = "🔴"
                
                st.metric("Calibration Quality", f"{color} {calibration_quality}")
        
        st.divider()
        
        st.info("""
        **Perfect calibration:** Points fall on diagonal line
        - **Below diagonal:** Under-confident (predicted probabilities too low)
        - **Above diagonal:** Over-confident (predicted probabilities too high)
        
        Well-calibrated models are essential for accurate bet sizing and edge calculation.
        """)
    
    else:
        st.info("""
        **No calibration data available yet**
        
        Calibration analysis requires settled bets. Once you have bet history:
        - We'll compare predicted win probabilities to actual outcomes
        - Calibration curves help identify model bias
        - Well-calibrated models = better betting decisions
        """)

with tab4:
    st.subheader("ROI by Confidence Level")
    
    bets = get_settled_bets(days=days)
    
    if len(bets) > 0 and 'confidence' in bets.columns:
        create_roi_by_confidence_chart(bets)
        
        # Summary table
        st.subheader("Performance by Confidence")
        
        conf_summary = bets.groupby('confidence').agg({
            'profit': ['sum', 'mean'],
            'stake': 'sum',
            'result': lambda x: (x == 'won').sum() / len(x) * 100
        })
        
        conf_summary.columns = ['Total Profit', 'Avg Profit', 'Total Staked', 'Win Rate']
        conf_summary['ROI'] = (conf_summary['Total Profit'] / conf_summary['Total Staked'] * 100).fillna(0)
        
        conf_summary = conf_summary.reset_index()
        conf_summary['Total Profit'] = conf_summary['Total Profit'].apply(lambda x: f"${x:.2f}")
        conf_summary['Avg Profit'] = conf_summary['Avg Profit'].apply(lambda x: f"${x:.2f}")
        conf_summary['Total Staked'] = conf_summary['Total Staked'].apply(lambda x: f"${x:.2f}")
        conf_summary['Win Rate'] = conf_summary['Win Rate'].apply(lambda x: f"{x:.1f}%")
        conf_summary['ROI'] = conf_summary['ROI'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(conf_summary, use_container_width=True, hide_index=True)
        
        # Recommendations
        st.success("""
        **Strategy Recommendations:**
        - Focus on high-confidence bets for maximum ROI
        - Use medium-confidence for portfolio diversification
        - Consider skipping low-confidence opportunities
        """)
    
    else:
        st.info("No betting history available to analyze confidence levels")

# Footer
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.caption("📊 Performance metrics update daily")
    st.caption("🔄 Charts refresh with each new settled bet")

with col2:
    if st.button("🔄 Refresh Metrics", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
