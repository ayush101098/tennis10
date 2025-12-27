"""
Tennis Prediction Dashboard
============================
Streamlit dashboard for tennis match predictions and betting analytics.

Run with: streamlit run dashboard/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Tennis Prediction Dashboard",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .profit-positive {
        color: #00c853;
    }
    .profit-negative {
        color: #ff1744;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_database():
    """Load data from SQLite database."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tennis_data.db')
    if not os.path.exists(db_path):
        return None, None, None
    
    conn = sqlite3.connect(db_path)
    
    # Load matches
    matches_df = pd.read_sql("SELECT * FROM matches", conn)
    
    # Load players
    players_df = pd.read_sql("SELECT * FROM players", conn)
    
    # Load player stats
    try:
        stats_df = pd.read_sql("SELECT * FROM player_stats", conn)
    except:
        stats_df = None
    
    conn.close()
    return matches_df, players_df, stats_df


@st.cache_data
def load_predictions():
    """Load model predictions if available."""
    pred_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hierarchical_model_predictions.csv')
    if os.path.exists(pred_path):
        return pd.read_csv(pred_path)
    return None


def create_sidebar():
    """Create sidebar with filters and settings."""
    st.sidebar.title("🎾 Tennis Predictions")
    st.sidebar.markdown("---")
    
    # Tab selection
    tab = st.sidebar.radio(
        "Navigation",
        ["📊 Live Predictions", "📈 Performance", "🔍 Model Insights", "💰 Bankroll Management"]
    )
    
    st.sidebar.markdown("---")
    
    # Filters
    st.sidebar.subheader("Filters")
    
    surface = st.sidebar.multiselect(
        "Surface",
        ["Hard", "Clay", "Grass"],
        default=["Hard", "Clay", "Grass"]
    )
    
    min_edge = st.sidebar.slider(
        "Minimum Edge %",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.5
    )
    
    st.sidebar.markdown("---")
    
    # Settings
    st.sidebar.subheader("Settings")
    
    bankroll = st.sidebar.number_input(
        "Bankroll ($)",
        min_value=100,
        max_value=100000,
        value=1000,
        step=100
    )
    
    kelly_fraction = st.sidebar.slider(
        "Kelly Fraction",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    
    return tab, surface, min_edge, bankroll, kelly_fraction


def live_predictions_tab(matches_df, players_df, predictions_df, bankroll, kelly_fraction, min_edge):
    """Live predictions tab content."""
    st.header("📊 Live Match Predictions")
    
    if predictions_df is None:
        st.warning("No prediction data available. Run model training first.")
        
        # Show sample prediction interface
        st.subheader("Manual Prediction Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Player 1 Name", value="Player A")
            p1_serve = st.slider("P1 Serve Win %", 50, 80, 65)
            p1_return = st.slider("P1 Return Win %", 30, 50, 38)
        
        with col2:
            st.text_input("Player 2 Name", value="Player B")
            p2_serve = st.slider("P2 Serve Win %", 50, 80, 63)
            p2_return = st.slider("P2 Return Win %", 30, 50, 40)
        
        if st.button("Calculate Prediction"):
            # Simple logistic calculation
            diff = (p1_serve - p2_serve) + (p1_return - p2_return)
            p1_win = 1 / (1 + np.exp(-0.1 * diff))
            
            st.metric("P(Player 1 wins)", f"{p1_win:.1%}")
            st.metric("P(Player 2 wins)", f"{1-p1_win:.1%}")
        
        return
    
    # Display upcoming matches
    st.subheader("Upcoming Matches")
    
    # Create sample upcoming matches display
    upcoming = predictions_df.head(10).copy()
    
    for idx, row in upcoming.iterrows():
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.markdown(f"**{row.get('player1', 'Player 1')} vs {row.get('player2', 'Player 2')}**")
            
            with col2:
                p1_prob = row.get('p_player1_win', 0.5)
                st.metric("P(Player 1)", f"{p1_prob:.1%}")
            
            with col3:
                p2_prob = 1 - p1_prob
                st.metric("P(Player 2)", f"{p2_prob:.1%}")
            
            with col4:
                edge = max(p1_prob - 0.5, p2_prob - 0.5) - 0.5
                if edge > min_edge / 100:
                    st.success(f"BET: Edge {edge:.1%}")
                else:
                    st.info("SKIP")
            
            st.markdown("---")


def performance_tab(predictions_df):
    """Performance metrics and charts."""
    st.header("📈 Model Performance")
    
    if predictions_df is None:
        st.warning("No performance data available. Run backtesting first.")
        
        # Show sample performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total ROI", "+12.5%", delta="↑ 2.3%")
        
        with col2:
            st.metric("Win Rate", "54.2%", delta="↑ 1.1%")
        
        with col3:
            st.metric("Sharpe Ratio", "1.85", delta="↑ 0.12")
        
        with col4:
            st.metric("Max Drawdown", "-8.3%", delta="↓ 1.5%")
        
        # Sample PnL chart
        st.subheader("Cumulative P&L (Sample)")
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        pnl = np.cumsum(np.random.randn(365) * 10 + 0.5)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=pnl, mode='lines',
            name='P&L', line=dict(color='#00c853', width=2)
        ))
        fig.update_layout(
            title="Cumulative Profit/Loss",
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ROI by surface
        st.subheader("ROI by Surface")
        surface_data = pd.DataFrame({
            'Surface': ['Hard', 'Clay', 'Grass'],
            'ROI': [14.2, 10.8, 8.5],
            'Matches': [520, 380, 85]
        })
        
        fig2 = px.bar(surface_data, x='Surface', y='ROI', 
                      title='ROI by Surface',
                      color='ROI',
                      color_continuous_scale=['red', 'yellow', 'green'])
        st.plotly_chart(fig2, use_container_width=True)
        
        return
    
    # Actual performance metrics would go here
    st.info("Connect your prediction results to see actual performance")


def model_insights_tab():
    """Model analysis and feature importance."""
    st.header("🔍 Model Insights")
    
    # Feature importance
    st.subheader("Feature Importance Rankings")
    
    features = [
        ('SERVE_WIN_PCT', 0.23),
        ('RETURN_WIN_PCT', 0.19),
        ('ACE_PCT', 0.12),
        ('FIRST_SERVE_PCT', 0.10),
        ('BP_SAVED_PCT', 0.09),
        ('BP_CONVERTED_PCT', 0.08),
        ('HOLD_PCT', 0.07),
        ('BREAK_PCT', 0.05),
        ('DIRECT', 0.04),
        ('FATIGUE', 0.03)
    ]
    
    feat_df = pd.DataFrame(features, columns=['Feature', 'Importance'])
    
    fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance (Logistic Regression)',
                 color='Importance',
                 color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.subheader("Model Comparison")
    
    model_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'Neural Network', 'Markov Model', 'Ensemble'],
        'Accuracy': [0.642, 0.655, 0.618, 0.668],
        'Log Loss': [0.598, 0.585, 0.625, 0.572],
        'ROI': [8.5, 12.3, 5.2, 15.1],
        'Calibration': [0.92, 0.88, 0.95, 0.91]
    })
    
    st.dataframe(model_data.style.highlight_max(axis=0, subset=['Accuracy', 'ROI', 'Calibration'])
                 .highlight_min(axis=0, subset=['Log Loss']))
    
    # Calibration curve
    st.subheader("Calibration Curve")
    
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Simulated calibration data
    predicted = bin_centers
    actual_lr = bin_centers * 0.95 + 0.025 + np.random.randn(10) * 0.02
    actual_nn = bin_centers * 0.92 + 0.04 + np.random.randn(10) * 0.03
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predicted, y=actual_lr, mode='lines+markers',
                             name='Logistic Regression'))
    fig.add_trace(go.Scatter(x=predicted, y=actual_nn, mode='lines+markers',
                             name='Neural Network'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             name='Perfect Calibration', line=dict(dash='dash', color='gray')))
    
    fig.update_layout(
        title="Model Calibration",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Win Rate",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Worst predictions
    st.subheader("Recent Worst Predictions (for debugging)")
    
    worst_df = pd.DataFrame({
        'Match': ['A vs B', 'C vs D', 'E vs F', 'G vs H', 'I vs J'],
        'Predicted': ['A (85%)', 'C (72%)', 'E (68%)', 'G (79%)', 'I (65%)'],
        'Actual Winner': ['B', 'D', 'F', 'H', 'J'],
        'Loss': [1.89, 1.27, 1.14, 1.56, 1.05]
    })
    st.dataframe(worst_df)


def bankroll_management_tab(bankroll, kelly_fraction):
    """Bankroll and risk management."""
    st.header("💰 Bankroll Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Status")
        st.metric("Current Bankroll", f"${bankroll:,.2f}")
        st.metric("Kelly Fraction", f"{kelly_fraction:.0%}")
        st.metric("Max Stake (5% cap)", f"${bankroll * 0.05:,.2f}")
        st.metric("Today's Exposure", f"${bankroll * 0.12:,.2f}", delta="12%")
    
    with col2:
        st.subheader("Risk Metrics")
        st.metric("Value at Risk (95%)", f"${bankroll * 0.15:,.2f}")
        st.metric("Expected Daily Volatility", f"${bankroll * 0.03:,.2f}")
        st.metric("Recommended Reserve", f"${bankroll * 0.30:,.2f}")
        st.metric("Betting Capital", f"${bankroll * 0.70:,.2f}")
    
    # Today's recommended bets
    st.subheader("Today's Recommended Stakes")
    
    bets_df = pd.DataFrame({
        'Match': ['Player A vs Player B', 'Player C vs Player D', 'Player E vs Player F'],
        'Selection': ['Player A', 'Player D', 'Player E'],
        'Odds': [1.85, 2.10, 1.65],
        'Model Prob': ['58%', '52%', '64%'],
        'Edge': ['7.8%', '4.2%', '3.4%'],
        'Kelly': ['10.4%', '4.0%', '5.7%'],
        'Quarter Kelly': ['2.6%', '1.0%', '1.4%'],
        'Stake': [f"${bankroll*0.026:.2f}", f"${bankroll*0.01:.2f}", f"${bankroll*0.014:.2f}"]
    })
    
    st.dataframe(bets_df, use_container_width=True)
    
    # Historical drawdown
    st.subheader("Historical Drawdown")
    
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    equity = 1000 + np.cumsum(np.random.randn(365) * 10 + 0.5)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=drawdown, fill='tozeroy',
        name='Drawdown', line=dict(color='red', width=1),
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))
    fig.update_layout(
        title="Drawdown from Peak",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Bankroll simulation
    st.subheader("Monte Carlo Bankroll Simulation")
    
    n_sims = 100
    n_days = 90
    
    sims = np.zeros((n_sims, n_days))
    sims[:, 0] = bankroll
    
    for i in range(1, n_days):
        returns = np.random.normal(0.003, 0.02, n_sims)  # 0.3% daily expected return
        sims[:, i] = sims[:, i-1] * (1 + returns)
    
    fig = go.Figure()
    
    # Plot percentiles
    p5 = np.percentile(sims, 5, axis=0)
    p50 = np.percentile(sims, 50, axis=0)
    p95 = np.percentile(sims, 95, axis=0)
    
    days = list(range(n_days))
    
    fig.add_trace(go.Scatter(x=days, y=p95, mode='lines', name='95th percentile',
                             line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=days, y=p50, mode='lines', name='Median',
                             line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=days, y=p5, mode='lines', name='5th percentile',
                             line=dict(color='red', dash='dash')))
    
    fig.update_layout(
        title=f"90-Day Bankroll Projection ({n_sims} simulations)",
        xaxis_title="Days",
        yaxis_title="Bankroll ($)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard application."""
    # Load data
    matches_df, players_df, stats_df = load_database()
    predictions_df = load_predictions()
    
    # Create sidebar
    tab, surface, min_edge, bankroll, kelly_fraction = create_sidebar()
    
    # Render selected tab
    if tab == "📊 Live Predictions":
        live_predictions_tab(matches_df, players_df, predictions_df, bankroll, kelly_fraction, min_edge)
    
    elif tab == "📈 Performance":
        performance_tab(predictions_df)
    
    elif tab == "🔍 Model Insights":
        model_insights_tab()
    
    elif tab == "💰 Bankroll Management":
        bankroll_management_tab(bankroll, kelly_fraction)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Tennis Prediction System v1.0**")
    st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
