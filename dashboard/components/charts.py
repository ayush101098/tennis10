"""
Charts Component - Reusable visualization components
====================================================
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_pnl_chart(pnl_data: pd.DataFrame):
    """
    Create cumulative PnL chart
    
    Args:
        pnl_data: DataFrame with columns [timestamp, bankroll, daily_pnl, roi]
    """
    
    if len(pnl_data) == 0:
        st.info("No PnL data available yet")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pnl_data['timestamp'],
        y=pnl_data['bankroll'],
        mode='lines+markers',
        name='Bankroll',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    
    # Add starting bankroll line
    if len(pnl_data) > 0:
        start_bankroll = pnl_data.iloc[0]['bankroll']
        fig.add_hline(
            y=start_bankroll,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Starting: ${start_bankroll:.0f}"
        )
    
    fig.update_layout(
        title="Bankroll Growth Over Time",
        xaxis_title="Date",
        yaxis_title="Bankroll ($)",
        hovermode='x unified',
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_drawdown_chart(pnl_data: pd.DataFrame):
    """
    Create drawdown analysis chart
    
    Args:
        pnl_data: DataFrame with columns [timestamp, bankroll]
    """
    
    if len(pnl_data) == 0:
        st.info("No drawdown data available yet")
        return
    
    # Calculate running maximum and drawdown
    pnl_data = pnl_data.copy()
    pnl_data['running_max'] = pnl_data['bankroll'].cummax()
    pnl_data['drawdown'] = (pnl_data['bankroll'] - pnl_data['running_max']) / pnl_data['running_max'] * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pnl_data['timestamp'],
        y=pnl_data['drawdown'],
        mode='lines',
        name='Drawdown %',
        line=dict(color='#d62728', width=2),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.2)'
    ))
    
    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=300,
        showlegend=False,
        template='plotly_white'
    )
    
    # Add max drawdown annotation
    max_dd = pnl_data['drawdown'].min()
    max_dd_date = pnl_data.loc[pnl_data['drawdown'].idxmin(), 'timestamp']
    
    fig.add_annotation(
        x=max_dd_date,
        y=max_dd,
        text=f"Max DD: {max_dd:.1f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor='red',
        bgcolor='white',
        bordercolor='red'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_model_comparison_radar(comparison_df: pd.DataFrame):
    """
    Create radar chart comparing models
    
    Args:
        comparison_df: DataFrame with columns [Model, ROI, Win Rate, Log Loss, Sharpe]
    """
    
    fig = go.Figure()
    
    categories = ['ROI', 'Win Rate', 'Sharpe Ratio', '1/Log Loss']
    
    for _, row in comparison_df.iterrows():
        model_name = row['Model']
        
        # Normalize metrics to 0-100 scale
        values = [
            row['ROI'] * 10,  # Scale ROI
            row['Win Rate'],
            row['Sharpe'] * 20,  # Scale Sharpe
            (1 / row['Log Loss']) * 100 if row['Log Loss'] > 0 else 0  # Invert and scale log loss
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        height=500,
        title="Model Performance Comparison"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_calibration_plot(calibration_data: pd.DataFrame):
    """
    Create calibration curve
    
    Args:
        calibration_data: DataFrame with columns [predicted_prob, actual_freq, count]
    """
    
    if len(calibration_data) == 0:
        st.info("No calibration data available yet (need settled bets)")
        return
    
    fig = go.Figure()
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash', width=2)
    ))
    
    # Actual calibration
    fig.add_trace(go.Scatter(
        x=calibration_data['predicted_prob'],
        y=calibration_data['actual_freq'],
        mode='markers+lines',
        name='Actual Calibration',
        marker=dict(
            size=calibration_data['count'] * 2,  # Size by sample count
            color='#1f77b4',
            line=dict(color='white', width=1)
        ),
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Model Calibration Curve",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        showlegend=True,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_odds_movement_chart(odds_history: pd.DataFrame):
    """
    Create odds movement over time chart
    
    Args:
        odds_history: DataFrame with columns [timestamp, bookmaker, player1_odds, player2_odds]
    """
    
    if len(odds_history) == 0:
        st.info("No odds history available yet")
        return
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Player 1 Odds Movement", "Player 2 Odds Movement")
    )
    
    # Get unique bookmakers
    bookmakers = odds_history['bookmaker'].unique()
    
    for bookmaker in bookmakers:
        bm_data = odds_history[odds_history['bookmaker'] == bookmaker]
        
        # Player 1 odds
        fig.add_trace(
            go.Scatter(
                x=bm_data['timestamp'],
                y=bm_data['player1_odds'],
                mode='lines+markers',
                name=f"{bookmaker} - P1",
                legendgroup=bookmaker
            ),
            row=1, col=1
        )
        
        # Player 2 odds
        fig.add_trace(
            go.Scatter(
                x=bm_data['timestamp'],
                y=bm_data['player2_odds'],
                mode='lines+markers',
                name=f"{bookmaker} - P2",
                legendgroup=bookmaker,
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Odds", row=1, col=1)
    fig.update_yaxes(title_text="Odds", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def create_edge_distribution_chart(predictions: pd.DataFrame):
    """
    Create histogram of edge distribution
    
    Args:
        predictions: DataFrame with 'edge' column
    """
    
    if len(predictions) == 0 or 'edge' not in predictions.columns:
        st.info("No edge data available")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=predictions['edge'] * 100,
        nbinsx=30,
        name='Edge Distribution',
        marker=dict(
            color='#1f77b4',
            line=dict(color='white', width=1)
        )
    ))
    
    # Add vertical line at 2.5% (min edge threshold)
    fig.add_vline(
        x=2.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Min Edge (2.5%)"
    )
    
    fig.update_layout(
        title="Edge Distribution Across All Matches",
        xaxis_title="Edge (%)",
        yaxis_title="Number of Matches",
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_roi_by_confidence_chart(bets: pd.DataFrame):
    """
    Create bar chart showing ROI by confidence level
    
    Args:
        bets: DataFrame with columns [confidence, profit, stake]
    """
    
    if len(bets) == 0:
        st.info("No betting data available yet")
        return
    
    # Calculate ROI by confidence
    roi_by_conf = bets.groupby('confidence').agg({
        'profit': 'sum',
        'stake': 'sum'
    })
    
    roi_by_conf['roi'] = (roi_by_conf['profit'] / roi_by_conf['stake'] * 100).fillna(0)
    roi_by_conf = roi_by_conf.reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=roi_by_conf['confidence'],
        y=roi_by_conf['roi'],
        marker=dict(
            color=roi_by_conf['roi'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="ROI (%)")
        ),
        text=[f"{x:.1f}%" for x in roi_by_conf['roi']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="ROI by Confidence Level",
        xaxis_title="Confidence Level",
        yaxis_title="ROI (%)",
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_win_rate_by_surface_chart(bets: pd.DataFrame, matches: pd.DataFrame):
    """
    Create bar chart showing win rate by surface
    
    Args:
        bets: DataFrame with bet results
        matches: DataFrame with match details including surface
    """
    
    if len(bets) == 0 or len(matches) == 0:
        st.info("Insufficient data for surface analysis")
        return
    
    # Merge bets with matches to get surface
    merged = bets.merge(matches, on='match_id', how='left')
    
    if 'surface' not in merged.columns:
        st.warning("Surface data not available")
        return
    
    # Calculate win rate by surface
    surface_stats = merged.groupby('surface').agg({
        'result': lambda x: (x == 'won').sum() / len(x) * 100 if len(x) > 0 else 0
    }).reset_index()
    
    surface_stats.columns = ['surface', 'win_rate']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=surface_stats['surface'],
        y=surface_stats['win_rate'],
        marker=dict(color='#1f77b4'),
        text=[f"{x:.1f}%" for x in surface_stats['win_rate']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Win Rate by Surface",
        xaxis_title="Surface",
        yaxis_title="Win Rate (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
