"""
Tables Component - Reusable table components
============================================
"""

import streamlit as st
import pandas as pd


def render_predictions_table(predictions: pd.DataFrame, show_actions: bool = True):
    """
    Render formatted predictions table
    
    Args:
        predictions: DataFrame with prediction data
        show_actions: Whether to show action buttons
    """
    
    if len(predictions) == 0:
        st.info("No predictions available")
        return
    
    # Prepare display DataFrame
    display_df = predictions[[
        'player1_name', 'player2_name',
        'ensemble_p1_win', 'best_p1_odds',
        'edge', 'confidence'
    ]].copy()
    
    display_df.columns = [
        'Player 1', 'Player 2',
        'Win Prob', 'Best Odds',
        'Edge', 'Confidence'
    ]
    
    # Format percentages
    display_df['Win Prob'] = display_df['Win Prob'].apply(lambda x: f"{x:.1%}")
    display_df['Edge'] = display_df['Edge'].apply(lambda x: f"{x:.1%}")
    
    # Format odds
    display_df['Best Odds'] = display_df['Best Odds'].apply(lambda x: f"{x:.2f}")
    
    # Color code confidence
    def color_confidence(val):
        if val == 'high':
            return 'background-color: #c8e6c9'
        elif val == 'medium':
            return 'background-color: #fff9c4'
        else:
            return 'background-color: #ffcdd2'
    
    def color_edge(val):
        val_float = float(val.strip('%')) / 100
        if val_float >= 0.05:
            return 'background-color: #c8e6c9; font-weight: bold'
        elif val_float >= 0.025:
            return 'background-color: #fff9c4'
        else:
            return 'background-color: #ffcdd2'
    
    styled_df = display_df.style\
        .applymap(color_confidence, subset=['Confidence'])\
        .applymap(color_edge, subset=['Edge'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )


def render_bets_table(bets: pd.DataFrame, bet_type: str = 'active'):
    """
    Render formatted bets table
    
    Args:
        bets: DataFrame with bet data
        bet_type: 'active' or 'settled'
    """
    
    if len(bets) == 0:
        st.info(f"No {bet_type} bets")
        return
    
    if bet_type == 'active':
        # Active bets table
        display_df = bets[[
            'match', 'selection', 'odds',
            'stake', 'potential_profit', 'confidence'
        ]].copy()
        
        display_df.columns = [
            'Match', 'Selection', 'Odds',
            'Stake ($)', 'Potential ($)', 'Confidence'
        ]
        
        # Format currency
        display_df['Stake ($)'] = display_df['Stake ($)'].apply(lambda x: f"${x:.2f}")
        display_df['Potential ($)'] = display_df['Potential ($)'].apply(lambda x: f"${x:.2f}")
        
        # Format odds
        display_df['Odds'] = display_df['Odds'].apply(lambda x: f"{x:.2f}")
        
    else:
        # Settled bets table
        display_df = bets[[
            'match', 'selection', 'odds',
            'stake', 'result', 'profit', 'confidence'
        ]].copy()
        
        display_df.columns = [
            'Match', 'Selection', 'Odds',
            'Stake ($)', 'Result', 'Profit ($)', 'Confidence'
        ]
        
        # Format currency
        display_df['Stake ($)'] = display_df['Stake ($)'].apply(lambda x: f"${x:.2f}")
        display_df['Profit ($)'] = display_df['Profit ($)'].apply(
            lambda x: f"+${x:.2f}" if x >= 0 else f"-${abs(x):.2f}"
        )
        
        # Format odds
        display_df['Odds'] = display_df['Odds'].apply(lambda x: f"{x:.2f}")
        
        # Color code result
        def color_result(val):
            if val == 'won':
                return 'background-color: #c8e6c9; color: #2e7d32'
            else:
                return 'background-color: #ffcdd2; color: #c62828'
        
        styled_df = display_df.style.applymap(color_result, subset=['Result'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
        return
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def render_performance_summary(bets: pd.DataFrame):
    """
    Render performance summary statistics table
    
    Args:
        bets: DataFrame with settled bets
    """
    
    if len(bets) == 0:
        st.info("No betting history available yet")
        return
    
    # Calculate metrics
    total_bets = len(bets)
    wins = (bets['result'] == 'won').sum()
    losses = (bets['result'] == 'lost').sum()
    win_rate = wins / total_bets * 100 if total_bets > 0 else 0
    
    total_staked = bets['stake'].sum()
    total_profit = bets['profit'].sum()
    roi = total_profit / total_staked * 100 if total_staked > 0 else 0
    
    avg_odds = bets['odds'].mean()
    avg_stake = bets['stake'].mean()
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Metric': [
            'Total Bets',
            'Wins',
            'Losses',
            'Win Rate',
            'Total Staked',
            'Total Profit',
            'ROI',
            'Avg Odds',
            'Avg Stake'
        ],
        'Value': [
            total_bets,
            wins,
            losses,
            f"{win_rate:.1f}%",
            f"${total_staked:.2f}",
            f"${total_profit:+.2f}",
            f"{roi:.2f}%",
            f"{avg_odds:.2f}",
            f"${avg_stake:.2f}"
        ]
    })
    
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


def render_model_performance_table(metrics_df: pd.DataFrame):
    """
    Render model performance comparison table
    
    Args:
        metrics_df: DataFrame with model performance metrics
    """
    
    # Format percentages and decimals
    formatted = metrics_df.copy()
    
    if 'ROI' in formatted.columns:
        formatted['ROI'] = formatted['ROI'].apply(lambda x: f"{x:.2f}%")
    
    if 'Win Rate' in formatted.columns:
        formatted['Win Rate'] = formatted['Win Rate'].apply(lambda x: f"{x:.1f}%")
    
    if 'Log Loss' in formatted.columns:
        formatted['Log Loss'] = formatted['Log Loss'].apply(lambda x: f"{x:.4f}")
    
    if 'Sharpe' in formatted.columns:
        formatted['Sharpe'] = formatted['Sharpe'].apply(lambda x: f"{x:.2f}")
    
    # Color code best performer
    def highlight_best(s):
        if s.name == 'ROI' or s.name == 'Win Rate' or s.name == 'Sharpe':
            # Higher is better
            is_max = s == s.max()
            return ['background-color: #c8e6c9' if v else '' for v in is_max]
        elif s.name == 'Log Loss':
            # Lower is better
            # Convert back to float for comparison
            values = [float(x) for x in s]
            is_min = [v == min(values) for v in values]
            return ['background-color: #c8e6c9' if v else '' for v in is_min]
        else:
            return ['' for _ in s]
    
    styled_df = formatted.style.apply(highlight_best, axis=0)
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )


def render_player_stats_table(player_stats: pd.DataFrame, surface: str = None):
    """
    Render player statistics table
    
    Args:
        player_stats: DataFrame with player statistics
        surface: Optional surface filter
    """
    
    if len(player_stats) == 0:
        st.info("No player statistics available")
        return
    
    # Filter by surface if specified
    if surface and 'surface' in player_stats.columns:
        player_stats = player_stats[player_stats['surface'] == surface]
    
    # Select relevant columns
    display_cols = [
        col for col in [
            'player_name', 'matches_played', 'win_rate',
            'avg_service_points_won', 'avg_return_points_won',
            'surface'
        ] if col in player_stats.columns
    ]
    
    display_df = player_stats[display_cols].copy()
    
    # Format percentages
    for col in display_df.columns:
        if 'rate' in col.lower() or 'won' in col.lower():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
                )
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def render_upcoming_matches_table(matches: pd.DataFrame):
    """
    Render upcoming matches table with minimal info
    
    Args:
        matches: DataFrame with match data
    """
    
    if len(matches) == 0:
        st.info("No upcoming matches")
        return
    
    display_df = matches[[
        'player1_name', 'player2_name',
        'tournament', 'surface', 'scheduled_time'
    ]].copy()
    
    display_df.columns = [
        'Player 1', 'Player 2',
        'Tournament', 'Surface', 'Start Time'
    ]
    
    # Format datetime
    if 'Start Time' in display_df.columns:
        display_df['Start Time'] = pd.to_datetime(display_df['Start Time']).dt.strftime('%b %d, %H:%M')
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=300
    )
