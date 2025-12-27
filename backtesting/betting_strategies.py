"""
Betting Strategies for Tennis Match Prediction

Implements three strategies:
1. Fixed stake on predicted winner
2. Value betting (bet when edge exists)
3. Kelly Criterion with fractional Kelly for risk management
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


def calculate_roi(predictions_df: pd.DataFrame, 
                  odds_df: pd.DataFrame,
                  strategy: str = 'kelly',
                  initial_bankroll: float = 1000.0,
                  fixed_stake: float = 10.0,
                  kelly_fraction: float = 0.25) -> Dict:
    """
    Calculate ROI and betting performance for a given strategy.
    
    Args:
        predictions_df: DataFrame with columns ['match_id', 'p_player1_win', 'actual_winner']
        odds_df: DataFrame with columns ['match_id', 'player1_odds', 'player2_odds']
        strategy: 'fixed', 'value', or 'kelly'
        initial_bankroll: Starting bankroll
        fixed_stake: Fixed stake amount for 'fixed' and 'value' strategies
        kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)
        
    Returns:
        Dictionary with ROI, final bankroll, number of bets, win rate, etc.
    """
    # Merge predictions and odds
    df = predictions_df.merge(odds_df, on='match_id')
    
    bankroll = initial_bankroll
    total_staked = 0.0
    total_profit = 0.0
    bets = []
    
    for _, row in df.iterrows():
        p_pred = row['p_player1_win']
        p1_odds = row['player1_odds']
        p2_odds = row['player2_odds']
        actual_winner = row['actual_winner']  # 1 or 2
        
        # Determine bet based on strategy
        bet_player = None
        stake = 0.0
        
        if strategy == 'fixed':
            # Strategy 1: Fixed stake on predicted winner
            if p_pred > 0.5:
                bet_player = 1
                odds = p1_odds
                stake = fixed_stake
            elif p_pred < 0.5:
                bet_player = 2
                odds = p2_odds
                stake = fixed_stake
        
        elif strategy == 'value':
            # Strategy 2: Bet when edge exists
            p1_implied = 1.0 / p1_odds
            p2_implied = 1.0 / p2_odds
            
            if p_pred > p1_implied:
                bet_player = 1
                odds = p1_odds
                stake = fixed_stake
            elif (1 - p_pred) > p2_implied:
                bet_player = 2
                odds = p2_odds
                stake = fixed_stake
        
        elif strategy == 'kelly':
            # Strategy 3: Kelly Criterion
            # Check both players for positive edge
            p1_edge = p_pred * p1_odds - 1
            p2_edge = (1 - p_pred) * p2_odds - 1
            
            if p1_edge > 0:
                bet_player = 1
                odds = p1_odds
                kelly_frac = p1_edge / (p1_odds - 1)
                stake = bankroll * kelly_frac * kelly_fraction
                stake = max(0, min(stake, bankroll * 0.05))  # Cap at 5% of bankroll
            elif p2_edge > 0:
                bet_player = 2
                odds = p2_odds
                kelly_frac = p2_edge / (p2_odds - 1)
                stake = bankroll * kelly_frac * kelly_fraction
                stake = max(0, min(stake, bankroll * 0.05))  # Cap at 5% of bankroll
        
        # Execute bet if we have one
        if bet_player is not None and stake > 0:
            win = (bet_player == actual_winner)
            profit = stake * (odds - 1) if win else -stake
            
            bankroll += profit
            total_staked += stake
            total_profit += profit
            
            bets.append({
                'match_id': row['match_id'],
                'bet_player': bet_player,
                'odds': odds,
                'stake': stake,
                'win': win,
                'profit': profit,
                'bankroll': bankroll
            })
    
    # Calculate metrics
    if total_staked == 0:
        return {
            'roi': 0.0,
            'final_bankroll': initial_bankroll,
            'total_staked': 0.0,
            'total_profit': 0.0,
            'num_bets': 0,
            'win_rate': 0.0,
            'avg_odds': 0.0,
            'bets_df': pd.DataFrame()
        }
    
    bets_df = pd.DataFrame(bets)
    
    return {
        'roi': total_profit / total_staked,
        'final_bankroll': bankroll,
        'total_staked': total_staked,
        'total_profit': total_profit,
        'num_bets': len(bets),
        'win_rate': bets_df['win'].mean(),
        'avg_odds': bets_df['odds'].mean(),
        'bets_df': bets_df
    }


def calculate_sharpe_ratio(returns_series: pd.Series, 
                          risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe Ratio from returns series.
    
    Args:
        returns_series: Series of returns (profit/stake for each bet)
        risk_free_rate: Annual risk-free rate (default 0)
        
    Returns:
        Sharpe ratio (annualized)
    """
    if len(returns_series) == 0 or returns_series.std() == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns_series - risk_free_rate
    
    # Sharpe ratio = mean / std
    sharpe = excess_returns.mean() / returns_series.std()
    
    # Annualize (assuming daily bets, ~250 trading days)
    sharpe_annual = sharpe * np.sqrt(250)
    
    return sharpe_annual


def calculate_max_drawdown(bankroll_series: pd.Series) -> float:
    """
    Calculate maximum drawdown from bankroll series.
    
    Args:
        bankroll_series: Series of bankroll values over time
        
    Returns:
        Maximum drawdown as percentage
    """
    if len(bankroll_series) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = bankroll_series.expanding().max()
    
    # Calculate drawdown
    drawdown = (bankroll_series - running_max) / running_max
    
    max_dd = drawdown.min()
    
    return max_dd


def plot_bankroll_evolution(bets_df: pd.DataFrame,
                           initial_bankroll: float = 1000.0,
                           title: str = "Bankroll Evolution",
                           save_path: Optional[str] = None) -> None:
    """
    Plot bankroll evolution over time.
    
    Args:
        bets_df: DataFrame with betting results (must have 'bankroll' column)
        initial_bankroll: Starting bankroll
        title: Plot title
        save_path: Path to save plot (optional)
    """
    if len(bets_df) == 0:
        print("No bets to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Add initial bankroll
    bankroll_evolution = [initial_bankroll] + bets_df['bankroll'].tolist()
    bet_numbers = list(range(len(bankroll_evolution)))
    
    # Plot 1: Bankroll evolution
    ax1.plot(bet_numbers, bankroll_evolution, 'b-', linewidth=2)
    ax1.axhline(y=initial_bankroll, color='r', linestyle='--', 
                linewidth=1, label='Initial Bankroll')
    ax1.fill_between(bet_numbers, initial_bankroll, bankroll_evolution,
                     where=np.array(bankroll_evolution) >= initial_bankroll,
                     alpha=0.3, color='green', label='Profit')
    ax1.fill_between(bet_numbers, initial_bankroll, bankroll_evolution,
                     where=np.array(bankroll_evolution) < initial_bankroll,
                     alpha=0.3, color='red', label='Loss')
    
    ax1.set_xlabel('Bet Number', fontsize=12)
    ax1.set_ylabel('Bankroll ($)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative profit
    cumulative_profit = bets_df['profit'].cumsum()
    ax2.plot(range(1, len(cumulative_profit) + 1), cumulative_profit, 
            'g-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax2.fill_between(range(1, len(cumulative_profit) + 1), 0, cumulative_profit,
                     where=cumulative_profit >= 0, alpha=0.3, color='green')
    ax2.fill_between(range(1, len(cumulative_profit) + 1), 0, cumulative_profit,
                     where=cumulative_profit < 0, alpha=0.3, color='red')
    
    ax2.set_xlabel('Bet Number', fontsize=12)
    ax2.set_ylabel('Cumulative Profit ($)', fontsize=12)
    ax2.set_title('Cumulative Profit/Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_monthly_pnl(bets_df: pd.DataFrame,
                    date_column: str = 'date',
                    title: str = "Monthly P&L",
                    save_path: Optional[str] = None) -> None:
    """
    Plot month-by-month profit and loss.
    
    Args:
        bets_df: DataFrame with betting results and dates
        date_column: Name of date column
        title: Plot title
        save_path: Path to save plot (optional)
    """
    if len(bets_df) == 0 or date_column not in bets_df.columns:
        print("No data to plot or missing date column")
        return
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(bets_df[date_column]):
        bets_df[date_column] = pd.to_datetime(bets_df[date_column])
    
    # Extract year-month
    bets_df['year_month'] = bets_df[date_column].dt.to_period('M')
    
    # Calculate monthly profit
    monthly_pnl = bets_df.groupby('year_month')['profit'].sum().reset_index()
    monthly_pnl['year_month_str'] = monthly_pnl['year_month'].astype(str)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['green' if p >= 0 else 'red' for p in monthly_pnl['profit']]
    bars = ax.bar(monthly_pnl['year_month_str'], monthly_pnl['profit'], 
                  color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Profit/Loss ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${height:.0f}',
               ha='center', va='bottom' if height >= 0 else 'top',
               fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_strategies(predictions_df: pd.DataFrame,
                      odds_df: pd.DataFrame,
                      strategies: List[str] = ['fixed', 'value', 'kelly'],
                      initial_bankroll: float = 1000.0) -> pd.DataFrame:
    """
    Compare multiple betting strategies.
    
    Args:
        predictions_df: DataFrame with predictions
        odds_df: DataFrame with odds
        strategies: List of strategies to compare
        initial_bankroll: Starting bankroll
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for strategy in strategies:
        metrics = calculate_roi(predictions_df, odds_df, strategy, initial_bankroll)
        
        # Calculate additional metrics
        if len(metrics['bets_df']) > 0:
            returns = metrics['bets_df']['profit'] / metrics['bets_df']['stake']
            sharpe = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(metrics['bets_df']['bankroll'])
        else:
            sharpe = 0.0
            max_dd = 0.0
        
        results.append({
            'Strategy': strategy.upper(),
            'ROI': metrics['roi'],
            'Final Bankroll': metrics['final_bankroll'],
            'Total Profit': metrics['total_profit'],
            'Total Staked': metrics['total_staked'],
            'Num Bets': metrics['num_bets'],
            'Win Rate': metrics['win_rate'],
            'Avg Odds': metrics['avg_odds'],
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd
        })
    
    return pd.DataFrame(results)


def backtest_model(model_predictions: pd.DataFrame,
                  odds_df: pd.DataFrame,
                  model_name: str = "Model",
                  strategy: str = 'kelly',
                  initial_bankroll: float = 1000.0) -> Dict:
    """
    Backtest a single model with a given strategy.
    
    Args:
        model_predictions: DataFrame with model predictions
        odds_df: DataFrame with betting odds
        model_name: Name of the model
        strategy: Betting strategy to use
        initial_bankroll: Starting bankroll
        
    Returns:
        Dictionary with backtest results
    """
    metrics = calculate_roi(model_predictions, odds_df, strategy, initial_bankroll)
    
    # Calculate additional metrics
    if len(metrics['bets_df']) > 0:
        returns = metrics['bets_df']['profit'] / metrics['bets_df']['stake']
        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(metrics['bets_df']['bankroll'])
    else:
        sharpe = 0.0
        max_dd = 0.0
    
    return {
        'model_name': model_name,
        'roi': metrics['roi'],
        'final_bankroll': metrics['final_bankroll'],
        'total_profit': metrics['total_profit'],
        'total_staked': metrics['total_staked'],
        'num_bets': metrics['num_bets'],
        'win_rate': metrics['win_rate'],
        'avg_odds': metrics['avg_odds'],
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'bets_df': metrics['bets_df']
    }
