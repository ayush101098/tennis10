"""
Model Comparison and Evaluation

Evaluates all models on test set with comprehensive metrics:
- Log Loss, Brier Score
- ROI for betting strategies
- Statistical significance tests
- Calibration analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import log_loss, brier_score_loss


def calculate_log_loss(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate log loss (cross-entropy loss).
    
    Lower values indicate better probability calibration.
    
    Args:
        predictions: Predicted probabilities for player 1
        actuals: Actual outcomes (1 if player 1 won, 2 if player 2 won)
        
    Returns:
        Log loss value
    """
    # Convert actuals to binary (1 if player 1 won, 0 otherwise)
    y_true = (actuals == 1).astype(int)
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(predictions, 1e-15, 1 - 1e-15)
    
    return log_loss(y_true, y_pred)


def calculate_brier_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Brier score.
    
    Measures mean squared difference between predicted probabilities
    and actual outcomes.
    
    Args:
        predictions: Predicted probabilities for player 1
        actuals: Actual outcomes (1 if player 1 won, 2 if player 2 won)
        
    Returns:
        Brier score (lower is better)
    """
    y_true = (actuals == 1).astype(int)
    y_pred = np.clip(predictions, 0, 1)
    
    return brier_score_loss(y_true, y_pred)


def calculate_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate prediction accuracy.
    
    Args:
        predictions: Predicted probabilities for player 1
        actuals: Actual outcomes (1 if player 1 won, 2 if player 2 won)
        
    Returns:
        Accuracy as fraction
    """
    predicted_winner = (predictions > 0.5).astype(int) + 1
    return (predicted_winner == actuals).mean()


def calculate_longest_losing_streak(bets_df: pd.DataFrame) -> int:
    """
    Calculate longest consecutive losing streak.
    
    Args:
        bets_df: DataFrame with bet outcomes
        
    Returns:
        Longest losing streak
    """
    if len(bets_df) == 0:
        return 0
    
    # Check if bet won
    bets_df = bets_df.copy()
    bets_df['won'] = bets_df['profit'] > 0
    
    max_streak = 0
    current_streak = 0
    
    for won in bets_df['won']:
        if not won:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak


def bootstrap_roi_confidence_interval(
    bets_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for ROI.
    
    Args:
        bets_df: DataFrame with bet outcomes
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(bets_df) == 0:
        return (0.0, 0.0)
    
    rois = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = bets_df.sample(n=len(bets_df), replace=True)
        
        # Calculate ROI for this sample
        total_profit = sample['profit'].sum()
        total_staked = sample['stake'].sum()
        
        if total_staked > 0:
            roi = total_profit / total_staked
            rois.append(roi)
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower = np.percentile(rois, 100 * alpha / 2)
    upper = np.percentile(rois, 100 * (1 - alpha / 2))
    
    return (lower, upper)


def mcnemar_test(
    predictions1: np.ndarray,
    predictions2: np.ndarray,
    actuals: np.ndarray
) -> Dict[str, float]:
    """
    Perform McNemar's test to compare two models.
    
    Tests if the two models have significantly different error rates.
    
    Args:
        predictions1: Model 1 predicted probabilities
        predictions2: Model 2 predicted probabilities
        actuals: Actual outcomes
        
    Returns:
        Dictionary with statistic and p-value
    """
    # Convert to binary predictions
    pred1 = (predictions1 > 0.5).astype(int) + 1
    pred2 = (predictions2 > 0.5).astype(int) + 1
    
    # Create contingency table
    # n00: both wrong, n01: model1 wrong model2 correct
    # n10: model1 correct model2 wrong, n11: both correct
    n01 = ((pred1 != actuals) & (pred2 == actuals)).sum()
    n10 = ((pred1 == actuals) & (pred2 != actuals)).sum()
    
    # McNemar's test statistic
    if n01 + n10 == 0:
        return {'statistic': 0.0, 'p_value': 1.0}
    
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return {'statistic': statistic, 'p_value': p_value}


def calculate_calibration_curve(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve.
    
    Divides predictions into bins and calculates actual win rate in each bin.
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual outcomes (1 or 2)
        n_bins: Number of bins
        
    Returns:
        Tuple of (bin_centers, actual_win_rates, bin_counts)
    """
    y_true = (actuals == 1).astype(int)
    
    # Define bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    actual_win_rates = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        # Find predictions in this bin
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        
        if i == n_bins - 1:  # Last bin includes upper edge
            mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])
        
        if mask.sum() > 0:
            actual_win_rates[i] = y_true[mask].mean()
            bin_counts[i] = mask.sum()
    
    return bin_centers, actual_win_rates, bin_counts


def plot_calibration_curve(
    predictions_dict: Dict[str, np.ndarray],
    actuals: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
):
    """
    Plot calibration curves for multiple models.
    
    Args:
        predictions_dict: Dictionary mapping model names to predictions
        actuals: Actual outcomes
        n_bins: Number of bins
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(predictions_dict)))
    
    for (model_name, predictions), color in zip(predictions_dict.items(), colors):
        bin_centers, actual_win_rates, bin_counts = calculate_calibration_curve(
            predictions, actuals, n_bins
        )
        
        # Plot only bins with data
        mask = bin_counts > 0
        plt.plot(
            bin_centers[mask],
            actual_win_rates[mask],
            'o-',
            label=model_name,
            color=color,
            linewidth=2,
            markersize=8,
            alpha=0.8
        )
        
        # Add bin counts as text
        for x, y, count in zip(bin_centers[mask], actual_win_rates[mask], bin_counts[mask]):
            plt.text(x, y, f'{int(count)}', fontsize=8, ha='center', va='bottom')
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)
    
    plt.xlabel('Predicted Win Probability', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Win Rate', fontsize=12, fontweight='bold')
    plt.title('Model Calibration Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_reliability_diagram(
    predictions: np.ndarray,
    actuals: np.ndarray,
    model_name: str,
    n_bins: int = 10,
    save_path: Optional[str] = None
):
    """
    Plot reliability diagram with histogram.
    
    Args:
        predictions: Predicted probabilities
        actuals: Actual outcomes
        model_name: Name of model
        n_bins: Number of bins
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Calculate calibration
    bin_centers, actual_win_rates, bin_counts = calculate_calibration_curve(
        predictions, actuals, n_bins
    )
    
    # Top plot: Calibration curve
    mask = bin_counts > 0
    bars = ax1.bar(
        bin_centers[mask],
        actual_win_rates[mask],
        width=0.08,
        alpha=0.6,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Color bars by calibration error
    for bar, pred, actual in zip(bars, bin_centers[mask], actual_win_rates[mask]):
        error = abs(pred - actual)
        if error < 0.05:
            bar.set_facecolor('green')
        elif error < 0.10:
            bar.set_facecolor('orange')
        else:
            bar.set_facecolor('red')
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    ax1.set_xlabel('Predicted Win Probability', fontsize=12)
    ax1.set_ylabel('Actual Win Rate', fontsize=12)
    ax1.set_title(f'{model_name} - Reliability Diagram', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Bottom plot: Histogram of predictions
    ax2.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Predictions', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def evaluate_model(
    predictions: np.ndarray,
    actuals: np.ndarray,
    odds_df: pd.DataFrame,
    model_name: str,
    initial_bankroll: float = 1000.0
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        predictions: Predicted probabilities for player 1
        actuals: Actual outcomes
        odds_df: DataFrame with odds
        model_name: Name of model
        initial_bankroll: Starting bankroll
        
    Returns:
        Dictionary with all metrics
    """
    from backtesting.betting_strategies import backtest_model
    
    # Prepare predictions dataframe
    pred_df = pd.DataFrame({
        'match_id': odds_df['match_id'],
        'p_player1_win': predictions,
        'actual_winner': actuals
    })
    
    # Probability metrics
    logloss = calculate_log_loss(predictions, actuals)
    brier = calculate_brier_score(predictions, actuals)
    accuracy = calculate_accuracy(predictions, actuals)
    
    # Backtest with Kelly strategy
    result = backtest_model(
        pred_df,
        odds_df,
        model_name=model_name,
        strategy='kelly',
        initial_bankroll=initial_bankroll
    )
    
    # Calculate additional metrics
    longest_streak = calculate_longest_losing_streak(result['bets_df'])
    
    # Bootstrap confidence interval for ROI
    ci_lower, ci_upper = bootstrap_roi_confidence_interval(result['bets_df'])
    
    # Calculate average stake size
    avg_stake = result['bets_df']['stake'].mean() if len(result['bets_df']) > 0 else 0
    
    return {
        'model': model_name,
        'log_loss': logloss,
        'brier_score': brier,
        'accuracy': accuracy,
        'roi': result['roi'],
        'roi_ci_lower': ci_lower,
        'roi_ci_upper': ci_upper,
        'final_bankroll': result['final_bankroll'],
        'total_profit': result['total_profit'],
        'num_bets': result['num_bets'],
        'win_rate': result['win_rate'],
        'avg_odds': result['avg_odds'],
        'avg_stake': avg_stake,
        'sharpe_ratio': result['sharpe_ratio'],
        'max_drawdown': result['max_drawdown'],
        'longest_losing_streak': longest_streak,
        'bets_df': result['bets_df']
    }


def compare_all_models(
    predictions_dict: Dict[str, np.ndarray],
    actuals: np.ndarray,
    odds_df: pd.DataFrame,
    initial_bankroll: float = 1000.0
) -> pd.DataFrame:
    """
    Compare all models and create summary table.
    
    Args:
        predictions_dict: Dictionary mapping model names to predictions
        actuals: Actual outcomes
        odds_df: DataFrame with odds
        initial_bankroll: Starting bankroll
        
    Returns:
        DataFrame with comparison table
    """
    results = []
    
    for model_name, predictions in predictions_dict.items():
        print(f"\nEvaluating {model_name}...")
        result = evaluate_model(
            predictions,
            actuals,
            odds_df,
            model_name,
            initial_bankroll
        )
        results.append(result)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame([
        {
            'Model': r['model'],
            'Log Loss': f"{r['log_loss']:.4f}",
            'Brier Score': f"{r['brier_score']:.4f}",
            'Accuracy': f"{r['accuracy']:.2%}",
            'ROI (Kelly)': f"{r['roi']:+.2%}",
            'ROI 95% CI': f"[{r['roi_ci_lower']:+.2%}, {r['roi_ci_upper']:+.2%}]",
            '# Bets': r['num_bets'],
            'Win Rate': f"{r['win_rate']:.2%}",
            'Avg Stake': f"${r['avg_stake']:.2f}",
            'Sharpe': f"{r['sharpe_ratio']:.2f}",
            'Max DD': f"{r['max_drawdown']:.2%}",
            'Longest Streak': r['longest_losing_streak']
        }
        for r in results
    ])
    
    return comparison, results


def statistical_significance_tests(
    predictions_dict: Dict[str, np.ndarray],
    actuals: np.ndarray
) -> pd.DataFrame:
    """
    Perform pairwise statistical significance tests.
    
    Args:
        predictions_dict: Dictionary mapping model names to predictions
        actuals: Actual outcomes
        
    Returns:
        DataFrame with test results
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    test_results = []
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            name1 = model_names[i]
            name2 = model_names[j]
            
            # McNemar's test
            result = mcnemar_test(
                predictions_dict[name1],
                predictions_dict[name2],
                actuals
            )
            
            # Determine significance
            if result['p_value'] < 0.01:
                significance = '***'
            elif result['p_value'] < 0.05:
                significance = '**'
            elif result['p_value'] < 0.10:
                significance = '*'
            else:
                significance = 'n.s.'
            
            test_results.append({
                'Model 1': name1,
                'Model 2': name2,
                'Statistic': f"{result['statistic']:.2f}",
                'P-Value': f"{result['p_value']:.4f}",
                'Significant': significance
            })
    
    return pd.DataFrame(test_results)


def plot_model_comparison(
    results: List[Dict],
    save_path: Optional[str] = None
):
    """
    Create comprehensive model comparison visualization.
    
    Args:
        results: List of result dictionaries
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    model_names = [r['model'] for r in results]
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    
    # Plot 1: Log Loss
    ax = axes[0, 0]
    log_losses = [r['log_loss'] for r in results]
    bars = ax.bar(model_names, log_losses, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Log Loss', fontsize=11, fontweight='bold')
    ax.set_title('Probability Calibration', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, log_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Accuracy
    ax = axes[0, 1]
    accuracies = [r['accuracy'] * 100 for r in results]
    bars = ax.bar(model_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: ROI with confidence intervals
    ax = axes[0, 2]
    rois = [r['roi'] * 100 for r in results]
    ci_lowers = [r['roi_ci_lower'] * 100 for r in results]
    ci_uppers = [r['roi_ci_upper'] * 100 for r in results]
    errors = [[roi - lower for roi, lower in zip(rois, ci_lowers)],
              [upper - roi for roi, upper in zip(rois, ci_uppers)]]
    bars = ax.bar(model_names, rois, yerr=errors, color=colors, alpha=0.7,
                  edgecolor='black', capsize=5, error_kw={'linewidth': 2})
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('ROI (%)', fontsize=11, fontweight='bold')
    ax.set_title('Return on Investment (95% CI)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rois):
        ax.text(bar.get_x() + bar.get_width()/2, val,
                f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # Plot 4: Sharpe Ratio
    ax = axes[1, 0]
    sharpes = [r['sharpe_ratio'] for r in results]
    bars = ax.bar(model_names, sharpes, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax.set_title('Risk-Adjusted Returns', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 5: Max Drawdown
    ax = axes[1, 1]
    drawdowns = [r['max_drawdown'] * 100 for r in results]
    bars = ax.bar(model_names, drawdowns, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Max Drawdown (%)', fontsize=11, fontweight='bold')
    ax.set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, drawdowns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Win Rate
    ax = axes[1, 2]
    win_rates = [r['win_rate'] * 100 for r in results]
    bars = ax.bar(model_names, win_rates, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Break-even')
    ax.set_ylabel('Win Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Betting Win Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
