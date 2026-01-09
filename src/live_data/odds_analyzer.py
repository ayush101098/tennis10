"""
Odds Analyzer - Analyze betting odds for value and movement

Functions:
- Calculate implied probability
- Calculate overround (bookmaker margin)
- Detect odds movement
- Find value bets
- Track sharp money
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def calculate_implied_probability(decimal_odds: float) -> float:
    """
    Convert decimal odds to implied probability
    
    Args:
        decimal_odds: Decimal odds (e.g., 1.85)
    
    Returns:
        Implied probability (0-1)
    
    Example:
        1.85 odds → 54.05% implied probability
    """
    if decimal_odds <= 0:
        raise ValueError(f"Odds must be positive, got {decimal_odds}")
    
    return 1.0 / decimal_odds


def calculate_overround(odds1: float, odds2: float) -> float:
    """
    Calculate overround (bookmaker margin)
    
    Args:
        odds1: Player 1 decimal odds
        odds2: Player 2 decimal odds
    
    Returns:
        Overround value
    
    Fair market: 1.00
    Pinnacle: ~1.02 (2% margin - lowest in industry)
    Typical bookmaker: ~1.07 (7% margin)
    Poor bookmaker: ~1.10+ (10%+ margin)
    
    Example:
        Odds: 1.85 / 2.05
        Overround: (1/1.85) + (1/2.05) = 1.0284 (2.84% margin)
    """
    return calculate_implied_probability(odds1) + calculate_implied_probability(odds2)


def calculate_true_probabilities(odds1: float, odds2: float) -> Tuple[float, float]:
    """
    Remove bookmaker margin to get true probabilities
    
    Uses Shin's method for removing overround
    
    Args:
        odds1: Player 1 decimal odds
        odds2: Player 2 decimal odds
    
    Returns:
        (true_prob1, true_prob2)
    """
    impl_prob1 = calculate_implied_probability(odds1)
    impl_prob2 = calculate_implied_probability(odds2)
    
    overround = impl_prob1 + impl_prob2
    
    # Remove overround proportionally
    true_prob1 = impl_prob1 / overround
    true_prob2 = impl_prob2 / overround
    
    return true_prob1, true_prob2


def find_value_bets(our_probability: float, bookmaker_odds: float,
                    min_edge: float = 0.02) -> Optional[Dict]:
    """
    Determine if there's value in betting
    
    Value exists when:
    our_probability > implied_probability_from_odds
    
    Args:
        our_probability: Our calculated probability (0-1)
        bookmaker_odds: Bookmaker's decimal odds
        min_edge: Minimum edge to consider (default 2%)
    
    Returns:
        {
            'edge': float,
            'expected_value': float,
            'value_percent': float,
            'recommended': bool
        }
        or None if no value
    
    Example:
        Our prediction: 60% chance
        Bookmaker odds: 1.85 (implies 54%)
        Edge: 6%
        EV: (0.6 × 1.85) - 1 = 0.11 (11% expected value!)
    """
    implied_prob = calculate_implied_probability(bookmaker_odds)
    edge = our_probability - implied_prob
    
    # Calculate expected value
    expected_value = (our_probability * bookmaker_odds) - 1
    
    # Value as percentage
    value_percent = edge / implied_prob if implied_prob > 0 else 0
    
    # Determine if this is a recommended bet
    recommended = edge >= min_edge and expected_value > 0
    
    return {
        'edge': edge,
        'expected_value': expected_value,
        'value_percent': value_percent,
        'recommended': recommended,
        'our_probability': our_probability,
        'implied_probability': implied_prob
    }


def detect_odds_movement(historical_odds: List[Dict], window_hours: int = 24) -> Dict:
    """
    Analyze how odds have moved over time
    
    Args:
        historical_odds: List of odds snapshots, each with:
            {'timestamp': datetime, 'odds': float}
        window_hours: Time window to analyze (default 24 hours)
    
    Returns:
        {
            'direction': 'shortening' | 'drifting' | 'stable',
            'magnitude': float (percentage change),
            'velocity': float (change per hour),
            'volatility': float (std dev of changes),
            'current_odds': float,
            'starting_odds': float
        }
    
    Interpretation:
        - Shortening (decreasing odds): More money on this player, market confidence increasing
        - Drifting (increasing odds): Less confidence, money going elsewhere
        - Stable: Little movement
    """
    if not historical_odds or len(historical_odds) < 2:
        return {
            'direction': 'stable',
            'magnitude': 0,
            'velocity': 0,
            'volatility': 0,
            'current_odds': historical_odds[0]['odds'] if historical_odds else 0,
            'starting_odds': historical_odds[0]['odds'] if historical_odds else 0
        }
    
    # Sort by timestamp
    sorted_odds = sorted(historical_odds, key=lambda x: x['timestamp'])
    
    # Filter to time window
    cutoff_time = datetime.now() - timedelta(hours=window_hours)
    windowed_odds = [o for o in sorted_odds if o['timestamp'] >= cutoff_time]
    
    if len(windowed_odds) < 2:
        windowed_odds = sorted_odds[-2:]  # Use last 2 points
    
    starting_odds = windowed_odds[0]['odds']
    current_odds = windowed_odds[-1]['odds']
    
    # Calculate change
    change = current_odds - starting_odds
    magnitude = change / starting_odds if starting_odds > 0 else 0
    
    # Calculate velocity (change per hour)
    time_diff_hours = (windowed_odds[-1]['timestamp'] - windowed_odds[0]['timestamp']).total_seconds() / 3600
    velocity = magnitude / time_diff_hours if time_diff_hours > 0 else 0
    
    # Calculate volatility
    odds_values = [o['odds'] for o in windowed_odds]
    volatility = np.std(odds_values) / np.mean(odds_values) if odds_values else 0
    
    # Determine direction
    if magnitude < -0.02:  # > 2% decrease
        direction = 'shortening'
    elif magnitude > 0.02:  # > 2% increase
        direction = 'drifting'
    else:
        direction = 'stable'
    
    return {
        'direction': direction,
        'magnitude': magnitude,
        'velocity': velocity,
        'volatility': volatility,
        'current_odds': current_odds,
        'starting_odds': starting_odds
    }


def calculate_arbitrage_opportunity(p1_odds: float, p2_odds: float,
                                   p1_bookmaker: str, p2_bookmaker: str) -> Optional[Dict]:
    """
    Check for arbitrage (risk-free profit) opportunity
    
    Arbitrage exists when:
    (1 / odds1) + (1 / odds2) < 1.00
    
    Args:
        p1_odds: Player 1 best odds
        p2_odds: Player 2 best odds
        p1_bookmaker: Bookmaker offering p1_odds
        p2_bookmaker: Bookmaker offering p2_odds
    
    Returns:
        Arbitrage info dict or None if no arbitrage
    
    Example:
        Bookmaker A: Player 1 @ 2.10
        Bookmaker B: Player 2 @ 2.10
        Overround: 0.476 + 0.476 = 0.952 < 1.00 → ARBITRAGE!
        Profit: ~5%
    """
    overround = calculate_overround(p1_odds, p2_odds)
    
    if overround < 1.0:
        # Arbitrage exists!
        profit_margin = (1.0 - overround) / overround
        
        # Calculate optimal stakes (for $100 total)
        total_stake = 100
        stake1 = total_stake * calculate_implied_probability(p1_odds)
        stake2 = total_stake * calculate_implied_probability(p2_odds)
        
        # Calculate guaranteed profit
        payout_if_p1_wins = stake1 * p1_odds
        payout_if_p2_wins = stake2 * p2_odds
        
        profit = min(payout_if_p1_wins, payout_if_p2_wins) - total_stake
        
        return {
            'exists': True,
            'profit_margin': profit_margin,
            'profit_amount': profit,
            'total_stake': total_stake,
            'stake_player1': stake1,
            'stake_player2': stake2,
            'bookmaker1': p1_bookmaker,
            'bookmaker2': p2_bookmaker,
            'overround': overround
        }
    
    return None


def compare_bookmakers(odds_list: List[Dict]) -> pd.DataFrame:
    """
    Compare odds across multiple bookmakers
    
    Args:
        odds_list: List of odds from different bookmakers
            Each dict: {'bookmaker': str, 'player1_odds': float, 'player2_odds': float}
    
    Returns:
        DataFrame with comparison and rankings
    """
    if not odds_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(odds_list)
    
    # Calculate overround for each bookmaker
    df['overround'] = df.apply(
        lambda row: calculate_overround(row['player1_odds'], row['player2_odds']),
        axis=1
    )
    
    # Calculate margin
    df['margin_percent'] = (df['overround'] - 1.0) * 100
    
    # Rank bookmakers (lower margin = better)
    df['margin_rank'] = df['margin_percent'].rank(method='min')
    
    # Sort by margin
    df = df.sort_values('margin_percent')
    
    return df


def identify_sharp_money(odds_history: pd.DataFrame, volume_threshold: float = 0.05) -> List[Dict]:
    """
    Identify sharp money movements (professional bettors)
    
    Sharp money indicators:
    - Large odds movement on low volume (sharp bettors move lines efficiently)
    - Reverse line movement (odds move opposite to public betting %)
    - Steam moves (multiple bookmakers move odds simultaneously)
    
    Args:
        odds_history: DataFrame with odds over time
        volume_threshold: Threshold for significant movement (default 5%)
    
    Returns:
        List of detected sharp money events
    """
    sharp_events = []
    
    # Look for sharp moves (>5% odds change in <1 hour)
    for i in range(1, len(odds_history)):
        prev = odds_history.iloc[i-1]
        curr = odds_history.iloc[i]
        
        time_diff = (curr['timestamp'] - prev['timestamp']).total_seconds() / 3600
        
        if time_diff < 1:  # Within 1 hour
            # Check player 1 odds
            change_p1 = abs(curr['player1_odds'] - prev['player1_odds']) / prev['player1_odds']
            
            if change_p1 > volume_threshold:
                sharp_events.append({
                    'timestamp': curr['timestamp'],
                    'player': 'player1',
                    'old_odds': prev['player1_odds'],
                    'new_odds': curr['player1_odds'],
                    'change_percent': change_p1 * 100,
                    'time_window_hours': time_diff,
                    'type': 'sharp_move'
                })
            
            # Check player 2 odds
            change_p2 = abs(curr['player2_odds'] - prev['player2_odds']) / prev['player2_odds']
            
            if change_p2 > volume_threshold:
                sharp_events.append({
                    'timestamp': curr['timestamp'],
                    'player': 'player2',
                    'old_odds': prev['player2_odds'],
                    'new_odds': curr['player2_odds'],
                    'change_percent': change_p2 * 100,
                    'time_window_hours': time_diff,
                    'type': 'sharp_move'
                })
    
    return sharp_events


if __name__ == "__main__":
    print("🎾 Testing Odds Analyzer\n")
    
    # Test implied probability
    print("="*50)
    print("Implied Probability:")
    for odds in [1.50, 1.85, 2.00, 2.50, 3.00]:
        prob = calculate_implied_probability(odds)
        print(f"  Odds {odds:.2f} → {prob:.2%}")
    
    # Test overround
    print("\n" + "="*50)
    print("Overround (Bookmaker Margin):")
    test_odds = [
        (1.90, 1.90, "Pinnacle"),
        (1.85, 2.05, "Bet365"),
        (1.80, 2.10, "DraftKings"),
    ]
    
    for odds1, odds2, bookie in test_odds:
        overround = calculate_overround(odds1, odds2)
        margin = (overround - 1.0) * 100
        print(f"  {bookie:15s} {odds1:.2f} / {odds2:.2f} → {overround:.4f} ({margin:.2f}% margin)")
    
    # Test value betting
    print("\n" + "="*50)
    print("Value Bet Analysis:")
    
    our_prob = 0.60  # We think player has 60% chance
    bookie_odds = 1.85  # Bookmaker offers 1.85 odds
    
    value = find_value_bets(our_prob, bookie_odds)
    
    print(f"  Our probability: {value['our_probability']:.2%}")
    print(f"  Implied probability: {value['implied_probability']:.2%}")
    print(f"  Edge: {value['edge']:.2%}")
    print(f"  Expected Value: {value['expected_value']:.2%}")
    print(f"  Recommended: {'✅ YES' if value['recommended'] else '❌ NO'}")
    
    # Test arbitrage
    print("\n" + "="*50)
    print("Arbitrage Detection:")
    
    arb = calculate_arbitrage_opportunity(2.10, 2.10, "Bookmaker A", "Bookmaker B")
    
    if arb:
        print(f"  🚨 ARBITRAGE FOUND!")
        print(f"  Profit margin: {arb['profit_margin']:.2%}")
        print(f"  Guaranteed profit: ${arb['profit_amount']:.2f} on ${arb['total_stake']:.0f} stake")
        print(f"  Bet ${arb['stake_player1']:.2f} on Player 1 @ {arb['bookmaker1']}")
        print(f"  Bet ${arb['stake_player2']:.2f} on Player 2 @ {arb['bookmaker2']}")
    else:
        print("  No arbitrage opportunity")
