"""
Bet Calculator - Kelly Criterion and risk management

Calculates:
- Optimal bet sizes using Kelly Criterion
- Risk-adjusted stakes
- Bankroll management
- Portfolio optimization
"""

import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class BetCalculator:
    """Calculate optimal bet sizes and manage risk"""
    
    def __init__(self, bankroll: float = 1000, kelly_fraction: float = 0.25,
                 max_bet_percent: float = 0.15, min_edge: float = 0.02):
        """
        Initialize bet calculator
        
        Args:
            bankroll: Total bankroll ($)
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly, conservative)
            max_bet_percent: Maximum bet as % of bankroll (0.15 = 15%)
            min_edge: Minimum edge to consider betting (0.02 = 2%)
        """
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_percent = max_bet_percent
        self.min_edge = min_edge
        self.initial_bankroll = bankroll
    
    def calculate_bet_recommendation(
        self,
        our_probability: float,
        p1_odds: float,
        p2_odds: float,
        uncertainty: float = 0.0,
        model_agreement: float = 1.0
    ) -> Dict:
        """
        Calculate betting recommendation
        
        Args:
            our_probability: Our predicted probability for player 1 (0-1)
            p1_odds: Player 1 decimal odds
            p2_odds: Player 2 decimal odds
            uncertainty: Model uncertainty (0-1, higher = less confident)
            model_agreement: Agreement between models (0-1, higher = more agreement)
        
        Returns:
            {
                'action': 'bet_player1' | 'bet_player2' | 'no_bet',
                'recommended_stake': float,
                'edge': float,
                'expected_value': float,
                'confidence': 'high' | 'medium' | 'low',
                'kelly_stake': float,
                'reason': str
            }
        """
        
        # Calculate edges
        p1_implied = 1 / p1_odds
        p2_implied = 1 / p2_odds
        
        p1_edge = our_probability - p1_implied
        p2_edge = (1 - our_probability) - p2_implied
        
        # Check uncertainty threshold
        if uncertainty > 0.5:
            return {
                'action': 'no_bet',
                'recommended_stake': 0,
                'edge': max(p1_edge, p2_edge),
                'expected_value': 0,
                'confidence': 'low',
                'reason': f'High uncertainty ({uncertainty:.2%}) - insufficient data'
            }
        
        # Check model agreement
        if model_agreement < 0.85:
            return {
                'action': 'no_bet',
                'recommended_stake': 0,
                'edge': max(p1_edge, p2_edge),
                'expected_value': 0,
                'confidence': 'low',
                'reason': f'Low model agreement ({model_agreement:.2%}) - models disagree'
            }
        
        # Determine which bet (if any)
        if p1_edge > self.min_edge and p1_edge > p2_edge:
            # Bet on player 1
            kelly_stake = self._calculate_kelly_stake(our_probability, p1_odds)
            recommended_stake = self._apply_risk_adjustments(
                kelly_stake, p1_edge, uncertainty, model_agreement
            )
            ev = (our_probability * p1_odds) - 1
            
            return {
                'action': 'bet_player1',
                'recommended_stake': recommended_stake,
                'edge': p1_edge,
                'expected_value': ev,
                'confidence': self._get_confidence_level(p1_edge, uncertainty, model_agreement),
                'kelly_stake': kelly_stake,
                'odds': p1_odds,
                'reason': f'{p1_edge:.2%} edge, {ev:.2%} EV'
            }
        
        elif p2_edge > self.min_edge and p2_edge > p1_edge:
            # Bet on player 2
            kelly_stake = self._calculate_kelly_stake(1 - our_probability, p2_odds)
            recommended_stake = self._apply_risk_adjustments(
                kelly_stake, p2_edge, uncertainty, model_agreement
            )
            ev = ((1 - our_probability) * p2_odds) - 1
            
            return {
                'action': 'bet_player2',
                'recommended_stake': recommended_stake,
                'edge': p2_edge,
                'expected_value': ev,
                'confidence': self._get_confidence_level(p2_edge, uncertainty, model_agreement),
                'kelly_stake': kelly_stake,
                'odds': p2_odds,
                'reason': f'{p2_edge:.2%} edge, {ev:.2%} EV'
            }
        
        else:
            # No bet - insufficient edge
            return {
                'action': 'no_bet',
                'recommended_stake': 0,
                'edge': max(p1_edge, p2_edge),
                'expected_value': 0,
                'confidence': 'low',
                'reason': f'Insufficient edge (p1: {p1_edge:.2%}, p2: {p2_edge:.2%}, min: {self.min_edge:.2%})'
            }
    
    def _calculate_kelly_stake(self, win_prob: float, decimal_odds: float) -> float:
        """
        Calculate Kelly Criterion stake
        
        Kelly formula: f = (bp - q) / b
        where:
        - f = fraction of bankroll to bet
        - b = odds - 1 (net odds)
        - p = probability of winning
        - q = probability of losing (1 - p)
        
        Args:
            win_prob: Probability of winning (0-1)
            decimal_odds: Decimal odds
        
        Returns:
            Recommended stake in dollars
        """
        
        if win_prob <= 0 or win_prob >= 1:
            return 0
        
        b = decimal_odds - 1  # Net odds
        p = win_prob
        q = 1 - p
        
        # Kelly fraction
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly (e.g., 0.25 for quarter Kelly)
        kelly_fraction *= self.kelly_fraction
        
        # Cap at max bet percentage
        kelly_fraction = min(kelly_fraction, self.max_bet_percent)
        
        # Ensure non-negative
        kelly_fraction = max(kelly_fraction, 0)
        
        # Convert to dollar amount
        stake = kelly_fraction * self.bankroll
        
        return stake
    
    def _apply_risk_adjustments(
        self,
        base_stake: float,
        edge: float,
        uncertainty: float,
        model_agreement: float
    ) -> float:
        """
        Apply risk adjustments to base stake
        
        Reduce stake when:
        - High uncertainty
        - Low model agreement
        - Small edge
        
        Args:
            base_stake: Base Kelly stake
            edge: Betting edge
            uncertainty: Model uncertainty (0-1)
            model_agreement: Model agreement (0-1)
        
        Returns:
            Risk-adjusted stake
        """
        
        # Start with base stake
        adjusted_stake = base_stake
        
        # Uncertainty adjustment (reduce stake for high uncertainty)
        uncertainty_factor = 1 - (uncertainty * 0.5)  # Max 50% reduction
        adjusted_stake *= uncertainty_factor
        
        # Model agreement adjustment (reduce stake when models disagree)
        agreement_factor = 0.5 + (model_agreement * 0.5)  # 0.5 to 1.0 range
        adjusted_stake *= agreement_factor
        
        # Edge size adjustment (reduce stake for small edges)
        if edge < 0.05:  # Less than 5% edge
            edge_factor = edge / 0.05  # Scale down
            adjusted_stake *= edge_factor
        
        # Minimum stake threshold ($5)
        if adjusted_stake < 5:
            adjusted_stake = 0
        
        # Round to nearest dollar
        adjusted_stake = round(adjusted_stake, 0)
        
        return adjusted_stake
    
    def _get_confidence_level(
        self,
        edge: float,
        uncertainty: float,
        model_agreement: float
    ) -> str:
        """
        Classify bet confidence
        
        High: Large edge, low uncertainty, high agreement
        Medium: Moderate edge or moderate uncertainty
        Low: Small edge or high uncertainty or low agreement
        
        Args:
            edge: Betting edge
            uncertainty: Model uncertainty
            model_agreement: Model agreement
        
        Returns:
            'high' | 'medium' | 'low'
        """
        
        # High confidence criteria
        if edge > 0.05 and uncertainty < 0.3 and model_agreement > 0.95:
            return 'high'
        
        # Medium confidence criteria
        elif edge > 0.03 and uncertainty < 0.4 and model_agreement > 0.90:
            return 'medium'
        
        # Otherwise low confidence
        else:
            return 'low'
    
    def calculate_portfolio_stakes(self, bet_opportunities: List[Dict]) -> List[Dict]:
        """
        Optimize stakes across multiple simultaneous bets
        
        When betting on multiple matches at once, we need to:
        1. Ensure total stakes don't exceed safe limit
        2. Allocate bankroll proportionally to edge/confidence
        3. Account for correlation (matches at same tournament)
        
        Args:
            bet_opportunities: List of potential bets, each with:
                {
                    'match_id': str,
                    'recommended_stake': float,
                    'edge': float,
                    'confidence': str,
                    'expected_value': float
                }
        
        Returns:
            List of bets with adjusted stakes
        """
        
        if not bet_opportunities:
            return []
        
        # Calculate total recommended stakes
        total_stakes = sum(bet['recommended_stake'] for bet in bet_opportunities)
        
        # If total exceeds safe limit (50% of bankroll), scale down
        safe_limit = self.bankroll * 0.50
        
        if total_stakes > safe_limit:
            scale_factor = safe_limit / total_stakes
            
            for bet in bet_opportunities:
                bet['original_stake'] = bet['recommended_stake']
                bet['recommended_stake'] = round(bet['recommended_stake'] * scale_factor, 0)
                bet['stake_adjustment'] = f"Scaled by {scale_factor:.2%} (portfolio risk management)"
        
        return bet_opportunities
    
    def update_bankroll(self, result: str, stake: float, odds: float):
        """
        Update bankroll after bet result
        
        Args:
            result: 'win' or 'loss'
            stake: Amount bet
            odds: Decimal odds
        """
        
        if result == 'win':
            profit = stake * (odds - 1)
            self.bankroll += profit
            logger.info(f"✅ Win: ${profit:.2f} profit, bankroll now ${self.bankroll:.2f}")
        
        elif result == 'loss':
            self.bankroll -= stake
            logger.info(f"❌ Loss: ${stake:.2f}, bankroll now ${self.bankroll:.2f}")
        
        # Log progress toward goal
        roi = ((self.bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        logger.info(f"📊 ROI: {roi:+.2f}%")
    
    def get_bankroll_status(self) -> Dict:
        """
        Get current bankroll status
        
        Returns:
            {
                'current': float,
                'initial': float,
                'profit_loss': float,
                'roi_percent': float,
                'progress_to_goal': float (if goal set)
            }
        """
        
        profit_loss = self.bankroll - self.initial_bankroll
        roi = (profit_loss / self.initial_bankroll) * 100
        
        return {
            'current': self.bankroll,
            'initial': self.initial_bankroll,
            'profit_loss': profit_loss,
            'roi_percent': roi
        }


# Convenience function

def calculate_kelly_stake(win_probability: float, decimal_odds: float,
                         bankroll: float = 1000, kelly_fraction: float = 0.25) -> float:
    """
    Calculate Kelly Criterion stake
    
    Args:
        win_probability: Probability of winning (0-1)
        decimal_odds: Decimal odds
        bankroll: Total bankroll
        kelly_fraction: Fraction of Kelly to use (0.25 = conservative)
    
    Returns:
        Recommended stake
    """
    calculator = BetCalculator(bankroll=bankroll, kelly_fraction=kelly_fraction)
    return calculator._calculate_kelly_stake(win_probability, decimal_odds)


if __name__ == "__main__":
    print("🎾 Testing Bet Calculator\n")
    
    # Initialize with $1000 bankroll
    calc = BetCalculator(bankroll=1000, kelly_fraction=0.25, min_edge=0.02)
    
    # Test case 1: Strong bet
    print("="*50)
    print("Test Case 1: Strong Value Bet")
    print("Our prediction: 65% win probability")
    print("Bookmaker odds: 1.85 (implies 54%)")
    
    rec = calc.calculate_bet_recommendation(
        our_probability=0.65,
        p1_odds=1.85,
        p2_odds=2.20,
        uncertainty=0.2,
        model_agreement=0.95
    )
    
    print(f"\nRecommendation:")
    print(f"  Action: {rec['action']}")
    print(f"  Stake: ${rec['recommended_stake']:.2f}")
    print(f"  Edge: {rec['edge']:.2%}")
    print(f"  Expected Value: {rec['expected_value']:.2%}")
    print(f"  Confidence: {rec['confidence']}")
    print(f"  Reason: {rec['reason']}")
    
    # Test case 2: Marginal bet
    print("\n" + "="*50)
    print("Test Case 2: Marginal Bet (High Uncertainty)")
    
    rec = calc.calculate_bet_recommendation(
        our_probability=0.55,
        p1_odds=1.90,
        p2_odds=2.00,
        uncertainty=0.6,  # High uncertainty
        model_agreement=0.80  # Low agreement
    )
    
    print(f"\nRecommendation:")
    print(f"  Action: {rec['action']}")
    print(f"  Reason: {rec['reason']}")
    
    # Test case 3: Portfolio management
    print("\n" + "="*50)
    print("Test Case 3: Portfolio Management (3 simultaneous bets)")
    
    opportunities = [
        {'match_id': 'match1', 'recommended_stake': 100, 'edge': 0.08, 'confidence': 'high', 'expected_value': 0.15},
        {'match_id': 'match2', 'recommended_stake': 80, 'edge': 0.06, 'confidence': 'medium', 'expected_value': 0.10},
        {'match_id': 'match3', 'recommended_stake': 150, 'edge': 0.10, 'confidence': 'high', 'expected_value': 0.20},
    ]
    
    adjusted = calc.calculate_portfolio_stakes(opportunities)
    
    print("\nAdjusted Stakes:")
    for bet in adjusted:
        if 'original_stake' in bet:
            print(f"  {bet['match_id']}: ${bet['original_stake']:.0f} → ${bet['recommended_stake']:.0f}")
        else:
            print(f"  {bet['match_id']}: ${bet['recommended_stake']:.0f} (no adjustment)")
    
    print(f"\nTotal: ${sum(b['recommended_stake'] for b in adjusted):.0f} / ${calc.bankroll * 0.5:.0f} safe limit")
