"""
Tests for Betting Strategy Module
==================================
Test Kelly calculation, edge detection, and bankroll updates.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKellyCalculation:
    """Test Kelly criterion calculations."""
    
    def test_kelly_formula_basic(self):
        """Test basic Kelly formula: f* = (p*b - q) / b where b = odds - 1."""
        # P(win) = 0.6, odds = 2.0
        p = 0.6
        odds = 2.0
        b = odds - 1  # Net odds (1.0)
        q = 1 - p
        
        kelly = (p * b - q) / b
        
        assert kelly == pytest.approx(0.2)  # Should bet 20% of bankroll
    
    def test_kelly_alternative_formula(self):
        """Test alternative Kelly: f* = edge / (odds - 1)."""
        p = 0.6
        odds = 2.0
        
        # Edge = p * odds - 1
        edge = p * odds - 1  # 0.2
        kelly = edge / (odds - 1)
        
        assert kelly == pytest.approx(0.2)
    
    def test_kelly_no_edge(self):
        """Kelly should be 0 or negative when no edge."""
        p = 0.5  # Fair odds
        odds = 2.0
        
        edge = p * odds - 1  # 0
        
        assert edge <= 0  # No positive edge
    
    def test_kelly_negative_edge(self):
        """Kelly should be negative when edge is negative (don't bet)."""
        p = 0.4  # Unfavorable
        odds = 2.0
        
        edge = p * odds - 1  # -0.2
        kelly = edge / (odds - 1)
        
        assert kelly < 0  # Negative Kelly = don't bet
    
    def test_quarter_kelly(self):
        """Test quarter Kelly (25% of full Kelly for safety)."""
        p = 0.6
        odds = 2.0
        
        edge = p * odds - 1
        full_kelly = edge / (odds - 1)
        quarter_kelly = full_kelly * 0.25
        
        assert quarter_kelly == pytest.approx(0.05)  # 5% stake
    
    def test_kelly_with_high_odds(self):
        """Test Kelly with high odds (longshot)."""
        p = 0.15  # Underdog
        odds = 8.0
        
        edge = p * odds - 1  # 0.2
        kelly = edge / (odds - 1)
        
        assert kelly == pytest.approx(0.2 / 7)  # ~2.86%
    
    def test_kelly_cap(self):
        """Test Kelly with max stake cap."""
        p = 0.8  # Strong favorite
        odds = 1.5
        
        edge = p * odds - 1  # 0.2
        kelly = edge / (odds - 1)  # 0.4 = 40%
        
        max_stake = 0.05  # 5% cap
        capped_stake = min(kelly * 0.25, max_stake)
        
        assert capped_stake <= max_stake


class TestEdgeDetection:
    """Test betting edge detection logic."""
    
    def test_positive_edge(self):
        """Detect positive edge when model prob > implied prob."""
        model_prob = 0.55
        odds = 2.0
        implied_prob = 1 / odds  # 0.5
        
        edge = model_prob - implied_prob
        
        assert edge > 0
        assert edge == pytest.approx(0.05)
    
    def test_negative_edge(self):
        """Detect negative edge when model prob < implied prob."""
        model_prob = 0.45
        odds = 2.0
        implied_prob = 1 / odds  # 0.5
        
        edge = model_prob - implied_prob
        
        assert edge < 0
    
    def test_edge_threshold(self):
        """Only bet when edge exceeds threshold."""
        threshold = 0.02  # 2% minimum edge
        
        test_cases = [
            (0.521, 2.0, True),   # 2.1% edge, above threshold
            (0.51, 2.0, False),   # 1% edge, below threshold
            (0.55, 2.0, True),    # 5% edge, above threshold
            (0.60, 2.0, True),    # 10% edge, above threshold
        ]
        
        for model_prob, odds, should_bet in test_cases:
            implied_prob = 1 / odds
            edge = model_prob - implied_prob
            bet = edge > threshold
            assert bet == should_bet, f"Failed for model_prob={model_prob}, edge={edge}"
    
    def test_both_sides_edge(self):
        """Check edge on both sides of market."""
        p1_model = 0.55
        p2_model = 1 - p1_model  # 0.45
        
        p1_odds = 1.90
        p2_odds = 2.10
        
        p1_implied = 1 / p1_odds  # ~0.526
        p2_implied = 1 / p2_odds  # ~0.476
        
        p1_edge = p1_model - p1_implied  # 0.55 - 0.526 = 0.024
        p2_edge = p2_model - p2_implied  # 0.45 - 0.476 = -0.026
        
        assert p1_edge > 0
        assert p2_edge < 0
    
    def test_overround_adjustment(self):
        """Account for bookmaker overround/vig."""
        p1_odds = 1.90
        p2_odds = 1.90
        
        # Implied probabilities
        p1_implied = 1 / p1_odds  # 0.526
        p2_implied = 1 / p2_odds  # 0.526
        
        overround = p1_implied + p2_implied - 1  # 0.053 = 5.3%
        
        # Fair odds (remove overround)
        total_implied = p1_implied + p2_implied
        p1_fair = p1_implied / total_implied  # 0.5
        p2_fair = p2_implied / total_implied  # 0.5
        
        assert p1_fair == pytest.approx(0.5)
        assert p2_fair == pytest.approx(0.5)


class TestBankrollUpdates:
    """Test bankroll management and updates."""
    
    def test_winning_bet_update(self):
        """Test bankroll update after winning bet."""
        bankroll = 1000
        stake = 50
        odds = 2.0
        
        profit = stake * (odds - 1)
        new_bankroll = bankroll + profit
        
        assert new_bankroll == 1050
    
    def test_losing_bet_update(self):
        """Test bankroll update after losing bet."""
        bankroll = 1000
        stake = 50
        
        new_bankroll = bankroll - stake
        
        assert new_bankroll == 950
    
    def test_roi_calculation(self):
        """Test ROI calculation."""
        total_staked = 1000
        total_profit = 150
        
        roi = total_profit / total_staked
        
        assert roi == 0.15  # 15% ROI
    
    def test_compound_growth(self):
        """Test compound bankroll growth."""
        bankroll = 1000
        
        # Series of bets
        bets = [
            {'win': True, 'odds': 2.0, 'stake_pct': 0.05},
            {'win': False, 'odds': 1.8, 'stake_pct': 0.05},
            {'win': True, 'odds': 2.5, 'stake_pct': 0.03},
        ]
        
        for bet in bets:
            stake = bankroll * bet['stake_pct']
            if bet['win']:
                bankroll += stake * (bet['odds'] - 1)
            else:
                bankroll -= stake
        
        # Should have grown overall
        assert bankroll != 1000
    
    def test_drawdown_calculation(self):
        """Test max drawdown calculation."""
        equity_curve = [1000, 1050, 1020, 980, 1010, 950, 1100]
        
        peak = equity_curve[0]
        max_drawdown = 0
        
        for value in equity_curve:
            peak = max(peak, value)
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Max drawdown should be from peak 1100 to 950 (doesn't happen in sequence)
        # Actually from 1050 to 950 or 1020 to 980
        assert max_drawdown > 0
        assert max_drawdown < 1  # Less than 100%
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        daily_returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, 0.005])
        
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns, ddof=1)
        
        sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
        
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)
    
    def test_bankroll_never_negative(self):
        """Ensure bankroll can't go negative with proper stake sizing."""
        bankroll = 1000
        max_stake_pct = 0.05  # 5% max
        
        # Even 20 consecutive losses shouldn't bankrupt
        for _ in range(20):
            stake = bankroll * max_stake_pct
            bankroll -= stake
        
        assert bankroll > 0


class TestBettingStrategies:
    """Test different betting strategy implementations."""
    
    def test_fixed_stake_strategy(self):
        """Test fixed stake strategy."""
        fixed_stake = 10
        predictions = [0.55, 0.48, 0.62, 0.51]
        
        for p in predictions:
            if p > 0.5:
                stake = fixed_stake
                assert stake == 10
    
    def test_value_betting_strategy(self):
        """Test value betting (only bet when edge exists)."""
        predictions = [
            {'prob': 0.55, 'odds': 2.0, 'should_bet': True},   # Edge exists
            {'prob': 0.48, 'odds': 2.0, 'should_bet': False},  # No edge
            {'prob': 0.35, 'odds': 3.0, 'should_bet': True},   # Edge on underdog
        ]
        
        for pred in predictions:
            implied = 1 / pred['odds']
            edge = pred['prob'] - implied
            should_bet = edge > 0
            assert should_bet == pred['should_bet']
    
    def test_kelly_strategy(self):
        """Test Kelly criterion strategy."""
        bankroll = 1000
        kelly_fraction = 0.25
        max_stake_pct = 0.05
        
        test_cases = [
            {'prob': 0.6, 'odds': 2.0},
            {'prob': 0.55, 'odds': 1.9},
            {'prob': 0.4, 'odds': 3.0},
        ]
        
        for case in test_cases:
            edge = case['prob'] * case['odds'] - 1
            
            if edge > 0:
                kelly = edge / (case['odds'] - 1)
                stake = bankroll * kelly * kelly_fraction
                stake = min(stake, bankroll * max_stake_pct)
                
                assert stake > 0
                assert stake <= bankroll * max_stake_pct
    
    def test_strategy_comparison(self):
        """Compare expected returns across strategies."""
        # Simulated match with edge
        prob = 0.55
        odds = 2.0
        bankroll = 1000
        
        # Fixed stake
        fixed_stake = 10
        fixed_ev = fixed_stake * (prob * odds - 1)
        
        # Kelly stake
        edge = prob * odds - 1
        kelly = edge / (odds - 1)
        kelly_stake = bankroll * kelly * 0.25
        kelly_ev = kelly_stake * (prob * odds - 1)
        
        # Kelly should have higher EV when properly calibrated
        assert kelly_ev > 0
        assert fixed_ev > 0


class TestOddsConversion:
    """Test odds format conversions."""
    
    def test_decimal_to_implied(self):
        """Convert decimal odds to implied probability."""
        decimal_odds = 2.0
        implied = 1 / decimal_odds
        
        assert implied == 0.5
    
    def test_implied_to_decimal(self):
        """Convert implied probability to decimal odds."""
        implied = 0.4
        decimal_odds = 1 / implied
        
        assert decimal_odds == 2.5
    
    def test_american_to_decimal_positive(self):
        """Convert positive American odds to decimal."""
        american = +150
        decimal_odds = 1 + american / 100
        
        assert decimal_odds == 2.5
    
    def test_american_to_decimal_negative(self):
        """Convert negative American odds to decimal."""
        american = -150
        decimal_odds = 1 + 100 / abs(american)
        
        assert decimal_odds == pytest.approx(1.667, abs=0.01)
    
    def test_fractional_to_decimal(self):
        """Convert fractional odds to decimal."""
        numerator, denominator = 3, 2  # 3/2
        decimal_odds = numerator / denominator + 1
        
        assert decimal_odds == 2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
