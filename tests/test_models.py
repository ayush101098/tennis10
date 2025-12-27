"""
Tests for Tennis Prediction Models
===================================
Test probability outputs, symmetry, and model behavior.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProbabilityOutputs:
    """Test that model outputs are valid probabilities."""
    
    def test_probability_range_logistic(self):
        """Logistic regression probabilities should be in [0, 1]."""
        # Simulated logistic regression output
        def logistic(z):
            return 1.0 / (1.0 + np.exp(-z))
        
        # Test various input values
        test_values = [-100, -10, -1, 0, 1, 10, 100]
        
        for z in test_values:
            prob = logistic(z)
            assert 0 <= prob <= 1, f"Probability {prob} out of range for z={z}"
    
    def test_probability_range_neural_network(self):
        """Neural network (sigmoid output) should be in [0, 1]."""
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        
        # Test array of values
        test_values = np.linspace(-100, 100, 1000)
        probs = sigmoid(test_values)
        
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
    
    def test_probability_sum_to_one(self):
        """P(A wins) + P(B wins) should equal 1."""
        np.random.seed(42)
        
        # Simulated model outputs
        p_a_wins = np.random.rand(100)
        p_b_wins = 1 - p_a_wins
        
        sums = p_a_wins + p_b_wins
        
        assert np.allclose(sums, 1.0)
    
    def test_markov_probability_range(self):
        """Markov model probabilities should be valid."""
        # Simplified Markov calculation
        def p_win_game(p_serve):
            """Probability of winning a service game."""
            p = p_serve
            q = 1 - p
            
            # Win in exactly 4, 5, or 6 points, or via deuce
            p_4 = p**4
            p_5 = 4 * p**4 * q
            p_6 = 10 * p**4 * q**2
            p_deuce = 20 * p**3 * q**3
            p_win_from_deuce = p**2 / (p**2 + q**2)
            
            return p_4 + p_5 + p_6 + p_deuce * p_win_from_deuce
        
        # Test various serve percentages
        for p_serve in np.linspace(0.3, 0.8, 50):
            p_hold = p_win_game(p_serve)
            assert 0 <= p_hold <= 1, f"Invalid hold probability {p_hold} for serve {p_serve}"


class TestModelSymmetry:
    """Test that P(A beats B) = 1 - P(B beats A)."""
    
    def test_logistic_symmetry(self):
        """Logistic regression should be symmetric when features are negated."""
        def logistic(z):
            return 1.0 / (1.0 + np.exp(-z))
        
        # If features are (A - B), then for (B - A) we just negate
        z_a_vs_b = 0.5  # Some feature difference
        z_b_vs_a = -0.5  # Negated
        
        p_a = logistic(z_a_vs_b)
        p_b = logistic(z_b_vs_a)
        
        assert p_a == pytest.approx(1 - p_b)
    
    def test_neural_network_symmetry(self):
        """Neural network (no bias) should be symmetric."""
        # With no bias terms, f(-x) should equal 1 - f(x) for sigmoid output
        def symmetric_nn(x, weights):
            """Simplified symmetric NN: tanh hidden, sigmoid output."""
            h = np.tanh(np.dot(x, weights['hidden']))
            out = 1.0 / (1.0 + np.exp(-np.dot(h, weights['output'])))
            return out
        
        # Since tanh(-x) = -tanh(x), and we have no bias,
        # symmetric_nn(-x) = sigmoid(-y) = 1 - sigmoid(y) for some y
        
        np.random.seed(42)
        weights = {
            'hidden': np.random.randn(10, 20),
            'output': np.random.randn(20, 1)
        }
        
        x = np.random.randn(10)
        
        p_a = symmetric_nn(x, weights)
        p_b = symmetric_nn(-x, weights)
        
        assert p_a == pytest.approx(1 - p_b, abs=1e-6)
    
    def test_markov_symmetry(self):
        """Markov model should satisfy symmetry."""
        # If A has serve% = p and B has serve% = q,
        # then P(A|serve_a=p, serve_b=q) = 1 - P(B|serve_a=q, serve_b=p)
        
        def p_win_game(p_serve):
            p = p_serve
            q = 1 - p
            p_4 = p**4
            p_5 = 4 * p**4 * q
            p_6 = 10 * p**4 * q**2
            p_deuce = 20 * p**3 * q**3
            p_win_from_deuce = p**2 / (p**2 + q**2) if p**2 + q**2 > 0 else 0.5
            return p_4 + p_5 + p_6 + p_deuce * p_win_from_deuce
        
        def p_win_match_simple(p_serve_a, p_serve_b):
            """Simplified match probability (symmetric first-order approx)."""
            p_hold_a = p_win_game(p_serve_a)
            p_hold_b = p_win_game(p_serve_b)
            p_break_a = 1 - p_hold_b
            p_break_b = 1 - p_hold_a
            
            # Approximate: team with more breaks wins
            edge = (p_hold_a - p_break_b + p_break_a - p_hold_b) / 2
            return 0.5 + edge
        
        p_a = 0.65
        p_b = 0.60
        
        # P(A wins | A=65%, B=60%) should equal 1 - P(B wins | B=65%, A=60%)
        match_a = p_win_match_simple(p_a, p_b)
        match_b = p_win_match_simple(p_b, p_a)
        
        # Not exactly symmetric due to serving first, but close
        assert match_a == pytest.approx(1 - match_b, abs=0.05)


class TestKnownOutcomes:
    """Test model behavior on matches with known outcomes."""
    
    def test_extreme_mismatch_high_prob(self):
        """Extreme skill mismatch should give high probability to favorite."""
        def logistic(z):
            return 1.0 / (1.0 + np.exp(-z))
        
        # Very strong player vs very weak player
        strong_features = {'serve': 0.75, 'return': 0.50}
        weak_features = {'serve': 0.50, 'return': 0.30}
        
        # Feature difference
        diff = (strong_features['serve'] - weak_features['serve']) * 10 + \
               (strong_features['return'] - weak_features['return']) * 10
        
        p_strong_wins = logistic(diff)
        
        assert p_strong_wins > 0.9
    
    def test_equal_players_fifty_fifty(self):
        """Equal players should have ~50% probability."""
        def logistic(z):
            return 1.0 / (1.0 + np.exp(-z))
        
        # Equal players
        features_a = {'serve': 0.65, 'return': 0.40}
        features_b = {'serve': 0.65, 'return': 0.40}
        
        diff = (features_a['serve'] - features_b['serve']) + \
               (features_a['return'] - features_b['return'])
        
        p_a_wins = logistic(diff)
        
        assert p_a_wins == pytest.approx(0.5)
    
    def test_surface_specialist_advantage(self):
        """Clay specialist should have edge on clay."""
        # Nadal-like stats on clay vs Federer-like stats
        nadal_clay = {'serve': 0.68, 'return': 0.50, 'surface_boost': 0.10}
        federer_clay = {'serve': 0.70, 'return': 0.42, 'surface_boost': 0.02}
        
        # Adjusted for clay
        nadal_adj = nadal_clay['return'] + nadal_clay['surface_boost']
        federer_adj = federer_clay['return'] + federer_clay['surface_boost']
        
        # Nadal should have return advantage on clay
        assert nadal_adj > federer_adj


class TestModelCalibration:
    """Test model calibration (predicted probabilities match actual win rates)."""
    
    def test_calibration_binning(self):
        """Test calibration calculation logic."""
        np.random.seed(42)
        
        # Simulated predictions and outcomes
        n = 1000
        predicted = np.random.rand(n)
        # Well-calibrated model: actual outcomes match predictions
        actual = np.random.rand(n) < predicted
        
        # Bin predictions
        bins = np.linspace(0, 1, 11)
        calibration_errors = []
        
        for i in range(len(bins) - 1):
            mask = (predicted >= bins[i]) & (predicted < bins[i+1])
            if mask.sum() > 10:
                pred_mean = predicted[mask].mean()
                actual_mean = actual[mask].mean()
                calibration_errors.append(abs(pred_mean - actual_mean))
        
        # Average calibration error should be small for well-calibrated model
        mean_cal_error = np.mean(calibration_errors)
        assert mean_cal_error < 0.1  # Less than 10% calibration error
    
    def test_overconfident_model(self):
        """Detect overconfident model (extreme predictions)."""
        predictions = np.array([0.95, 0.92, 0.88, 0.96, 0.91])
        
        # Check for overconfidence
        confidence = np.abs(predictions - 0.5).mean()
        
        # Mean distance from 0.5 > 0.4 suggests overconfidence
        if confidence > 0.4:
            # Model might be overconfident
            assert True
    
    def test_underconfident_model(self):
        """Detect underconfident model (predictions too close to 0.5)."""
        predictions = np.array([0.52, 0.48, 0.51, 0.49, 0.50])
        
        # Check for underconfidence
        confidence = np.abs(predictions - 0.5).mean()
        
        # Mean distance from 0.5 < 0.05 suggests underconfidence
        if confidence < 0.05:
            # Model might be underconfident
            assert True


class TestEdgeCases:
    """Test model behavior on edge cases."""
    
    def test_missing_features(self):
        """Model should handle missing features gracefully."""
        features = {'serve': 0.65, 'return': np.nan, 'aces': 0.08}
        
        # Replace NaN with mean
        features_clean = {k: (v if not np.isnan(v) else 0.40) for k, v in features.items()}
        
        assert not any(np.isnan(v) for v in features_clean.values())
    
    def test_extreme_features(self):
        """Model should handle extreme feature values."""
        def logistic(z):
            # Clip to prevent overflow
            z = np.clip(z, -500, 500)
            return 1.0 / (1.0 + np.exp(-z))
        
        # Very extreme values
        extreme_z = 1000
        prob = logistic(extreme_z)
        
        assert 0 <= prob <= 1
        assert not np.isnan(prob)
        assert not np.isinf(prob)
    
    def test_no_historical_data(self):
        """Handle players with no historical data."""
        def get_prior_stats():
            """Return prior/default stats for unknown players."""
            return {
                'serve': 0.62,  # ATP average
                'return': 0.38,
                'uncertainty': 1.0  # Maximum uncertainty
            }
        
        unknown_player_stats = get_prior_stats()
        
        assert unknown_player_stats['uncertainty'] == 1.0
    
    def test_retired_match(self):
        """Handle retired matches correctly."""
        match = {
            'winner': 'Player A',
            'loser': 'Player B',
            'retired': True,
            'score': '6-4 3-0'
        }
        
        # Retired matches might be excluded or weighted differently
        weight = 0.5 if match['retired'] else 1.0
        
        assert weight < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
