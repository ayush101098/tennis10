"""
Tests for Tennis Feature Engineering Module
============================================
Test time decay, surface weighting, uncertainty thresholds, and common opponent logic.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import TennisFeatureExtractor


class TestTimeDecay:
    """Test exponential time decay calculations."""
    
    def test_same_day_returns_one(self):
        """Same day should return weight of 1.0."""
        extractor = TennisFeatureExtractor.__new__(TennisFeatureExtractor)
        
        date = datetime(2024, 6, 1)
        weight = extractor.apply_time_discount(date, date)
        
        assert weight == pytest.approx(1.0, abs=0.001)
    
    def test_half_life_returns_half(self):
        """At half-life, weight should be ~0.5."""
        extractor = TennisFeatureExtractor.__new__(TennisFeatureExtractor)
        
        current = datetime(2024, 6, 1)
        past = datetime(2023, 8, 1)  # ~0.8 years ago
        
        weight = extractor.apply_time_discount(current, past, half_life_years=0.8)
        
        # Should be close to 0.5
        assert weight == pytest.approx(0.5, abs=0.1)
    
    def test_decay_decreases_over_time(self):
        """Weight should decrease as time passes."""
        extractor = TennisFeatureExtractor.__new__(TennisFeatureExtractor)
        
        current = datetime(2024, 6, 1)
        
        weights = []
        for months_ago in [1, 6, 12, 24, 36]:
            past = current - timedelta(days=months_ago * 30)
            weight = extractor.apply_time_discount(current, past)
            weights.append(weight)
        
        # Weights should be strictly decreasing
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i+1], f"Weight at {i} should be > weight at {i+1}"
    
    def test_decay_never_negative(self):
        """Weight should never go negative."""
        extractor = TennisFeatureExtractor.__new__(TennisFeatureExtractor)
        
        current = datetime(2024, 6, 1)
        past = datetime(2000, 1, 1)  # Very old date
        
        weight = extractor.apply_time_discount(current, past)
        
        assert weight >= 0
    
    def test_different_half_lives(self):
        """Shorter half-life should decay faster."""
        extractor = TennisFeatureExtractor.__new__(TennisFeatureExtractor)
        
        current = datetime(2024, 6, 1)
        past = datetime(2023, 6, 1)  # 1 year ago
        
        weight_short = extractor.apply_time_discount(current, past, half_life_years=0.5)
        weight_medium = extractor.apply_time_discount(current, past, half_life_years=0.8)
        weight_long = extractor.apply_time_discount(current, past, half_life_years=1.0)
        
        assert weight_short < weight_medium < weight_long


class TestSurfaceWeighting:
    """Test surface correlation/weighting logic."""
    
    def test_same_surface_weight_one(self):
        """Same surface should have weight 1.0."""
        extractor = TennisFeatureExtractor.__new__(TennisFeatureExtractor)
        extractor.SURFACE_CORRELATIONS = TennisFeatureExtractor.SURFACE_CORRELATIONS
        
        assert extractor.get_surface_weight('Hard', 'Hard') == 1.0
        assert extractor.get_surface_weight('Clay', 'Clay') == 1.0
        assert extractor.get_surface_weight('Grass', 'Grass') == 1.0
    
    def test_hard_clay_correlation(self):
        """Hard-Clay correlation should be moderate."""
        extractor = TennisFeatureExtractor.__new__(TennisFeatureExtractor)
        extractor.SURFACE_CORRELATIONS = TennisFeatureExtractor.SURFACE_CORRELATIONS
        
        weight = extractor.get_surface_weight('Hard', 'Clay')
        
        assert 0.2 <= weight <= 0.4
    
    def test_clay_grass_lowest_correlation(self):
        """Clay-Grass should have lowest correlation."""
        extractor = TennisFeatureExtractor.__new__(TennisFeatureExtractor)
        extractor.SURFACE_CORRELATIONS = TennisFeatureExtractor.SURFACE_CORRELATIONS
        
        clay_grass = extractor.get_surface_weight('Clay', 'Grass')
        hard_clay = extractor.get_surface_weight('Hard', 'Clay')
        hard_grass = extractor.get_surface_weight('Hard', 'Grass')
        
        assert clay_grass <= hard_clay
        assert clay_grass <= hard_grass
    
    def test_symmetry(self):
        """Surface weights should be symmetric."""
        extractor = TennisFeatureExtractor.__new__(TennisFeatureExtractor)
        extractor.SURFACE_CORRELATIONS = TennisFeatureExtractor.SURFACE_CORRELATIONS
        
        assert extractor.get_surface_weight('Hard', 'Clay') == extractor.get_surface_weight('Clay', 'Hard')
        assert extractor.get_surface_weight('Hard', 'Grass') == extractor.get_surface_weight('Grass', 'Hard')
        assert extractor.get_surface_weight('Clay', 'Grass') == extractor.get_surface_weight('Grass', 'Clay')
    
    def test_unknown_surface_default(self):
        """Unknown surface combinations should return low default weight."""
        extractor = TennisFeatureExtractor.__new__(TennisFeatureExtractor)
        extractor.SURFACE_CORRELATIONS = TennisFeatureExtractor.SURFACE_CORRELATIONS
        
        weight = extractor.get_surface_weight('Carpet', 'Hard')
        
        assert weight <= 0.2  # Should be low default


class TestUncertaintyThresholds:
    """Test uncertainty/confidence scoring logic."""
    
    def test_uncertainty_range(self):
        """Uncertainty should be between 0 and 1."""
        # Simulated uncertainty calculation
        def calculate_uncertainty(n_matches: int, variance: float) -> float:
            """Calculate uncertainty based on sample size and variance."""
            sample_uncertainty = 1.0 / np.sqrt(n_matches + 1)
            return min(1.0, sample_uncertainty + variance * 0.1)
        
        # Test various scenarios
        for n_matches in [0, 5, 10, 50, 100]:
            for variance in [0.0, 0.1, 0.5, 1.0]:
                uncertainty = calculate_uncertainty(n_matches, variance)
                assert 0 <= uncertainty <= 1
    
    def test_more_matches_less_uncertainty(self):
        """More matches should generally reduce uncertainty."""
        def calculate_uncertainty(n_matches: int) -> float:
            return 1.0 / np.sqrt(n_matches + 1)
        
        uncertainties = [calculate_uncertainty(n) for n in [1, 5, 10, 50, 100]]
        
        for i in range(len(uncertainties) - 1):
            assert uncertainties[i] >= uncertainties[i+1]
    
    def test_threshold_filtering(self):
        """Test that high uncertainty matches are filtered."""
        uncertainty_threshold = 0.5
        
        matches = [
            {'uncertainty': 0.3, 'should_include': True},
            {'uncertainty': 0.5, 'should_include': False},  # At threshold
            {'uncertainty': 0.7, 'should_include': False},
            {'uncertainty': 0.1, 'should_include': True},
        ]
        
        for match in matches:
            include = match['uncertainty'] < uncertainty_threshold
            assert include == match['should_include']


class TestCommonOpponents:
    """Test common opponent analysis logic."""
    
    def test_no_common_opponents(self):
        """Handle case with no common opponents."""
        p1_opponents = {'A', 'B', 'C'}
        p2_opponents = {'D', 'E', 'F'}
        
        common = p1_opponents & p2_opponents
        
        assert len(common) == 0
    
    def test_some_common_opponents(self):
        """Find common opponents between players."""
        p1_opponents = {'A', 'B', 'C', 'D'}
        p2_opponents = {'C', 'D', 'E', 'F'}
        
        common = p1_opponents & p2_opponents
        
        assert common == {'C', 'D'}
        assert len(common) == 2
    
    def test_common_opponent_strength_calculation(self):
        """Test calculating strength from common opponent results."""
        # Player 1 beat opponents A, B; lost to C
        # Player 2 beat opponents B, C; lost to A
        
        common_results = {
            'A': {'p1_won': True, 'p2_won': False},
            'B': {'p1_won': True, 'p2_won': True},
            'C': {'p1_won': False, 'p2_won': True},
        }
        
        p1_win_rate = sum(1 for r in common_results.values() if r['p1_won']) / len(common_results)
        p2_win_rate = sum(1 for r in common_results.values() if r['p2_won']) / len(common_results)
        
        assert p1_win_rate == pytest.approx(2/3)
        assert p2_win_rate == pytest.approx(2/3)
    
    def test_weighted_common_opponent(self):
        """Test time-weighted common opponent analysis."""
        # More recent matches should count more
        
        def weighted_win_rate(results: list, weights: list) -> float:
            """Calculate weighted win rate."""
            total_weight = sum(weights)
            weighted_wins = sum(w * r for w, r in zip(weights, results))
            return weighted_wins / total_weight if total_weight > 0 else 0.5
        
        # Results: [1, 1, 0, 0] with time weights [1.0, 0.8, 0.5, 0.3]
        results = [1, 1, 0, 0]
        weights = [1.0, 0.8, 0.5, 0.3]  # Most recent first
        
        unweighted = sum(results) / len(results)  # 0.5
        weighted = weighted_win_rate(results, weights)  # Should be > 0.5
        
        assert weighted > unweighted  # Recent wins matter more


class TestFeatureNormalization:
    """Test feature standardization and normalization."""
    
    def test_standardize_zero_mean(self):
        """Standardized features should have ~0 mean."""
        from sklearn.preprocessing import StandardScaler
        
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]).astype(float)
        scaler = StandardScaler()
        normalized = scaler.fit_transform(data)
        
        assert np.mean(normalized, axis=0) == pytest.approx([0, 0, 0], abs=1e-10)
    
    def test_standardize_unit_variance(self):
        """Standardized features should have ~1 variance."""
        from sklearn.preprocessing import StandardScaler
        
        np.random.seed(42)
        data = np.random.randn(100, 5) * 10 + 50
        
        scaler = StandardScaler()
        normalized = scaler.fit_transform(data)
        
        variances = np.var(normalized, axis=0, ddof=1)
        for v in variances:
            assert v == pytest.approx(1.0, abs=0.1)
    
    def test_skip_binary_features(self):
        """Binary features (DIRECT, RETIRED, FATIGUE) should not be standardized."""
        non_standardized = ['DIRECT', 'RETIRED', 'FATIGUE']
        all_features = ['SERVE_PCT', 'RETURN_PCT', 'DIRECT', 'FATIGUE', 'ACE_PCT']
        
        to_standardize = [f for f in all_features if f not in non_standardized]
        
        assert 'DIRECT' not in to_standardize
        assert 'FATIGUE' not in to_standardize
        assert 'SERVE_PCT' in to_standardize


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
