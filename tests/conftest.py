"""
Pytest configuration and fixtures for Tennis Prediction Tests
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_match_data():
    """Sample match data for testing."""
    return pd.DataFrame({
        'match_id': [1, 2, 3, 4, 5],
        'player1_id': [101, 102, 103, 101, 104],
        'player2_id': [102, 103, 104, 103, 101],
        'winner': [1, 2, 1, 1, 2],
        'surface': ['Hard', 'Clay', 'Grass', 'Hard', 'Clay'],
        'player1_serve': [0.65, 0.62, 0.68, 0.66, 0.60],
        'player2_serve': [0.62, 0.64, 0.63, 0.61, 0.67],
        'player1_return': [0.40, 0.38, 0.42, 0.41, 0.36],
        'player2_return': [0.38, 0.41, 0.38, 0.37, 0.43],
    })


@pytest.fixture
def sample_predictions():
    """Sample model predictions for testing."""
    return pd.DataFrame({
        'match_id': [1, 2, 3, 4, 5],
        'p_player1_win': [0.55, 0.48, 0.62, 0.58, 0.42],
        'actual_winner': [1, 2, 1, 1, 2],
    })


@pytest.fixture
def sample_odds():
    """Sample betting odds for testing."""
    return pd.DataFrame({
        'match_id': [1, 2, 3, 4, 5],
        'player1_odds': [1.85, 2.10, 1.70, 1.80, 2.30],
        'player2_odds': [2.00, 1.75, 2.20, 2.05, 1.65],
    })


@pytest.fixture
def model_weights():
    """Sample model weights for testing."""
    return {
        'serve_diff': 5.0,
        'return_diff': 4.0,
        'ace_diff': 2.0,
        'bp_saved_diff': 3.0,
    }


@pytest.fixture
def bankroll_config():
    """Bankroll management configuration."""
    return {
        'initial_bankroll': 1000,
        'kelly_fraction': 0.25,
        'max_stake_pct': 0.05,
        'min_edge': 0.02,
    }


# Random seed for reproducibility
@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
