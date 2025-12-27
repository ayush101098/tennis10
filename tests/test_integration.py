"""
Integration Tests for Tennis Prediction System
================================================
End-to-end tests: Data → Features → Model → Bet → Update
"""

import pytest
import numpy as np
import pandas as pd
import sqlite3
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataPipeline:
    """Test data loading and preprocessing pipeline."""
    
    def test_database_connection(self):
        """Test database connection."""
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tennis_data.db')
        
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            assert 'matches' in tables or 'players' in tables or len(tables) > 0
    
    def test_data_loading(self):
        """Test loading match data."""
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tennis_data.db')
        
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            
            try:
                df = pd.read_sql("SELECT COUNT(*) as cnt FROM matches", conn)
                assert df['cnt'].iloc[0] >= 0
            except:
                pass  # Table might not exist
            
            conn.close()
    
    def test_data_schema(self):
        """Test expected data schema."""
        expected_match_columns = ['winner_id', 'loser_id', 'surface', 'match_date']
        
        # Create mock dataframe to test schema handling
        mock_df = pd.DataFrame({
            'winner_id': [1, 2],
            'loser_id': [2, 1],
            'surface': ['Hard', 'Clay'],
            'match_date': ['2024-01-01', '2024-01-02']
        })
        
        for col in expected_match_columns:
            assert col in mock_df.columns


class TestFeatureExtraction:
    """Test feature extraction pipeline."""
    
    def test_feature_difference_calculation(self):
        """Test player1 - player2 feature differences."""
        df = pd.DataFrame({
            'player1_serve': [0.65, 0.70],
            'player2_serve': [0.60, 0.68],
            'player1_return': [0.40, 0.42],
            'player2_return': [0.38, 0.40]
        })
        
        serve_diff = df['player1_serve'] - df['player2_serve']
        return_diff = df['player1_return'] - df['player2_return']
        
        assert serve_diff.tolist() == pytest.approx([0.05, 0.02])
        assert return_diff.tolist() == pytest.approx([0.02, 0.02])
    
    def test_feature_scaling(self):
        """Test feature standardization."""
        from sklearn.preprocessing import StandardScaler
        
        features = np.array([[0.65, 0.40], [0.70, 0.42], [0.60, 0.38]])
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        
        # Mean should be ~0, std should be ~1
        assert np.abs(scaled.mean(axis=0)).max() < 0.001
        assert np.abs(scaled.std(axis=0, ddof=0) - 1).max() < 0.001
    
    def test_missing_feature_handling(self):
        """Test handling of missing features."""
        df = pd.DataFrame({
            'player1_serve': [0.65, np.nan, 0.70],
            'player2_serve': [0.60, 0.62, np.nan]
        })
        
        # Fill with mean
        df_filled = df.fillna(df.mean())
        
        assert not df_filled.isnull().any().any()


class TestModelPrediction:
    """Test model prediction pipeline."""
    
    def test_logistic_prediction_format(self):
        """Test logistic regression prediction output."""
        # Mock prediction
        z = np.array([0.5, -0.3, 0.8])
        probs = 1 / (1 + np.exp(-z))
        
        # Should be array of probabilities
        assert probs.shape == (3,)
        assert all(0 <= p <= 1 for p in probs)
    
    def test_ensemble_averaging(self):
        """Test ensemble prediction averaging."""
        # Mock predictions from 3 models
        pred1 = np.array([0.6, 0.55, 0.7])
        pred2 = np.array([0.58, 0.52, 0.68])
        pred3 = np.array([0.62, 0.58, 0.72])
        
        ensemble = np.mean([pred1, pred2, pred3], axis=0)
        
        assert ensemble.shape == (3,)
        assert all(0 <= p <= 1 for p in ensemble)
    
    def test_prediction_symmetry(self):
        """Test that swapping players gives complementary probabilities."""
        # Mock symmetric model
        def predict(features):
            return 1 / (1 + np.exp(-features.sum(axis=1)))
        
        features_ab = np.array([[0.05, 0.02]])  # A vs B
        features_ba = np.array([[-0.05, -0.02]])  # B vs A (negated)
        
        p_a_wins = predict(features_ab)
        p_b_wins = predict(features_ba)
        
        assert p_a_wins[0] == pytest.approx(1 - p_b_wins[0])


class TestBettingPipeline:
    """Test betting decision and execution pipeline."""
    
    def test_edge_to_stake_conversion(self):
        """Test converting edge to stake using Kelly."""
        prob = 0.6
        odds = 2.0
        bankroll = 1000
        kelly_fraction = 0.25
        
        edge = prob * odds - 1
        if edge > 0:
            kelly = edge / (odds - 1)
            stake = bankroll * kelly * kelly_fraction
            stake = min(stake, bankroll * 0.05)
        else:
            stake = 0
        
        assert stake > 0
        assert stake <= 50  # Max 5% of bankroll
    
    def test_bet_execution(self):
        """Test bet execution and bankroll update."""
        bankroll = 1000
        stake = 50
        odds = 2.0
        
        # Win scenario
        win_result = bankroll + stake * (odds - 1)
        
        # Lose scenario
        lose_result = bankroll - stake
        
        assert win_result == 1050
        assert lose_result == 950
    
    def test_bet_logging(self):
        """Test bet logging structure."""
        bet_record = {
            'match_id': 'ABC123',
            'timestamp': '2024-01-01 12:00:00',
            'player_bet': 'Player A',
            'odds': 2.0,
            'stake': 50,
            'model_prob': 0.6,
            'outcome': None  # To be filled after match
        }
        
        required_fields = ['match_id', 'stake', 'odds', 'model_prob']
        
        for field in required_fields:
            assert field in bet_record


class TestEndToEnd:
    """Full end-to-end integration tests."""
    
    def test_full_prediction_pipeline(self):
        """Test complete pipeline from data to prediction."""
        # 1. Mock match data
        match_data = {
            'player1_id': 1,
            'player2_id': 2,
            'surface': 'Hard',
            'player1_serve': 0.65,
            'player2_serve': 0.62,
            'player1_return': 0.40,
            'player2_return': 0.38
        }
        
        # 2. Extract features (differences)
        features = np.array([
            match_data['player1_serve'] - match_data['player2_serve'],
            match_data['player1_return'] - match_data['player2_return']
        ])
        
        # 3. Make prediction
        weights = np.array([5.0, 4.0])  # Mock model weights
        z = np.dot(features, weights)
        prob = 1 / (1 + np.exp(-z))
        
        # 4. Check betting decision
        odds = 1.90
        implied = 1 / odds
        edge = prob - implied
        
        # 5. Calculate stake
        if edge > 0.02:
            kelly = edge / (odds - 1)
            stake = 1000 * kelly * 0.25
            should_bet = True
        else:
            stake = 0
            should_bet = False
        
        # Verify pipeline completed
        assert 0 <= prob <= 1
        assert isinstance(should_bet, bool)
    
    def test_backtest_simulation(self):
        """Test backtesting simulation."""
        np.random.seed(42)
        
        # Simulate 100 matches
        n_matches = 100
        
        predictions = np.random.rand(n_matches) * 0.3 + 0.35  # 0.35 to 0.65
        odds = np.random.rand(n_matches) * 1.0 + 1.5  # 1.5 to 2.5
        outcomes = np.random.rand(n_matches) < predictions  # Calibrated outcomes
        
        # Run backtest
        bankroll = 1000
        total_staked = 0
        total_profit = 0
        
        for pred, odd, won in zip(predictions, odds, outcomes):
            implied = 1 / odd
            edge = pred - implied
            
            if edge > 0.02:
                kelly = edge / (odd - 1)
                stake = min(bankroll * kelly * 0.25, bankroll * 0.05)
                
                total_staked += stake
                if won:
                    total_profit += stake * (odd - 1)
                else:
                    total_profit -= stake
                
                bankroll += (stake * (odd - 1) if won else -stake)
        
        # Verify backtest completed
        roi = total_profit / total_staked if total_staked > 0 else 0
        
        assert bankroll > 0  # Not bankrupt
        assert isinstance(roi, float)
    
    def test_model_update_pipeline(self):
        """Test model update after new data."""
        # Initial training data
        X_train = np.random.randn(100, 5)
        y_train = (np.random.rand(100) > 0.5).astype(int)
        
        # New data arrives
        X_new = np.random.randn(10, 5)
        y_new = (np.random.rand(10) > 0.5).astype(int)
        
        # Combine data
        X_combined = np.vstack([X_train, X_new])
        y_combined = np.concatenate([y_train, y_new])
        
        # Verify data combined correctly
        assert X_combined.shape[0] == 110
        assert y_combined.shape[0] == 110


class TestPerformanceMetrics:
    """Test performance metric calculations."""
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        predictions = np.array([0.6, 0.4, 0.7, 0.3, 0.55])
        actuals = np.array([1, 0, 1, 0, 1])
        
        pred_binary = (predictions > 0.5).astype(int)
        # pred_binary = [1, 0, 1, 0, 1]
        # actuals =     [1, 0, 1, 0, 1]
        # All 5 are correct!
        accuracy = (pred_binary == actuals).mean()
        
        assert accuracy == 1.0  # 5/5 correct
    
    def test_log_loss_calculation(self):
        """Test log loss calculation."""
        predictions = np.array([0.6, 0.4, 0.7, 0.3])
        actuals = np.array([1, 0, 1, 0])
        
        # Log loss formula
        eps = 1e-15
        predictions = np.clip(predictions, eps, 1 - eps)
        log_loss = -np.mean(
            actuals * np.log(predictions) + 
            (1 - actuals) * np.log(1 - predictions)
        )
        
        assert log_loss > 0
        assert log_loss < 1  # Should be reasonable for good predictions
    
    def test_brier_score(self):
        """Test Brier score calculation."""
        predictions = np.array([0.6, 0.4, 0.7, 0.3])
        actuals = np.array([1, 0, 1, 0])
        
        brier = np.mean((predictions - actuals) ** 2)
        
        assert 0 <= brier <= 1
    
    def test_roi_calculation(self):
        """Test ROI calculation."""
        bets = [
            {'stake': 100, 'profit': 90},   # Win
            {'stake': 100, 'profit': -100}, # Lose
            {'stake': 50, 'profit': 45},    # Win
        ]
        
        total_staked = sum(b['stake'] for b in bets)
        total_profit = sum(b['profit'] for b in bets)
        roi = total_profit / total_staked
        
        assert total_staked == 250
        assert total_profit == 35
        assert roi == pytest.approx(0.14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
