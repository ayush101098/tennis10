"""
Live Tennis Match Prediction Tool
==================================
Use this script to get predictions for upcoming matches.

Usage:
    python live_prediction.py "Novak Djokovic" "Carlos Alcaraz" "Hard" --odds1 1.85 --odds2 2.10
"""

import pickle
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import argparse
from datetime import datetime
from features import TennisFeatureExtractor
from hierarchical_model import HierarchicalTennisModel


class LivePredictor:
    """Live prediction engine combining all models."""
    
    def __init__(self, db_path='tennis_data.db'):
        print("Loading prediction models...")
        
        # Load feature extractor
        self.feature_extractor = TennisFeatureExtractor(db_path)
        
        # Load Markov model
        self.markov_model = HierarchicalTennisModel(db_path)
        
        # Load Logistic Regression
        with open('ml_models/logistic_regression_trained.pkl', 'rb') as f:
            lr_data = pickle.load(f)
            self.lr_model = lr_data['model']
            self.lr_features = lr_data['selected_features']
        
        # Load Neural Network Ensemble
        with open('ml_models/neural_network_ensemble.pkl', 'rb') as f:
            nn_data = pickle.load(f)
            self.nn_scaler = nn_data['scaler']
            self.nn_features = nn_data['features']
            self.nn_hidden_dim = nn_data['hidden_dim']
            self.nn_model_states = nn_data['models']
        
        # Recreate NN models
        self.nn_models = []
        for state_dict in self.nn_model_states:
            model = self._create_nn(len(self.nn_features), self.nn_hidden_dim)
            model.load_state_dict(state_dict)
            model.eval()
            self.nn_models.append(model)
        
        # Database connection for player lookup
        self.conn = sqlite3.connect(db_path)
        
        print("✅ All models loaded!")
    
    def _create_nn(self, input_dim, hidden_dim):
        """Create neural network architecture."""
        class SymmetricNN(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
                self.fc2 = nn.Linear(hidden_dim, 1, bias=False)
            def forward(self, x):
                return torch.sigmoid(self.fc2(torch.tanh(self.fc1(x)))).squeeze()
        return SymmetricNN(input_dim, hidden_dim)
    
    def find_player(self, name_query):
        """Find player by name (fuzzy search)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT player_id, player_name 
            FROM players 
            WHERE player_name LIKE ? 
            ORDER BY LENGTH(player_name)
            LIMIT 5
        """, (f'%{name_query}%',))
        results = cursor.fetchall()
        return results
    
    def get_player_id(self, name_query):
        """Get best matching player ID."""
        results = self.find_player(name_query)
        if results:
            return results[0][0], results[0][1]
        return None, None
    
    def predict(self, player1_name, player2_name, surface, best_of=3):
        """
        Get predictions for a match.
        
        Args:
            player1_name: Name of player 1
            player2_name: Name of player 2
            surface: 'Hard', 'Clay', or 'Grass'
            best_of: 3 or 5
            
        Returns:
            dict with predictions from all models
        """
        # Find players
        p1_id, p1_full_name = self.get_player_id(player1_name)
        p2_id, p2_full_name = self.get_player_id(player2_name)
        
        if not p1_id:
            raise ValueError(f"Player not found: {player1_name}")
        if not p2_id:
            raise ValueError(f"Player not found: {player2_name}")
        
        print(f"\n{'='*60}")
        print(f"MATCH: {p1_full_name} vs {p2_full_name}")
        print(f"Surface: {surface} | Best of {best_of}")
        print(f"{'='*60}")
        
        # Get features using player IDs
        features = self._extract_features_for_players(p1_id, p2_id, surface)
        
        predictions = {}
        
        # 1. Logistic Regression
        import pandas as pd
        df = pd.DataFrame([features])
        lr_prob = self.lr_model.predict_proba(df)[0]
        predictions['logistic'] = lr_prob
        
        # 2. Neural Network Ensemble
        X = np.array([[features.get(f, 0) for f in self.nn_features]])
        X_scaled = self.nn_scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        nn_preds = []
        for model in self.nn_models:
            with torch.no_grad():
                pred = model(X_tensor).item()
                nn_preds.append(pred)
        predictions['neural_net'] = np.mean(nn_preds)
        
        # 3. Markov Model
        markov_result = self.markov_model.predict_match(p1_id, p2_id, surface, num_sets=best_of)
        predictions['markov'] = markov_result['p_player1_win']
        
        # 4. Meta-Ensemble (weighted)
        weights = {'logistic': 0.35, 'neural_net': 0.45, 'markov': 0.20}
        ensemble_prob = sum(predictions[m] * weights[m] for m in weights)
        predictions['ensemble'] = ensemble_prob
        
        # Calculate confidence
        model_std = np.std([predictions['logistic'], predictions['neural_net'], predictions['markov']])
        
        return {
            'player1': p1_full_name,
            'player2': p2_full_name,
            'player1_id': p1_id,
            'player2_id': p2_id,
            'surface': surface,
            'best_of': best_of,
            'predictions': predictions,
            'model_agreement': 1 - model_std,  # Higher = more agreement
            'features': features
        }
    
    def _extract_features_for_players(self, p1_id, p2_id, surface):
        """Extract features for two players."""
        cursor = self.conn.cursor()
        
        # Get player stats from statistics table
        def get_player_stats(pid):
            # Win rate from matches
            cursor.execute("""
                SELECT 
                    COUNT(*) as matches,
                    SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
                FROM matches
                WHERE (winner_id = ? OR loser_id = ?)
                AND tournament_date >= date('now', '-2 years')
            """, (pid, pid, pid))
            match_stats = cursor.fetchone()
            
            # Surface win rate
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as surface_win_rate
                FROM matches
                WHERE (winner_id = ? OR loser_id = ?)
                AND surface = ?
                AND tournament_date >= date('now', '-2 years')
            """, (pid, pid, pid, surface))
            surface_stats = cursor.fetchone()
            
            # Serve stats from statistics table
            cursor.execute("""
                SELECT 
                    AVG(first_serve_win_pct) as first_serve_pct,
                    AVG(second_serve_win_pct) as second_serve_pct,
                    AVG(break_point_save_pct) as bp_save_pct
                FROM statistics s
                JOIN matches m ON s.match_id = m.match_id
                WHERE s.player_id = ?
                AND m.tournament_date >= date('now', '-2 years')
            """, (pid,))
            serve_stats = cursor.fetchone()
            
            return (
                match_stats[0] if match_stats else 0,  # matches
                match_stats[1] if match_stats and match_stats[1] else 0.5,  # win_rate
                surface_stats[0] if surface_stats and surface_stats[0] else 0.5,  # surface_win_rate
                serve_stats[0] if serve_stats and serve_stats[0] else 0.65,  # first_serve_pct
                serve_stats[1] if serve_stats and serve_stats[1] else 0.50,  # second_serve_pct
                serve_stats[2] if serve_stats and serve_stats[2] else 0.60,  # bp_save_pct
            )
        
        p1_stats = get_player_stats(p1_id)
        p2_stats = get_player_stats(p2_id)
        
        # Build feature dict (differences)
        features = {
            'player1_id': p1_id,
            'player2_id': p2_id,
            'surface': surface,
            'WIN_RATE_DIFF': (p1_stats[1] or 0.5) - (p2_stats[1] or 0.5),
            'SURFACE_EXP_DIFF': (p1_stats[2] or 0.5) - (p2_stats[2] or 0.5),
            'FIRST_SERVE_WIN_PCT_DIFF': (p1_stats[3] or 0.65) - (p2_stats[3] or 0.65),
            'SECOND_SERVE_WIN_PCT_DIFF': (p1_stats[4] or 0.50) - (p2_stats[4] or 0.50),
            'BP_SAVE_DIFF': (p1_stats[5] or 0.60) - (p2_stats[5] or 0.60),
            'MATCHES_PLAYED_DIFF': (p1_stats[0] or 0) - (p2_stats[0] or 0),
        }
        
        # Calculate WSP (Weighted Serve Points)
        p1_wsp = 0.6 * (p1_stats[3] or 0.65) + 0.4 * (p1_stats[4] or 0.50)
        p2_wsp = 0.6 * (p2_stats[3] or 0.65) + 0.4 * (p2_stats[4] or 0.50)
        features['WSP_DIFF'] = p1_wsp - p2_wsp
        
        # Head-to-head
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as p1_wins,
                COUNT(*) as total
            FROM matches
            WHERE (winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)
        """, (p1_id, p1_id, p2_id, p2_id, p1_id))
        h2h = cursor.fetchone()
        if h2h[1] > 0:
            features['DIRECT_H2H'] = (h2h[0] / h2h[1]) - 0.5
        else:
            features['DIRECT_H2H'] = 0.0
        
        # Complete rate (placeholder)
        features['COMPLETE_DIFF'] = 0.0
        
        return features
    
    def calculate_edge(self, prediction, bookmaker_odds):
        """
        Calculate betting edge.
        
        Args:
            prediction: Our probability (0-1)
            bookmaker_odds: Decimal odds (e.g., 1.85)
            
        Returns:
            dict with edge calculation
        """
        implied_prob = 1 / bookmaker_odds
        edge = prediction - implied_prob
        
        # Kelly criterion
        kelly_fraction = edge / (bookmaker_odds - 1) if edge > 0 else 0
        kelly_fraction = min(kelly_fraction, 0.25)  # Cap at 25%
        
        return {
            'our_probability': prediction,
            'implied_probability': implied_prob,
            'edge': edge,
            'edge_pct': edge * 100,
            'kelly_fraction': kelly_fraction,
            'recommended_bet_pct': kelly_fraction * 100,
            'has_value': edge > 0.02  # At least 2% edge
        }
    
    def full_analysis(self, player1_name, player2_name, surface, odds1=None, odds2=None, best_of=3):
        """
        Full match analysis with betting recommendations.
        """
        # Get predictions
        result = self.predict(player1_name, player2_name, surface, best_of)
        
        p1_prob = result['predictions']['ensemble']
        p2_prob = 1 - p1_prob
        
        print(f"\n📊 MODEL PREDICTIONS:")
        print(f"   Logistic Regression: {result['predictions']['logistic']:.1%} - {1-result['predictions']['logistic']:.1%}")
        print(f"   Neural Network:      {result['predictions']['neural_net']:.1%} - {1-result['predictions']['neural_net']:.1%}")
        print(f"   Markov Model:        {result['predictions']['markov']:.1%} - {1-result['predictions']['markov']:.1%}")
        print(f"\n   🎯 ENSEMBLE:         {p1_prob:.1%} - {p2_prob:.1%}")
        print(f"   Model Agreement:     {result['model_agreement']:.1%}")
        
        # Betting analysis if odds provided
        if odds1 and odds2:
            print(f"\n💰 BETTING ANALYSIS:")
            print(f"   Bookmaker odds: {result['player1']} @ {odds1:.2f} | {result['player2']} @ {odds2:.2f}")
            
            edge1 = self.calculate_edge(p1_prob, odds1)
            edge2 = self.calculate_edge(p2_prob, odds2)
            
            print(f"\n   {result['player1']}:")
            print(f"      Our prob: {edge1['our_probability']:.1%} vs Implied: {edge1['implied_probability']:.1%}")
            print(f"      Edge: {edge1['edge_pct']:+.1f}%")
            if edge1['has_value']:
                print(f"      ✅ VALUE BET! Kelly: {edge1['recommended_bet_pct']:.1f}% of bankroll")
            else:
                print(f"      ❌ No value")
            
            print(f"\n   {result['player2']}:")
            print(f"      Our prob: {edge2['our_probability']:.1%} vs Implied: {edge2['implied_probability']:.1%}")
            print(f"      Edge: {edge2['edge_pct']:+.1f}%")
            if edge2['has_value']:
                print(f"      ✅ VALUE BET! Kelly: {edge2['recommended_bet_pct']:.1f}% of bankroll")
            else:
                print(f"      ❌ No value")
            
            # Final recommendation
            print(f"\n{'='*60}")
            if edge1['has_value'] and edge1['edge'] > edge2['edge']:
                print(f"🎯 RECOMMENDATION: Bet on {result['player1']} @ {odds1:.2f}")
                print(f"   Stake: {edge1['recommended_bet_pct']:.1f}% of bankroll")
            elif edge2['has_value'] and edge2['edge'] > edge1['edge']:
                print(f"🎯 RECOMMENDATION: Bet on {result['player2']} @ {odds2:.2f}")
                print(f"   Stake: {edge2['recommended_bet_pct']:.1f}% of bankroll")
            else:
                print(f"⏸️  RECOMMENDATION: No bet - insufficient edge")
            print(f"{'='*60}")
            
            result['betting'] = {
                'player1_edge': edge1,
                'player2_edge': edge2
            }
        
        return result
    
    def close(self):
        """Clean up resources."""
        self.conn.close()
        self.feature_extractor.close()
        self.markov_model.close()


def main():
    parser = argparse.ArgumentParser(description='Tennis Match Prediction')
    parser.add_argument('player1', help='Name of player 1')
    parser.add_argument('player2', help='Name of player 2')
    parser.add_argument('surface', choices=['Hard', 'Clay', 'Grass'], help='Court surface')
    parser.add_argument('--odds1', type=float, help='Bookmaker odds for player 1')
    parser.add_argument('--odds2', type=float, help='Bookmaker odds for player 2')
    parser.add_argument('--best-of', type=int, default=3, choices=[3, 5], help='Best of 3 or 5')
    
    args = parser.parse_args()
    
    predictor = LivePredictor()
    
    try:
        predictor.full_analysis(
            args.player1,
            args.player2,
            args.surface,
            odds1=args.odds1,
            odds2=args.odds2,
            best_of=args.best_of
        )
    finally:
        predictor.close()


if __name__ == '__main__':
    main()
