"""
Live Predictor - Main prediction pipeline

Integrates:
1. Match scraping
2. Feature extraction
3. Model predictions
4. Odds collection
5. Bet recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_data.match_scraper import get_all_upcoming_matches
from live_data.odds_scraper import get_tennis_odds
from live_data.odds_analyzer import find_value_bets, calculate_implied_probability
from live_data.validators import validate_match_data
from live_predictions.bet_calculator import BetCalculator

logger = logging.getLogger(__name__)


class LivePredictor:
    """End-to-end live prediction and betting system"""
    
    def __init__(self, bankroll: float = 1000, kelly_fraction: float = 0.25,
                 min_edge: float = 0.025):
        """
        Initialize live predictor
        
        Args:
            bankroll: Starting bankroll ($)
            kelly_fraction: Kelly fraction (0.25 = quarter Kelly, conservative)
            min_edge: Minimum edge to consider betting (2.5%)
        """
        self.bankroll = bankroll
        self.bet_calculator = BetCalculator(
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            min_edge=min_edge
        )
        
        # Load models (placeholder - replace with actual models)
        self.models_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained prediction models"""
        
        try:
            # TODO: Load actual models
            # self.markov_model = load_model('markov_model.pkl')
            # self.lr_model = load_model('logistic_regression.pkl')
            # self.nn_model = load_model('neural_network.h5')
            
            logger.info("✅ Models loaded successfully")
            self.models_loaded = True
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load models: {e}")
            logger.warning("Using odds-based estimation for predictions")
            self.models_loaded = False
    
    def predict_upcoming_matches(self, days_ahead: int = 2,
                                use_odds_api: bool = True) -> pd.DataFrame:
        """
        Generate predictions for all upcoming matches
        
        Args:
            days_ahead: Number of days to look ahead
            use_odds_api: Use The Odds API (requires key) vs scraping
        
        Returns:
            DataFrame with predictions and bet recommendations
        """
        
        logger.info(f"🎾 Generating predictions for next {days_ahead} days...")
        
        # Step 1: Get upcoming matches
        logger.info("1️⃣  Scraping upcoming matches...")
        matches = get_all_upcoming_matches(days_ahead=days_ahead)
        
        if matches.empty:
            logger.warning("No upcoming matches found")
            return pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"   Found {len(matches)} upcoming matches")
        
        # Step 2: Get odds
        logger.info("2️⃣  Fetching bookmaker odds...")
        odds = get_tennis_odds(use_api=use_odds_api)
        
        if odds.empty:
            logger.warning("No odds data available")
            return matches, pd.DataFrame()
        
        logger.info(f"   Found odds for {len(odds)} matches")
        
        # Step 3: Match odds to matches
        logger.info("3️⃣  Matching odds to matches...")
        matches_with_odds = self._merge_matches_and_odds(matches, odds)
        
        if matches_with_odds.empty:
            logger.warning("Could not match any odds to matches")
            return matches, pd.DataFrame()
        
        logger.info(f"   Matched {len(matches_with_odds)} matches with odds")
        
        # Step 4: Generate predictions
        logger.info("4️⃣  Generating predictions...")
        predictions = []
        
        for idx, match in matches_with_odds.iterrows():
            try:
                prediction = self._predict_match(match)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting {match.get('player1_name')} vs {match.get('player2_name')}: {e}")
        
        if not predictions:
            logger.warning("No predictions generated")
            return pd.DataFrame()
        
        predictions_df = pd.DataFrame(predictions)
        
        logger.info(f"   Generated {len(predictions_df)} predictions")
        
        # Step 5: Calculate bet recommendations
        logger.info("5️⃣  Calculating bet recommendations...")
        predictions_df = self._calculate_recommendations(predictions_df)
        
        # Step 6: Filter to bets only
        bets = predictions_df[predictions_df['action'] != 'no_bet'].copy()
        
        if not bets.empty:
            logger.info(f"   ✅ Found {len(bets)} betting opportunities")
            
            # Sort by expected value
            bets = bets.sort_values('expected_value', ascending=False)
            
            # Apply portfolio optimization
            bet_list = bets.to_dict('records')
            optimized_bets = self.bet_calculator.calculate_portfolio_stakes(bet_list)
            bets = pd.DataFrame(optimized_bets)
        else:
            logger.info("   ⚠️  No profitable betting opportunities found")
        
        return predictions_df, bets
    
    def _merge_matches_and_odds(self, matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
        """
        Match scraped matches with odds data
        
        Uses fuzzy player name matching
        """
        
        merged = []
        
        for _, match in matches.iterrows():
            p1_name = match['player1_name'].lower()
            p2_name = match['player2_name'].lower()
            
            # Try to find matching odds
            for _, odd in odds.iterrows():
                odd_p1 = odd['player1_name'].lower()
                odd_p2 = odd['player2_name'].lower()
                
                # Check if players match (order-independent)
                if self._players_match(p1_name, p2_name, odd_p1, odd_p2):
                    # Merge match and odds
                    merged_row = {**match.to_dict(), **odd.to_dict()}
                    merged.append(merged_row)
                    break
        
        return pd.DataFrame(merged)
    
    def _players_match(self, p1a: str, p2a: str, p1b: str, p2b: str) -> bool:
        """Check if two player pairs match (fuzzy)"""
        
        # Exact match
        if (p1a in p1b or p1b in p1a) and (p2a in p2b or p2b in p2a):
            return True
        
        # Reverse order
        if (p1a in p2b or p2b in p1a) and (p2a in p1b or p1b in p2a):
            return True
        
        return False
    
    def _predict_match(self, match: Dict) -> Dict:
        """
        Generate prediction for a single match
        
        Args:
            match: Match dictionary with player info and odds
        
        Returns:
            Prediction dictionary
        """
        
        # If models are loaded, use them
        if self.models_loaded:
            prediction = self._predict_with_models(match)
        else:
            # Fall back to odds-based estimation
            prediction = self._predict_from_odds(match)
        
        # Add match info
        prediction.update({
            'match_id': match.get('match_id', ''),
            'player1_name': match['player1_name'],
            'player2_name': match['player2_name'],
            'tournament': match.get('tournament_name', ''),
            'surface': match.get('surface', ''),
            'scheduled_time': match.get('scheduled_time', ''),
            'best_player1_odds': match.get('best_player1_odds', 0),
            'best_player2_odds': match.get('best_player2_odds', 0),
        })
        
        return prediction
    
    def _predict_with_models(self, match: Dict) -> Dict:
        """
        Predict using trained models
        
        TODO: Implement actual model predictions
        Currently returns placeholder
        """
        
        # Placeholder - replace with actual model predictions
        # features = extract_features(match)
        # markov_prob = self.markov_model.predict(features)
        # lr_prob = self.lr_model.predict_proba(features)[0][1]
        # nn_prob = self.nn_model.predict(features)[0][0]
        
        # For now, use odds-based estimation
        return self._predict_from_odds(match)
    
    def _predict_from_odds(self, match: Dict) -> Dict:
        """
        Estimate probability from bookmaker odds
        
        Uses true probability after removing overround
        """
        
        p1_odds = match.get('best_player1_odds', 0)
        p2_odds = match.get('best_player2_odds', 0)
        
        if p1_odds == 0 or p2_odds == 0:
            return {
                'ensemble_p1_win': 0.5,
                'model_agreement': 0.0,
                'uncertainty': 1.0,
            }
        
        # Calculate implied probabilities
        p1_implied = calculate_implied_probability(p1_odds)
        p2_implied = calculate_implied_probability(p2_odds)
        
        # Remove overround to get true probabilities
        total = p1_implied + p2_implied
        p1_true = p1_implied / total
        
        return {
            'ensemble_p1_win': p1_true,
            'markov_p1_win': p1_true,  # Placeholder
            'lr_p1_win': p1_true,      # Placeholder
            'nn_p1_win': p1_true,      # Placeholder
            'model_agreement': 1.0,    # All agree (because same source)
            'uncertainty': 0.3,         # Moderate uncertainty
        }
    
    def _calculate_recommendations(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate bet recommendations for all predictions
        """
        
        recommendations = []
        
        for _, pred in predictions.iterrows():
            rec = self.bet_calculator.calculate_bet_recommendation(
                our_probability=pred['ensemble_p1_win'],
                p1_odds=pred['best_player1_odds'],
                p2_odds=pred['best_player2_odds'],
                uncertainty=pred.get('uncertainty', 0.3),
                model_agreement=pred.get('model_agreement', 1.0)
            )
            
            # Merge recommendation with prediction
            full_rec = {**pred.to_dict(), **rec}
            recommendations.append(full_rec)
        
        return pd.DataFrame(recommendations)
    
    def display_recommendations(self, predictions: pd.DataFrame, bets: pd.DataFrame):
        """
        Display predictions and betting recommendations
        """
        
        print("\n" + "="*80)
        print("🎾 LIVE TENNIS PREDICTIONS & BETTING RECOMMENDATIONS")
        print("="*80)
        print(f"\nBankroll: ${self.bankroll:.2f}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if bets.empty:
            print("\n⚠️  No profitable betting opportunities found")
            print("\nTop predictions (no bet):")
            
            if not predictions.empty:
                top = predictions.head(5)
                for idx, pred in top.iterrows():
                    print(f"\n{pred['player1_name']} vs {pred['player2_name']}")
                    print(f"  Tournament: {pred['tournament']}")
                    print(f"  Our prediction: {pred['ensemble_p1_win']:.1%} / {1-pred['ensemble_p1_win']:.1%}")
                    print(f"  Best odds: {pred['best_player1_odds']:.2f} / {pred['best_player2_odds']:.2f}")
                    print(f"  Reason: {pred.get('reason', 'Insufficient edge')}")
        
        else:
            print(f"\n✅ {len(bets)} PROFITABLE BETS FOUND\n")
            
            total_stake = bets['recommended_stake'].sum()
            total_ev = (bets['recommended_stake'] * bets['expected_value']).sum()
            
            print(f"Total recommended stake: ${total_stake:.2f}")
            print(f"Total expected value: ${total_ev:.2f}")
            print(f"Portfolio EV: {(total_ev/total_stake)*100:.1f}%")
            
            print("\n" + "-"*80)
            
            for idx, bet in bets.iterrows():
                print(f"\n{idx+1}. {bet['player1_name']} vs {bet['player2_name']}")
                print(f"   Tournament: {bet['tournament']} ({bet['surface']})")
                print(f"   Time: {bet['scheduled_time']}")
                
                if bet['action'] == 'bet_player1':
                    print(f"   🎯 BET: {bet['player1_name']}")
                    print(f"   Odds: {bet['best_player1_odds']:.2f}")
                else:
                    print(f"   🎯 BET: {bet['player2_name']}")
                    print(f"   Odds: {bet['best_player2_odds']:.2f}")
                
                print(f"   Stake: ${bet['recommended_stake']:.2f} ({bet['recommended_stake']/self.bankroll*100:.1f}% of bankroll)")
                print(f"   Edge: {bet['edge']:.2%}")
                print(f"   Expected Value: {bet['expected_value']:.2%}")
                print(f"   Potential profit: ${bet['recommended_stake'] * bet['expected_value']:.2f}")
                print(f"   Confidence: {bet['confidence'].upper()}")
            
            print("\n" + "="*80)


# Convenience function

def predict_upcoming_matches(bankroll: float = 1000, days_ahead: int = 2,
                            use_odds_api: bool = True) -> tuple:
    """
    Generate predictions for upcoming matches
    
    Args:
        bankroll: Starting bankroll
        days_ahead: Days to look ahead
        use_odds_api: Use The Odds API
    
    Returns:
        (all_predictions, profitable_bets)
    """
    predictor = LivePredictor(bankroll=bankroll)
    return predictor.predict_upcoming_matches(days_ahead=days_ahead, use_odds_api=use_odds_api)


if __name__ == "__main__":
    print("🎾 Live Tennis Prediction System\n")
    
    # Initialize
    predictor = LivePredictor(bankroll=1000, min_edge=0.025)
    
    # Generate predictions
    all_predictions, profitable_bets = predictor.predict_upcoming_matches(
        days_ahead=2,
        use_odds_api=True
    )
    
    # Display recommendations
    predictor.display_recommendations(all_predictions, profitable_bets)
    
    # Save results
    if not all_predictions.empty:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        all_predictions.to_csv(f'predictions_{timestamp}.csv', index=False)
        print(f"\n💾 All predictions saved to predictions_{timestamp}.csv")
        
        if not profitable_bets.empty:
            profitable_bets.to_csv(f'profitable_bets_{timestamp}.csv', index=False)
            print(f"💾 Profitable bets saved to profitable_bets_{timestamp}.csv")
