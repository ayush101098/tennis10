"""
Tennis Match Feature Engineering Module

Extracts features for ATP match prediction using:
- Basic features (ranking, points differences)
- Performance features with exponential time decay
- Constructed features (serve advantage, fatigue, h2h)
- Surface weighting and uncertainty scoring
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TennisFeatureExtractor:
    """Extract features for tennis match prediction"""
    
    # Surface correlation matrix (similarity between surfaces)
    SURFACE_CORRELATIONS = {
        ('Hard', 'Hard'): 1.0,
        ('Clay', 'Clay'): 1.0,
        ('Grass', 'Grass'): 1.0,
        ('Hard', 'Clay'): 0.28,
        ('Clay', 'Hard'): 0.28,
        ('Hard', 'Grass'): 0.24,
        ('Grass', 'Hard'): 0.24,
        ('Clay', 'Grass'): 0.15,
        ('Grass', 'Clay'): 0.15,
    }
    
    def __init__(self, db_path: str = 'tennis_data.db'):
        """Initialize feature extractor with database connection"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def get_surface_weight(self, surface1: str, surface2: str) -> float:
        """Get correlation weight between two surfaces"""
        key = (surface1, surface2)
        return self.SURFACE_CORRELATIONS.get(key, 0.1)  # Default low correlation
        
    def apply_time_discount(self, 
                           date_current: datetime, 
                           date_past: datetime, 
                           half_life_years: float = 0.8) -> float:
        """
        Apply exponential time decay to past matches
        
        Args:
            date_current: Current match date
            date_past: Past match date
            half_life_years: Time for weight to decay to 50%
            
        Returns:
            Weight factor (0 to 1)
        """
        days_diff = (date_current - date_past).days
        years_diff = days_diff / 365.25
        
        # Exponential decay: weight = 0.5^(years_diff / half_life)
        decay_factor = 0.5 ** (years_diff / half_life_years)
        return decay_factor
        
    def get_player_matches(self, 
                          player_id: int, 
                          before_date: datetime,
                          lookback_months: int = 36) -> pd.DataFrame:
        """
        Get all matches for a player within lookback period
        
        Args:
            player_id: Player ID
            before_date: Get matches before this date
            lookback_months: How many months to look back
            
        Returns:
            DataFrame of player's matches
        """
        cutoff_date = before_date - timedelta(days=lookback_months * 30)
        
        query = """
        SELECT 
            m.*,
            pw.player_name as winner_name,
            pl.player_name as loser_name,
            CASE 
                WHEN m.winner_id = ? THEN 1 
                ELSE 0 
            END as player_won
        FROM matches m
        LEFT JOIN players pw ON m.winner_id = pw.player_id
        LEFT JOIN players pl ON m.loser_id = pl.player_id
        WHERE (m.winner_id = ? OR m.loser_id = ?)
        AND m.tournament_date < ?
        AND m.tournament_date >= ?
        ORDER BY m.tournament_date DESC
        """
        
        df = pd.read_sql_query(
            query, 
            self.conn,
            params=(player_id, player_id, player_id, 
                   before_date.strftime('%Y-%m-%d'),
                   cutoff_date.strftime('%Y-%m-%d'))
        )
        
        if len(df) > 0:
            df['tournament_date'] = pd.to_datetime(df['tournament_date'])
            
        return df
        
    def calculate_performance_features(self,
                                      player_id: int,
                                      opponent_id: int,
                                      match_date: datetime,
                                      match_surface: str,
                                      lookback_months: int = 36) -> Dict[str, float]:
        """
        Calculate weighted performance features using REAL statistics with time decay and surface weighting
        
        Args:
            player_id: Player ID
            opponent_id: Opponent player ID (not used currently)
            match_date: Current match date
            match_surface: Surface of upcoming match
            lookback_months: Months to look back for match history
            
        Returns:
            Dictionary of REAL performance metrics (not proxies)
        """
        cutoff_date = match_date - timedelta(days=lookback_months * 30)
        
        # Get matches with REAL statistics from statistics table
        query = """
        SELECT 
            m.tournament_date,
            m.surface,
            s.aces,
            s.double_faults,
            s.serve_points_total,
            s.first_serve_in,
            s.first_serve_won,
            s.second_serve_won,
            s.serve_games,
            s.break_points_saved,
            s.break_points_faced,
            s.first_serve_pct,
            s.first_serve_win_pct,
            s.second_serve_win_pct,
            s.break_point_save_pct,
            s.is_winner,
            CASE WHEN s.is_winner = 1 THEN 1 ELSE 0 END as player_won
        FROM statistics s
        JOIN matches m ON s.match_id = m.match_id
        WHERE s.player_id = ?
        AND m.tournament_date < ?
        AND m.tournament_date >= ?
        AND s.aces IS NOT NULL
        ORDER BY m.tournament_date DESC
        """
        
        matches = pd.read_sql_query(
            query,
            self.conn,
            params=(player_id, 
                   match_date.strftime('%Y-%m-%d'),
                   cutoff_date.strftime('%Y-%m-%d'))
        )
        
        if len(matches) == 0:
            # Return baseline values if no match history
            return {
                'wsp': 0.65,  # Typical serve hold %
                'wrp': 0.35,  # Typical return points won %
                'aces_per_game': 0.5,
                'df_per_game': 0.2,
                'bp_save': 0.60,  # Break point save %
                'first_serve_pct': 0.62,
                'first_serve_win_pct': 0.70,
                'second_serve_win_pct': 0.50,
                'win_rate': 0.5,
                'matches_played': 0,
                'surface_matches': 0,
                'surface_win_rate': 0.5
            }
        
        matches['tournament_date'] = pd.to_datetime(matches['tournament_date'])
        
        # Calculate weights for each match (time decay + surface similarity)
        weights = []
        for _, match in matches.iterrows():
            time_weight = self.apply_time_discount(match_date, match['tournament_date'])
            surface_weight = self.get_surface_weight(match_surface, match['surface'])
            combined_weight = time_weight * surface_weight
            weights.append(combined_weight)
            
        matches['weight'] = weights
        
        # Calculate weighted REAL statistics
        # Clean data: remove NaN values for each metric before averaging
        def weighted_avg(series, weights_series):
            """Calculate weighted average, handling NaN values"""
            valid_mask = ~series.isna()
            if valid_mask.sum() == 0:
                return 0.0
            return np.average(series[valid_mask], weights=weights_series[valid_mask])
        
        # REAL metrics from actual match statistics
        first_serve_pct = weighted_avg(matches['first_serve_pct'], matches['weight'])
        first_serve_win_pct = weighted_avg(matches['first_serve_win_pct'], matches['weight'])
        second_serve_win_pct = weighted_avg(matches['second_serve_win_pct'], matches['weight'])
        bp_save_pct = weighted_avg(matches['break_point_save_pct'], matches['weight'])
        
        # Calculate aces and DFs per serve game
        valid_games = matches[matches['serve_games'].notna() & (matches['serve_games'] > 0)]
        if len(valid_games) > 0:
            aces_per_game = weighted_avg(
                valid_games['aces'] / valid_games['serve_games'],
                valid_games['weight']
            )
            df_per_game = weighted_avg(
                valid_games['double_faults'] / valid_games['serve_games'],
                valid_games['weight']
            )
        else:
            aces_per_game = 0.5
            df_per_game = 0.2
        
        # Calculate serve points won % (wsp)
        valid_serve = matches[matches['serve_points_total'].notna() & (matches['serve_points_total'] > 0)]
        if len(valid_serve) > 0:
            # Total serve points won = first serve won + second serve won
            serve_points_won = (
                valid_serve['first_serve_won'].fillna(0) + 
                valid_serve['second_serve_won'].fillna(0)
            ) / valid_serve['serve_points_total']
            wsp = weighted_avg(serve_points_won, valid_serve['weight'])
        else:
            wsp = 0.65
        
        # Calculate weighted win rate
        weighted_win_rate = weighted_avg(matches['player_won'], matches['weight'])
        
        # Calculate surface-specific performance
        surface_matches = matches[matches['surface'] == match_surface]
        if len(surface_matches) > 0:
            surface_win_rate = weighted_avg(
                surface_matches['player_won'],
                surface_matches['weight']
            )
            surface_count = len(surface_matches)
        else:
            surface_win_rate = weighted_win_rate
            surface_count = 0
        
        features = {
            'wsp': wsp,  # REAL: Serve points won %
            'wrp': 1.0 - wsp,  # Approximate: opponent's serve points won becomes our return points won
            'aces_per_game': aces_per_game,  # REAL: Aces per service game
            'df_per_game': df_per_game,  # REAL: Double faults per service game
            'bp_save': bp_save_pct,  # REAL: Break point save %
            'first_serve_pct': first_serve_pct,  # REAL: First serve in %
            'first_serve_win_pct': first_serve_win_pct,  # REAL: First serve points won %
            'second_serve_win_pct': second_serve_win_pct,  # REAL: Second serve points won %
            'win_rate': weighted_win_rate,  # Win rate (for context)
            'matches_played': len(matches),
            'surface_matches': surface_count,
            'surface_win_rate': surface_win_rate
        }
        
        return features
        
    def calculate_fatigue(self,
                         player_id: int,
                         match_date: datetime,
                         decay: float = 0.75) -> float:
        """
        Calculate fatigue based on matches played in last 3 days
        
        Args:
            player_id: Player ID
            match_date: Current match date
            decay: Decay factor for recency weighting
            
        Returns:
            Fatigue score (higher = more fatigued)
        """
        # Get matches in last 3 days
        three_days_ago = match_date - timedelta(days=3)
        
        query = """
        SELECT 
            m.tournament_date,
            m.minutes,
            CASE WHEN m.winner_id = ? THEN 1 ELSE 0 END as won
        FROM matches m
        WHERE (m.winner_id = ? OR m.loser_id = ?)
        AND m.tournament_date >= ?
        AND m.tournament_date < ?
        ORDER BY m.tournament_date DESC
        """
        
        recent_matches = pd.read_sql_query(
            query,
            self.conn,
            params=(player_id, player_id, player_id,
                   three_days_ago.strftime('%Y-%m-%d'),
                   match_date.strftime('%Y-%m-%d'))
        )
        
        if len(recent_matches) == 0:
            return 0.0
            
        recent_matches['tournament_date'] = pd.to_datetime(recent_matches['tournament_date'])
        
        # Calculate fatigue score based on match count and duration
        fatigue_score = 0.0
        for _, match in recent_matches.iterrows():
            # Use match duration if available, otherwise estimate
            if pd.notna(match['minutes']) and match['minutes'] > 0:
                # Weight by match duration (in hours)
                match_weight = match['minutes'] / 60.0
            else:
                # Estimate: average match is ~2 hours
                match_weight = 2.0
            
            # Apply recency decay
            days_ago = (match_date - match['tournament_date']).days
            weight = decay ** days_ago
            fatigue_score += match_weight * weight
            
        return fatigue_score
        
    def calculate_head_to_head(self,
                              player1_id: int,
                              player2_id: int,
                              match_date: datetime,
                              lookback_months: int = 36) -> Dict[str, float]:
        """
        Calculate head-to-head statistics between two players
        
        Returns:
            Dictionary with h2h win rate and match count
        """
        cutoff_date = match_date - timedelta(days=lookback_months * 30)
        
        query = """
        SELECT 
            m.*,
            CASE 
                WHEN m.winner_id = ? THEN 1 
                ELSE 0 
            END as player1_won
        FROM matches m
        WHERE ((m.winner_id = ? AND m.loser_id = ?) 
               OR (m.winner_id = ? AND m.loser_id = ?))
        AND m.tournament_date < ?
        AND m.tournament_date >= ?
        ORDER BY m.tournament_date DESC
        """
        
        h2h_matches = pd.read_sql_query(
            query,
            self.conn,
            params=(player1_id, player1_id, player2_id, 
                   player2_id, player1_id,
                   match_date.strftime('%Y-%m-%d'),
                   cutoff_date.strftime('%Y-%m-%d'))
        )
        
        if len(h2h_matches) == 0:
            return {
                'h2h_win_rate': 0.5,  # Neutral if no history
                'h2h_matches': 0
            }
        
        # Apply time decay to h2h matches
        h2h_matches['tournament_date'] = pd.to_datetime(h2h_matches['tournament_date'])
        weights = [self.apply_time_discount(match_date, date, half_life_years=1.5) 
                  for date in h2h_matches['tournament_date']]
        
        weighted_win_rate = np.average(h2h_matches['player1_won'], weights=weights)
        
        return {
            'h2h_win_rate': weighted_win_rate,
            'h2h_matches': len(h2h_matches)
        }
        
    def check_retirement_status(self,
                               player_id: int,
                               match_date: datetime) -> bool:
        """
        Check if this is first match since a long retirement (>90 days)
        
        Returns:
            1 if first match after retirement, 0 otherwise
        """
        # Get player's most recent match before this one
        query = """
        SELECT MAX(tournament_date) as last_match
        FROM matches
        WHERE (winner_id = ? OR loser_id = ?)
        AND tournament_date < ?
        """
        
        result = pd.read_sql_query(
            query,
            self.conn,
            params=(player_id, player_id, match_date.strftime('%Y-%m-%d'))
        )
        
        if result.empty or pd.isna(result.iloc[0]['last_match']):
            return True  # No previous match found
            
        last_match_date = pd.to_datetime(result.iloc[0]['last_match'])
        days_since_last = (match_date - last_match_date).days
        
        return days_since_last > 90  # 3 months = retirement
        
    def calculate_uncertainty(self, 
                            player1_features: Dict,
                            player2_features: Dict) -> float:
        """
        Calculate uncertainty score based on data availability
        
        Lower score = more confident predictions
        Higher score = less data available
        
        Returns:
            Uncertainty score (0 to 1)
        """
        # Factors contributing to uncertainty
        p1_matches = player1_features.get('matches_played', 0)
        p2_matches = player2_features.get('matches_played', 0)
        p1_surface = player1_features.get('surface_matches', 0)
        p2_surface = player2_features.get('surface_matches', 0)
        h2h_count = player1_features.get('h2h_matches', 0)
        
        # Calculate uncertainty components
        # Low match count = high uncertainty
        match_uncertainty = 1 / (1 + min(p1_matches, p2_matches) / 20)
        surface_uncertainty = 1 / (1 + min(p1_surface, p2_surface) / 10)
        h2h_uncertainty = 1 / (1 + h2h_count / 3)
        
        # Combined uncertainty (weighted average)
        uncertainty = (
            0.4 * match_uncertainty +
            0.3 * surface_uncertainty +
            0.3 * h2h_uncertainty
        )
        
        return uncertainty
        
    def extract_features(self,
                        match_id: int = None,
                        player1_id: int = None,
                        player2_id: int = None,
                        match_date: datetime = None,
                        surface: str = None,
                        lookback_months: int = 36) -> Dict[str, float]:
        """
        Extract complete feature vector for a match
        
        Args:
            match_id: Match ID (if loading existing match)
            OR provide: player1_id, player2_id, match_date, surface
            lookback_months: Historical data window
            
        Returns:
            Dictionary of features
        """
        # Load match data if match_id provided
        if match_id is not None:
            query = """
            SELECT 
                m.*,
                pw.player_name as winner_name,
                pl.player_name as loser_name
            FROM matches m
            LEFT JOIN players pw ON m.winner_id = pw.player_id
            LEFT JOIN players pl ON m.loser_id = pl.player_id
            WHERE m.match_id = ?
            """
            match_data = pd.read_sql_query(query, self.conn, params=(match_id,))
            
            if len(match_data) == 0:
                raise ValueError(f"Match {match_id} not found in database")
                
            match = match_data.iloc[0]
            # Player 1 = eventual winner (for training with label=1)
            player1_id = int(match['winner_id'])
            player2_id = int(match['loser_id'])
            match_date = pd.to_datetime(match['tournament_date'])
            surface = match['surface']
            
        if any(x is None for x in [player1_id, player2_id, match_date, surface]):
            raise ValueError("Must provide either match_id or all of: player1_id, player2_id, match_date, surface")
            
        # Get basic features from database (handle both winner_id/loser_id orders)
        query_basic = """
        SELECT 
            CASE WHEN winner_id = ? THEN winner_rank ELSE loser_rank END as p1_rank,
            CASE WHEN winner_id = ? THEN loser_rank ELSE winner_rank END as p2_rank,
            CASE WHEN winner_id = ? THEN winner_rank_points ELSE loser_rank_points END as p1_points,
            CASE WHEN winner_id = ? THEN loser_rank_points ELSE winner_rank_points END as p2_points
        FROM matches
        WHERE match_id = ?
        """
        
        # If we don't have match_id, try to find by players and date
        if match_id is not None:
            basic_data = pd.read_sql_query(
                query_basic, 
                self.conn,
                params=(player1_id, player1_id, player1_id, player1_id, match_id)
            )
        else:
            # For new predictions, use a simpler query
            basic_data = pd.DataFrame()
        
        # Get player performance features
        p1_perf = self.calculate_performance_features(
            player1_id, player2_id, match_date, surface, lookback_months
        )
        p2_perf = self.calculate_performance_features(
            player2_id, player1_id, match_date, surface, lookback_months
        )
        
        # Get head-to-head
        h2h = self.calculate_head_to_head(player1_id, player2_id, match_date, lookback_months)
        p1_perf['h2h_matches'] = h2h['h2h_matches']
        
        # Calculate fatigue
        p1_fatigue = self.calculate_fatigue(player1_id, match_date)
        p2_fatigue = self.calculate_fatigue(player2_id, match_date)
        
        # Check retirement status
        p1_retired = self.check_retirement_status(player1_id, match_date)
        p2_retired = self.check_retirement_status(player2_id, match_date)
        
        # Extract basic features (with defaults if not available)
        if len(basic_data) > 0 and not basic_data.empty:
            p1_rank = basic_data.iloc[0]['p1_rank']
            p2_rank = basic_data.iloc[0]['p2_rank']
            p1_points = basic_data.iloc[0]['p1_points']
            p2_points = basic_data.iloc[0]['p2_points']
        else:
            # Defaults if no basic data
            p1_rank = p2_rank = 100
            p1_points = p2_points = 1000
            
        # Build feature vector (differences between players)
        features = {
            # Basic features
            'RANK_DIFF': p2_rank - p1_rank if pd.notna(p1_rank) and pd.notna(p2_rank) else 0,
            'POINTS_DIFF': p1_points - p2_points if pd.notna(p1_points) and pd.notna(p2_points) else 0,
            
            # Performance features (differences) - ALL FROM REAL STATISTICS
            'WSP_DIFF': p1_perf['wsp'] - p2_perf['wsp'],
            'WRP_DIFF': p1_perf['wrp'] - p2_perf['wrp'],
            'ACES_DIFF': p1_perf['aces_per_game'] - p2_perf['aces_per_game'],
            'DF_DIFF': p2_perf['df_per_game'] - p1_perf['df_per_game'],  # Reversed (less DF is better)
            'BP_SAVE_DIFF': p1_perf['bp_save'] - p2_perf['bp_save'],
            'FIRST_SERVE_PCT_DIFF': p1_perf['first_serve_pct'] - p2_perf['first_serve_pct'],
            'FIRST_SERVE_WIN_PCT_DIFF': p1_perf['first_serve_win_pct'] - p2_perf['first_serve_win_pct'],
            'SECOND_SERVE_WIN_PCT_DIFF': p1_perf['second_serve_win_pct'] - p2_perf['second_serve_win_pct'],
            
            # Win rate features
            'WIN_RATE_DIFF': p1_perf['win_rate'] - p2_perf['win_rate'],
            'SURFACE_WIN_RATE_DIFF': p1_perf['surface_win_rate'] - p2_perf['surface_win_rate'],
            
            # Constructed features
            'SERVEADV': (p1_perf['wsp'] - p2_perf['wrp']) - (p2_perf['wsp'] - p1_perf['wrp']),
            'COMPLETE_DIFF': (p1_perf['wsp'] * p1_perf['wrp']) - (p2_perf['wsp'] * p2_perf['wrp']),
            
            # Fatigue
            'FATIGUE_DIFF': p2_fatigue - p1_fatigue,  # Higher opponent fatigue is good
            
            # Retirement status
            'RETIRED_DIFF': int(p2_retired) - int(p1_retired),
            
            # Head-to-head
            'DIRECT_H2H': h2h['h2h_win_rate'] - 0.5,  # Centered at 0
            
            # Experience features
            'MATCHES_PLAYED_DIFF': p1_perf['matches_played'] - p2_perf['matches_played'],
            'SURFACE_EXP_DIFF': p1_perf['surface_matches'] - p2_perf['surface_matches'],
        }
        
        # Calculate uncertainty
        uncertainty = self.calculate_uncertainty(p1_perf, p2_perf)
        features['UNCERTAINTY'] = uncertainty
        
        # Add metadata (not for model, but for filtering/analysis)
        features['match_id'] = match_id
        features['player1_id'] = player1_id
        features['player2_id'] = player2_id
        features['surface'] = surface
        features['match_date'] = match_date
        
        return features
        
    def extract_features_batch(self,
                              match_ids: List[int] = None,
                              lookback_months: int = 36,
                              uncertainty_threshold: float = 0.7) -> pd.DataFrame:
        """
        Extract features for multiple matches
        
        Args:
            match_ids: List of match IDs (if None, process all matches)
            lookback_months: Historical data window
            uncertainty_threshold: Only include matches below this threshold
            
        Returns:
            DataFrame with features for each match
        """
        if match_ids is None:
            # Get all match IDs
            query = "SELECT match_id FROM matches ORDER BY tournament_date"
            match_ids = pd.read_sql_query(query, self.conn)['match_id'].tolist()
            
        logger.info(f"Extracting features for {len(match_ids)} matches...")
        
        features_list = []
        skipped_uncertainty = 0
        skipped_error = 0
        
        for i, match_id in enumerate(match_ids):
            try:
                features = self.extract_features(match_id, lookback_months=lookback_months)
                
                # Filter by uncertainty
                if features['UNCERTAINTY'] <= uncertainty_threshold:
                    features_list.append(features)
                else:
                    skipped_uncertainty += 1
                    
                if (i + 1) % 500 == 0:
                    logger.info(f"Processed {i + 1}/{len(match_ids)} matches...")
                    
            except Exception as e:
                logger.warning(f"Error processing match {match_id}: {str(e)}")
                skipped_error += 1
                continue
                
        logger.info(f"Extraction complete!")
        logger.info(f"  Successfully extracted: {len(features_list)} matches")
        logger.info(f"  Skipped (high uncertainty): {skipped_uncertainty}")
        logger.info(f"  Skipped (errors): {skipped_error}")
        
        return pd.DataFrame(features_list)


def main():
    """Test feature extraction"""
    extractor = TennisFeatureExtractor()
    
    # Test on a single match
    print("Testing feature extraction on match ID 100...")
    features = extractor.extract_features(match_id=100, lookback_months=36)
    
    print("\nExtracted Features:")
    for key, value in features.items():
        if key not in ['match_id', 'player1_id', 'player2_id', 'surface', 'match_date']:
            print(f"  {key:25s}: {value:8.4f}")
            
    print(f"\nUncertainty Score: {features['UNCERTAINTY']:.4f}")
    
    extractor.close()


if __name__ == "__main__":
    main()
