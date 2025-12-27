"""
Hierarchical Tennis Match Prediction Model (Barnett-Clarke)

Based on:
- Barnett & Clarke (2005) "Combining player statistics to predict outcomes"
- O'Malley (2008) "Probability formulas and statistical analysis in tennis"
- McHale & Morton (2011) "A Bradley-Terry type model for forecasting tennis"

Hierarchy:
1. Point-level: Probability of winning a point on serve
2. Game-level: Probability of winning a game (Markov chain)
3. Set-level: Probability of winning a set
4. Match-level: Probability of winning a match (best of 3 or 5)
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalTennisModel:
    """Hierarchical probability model for tennis match prediction"""
    
    def __init__(self, db_path: str = 'tennis_data.db'):
        """Initialize model with database connection"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_player_serve_stats(self, player_id: int, surface: str, 
                               lookback_days: int = 365,
                               reference_date: Optional[str] = None) -> Dict[str, float]:
        """
        Get player's serve statistics from recent matches on similar surface
        
        Args:
            reference_date: Optional date to look back from (YYYY-MM-DD format)
                          If None, uses current date
        
        Returns:
            Dict with: first_serve_pct, first_serve_win_pct, second_serve_win_pct,
                      aces_per_game, df_per_game
        """
        if reference_date:
            date_filter = f"AND m.tournament_date < '{reference_date}' AND m.tournament_date >= date('{reference_date}', '-{lookback_days} days')"
        else:
            date_filter = f"AND m.tournament_date >= date('now', '-{lookback_days} days')"
        
        query = f"""
        SELECT 
            AVG(s.first_serve_pct) as first_serve_pct,
            AVG(s.first_serve_win_pct) as first_serve_win_pct,
            AVG(s.second_serve_win_pct) as second_serve_win_pct,
            AVG(CAST(s.aces AS FLOAT) / NULLIF(s.serve_games, 0)) as aces_per_game,
            AVG(CAST(s.double_faults AS FLOAT) / NULLIF(s.serve_games, 0)) as df_per_game,
            COUNT(*) as match_count
        FROM statistics s
        JOIN matches m ON s.match_id = m.match_id
        WHERE s.player_id = ?
            AND m.surface = ?
            {date_filter}
            AND s.first_serve_pct IS NOT NULL
        """
        
        df = pd.read_sql_query(query, self.conn, params=(player_id, surface))
        
        if df.empty or df['match_count'].iloc[0] < 5:
            # Fall back to all surfaces if insufficient data
            query_all = f"""
            SELECT 
                AVG(s.first_serve_pct) as first_serve_pct,
                AVG(s.first_serve_win_pct) as first_serve_win_pct,
                AVG(s.second_serve_win_pct) as second_serve_win_pct,
                AVG(CAST(s.aces AS FLOAT) / NULLIF(s.serve_games, 0)) as aces_per_game,
                AVG(CAST(s.double_faults AS FLOAT) / NULLIF(s.serve_games, 0)) as df_per_game,
                COUNT(*) as match_count
            FROM statistics s
            JOIN matches m ON s.match_id = m.match_id
            WHERE s.player_id = ?
                {date_filter}
                AND s.first_serve_pct IS NOT NULL
            """
            df = pd.read_sql_query(query_all, self.conn, params=(player_id,))
        
        if df.empty or df['match_count'].iloc[0] == 0:
            # Return ATP averages if no data
            return {
                'first_serve_pct': 0.625,
                'first_serve_win_pct': 0.715,
                'second_serve_win_pct': 0.510,
                'aces_per_game': 0.5,
                'df_per_game': 0.25,
                'match_count': 0
            }
        
        return df.iloc[0].to_dict()
    
    def get_player_return_stats(self, player_id: int, surface: str,
                               lookback_days: int = 365,
                               reference_date: Optional[str] = None) -> Dict[str, float]:
        """
        Get player's return statistics (opponent's serve stats when playing against this player)
        
        Args:
            reference_date: Optional date to look back from (YYYY-MM-DD format)
        
        Returns:
            Dict with return performance metrics
        """
        if reference_date:
            date_filter = f"AND m.tournament_date < '{reference_date}' AND m.tournament_date >= date('{reference_date}', '-{lookback_days} days')"
        else:
            date_filter = f"AND m.tournament_date >= date('now', '-{lookback_days} days')"
        
        # Get stats of opponents when playing against this player
        query = f"""
        SELECT 
            AVG(s.first_serve_win_pct) as opp_first_serve_win_pct,
            AVG(s.second_serve_win_pct) as opp_second_serve_win_pct,
            COUNT(*) as match_count
        FROM statistics s
        JOIN matches m ON s.match_id = m.match_id
        WHERE s.player_id != ?
            AND (m.winner_id = ? OR m.loser_id = ?)
            AND m.surface = ?
            {date_filter}
            AND s.first_serve_win_pct IS NOT NULL
        """
        
        df = pd.read_sql_query(query, self.conn, 
                              params=(player_id, player_id, player_id, surface))
        
        if df.empty or df['match_count'].iloc[0] < 5:
            # Fall back to all surfaces
            query_all = f"""
            SELECT 
                AVG(s.first_serve_win_pct) as opp_first_serve_win_pct,
                AVG(s.second_serve_win_pct) as opp_second_serve_win_pct,
                COUNT(*) as match_count
            FROM statistics s
            JOIN matches m ON s.match_id = m.match_id
            WHERE s.player_id != ?
                AND (m.winner_id = ? OR m.loser_id = ?)
                {date_filter}
                AND s.first_serve_win_pct IS NOT NULL
            """
            df = pd.read_sql_query(query_all, self.conn, 
                                  params=(player_id, player_id, player_id))
        
        if df.empty or df['match_count'].iloc[0] == 0:
            # Return neutral values
            return {
                'return_first_serve_pct': 0.285,  # 1 - 0.715
                'return_second_serve_pct': 0.490,  # 1 - 0.510
                'match_count': 0
            }
        
        # Convert opponent's serve win % to this player's return win %
        return {
            'return_first_serve_pct': 1.0 - df['opp_first_serve_win_pct'].iloc[0],
            'return_second_serve_pct': 1.0 - df['opp_second_serve_win_pct'].iloc[0],
            'match_count': df['match_count'].iloc[0]
        }
    
    def estimate_point_prob(self, server_stats: Dict[str, float], 
                           returner_stats: Dict[str, float],
                           surface: str) -> float:
        """
        Estimate probability that server wins a point
        
        Uses McHale & Morton (2011) approach:
        P(server wins point) = P(1st serve in) × P(win | 1st serve in) + 
                              P(2nd serve in) × P(win | 2nd serve in)
        
        Adjusted for opponent return strength using a mixing parameter
        """
        # Server's serve probabilities
        p_first_in = server_stats['first_serve_pct']
        p_win_first = server_stats['first_serve_win_pct']
        p_win_second = server_stats['second_serve_win_pct']
        
        # Returner's return strength
        return_first_strength = returner_stats['return_first_serve_pct']
        return_second_strength = returner_stats['return_second_serve_pct']
        
        # Mix server ability with opponent return ability (60-40 weighting)
        # Server's skill is slightly more important on serve
        alpha = 0.60  # Weight for server's stats
        
        # Adjusted probabilities
        p_win_first_adj = alpha * p_win_first + (1 - alpha) * (1 - return_first_strength)
        p_win_second_adj = alpha * p_win_second + (1 - alpha) * (1 - return_second_strength)
        
        # Second serve in probability (assuming ~95% get second serve in)
        p_second_in = 0.95
        
        # Combined point win probability
        p_point = p_first_in * p_win_first_adj + (1 - p_first_in) * p_second_in * p_win_second_adj
        
        # Ensure probability is in valid range
        p_point = np.clip(p_point, 0.45, 0.85)
        
        return p_point
    
    def prob_game_win(self, p_point: float, is_tiebreak: bool = False) -> float:
        """
        Probability of winning a game given point win probability
        
        Uses recursive Markov chain formulas from Barnett (2005)
        
        For regular game:
        - Win from 0-0: need to reach 40-X and win
        - States: (server_points, returner_points)
        
        For tiebreak:
        - First to 7 points (with 2-point margin)
        """
        if is_tiebreak:
            return self._prob_tiebreak_win(p_point)
        
        # Regular game probabilities using dynamic programming
        # States: (server_score, opponent_score)
        # Scores: 0=0, 1=15, 2=30, 3=40, 4=game
        
        # Probability of winning from each state
        prob = {}
        
        # Base cases
        prob[(4, 0)] = prob[(4, 1)] = prob[(4, 2)] = 1.0  # Server won
        prob[(0, 4)] = prob[(1, 4)] = prob[(2, 4)] = 0.0  # Server lost
        
        # Deuce and advantage
        # P(win from deuce) = p * P(win from AD-in) + (1-p) * P(win from AD-out)
        # P(win from AD-in) = p * 1 + (1-p) * P(win from deuce)
        # P(win from AD-out) = p * P(win from deuce) + (1-p) * 0
        
        # Solving the system:
        # Let D = P(win from deuce)
        # D = p * [p + (1-p)*D] + (1-p) * [p*D]
        # D = p² + p(1-p)D + p(1-p)D
        # D = p² + 2p(1-p)D
        # D[1 - 2p(1-p)] = p²
        # D = p² / [1 - 2p(1-p)]
        # D = p² / [p² + (1-p)²]
        
        prob[(3, 3)] = (p_point ** 2) / (p_point ** 2 + (1 - p_point) ** 2)
        
        # Other states (working backwards)
        prob[(3, 2)] = p_point * 1.0 + (1 - p_point) * prob[(3, 3)]
        prob[(3, 1)] = p_point * 1.0 + (1 - p_point) * prob[(3, 2)]
        prob[(3, 0)] = p_point * 1.0 + (1 - p_point) * prob[(3, 1)]
        
        prob[(2, 3)] = p_point * prob[(3, 3)] + (1 - p_point) * 0.0
        prob[(1, 3)] = p_point * prob[(2, 3)] + (1 - p_point) * 0.0
        prob[(0, 3)] = p_point * prob[(1, 3)] + (1 - p_point) * 0.0
        
        prob[(2, 2)] = p_point * prob[(3, 2)] + (1 - p_point) * prob[(2, 3)]
        prob[(2, 1)] = p_point * prob[(3, 1)] + (1 - p_point) * prob[(2, 2)]
        prob[(2, 0)] = p_point * prob[(3, 0)] + (1 - p_point) * prob[(2, 1)]
        
        prob[(1, 2)] = p_point * prob[(2, 2)] + (1 - p_point) * prob[(1, 3)]
        prob[(1, 1)] = p_point * prob[(2, 1)] + (1 - p_point) * prob[(1, 2)]
        prob[(1, 0)] = p_point * prob[(2, 0)] + (1 - p_point) * prob[(1, 1)]
        
        prob[(0, 2)] = p_point * prob[(1, 2)] + (1 - p_point) * prob[(0, 3)]
        prob[(0, 1)] = p_point * prob[(1, 1)] + (1 - p_point) * prob[(0, 2)]
        prob[(0, 0)] = p_point * prob[(1, 0)] + (1 - p_point) * prob[(0, 1)]
        
        return prob[(0, 0)]
    
    def _prob_tiebreak_win(self, p_point: float) -> float:
        """
        Probability of winning a tiebreak
        
        First to 7 points with 2-point margin
        Uses exact calculation for scores up to 7-5, then deuce formula for 6-6+
        """
        from math import comb
        
        total_prob = 0.0
        
        # Win 7-0 through 7-5 (decisive wins before potential deuce)
        for opp_points in range(0, 6):
            # To win 7-X, must win exactly 7 points and opponent wins X
            # Among first (6+X) points, we win 6, then we win the 7th point
            # This is: C(6+X, 6) * p^6 * (1-p)^X * p = C(6+X, X) * p^7 * (1-p)^X
            total_points = 6 + opp_points
            prob = comb(total_points, opp_points) * (p_point ** 7) * ((1 - p_point) ** opp_points)
            total_prob += prob
        
        # If reach 6-6, use deuce formula (same as game deuce)
        # P(reach 6-6) * P(win from 6-6)
        prob_6_6 = comb(12, 6) * (p_point ** 6) * ((1 - p_point) ** 6)
        prob_win_from_6_6 = (p_point ** 2) / (p_point ** 2 + (1 - p_point) ** 2)
        total_prob += prob_6_6 * prob_win_from_6_6
        
        return total_prob
    
    def prob_set_win(self, p_game_server: float, p_game_returner: float) -> float:
        """
        Probability of winning a set
        
        Accounts for:
        - Service alternation
        - Tiebreak at 6-6
        
        Simplified model: 
        - Approximate by averaging server/returner game probabilities
        - Use similar Markov approach as game-level
        """
        # Average probability of winning a game (accounting for service alternation)
        # In a set, player serves ~half the games
        p_avg = (p_game_server + (1 - p_game_returner)) / 2
        
        # Probability of winning set to 6 (before tiebreak)
        total_prob = 0.0
        
        # Win 6-0, 6-1, 6-2, 6-3, 6-4
        for opp_games in range(0, 5):
            # Probability of reaching 6-X
            from math import comb
            total_games = 6 + opp_games
            # Need to win 5 of first (total_games - 1), then win last
            prob = comb(total_games - 1, 5) * (p_avg ** 6) * ((1 - p_avg) ** opp_games)
            total_prob += prob
        
        # Win 7-5 (no tiebreak)
        # Reach 5-5, then win next 2
        from math import comb
        prob_5_5 = comb(10, 5) * (p_avg ** 5) * ((1 - p_avg) ** 5)
        prob_win_7_5 = prob_5_5 * (p_avg ** 2)
        total_prob += prob_win_7_5
        
        # Win 7-6 (tiebreak)
        # Reach 6-6, then win tiebreak
        # From 5-5, win 1 and lose 1 to get to 6-6
        prob_6_6 = prob_5_5 * 2 * p_avg * (1 - p_avg)
        
        # In tiebreak, service alternates more frequently
        # Simplified: use average point probability
        p_point_tb = (p_game_server ** 0.5 + (1 - p_game_returner) ** 0.5) / 2
        p_point_tb = p_point_tb ** 2  # Square back to get point prob from game prob approx
        
        # This is rough - use point probs from parent
        # For now, estimate from game probs
        p_tb = 0.5 + (p_avg - 0.5) * 0.8  # Tiebreak is closer to 50-50
        total_prob += prob_6_6 * p_tb
        
        return np.clip(total_prob, 0.0, 1.0)
    
    def prob_match_win(self, p_set: float, num_sets: int = 3) -> float:
        """
        Probability of winning a match (best of 3 or best of 5)
        
        Args:
            p_set: Probability of winning a set
            num_sets: 3 for best-of-3, 5 for best-of-5
        """
        sets_to_win = (num_sets + 1) // 2  # 2 for BO3, 3 for BO5
        
        total_prob = 0.0
        
        # Win in minimum sets (2-0 or 3-0)
        total_prob += p_set ** sets_to_win
        
        # Win in sets_to_win + 1 (2-1 or 3-1)
        from math import comb
        if num_sets >= 3:
            # Win (sets_to_win - 1) of first (sets_to_win), then win last
            prob = comb(sets_to_win, sets_to_win - 1) * (p_set ** sets_to_win) * ((1 - p_set) ** 1)
            total_prob += prob
        
        # Win in sets_to_win + 2 (3-2 for BO5)
        if num_sets == 5:
            # Win (sets_to_win - 1) of first (sets_to_win + 1), then win last
            prob = comb(sets_to_win + 1, sets_to_win - 1) * (p_set ** sets_to_win) * ((1 - p_set) ** 2)
            total_prob += prob
        
        return np.clip(total_prob, 0.0, 1.0)
    
    def predict_match(self, player1_id: int, player2_id: int, 
                     surface: str, num_sets: int = 3,
                     match_date: Optional[str] = None) -> Dict[str, float]:
        """
        Predict match outcome using hierarchical model
        
        Args:
            player1_id: ID of first player
            player2_id: ID of second player  
            surface: 'Hard', 'Clay', or 'Grass'
            num_sets: 3 for best-of-3, 5 for best-of-5
            match_date: Optional match date for historical predictions (YYYY-MM-DD format)
            
        Returns:
            Dict with:
                - p_player1_win: Probability player 1 wins
                - p_player2_win: Probability player 2 wins
                - p_point_1_serve: Point prob when player 1 serves
                - p_point_2_serve: Point prob when player 2 serves
                - p_game_1_serve: Game prob when player 1 serves
                - p_game_2_serve: Game prob when player 2 serves
                - p_set_player1: Set prob for player 1
        """
        # Get player statistics
        p1_serve = self.get_player_serve_stats(player1_id, surface, reference_date=match_date)
        p2_serve = self.get_player_serve_stats(player2_id, surface, reference_date=match_date)
        
        p1_return = self.get_player_return_stats(player1_id, surface, reference_date=match_date)
        p2_return = self.get_player_return_stats(player2_id, surface, reference_date=match_date)
        
        # Point-level probabilities
        p_point_1_serve = self.estimate_point_prob(p1_serve, p2_return, surface)
        p_point_2_serve = self.estimate_point_prob(p2_serve, p1_return, surface)
        
        # Game-level probabilities
        p_game_1_serve = self.prob_game_win(p_point_1_serve)
        p_game_2_serve = self.prob_game_win(p_point_2_serve)
        
        # Set-level probability
        p_set_player1 = self.prob_set_win(p_game_1_serve, p_game_2_serve)
        
        # Match-level probability
        p_match_player1 = self.prob_match_win(p_set_player1, num_sets)
        
        return {
            'p_player1_win': p_match_player1,
            'p_player2_win': 1 - p_match_player1,
            'p_point_1_serve': p_point_1_serve,
            'p_point_2_serve': p_point_2_serve,
            'p_game_1_serve': p_game_1_serve,
            'p_game_2_serve': p_game_2_serve,
            'p_set_player1': p_set_player1,
            'player1_serve_stats': p1_serve,
            'player2_serve_stats': p2_serve,
        }
    
    def get_player_name(self, player_id: int) -> str:
        """Get player name from ID"""
        query = "SELECT player_name FROM players WHERE player_id = ?"
        df = pd.read_sql_query(query, self.conn, params=(player_id,))
        if df.empty:
            return f"Player {player_id}"
        return df['player_name'].iloc[0]


if __name__ == "__main__":
    # Test the model
    model = HierarchicalTennisModel()
    
    # Get a recent match for testing
    query = """
    SELECT match_id, winner_id, loser_id, surface, tournament_name, tournament_date
    FROM matches 
    WHERE tournament_date >= '2024-01-01'
    LIMIT 5
    """
    matches = pd.read_sql_query(query, model.conn)
    
    print("Testing Hierarchical Model on Recent Matches:\n")
    print("=" * 80)
    
    for _, match in matches.iterrows():
        player1_id = match['winner_id']
        player2_id = match['loser_id']
        
        player1_name = model.get_player_name(player1_id)
        player2_name = model.get_player_name(player2_id)
        
        result = model.predict_match(
            player1_id, 
            player2_id, 
            match['surface'],
            match_date=match['tournament_date']
        )
        
        print(f"\n{match['tournament_name']} ({match['tournament_date']})")
        print(f"Surface: {match['surface']}")
        print(f"\n{player1_name} vs {player2_name}")
        print(f"Actual winner: {player1_name}")
        print(f"\nPredicted probabilities:")
        print(f"  {player1_name}: {result['p_player1_win']:.1%}")
        print(f"  {player2_name}: {result['p_player2_win']:.1%}")
        print(f"\nPoint win probabilities (on serve):")
        print(f"  {player1_name}: {result['p_point_1_serve']:.1%}")
        print(f"  {player2_name}: {result['p_point_2_serve']:.1%}")
        print(f"\nGame win probabilities (on serve):")
        print(f"  {player1_name}: {result['p_game_1_serve']:.1%}")
        print(f"  {player2_name}: {result['p_game_2_serve']:.1%}")
        print(f"\nSet win probability for {player1_name}: {result['p_set_player1']:.1%}")
        print("-" * 80)
    
    model.close()
