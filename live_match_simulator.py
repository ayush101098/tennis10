"""
🎾 Live Tennis Match Simulator with Profitable Betting Opportunities
====================================================================

Simulates a live tennis match between two players with:
- Markov chain probability calculations at every point
- Real player data from database
- Live odds simulation
- Kelly criterion optimal bet sizing
- Profit/Loss tracking
- Target: Turn $1000 → $5000

Usage:
    python live_match_simulator.py
"""

import numpy as np
import sqlite3
from datetime import datetime
from typing import Dict, Tuple, Optional
import random
from hierarchical_model import HierarchicalTennisModel
from scipy.special import comb


class LiveMatchSimulator:
    """Simulates live tennis matches with betting opportunities"""
    
    POINT_NAMES = {0: '0', 1: '15', 2: '30', 3: '40'}
    
    def __init__(self, db_path='tennis_data.db', bankroll=1000.0, target=5000.0):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.markov = HierarchicalTennisModel(db_path)
        
        # Bankroll management
        self.starting_bankroll = bankroll
        self.bankroll = bankroll
        self.target = target
        self.kelly_fraction = 0.25  # Quarter Kelly for safety
        self.min_edge = 0.03  # 3% minimum edge to bet
        
        # Match state
        self.player1_name = ""
        self.player2_name = ""
        self.player1_id = None
        self.player2_id = None
        self.surface = "Hard"
        self.best_of = 3
        
        # True probabilities (from player stats)
        self.p_point_p1_serve = 0.63
        self.p_point_p2_serve = 0.63
        
        # Current score
        self.sets_p1 = 0
        self.sets_p2 = 0
        self.games_p1 = 0
        self.games_p2 = 0
        self.points_p1 = 0
        self.points_p2 = 0
        self.server = 1  # 1 or 2
        self.is_tiebreak = False
        
        # Betting tracking
        self.bets_placed = []
        self.total_wagered = 0
        self.total_profit = 0
        
    def find_player(self, name: str) -> Optional[int]:
        """Find player ID by name (fuzzy match)"""
        query = """
        SELECT player_id, player_name 
        FROM players 
        WHERE LOWER(player_name) LIKE ? 
        LIMIT 5
        """
        cursor = self.conn.execute(query, (f'%{name.lower()}%',))
        results = cursor.fetchall()
        
        if not results:
            return None
        
        if len(results) == 1:
            return results[0][0]
        
        # Multiple matches - show options
        print(f"\n🔍 Multiple players found for '{name}':")
        for i, (pid, pname) in enumerate(results, 1):
            print(f"  {i}. {pname}")
        
        choice = input("Enter number (or 0 to skip): ")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                return results[idx][0]
        except:
            pass
        
        return None
    
    def setup_match(self, player1_name: str, player2_name: str, 
                   surface: str = "Hard", best_of: int = 3):
        """Initialize match with real player data"""
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.surface = surface
        self.best_of = best_of
        
        print(f"\n{'='*70}")
        print(f"🎾 LIVE MATCH SIMULATOR")
        print(f"{'='*70}")
        print(f"  {player1_name} vs {player2_name}")
        print(f"  Surface: {surface} | Best of {best_of}")
        print(f"{'='*70}")
        
        # Find players in database
        self.player1_id = self.find_player(player1_name)
        self.player2_id = self.find_player(player2_name)
        
        if not self.player1_id or not self.player2_id:
            print(f"\n⚠️  Using default probabilities (players not in database)")
            self.p_point_p1_serve = 0.63
            self.p_point_p2_serve = 0.63
        else:
            # Get real probabilities from database
            result = self.markov.predict_match(
                self.player1_id, self.player2_id,
                surface, num_sets=best_of
            )
            self.p_point_p1_serve = result['p_point_1_serve']
            self.p_point_p2_serve = result['p_point_2_serve']
            
            print(f"\n📊 Player Statistics (from database):")
            print(f"  {player1_name} serve strength: {self.p_point_p1_serve:.1%}")
            print(f"  {player2_name} serve strength: {self.p_point_p2_serve:.1%}")
        
        print(f"\n💰 Bankroll: ${self.bankroll:,.0f}")
        print(f"🎯 Target: ${self.target:,.0f}")
        print(f"{'='*70}\n")
    
    def p_game_from_score(self, p_point: float, score_a: int, score_b: int) -> float:
        """
        Calculate P(server wins game) from current score using Markov chain.
        
        Based on Barnett & Clarke (2005) recursive formula.
        """
        if score_a >= 4 and score_a >= score_b + 2:
            return 1.0
        if score_b >= 4 and score_b >= score_a + 2:
            return 0.0
        
        # Deuce situations
        if score_a >= 3 and score_b >= 3:
            if score_a == score_b:  # Deuce
                # P(win from deuce) = p² / (1 - 2pq)
                q = 1 - p_point
                return (p_point ** 2) / (1 - 2 * p_point * q)
            elif score_a == score_b + 1:  # Advantage server
                q = 1 - p_point
                p_deuce = (p_point ** 2) / (1 - 2 * p_point * q)
                return p_point + (1 - p_point) * p_deuce
            else:  # Advantage returner
                q = 1 - p_point
                p_deuce = (p_point ** 2) / (1 - 2 * p_point * q)
                return p_point * p_deuce
        
        # Before deuce - recursive
        p_win = p_point * self.p_game_from_score(p_point, score_a + 1, score_b)
        p_lose = (1 - p_point) * self.p_game_from_score(p_point, score_a, score_b + 1)
        return p_win + p_lose
    
    def p_set_from_games(self, p_game_hold: float, games_a: int, games_b: int) -> float:
        """
        Calculate P(player A wins set) from current game score.
        
        Assumes alternating serve. Formula from Barnett & Clarke (2005).
        """
        # Set already won
        if games_a >= 6 and games_a >= games_b + 2:
            return 1.0
        if games_b >= 6 and games_b >= games_a + 2:
            return 0.0
        
        # Tiebreak at 6-6
        if games_a == 6 and games_b == 6:
            # Simplified tiebreak model (can be improved)
            return 0.5
        
        # Simplified set calculation
        # This is complex analytically - using approximation
        total_games = games_a + games_b
        is_a_serving = (total_games % 2 == 0)
        
        if is_a_serving:
            current_p_game = p_game_hold
        else:
            current_p_game = 1 - p_game_hold
        
        # Monte Carlo simulation for remaining games (simplified)
        # For production, use exact Markov chain calculation
        return 0.5  # Placeholder - implement full calculation if needed
    
    def p_match_from_sets(self, p_set: float, sets_a: int, sets_b: int) -> float:
        """Calculate P(player A wins match) from current set score"""
        sets_to_win = (self.best_of + 1) // 2
        
        if sets_a >= sets_to_win:
            return 1.0
        if sets_b >= sets_to_win:
            return 0.0
        
        # Remaining sets needed
        need_a = sets_to_win - sets_a
        need_b = sets_to_win - sets_b
        
        # Sum over all ways A can win
        prob = 0.0
        for a_wins in range(need_a, need_a + need_b):
            # A wins exactly `a_wins` of next `a_wins + need_b - 1` sets
            # Then wins the final set
            if a_wins + need_b - 1 >= a_wins:
                ways = comb(a_wins + need_b - 1, a_wins, exact=True)
                prob += ways * (p_set ** a_wins) * ((1 - p_set) ** (need_b - 1)) * p_set
        
        return prob
    
    def get_current_probabilities(self) -> Dict[str, float]:
        """Get all current win probabilities"""
        # Determine who is serving
        if self.server == 1:
            p_point_server = self.p_point_p1_serve
            p_game_hold = self.p_game_from_score(p_point_server, self.points_p1, self.points_p2)
        else:
            p_point_server = self.p_point_p2_serve
            p_game_hold = self.p_game_from_score(p_point_server, self.points_p2, self.points_p1)
        
        # Current game probability
        p_p1_wins_game = p_game_hold if self.server == 1 else (1 - p_game_hold)
        
        # Set probability (simplified - use games count)
        # More accurate calculation would use Markov chain for sets
        p_p1_wins_set = 0.5  # Placeholder
        
        # Match probability (simplified)
        p_p1_wins_match = 0.5  # Placeholder
        
        return {
            'p_p1_wins_game': p_p1_wins_game,
            'p_p2_wins_game': 1 - p_p1_wins_game,
            'p_p1_wins_set': p_p1_wins_set,
            'p_p2_wins_set': 1 - p_p1_wins_set,
            'p_p1_wins_match': p_p1_wins_match,
            'p_p2_wins_match': 1 - p_p1_wins_match
        }
    
    def generate_market_odds(self, true_prob: float, margin: float = 0.05) -> Tuple[float, float]:
        """
        Generate bookmaker odds with random variation.
        
        Returns (back_odds, lay_odds) where back > lay (bookmaker spread)
        """
        # Add bookmaker margin
        implied_prob_back = true_prob * (1 + margin/2)
        implied_prob_lay = true_prob * (1 - margin/2)
        
        # Add random noise (market inefficiency)
        noise = np.random.normal(0, 0.02)
        implied_prob_back += noise
        implied_prob_lay += noise
        
        # Clamp
        implied_prob_back = np.clip(implied_prob_back, 0.01, 0.99)
        implied_prob_lay = np.clip(implied_prob_lay, 0.01, 0.99)
        
        # Convert to decimal odds
        back_odds = 1.0 / implied_prob_back
        lay_odds = 1.0 / implied_prob_lay
        
        return (back_odds, lay_odds)
    
    def calculate_edge(self, true_prob: float, odds: float) -> float:
        """Calculate betting edge: EV% = (true_prob * odds) - 1"""
        return (true_prob * odds) - 1.0
    
    def kelly_bet_size(self, edge: float, odds: float) -> float:
        """
        Calculate Kelly criterion bet size.
        
        Kelly% = edge / (odds - 1)
        We use fractional Kelly for safety.
        """
        if edge <= 0 or odds <= 1.0:
            return 0.0
        
        kelly_pct = edge / (odds - 1.0)
        fractional_kelly = kelly_pct * self.kelly_fraction
        
        # Don't bet more than 10% of bankroll on single bet
        max_bet_pct = 0.10
        bet_pct = min(fractional_kelly, max_bet_pct)
        
        return max(0.0, bet_pct)
    
    def check_betting_opportunities(self):
        """Check for profitable betting opportunities at current state"""
        probs = self.get_current_probabilities()
        opportunities = []
        
        # Check match winner market
        p_p1 = probs['p_p1_wins_match']
        back_odds_p1, _ = self.generate_market_odds(p_p1)
        edge_p1 = self.calculate_edge(p_p1, back_odds_p1)
        
        if edge_p1 >= self.min_edge:
            bet_size_pct = self.kelly_bet_size(edge_p1, back_odds_p1)
            bet_amount = self.bankroll * bet_size_pct
            
            if bet_amount >= 10:  # Minimum $10 bet
                opportunities.append({
                    'market': f'{self.player1_name} to win match',
                    'odds': back_odds_p1,
                    'true_prob': p_p1,
                    'edge': edge_p1,
                    'bet_amount': bet_amount,
                    'player': 1
                })
        
        p_p2 = probs['p_p2_wins_match']
        back_odds_p2, _ = self.generate_market_odds(p_p2)
        edge_p2 = self.calculate_edge(p_p2, back_odds_p2)
        
        if edge_p2 >= self.min_edge:
            bet_size_pct = self.kelly_bet_size(edge_p2, back_odds_p2)
            bet_amount = self.bankroll * bet_size_pct
            
            if bet_amount >= 10:
                opportunities.append({
                    'market': f'{self.player2_name} to win match',
                    'odds': back_odds_p2,
                    'true_prob': p_p2,
                    'edge': edge_p2,
                    'bet_amount': bet_amount,
                    'player': 2
                })
        
        # Check current game market
        p_p1_game = probs['p_p1_wins_game']
        back_odds_p1_game, _ = self.generate_market_odds(p_p1_game)
        edge_p1_game = self.calculate_edge(p_p1_game, back_odds_p1_game)
        
        if edge_p1_game >= self.min_edge:
            bet_size_pct = self.kelly_bet_size(edge_p1_game, back_odds_p1_game)
            bet_amount = self.bankroll * bet_size_pct
            
            if bet_amount >= 10:
                opportunities.append({
                    'market': f'{self.player1_name} to win current game',
                    'odds': back_odds_p1_game,
                    'true_prob': p_p1_game,
                    'edge': edge_p1_game,
                    'bet_amount': bet_amount,
                    'player': 1,
                    'game_market': True
                })
        
        return opportunities
    
    def place_bet(self, opportunity: Dict):
        """Place a bet and track it"""
        bet = {
            'timestamp': datetime.now(),
            'market': opportunity['market'],
            'odds': opportunity['odds'],
            'stake': opportunity['bet_amount'],
            'edge': opportunity['edge'],
            'score': self.get_score_string(),
            'player': opportunity['player'],
            'settled': False
        }
        
        self.bets_placed.append(bet)
        self.total_wagered += bet['stake']
        self.bankroll -= bet['stake']
        
        print(f"\n💸 BET PLACED:")
        print(f"   Market: {bet['market']}")
        print(f"   Odds: {bet['odds']:.2f}")
        print(f"   Stake: ${bet['stake']:.2f}")
        print(f"   Edge: {bet['edge']*100:.1f}%")
        print(f"   Score: {bet['score']}")
        print(f"   Bankroll: ${self.bankroll:,.2f}")
    
    def simulate_point(self):
        """Simulate one point being played"""
        # Determine probability based on server
        if self.server == 1:
            p_server_wins = self.p_point_p1_serve
        else:
            p_server_wins = self.p_point_p2_serve
        
        # Simulate point outcome
        server_wins = np.random.random() < p_server_wins
        
        if server_wins:
            if self.server == 1:
                self.points_p1 += 1
            else:
                self.points_p2 += 1
        else:
            if self.server == 1:
                self.points_p2 += 1
            else:
                self.points_p1 += 1
        
        # Check for game won
        self.check_game_complete()
    
    def check_game_complete(self):
        """Check if current game is complete"""
        if self.points_p1 >= 4 and self.points_p1 >= self.points_p2 + 2:
            # Player 1 wins game
            self.games_p1 += 1
            self.points_p1 = 0
            self.points_p2 = 0
            self.server = 2 if self.server == 1 else 1
            self.check_set_complete()
        elif self.points_p2 >= 4 and self.points_p2 >= self.points_p1 + 2:
            # Player 2 wins game
            self.games_p2 += 1
            self.points_p1 = 0
            self.points_p2 = 0
            self.server = 2 if self.server == 1 else 1
            self.check_set_complete()
    
    def check_set_complete(self):
        """Check if current set is complete"""
        if self.games_p1 >= 6 and self.games_p1 >= self.games_p2 + 2:
            self.sets_p1 += 1
            self.games_p1 = 0
            self.games_p2 = 0
        elif self.games_p2 >= 6 and self.games_p2 >= self.games_p1 + 2:
            self.sets_p2 += 1
            self.games_p1 = 0
            self.games_p2 = 0
        # TODO: Add tiebreak logic
    
    def is_match_complete(self) -> bool:
        """Check if match is complete"""
        sets_to_win = (self.best_of + 1) // 2
        return self.sets_p1 >= sets_to_win or self.sets_p2 >= sets_to_win
    
    def get_score_string(self) -> str:
        """Get current score as string"""
        point_names = {0: '0', 1: '15', 2: '30', 3: '40'}
        
        if self.points_p1 >= 3 and self.points_p2 >= 3:
            if self.points_p1 == self.points_p2:
                point_str = "40-40"
            elif self.points_p1 > self.points_p2:
                point_str = "AD-40"
            else:
                point_str = "40-AD"
        else:
            p1_pts = point_names.get(self.points_p1, '40')
            p2_pts = point_names.get(self.points_p2, '40')
            point_str = f"{p1_pts}-{p2_pts}"
        
        server_indicator = "●" if self.server == 1 else "○"
        return f"Sets: {self.sets_p1}-{self.sets_p2} | Games: {self.games_p1}-{self.games_p2} | {point_str} {server_indicator}"
    
    def print_status(self):
        """Print current match status"""
        print(f"\n{self.get_score_string()}")
        print(f"{'─'*70}")
        
        probs = self.get_current_probabilities()
        print(f"Win Probabilities:")
        print(f"  {self.player1_name}: {probs['p_p1_wins_match']*100:.1f}% match | {probs['p_p1_wins_game']*100:.1f}% game")
        print(f"  {self.player2_name}: {probs['p_p2_wins_match']*100:.1f}% match | {probs['p_p2_wins_game']*100:.1f}% game")
        
        print(f"\n💰 Bankroll: ${self.bankroll:,.2f} | P&L: ${self.bankroll - self.starting_bankroll:+,.2f}")
        print(f"🎯 Progress: {self.bankroll/self.target*100:.1f}% to target (${self.target:,.0f})")
    
    def run_simulation(self, auto_bet: bool = True):
        """Run full match simulation"""
        point_count = 0
        
        while not self.is_match_complete():
            point_count += 1
            
            # Check for betting opportunities every few points
            if point_count % 3 == 0:
                opportunities = self.check_betting_opportunities()
                
                if opportunities and auto_bet:
                    # Take the best opportunity
                    best_opp = max(opportunities, key=lambda x: x['edge'])
                    self.place_bet(best_opp)
            
            # Simulate the point
            self.simulate_point()
            
            # Print status every game
            if self.points_p1 == 0 and self.points_p2 == 0:
                self.print_status()
            
            # Check if we hit target
            if self.bankroll >= self.target:
                print(f"\n🎉 TARGET REACHED! ${self.bankroll:,.2f}")
                break
            
            # Check if we're broke
            if self.bankroll < 10:
                print(f"\n💔 BANKRUPT! ${self.bankroll:,.2f}")
                break
        
        self.print_final_report()
    
    def print_final_report(self):
        """Print final betting report"""
        print(f"\n{'='*70}")
        print(f"📊 FINAL REPORT")
        print(f"{'='*70}")
        print(f"Match: {self.player1_name} vs {self.player2_name}")
        print(f"Final Score: Sets {self.sets_p1}-{self.sets_p2}")
        print(f"\nBankroll:")
        print(f"  Starting: ${self.starting_bankroll:,.2f}")
        print(f"  Final: ${self.bankroll:,.2f}")
        print(f"  P&L: ${self.bankroll - self.starting_bankroll:+,.2f}")
        print(f"  ROI: {(self.bankroll/self.starting_bankroll - 1)*100:+.1f}%")
        print(f"\nBetting:")
        print(f"  Bets placed: {len(self.bets_placed)}")
        print(f"  Total wagered: ${self.total_wagered:,.2f}")
        print(f"={'='*70}\n")


def main():
    """Main entry point"""
    print("🎾 Live Tennis Match Simulator")
    print("=" * 70)
    
    # Get player names
    player1 = input("\nPlayer 1 name: ").strip()
    player2 = input("Player 2 name: ").strip()
    
    surface = input("Surface (Hard/Clay/Grass) [Hard]: ").strip() or "Hard"
    best_of = int(input("Best of (3 or 5) [3]: ").strip() or "3")
    
    bankroll = float(input("\nStarting bankroll [$1000]: ").strip() or "1000")
    target = float(input("Target bankroll [$5000]: ").strip() or "5000")
    
    # Create simulator
    sim = LiveMatchSimulator(bankroll=bankroll, target=target)
    sim.setup_match(player1, player2, surface, best_of)
    
    # Run simulation
    input("\nPress Enter to start simulation...")
    sim.run_simulation(auto_bet=True)


if __name__ == "__main__":
    main()
