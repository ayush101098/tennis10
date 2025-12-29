"""
Live Point-by-Point Tennis Match Tracker
=========================================
Track matches point by point and identify break opportunities.

This tool uses Markov chain probabilities to calculate:
- Win probability at every score state
- Break probability (returner winning the game)
- Optimal moments to bet on game/set/match outcomes

Usage:
    python point_tracker.py
    
Then follow the interactive prompts to track the match.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import sqlite3
from hierarchical_model import HierarchicalTennisModel


class PointByPointTracker:
    """
    Track tennis match point-by-point with live probability updates.
    
    Calculates at each point:
    - P(server holds) = probability server wins this game
    - P(break) = probability returner wins this game  
    - P(player1 wins match) = updated match win probability
    """
    
    # Score mapping
    POINT_NAMES = {0: '0', 1: '15', 2: '30', 3: '40', 4: 'Game'}
    
    def __init__(self, db_path='tennis_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.markov = HierarchicalTennisModel(db_path)
        
        # Match state
        self.player1_name = ""
        self.player2_name = ""
        self.player1_id = None
        self.player2_id = None
        self.surface = "Hard"
        self.best_of = 3
        
        # Point probabilities
        self.p_point_p1_serve = 0.63  # P(p1 wins point on p1 serve)
        self.p_point_p2_serve = 0.63  # P(p2 wins point on p2 serve)
        
        # Current score state
        self.sets = [0, 0]  # Sets won by each player
        self.games = [[0, 0]]  # Games in current set
        self.points = [0, 0]  # Points in current game
        self.server = 1  # 1 = player1 serving, 2 = player2 serving
        self.is_tiebreak = False
        
        # History
        self.history = []
        
    def setup_match(self, player1_name: str, player2_name: str, 
                   surface: str = "Hard", best_of: int = 3):
        """Initialize match with player information."""
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.surface = surface
        self.best_of = best_of
        
        # Look up players
        self.player1_id = self._find_player(player1_name)
        self.player2_id = self._find_player(player2_name)
        
        # Get serve probabilities from historical data
        if self.player1_id and self.player2_id:
            result = self.markov.predict_match(
                self.player1_id, self.player2_id, 
                surface, num_sets=best_of
            )
            self.p_point_p1_serve = result['p_point_1_serve']
            self.p_point_p2_serve = result['p_point_2_serve']
            
            print(f"\n📊 Based on historical data:")
            print(f"   {player1_name} serve: {self.p_point_p1_serve:.1%} point win rate")
            print(f"   {player2_name} serve: {self.p_point_p2_serve:.1%} point win rate")
        else:
            print(f"⚠️  Players not found in database, using default probabilities")
            self.p_point_p1_serve = 0.63
            self.p_point_p2_serve = 0.63
        
        # Reset score
        self.sets = [0, 0]
        self.games = [[0, 0]]
        self.points = [0, 0]
        self.server = 1
        self.is_tiebreak = False
        self.history = []
        
        print(f"\n🎾 Match Setup Complete")
        print(f"   {player1_name} vs {player2_name}")
        print(f"   Surface: {surface}")
        print(f"   Best of {best_of}")
        print(f"   {player1_name} serving first")
    
    def _find_player(self, name_query: str) -> Optional[int]:
        """Find player ID by name."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT player_id, player_name 
            FROM players 
            WHERE player_name LIKE ? 
            ORDER BY LENGTH(player_name)
            LIMIT 1
        """, (f'%{name_query}%',))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def get_current_p_point(self) -> float:
        """Get probability current server wins the point."""
        if self.server == 1:
            return self.p_point_p1_serve
        else:
            return 1 - self.p_point_p2_serve  # P2 serve, so P1 wins = 1 - P2 wins
    
    def prob_game_from_score(self, server_pts: int, returner_pts: int, 
                             p_point: float, is_tiebreak: bool = False) -> float:
        """
        Calculate probability server wins game from current point score.
        
        Uses backward induction on the Markov chain.
        """
        if is_tiebreak:
            return self._prob_tiebreak_from_score(server_pts, returner_pts, p_point)
        
        # Base cases
        if server_pts >= 4 and server_pts >= returner_pts + 2:
            return 1.0  # Server won
        if returner_pts >= 4 and returner_pts >= server_pts + 2:
            return 0.0  # Server lost (break)
        
        # Deuce situations (both at 3+)
        if server_pts >= 3 and returner_pts >= 3:
            # At deuce/ad, use deuce formula
            p_deuce = (p_point ** 2) / (p_point ** 2 + (1 - p_point) ** 2)
            
            if server_pts == returner_pts:
                return p_deuce  # Deuce
            elif server_pts > returner_pts:
                return p_point + (1 - p_point) * p_deuce  # Ad-in
            else:
                return p_point * p_deuce  # Ad-out
        
        # Regular game - recursive calculation
        # Cache for efficiency
        cache = {}
        
        def prob_from(s, r):
            if (s, r) in cache:
                return cache[(s, r)]
            
            # Win states
            if s == 4 and r <= 2:
                return 1.0
            if r == 4 and s <= 2:
                return 0.0
            
            # Deuce
            if s >= 3 and r >= 3:
                p_d = (p_point ** 2) / (p_point ** 2 + (1 - p_point) ** 2)
                if s == r:
                    cache[(s, r)] = p_d
                elif s > r:
                    cache[(s, r)] = p_point + (1 - p_point) * p_d
                else:
                    cache[(s, r)] = p_point * p_d
                return cache[(s, r)]
            
            # Regular
            next_win = prob_from(min(s + 1, 4), r)
            next_lose = prob_from(s, min(r + 1, 4))
            cache[(s, r)] = p_point * next_win + (1 - p_point) * next_lose
            return cache[(s, r)]
        
        return prob_from(server_pts, returner_pts)
    
    def _prob_tiebreak_from_score(self, p1_pts: int, p2_pts: int, p_point: float) -> float:
        """Calculate probability P1 wins tiebreak from current score."""
        # In tiebreak, server alternates every 2 points (after first point)
        # Simplified: use average point probability
        
        # Check if already won
        if p1_pts >= 7 and p1_pts >= p2_pts + 2:
            return 1.0
        if p2_pts >= 7 and p2_pts >= p1_pts + 2:
            return 0.0
        
        # If at 6-6 or beyond, use mini-deuce formula
        if p1_pts >= 6 and p2_pts >= 6:
            return (p_point ** 2) / (p_point ** 2 + (1 - p_point) ** 2)
        
        # Otherwise, approximate with recursive formula
        cache = {}
        
        def prob_from(p1, p2):
            if (p1, p2) in cache:
                return cache[(p1, p2)]
            
            if p1 >= 7 and p1 >= p2 + 2:
                return 1.0
            if p2 >= 7 and p2 >= p1 + 2:
                return 0.0
            
            if p1 >= 6 and p2 >= 6:
                return (p_point ** 2) / (p_point ** 2 + (1 - p_point) ** 2)
            
            next_win = prob_from(p1 + 1, p2)
            next_lose = prob_from(p1, p2 + 1)
            cache[(p1, p2)] = p_point * next_win + (1 - p_point) * next_lose
            return cache[(p1, p2)]
        
        return prob_from(p1_pts, p2_pts)
    
    def prob_set_from_score(self, p1_games: int, p2_games: int,
                           p_game_hold_p1: float, p_game_hold_p2: float,
                           current_server: int) -> float:
        """Calculate probability P1 wins set from current game score."""
        
        # Check if already won
        if p1_games >= 6 and p1_games >= p2_games + 2:
            return 1.0
        if p2_games >= 6 and p2_games >= p1_games + 2:
            return 0.0
        if p1_games == 7:
            return 1.0
        if p2_games == 7:
            return 0.0
        
        # Tiebreak scenario
        if p1_games == 6 and p2_games == 6:
            # Tiebreak probabilities are complex, use simplified model
            p_avg = (p_game_hold_p1 + (1 - p_game_hold_p2)) / 2
            return p_avg  # Simplified
        
        cache = {}
        
        def prob_from(g1, g2, server):
            if (g1, g2, server) in cache:
                return cache[(g1, g2, server)]
            
            # Win conditions
            if g1 >= 6 and g1 >= g2 + 2:
                return 1.0
            if g2 >= 6 and g2 >= g1 + 2:
                return 0.0
            if g1 == 7:
                return 1.0
            if g2 == 7:
                return 0.0
            
            # Tiebreak at 6-6
            if g1 == 6 and g2 == 6:
                p_avg = (p_game_hold_p1 + (1 - p_game_hold_p2)) / 2
                return p_avg
            
            # Current game probabilities
            if server == 1:
                p_p1_wins_game = p_game_hold_p1
            else:
                p_p1_wins_game = 1 - p_game_hold_p2
            
            next_server = 3 - server  # Switch server
            
            # P1 wins this game
            next_win = prob_from(g1 + 1, g2, next_server)
            # P2 wins this game
            next_lose = prob_from(g1, g2 + 1, next_server)
            
            result = p_p1_wins_game * next_win + (1 - p_p1_wins_game) * next_lose
            cache[(g1, g2, server)] = result
            return result
        
        return prob_from(p1_games, p2_games, current_server)
    
    def prob_match_from_score(self) -> float:
        """Calculate probability P1 wins match from current score."""
        
        # Get game probabilities
        p_game_hold_p1 = self.markov.prob_game_win(self.p_point_p1_serve)
        p_game_hold_p2 = self.markov.prob_game_win(self.p_point_p2_serve)
        
        # Set probability
        p_set_p1 = self.markov.prob_set_win(p_game_hold_p1, p_game_hold_p2)
        
        sets_to_win = (self.best_of + 1) // 2
        
        cache = {}
        
        def prob_from(s1, s2):
            if (s1, s2) in cache:
                return cache[(s1, s2)]
            
            if s1 >= sets_to_win:
                return 1.0
            if s2 >= sets_to_win:
                return 0.0
            
            # Need to account for current set state
            if s1 == self.sets[0] and s2 == self.sets[1]:
                # In current set, use current game score
                current_set_prob = self.prob_set_from_score(
                    self.games[-1][0], self.games[-1][1],
                    p_game_hold_p1, p_game_hold_p2,
                    self.server
                )
            else:
                current_set_prob = p_set_p1
            
            next_win = prob_from(s1 + 1, s2)
            next_lose = prob_from(s1, s2 + 1)
            
            result = current_set_prob * next_win + (1 - current_set_prob) * next_lose
            cache[(s1, s2)] = result
            return result
        
        return prob_from(self.sets[0], self.sets[1])
    
    def get_score_string(self) -> str:
        """Get human-readable score string."""
        # Sets
        sets_str = f"{self.sets[0]}-{self.sets[1]}"
        
        # Current set games
        games_str = f"{self.games[-1][0]}-{self.games[-1][1]}"
        
        # Points
        if self.is_tiebreak:
            points_str = f"{self.points[0]}-{self.points[1]}"
        else:
            # Handle deuce/advantage
            if self.points[0] >= 3 and self.points[1] >= 3:
                if self.points[0] == self.points[1]:
                    points_str = "Deuce"
                elif self.points[0] > self.points[1]:
                    points_str = f"Ad-{self.player1_name[:3]}"
                else:
                    points_str = f"Ad-{self.player2_name[:3]}"
            else:
                points_str = f"{self.POINT_NAMES.get(self.points[0], self.points[0])}-{self.POINT_NAMES.get(self.points[1], self.points[1])}"
        
        server_name = self.player1_name if self.server == 1 else self.player2_name
        
        return f"Sets: {sets_str} | Games: {games_str} | Points: {points_str} | Serving: {server_name[:15]}"
    
    def record_point(self, winner: int):
        """
        Record a point won by player 1 or 2.
        
        Args:
            winner: 1 if player1 won the point, 2 if player2 won
        """
        # Save state before
        state_before = self.get_state()
        
        # Update points
        self.points[winner - 1] += 1
        
        # Check for game won
        game_won = self._check_game_won()
        
        if game_won:
            self._handle_game_won(game_won)
        
        # Calculate probabilities
        probs = self.calculate_probabilities()
        
        # Record history
        self.history.append({
            'state_before': state_before,
            'point_winner': winner,
            'state_after': self.get_state(),
            'probabilities': probs
        })
        
        return probs
    
    def _check_game_won(self) -> Optional[int]:
        """Check if current game is won. Returns winner (1 or 2) or None."""
        p1, p2 = self.points
        
        if self.is_tiebreak:
            # Tiebreak: first to 7 with 2 point margin
            if p1 >= 7 and p1 >= p2 + 2:
                return 1
            if p2 >= 7 and p2 >= p1 + 2:
                return 2
        else:
            # Regular game: first to 4 with 2 point margin
            if p1 >= 4 and p1 >= p2 + 2:
                return 1
            if p2 >= 4 and p2 >= p1 + 2:
                return 2
        
        return None
    
    def _handle_game_won(self, winner: int):
        """Handle a game being won."""
        # Reset points
        self.points = [0, 0]
        
        # Update games
        self.games[-1][winner - 1] += 1
        
        # Check for set won
        g1, g2 = self.games[-1]
        set_won = None
        
        if self.is_tiebreak:
            # Tiebreak was just completed
            set_won = winner
            self.is_tiebreak = False
        elif g1 >= 6 and g1 >= g2 + 2:
            set_won = 1
        elif g2 >= 6 and g2 >= g1 + 2:
            set_won = 2
        elif g1 == 6 and g2 == 6:
            # Start tiebreak
            self.is_tiebreak = True
        
        if set_won:
            self._handle_set_won(set_won)
        
        # Switch server (unless tiebreak just started)
        if not self.is_tiebreak or (g1 == 6 and g2 == 6):
            self.server = 3 - self.server
    
    def _handle_set_won(self, winner: int):
        """Handle a set being won."""
        self.sets[winner - 1] += 1
        
        # Start new set
        if self.sets[0] < (self.best_of + 1) // 2 and self.sets[1] < (self.best_of + 1) // 2:
            self.games.append([0, 0])
    
    def calculate_probabilities(self) -> Dict[str, float]:
        """Calculate all relevant probabilities for current state."""
        
        # Point probability for current server
        p_point = self.get_current_p_point()
        
        # Game probability from current point score
        if self.server == 1:
            # P1 serving
            server_pts, returner_pts = self.points[0], self.points[1]
            p_hold = self.prob_game_from_score(server_pts, returner_pts, 
                                               self.p_point_p1_serve, self.is_tiebreak)
            p_break = 1 - p_hold  # P2 breaks
        else:
            # P2 serving
            server_pts, returner_pts = self.points[1], self.points[0]
            p_hold = self.prob_game_from_score(server_pts, returner_pts,
                                               self.p_point_p2_serve, self.is_tiebreak)
            p_break = 1 - p_hold  # P1 breaks
        
        # Match probability
        p_match = self.prob_match_from_score()
        
        # Break point detection
        is_break_point = False
        if not self.is_tiebreak:
            if self.server == 1 and self.points[1] >= 3 and self.points[1] > self.points[0]:
                is_break_point = True  # P2 has break point on P1 serve
            elif self.server == 2 and self.points[0] >= 3 and self.points[0] > self.points[1]:
                is_break_point = True  # P1 has break point on P2 serve
        
        return {
            'p_server_wins_game': p_hold,
            'p_break': p_break,
            'p_player1_wins_match': p_match,
            'p_player2_wins_match': 1 - p_match,
            'is_break_point': is_break_point,
            'server': self.server,
            'p_point_server': p_point,
        }
    
    def get_state(self) -> Dict:
        """Get current match state."""
        return {
            'sets': self.sets.copy(),
            'games': [g.copy() for g in self.games],
            'points': self.points.copy(),
            'server': self.server,
            'is_tiebreak': self.is_tiebreak
        }
    
    def display_status(self):
        """Display current match status with probabilities."""
        probs = self.calculate_probabilities()
        
        print(f"\n{'='*60}")
        print(f"📍 {self.get_score_string()}")
        print(f"{'='*60}")
        
        # Serving indicator
        server_name = self.player1_name if self.server == 1 else self.player2_name
        print(f"\n🎾 {server_name} serving")
        print(f"   Point win %: {probs['p_point_server']:.1%}")
        
        # Game probabilities
        print(f"\n📊 Current Game:")
        print(f"   P(Server holds): {probs['p_server_wins_game']:.1%}")
        print(f"   P(Break): {probs['p_break']:.1%}")
        
        if probs['is_break_point']:
            breaker = self.player2_name if self.server == 1 else self.player1_name
            print(f"   🔥 BREAK POINT for {breaker}!")
        
        # Match probabilities
        print(f"\n🏆 Match Win Probability:")
        print(f"   {self.player1_name}: {probs['p_player1_wins_match']:.1%}")
        print(f"   {self.player2_name}: {probs['p_player2_wins_match']:.1%}")
        
        # Betting insight
        print(f"\n💡 Insight:")
        if probs['p_break'] > 0.35:
            print(f"   High break opportunity ({probs['p_break']:.1%})")
        if probs['is_break_point']:
            print(f"   Consider live bet on game outcome")
        
        return probs
    
    def is_match_over(self) -> bool:
        """Check if match is complete."""
        sets_to_win = (self.best_of + 1) // 2
        return self.sets[0] >= sets_to_win or self.sets[1] >= sets_to_win
    
    def close(self):
        """Clean up resources."""
        self.conn.close()
        self.markov.close()


def interactive_tracker():
    """Run interactive point-by-point tracking session."""
    
    print("\n" + "="*60)
    print("🎾 LIVE POINT-BY-POINT TENNIS TRACKER")
    print("="*60)
    
    tracker = PointByPointTracker()
    
    # Setup match
    print("\nEnter match details:")
    player1 = input("Player 1 name (serving first): ").strip() or "Player 1"
    player2 = input("Player 2 name: ").strip() or "Player 2"
    surface = input("Surface (Hard/Clay/Grass) [Hard]: ").strip() or "Hard"
    best_of = int(input("Best of (3/5) [3]: ").strip() or "3")
    
    tracker.setup_match(player1, player2, surface, best_of)
    
    print("\n" + "="*60)
    print("MATCH STARTED!")
    print("Commands: 1 = Player 1 wins point")
    print("          2 = Player 2 wins point")
    print("          s = Show status")
    print("          u = Undo last point")
    print("          q = Quit")
    print("="*60)
    
    tracker.display_status()
    
    while not tracker.is_match_over():
        cmd = input("\nPoint winner (1/2/s/u/q): ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 's':
            tracker.display_status()
        elif cmd == 'u':
            if tracker.history:
                # Restore previous state
                prev = tracker.history.pop()
                state = prev['state_before']
                tracker.sets = state['sets']
                tracker.games = state['games']
                tracker.points = state['points']
                tracker.server = state['server']
                tracker.is_tiebreak = state['is_tiebreak']
                print("↩️  Point undone")
                tracker.display_status()
            else:
                print("No points to undo")
        elif cmd in ['1', '2']:
            winner = int(cmd)
            tracker.record_point(winner)
            winner_name = tracker.player1_name if winner == 1 else tracker.player2_name
            print(f"\n✓ Point to {winner_name}")
            tracker.display_status()
        else:
            print("Invalid command. Use 1, 2, s, u, or q")
    
    if tracker.is_match_over():
        winner_name = tracker.player1_name if tracker.sets[0] > tracker.sets[1] else tracker.player2_name
        print(f"\n🏆 MATCH OVER! {winner_name} wins!")
        print(f"Final Score: {tracker.sets[0]}-{tracker.sets[1]} sets")
    
    tracker.close()


def quick_analysis(player1: str, player2: str, surface: str = "Hard", best_of: int = 3):
    """Quick analysis of a match without point-by-point tracking."""
    
    tracker = PointByPointTracker()
    tracker.setup_match(player1, player2, surface, best_of)
    
    print("\n" + "="*60)
    print("QUICK MATCH ANALYSIS")
    print("="*60)
    
    # Pre-match probabilities
    p_game_p1 = tracker.markov.prob_game_win(tracker.p_point_p1_serve)
    p_game_p2 = tracker.markov.prob_game_win(tracker.p_point_p2_serve)
    
    print(f"\n📊 Pre-Match Analysis:")
    print(f"\n   {player1}:")
    print(f"     Point win on serve: {tracker.p_point_p1_serve:.1%}")
    print(f"     Game hold %: {p_game_p1:.1%}")
    print(f"     Expected breaks received per set: {(1-p_game_p2)*6:.2f}")
    
    print(f"\n   {player2}:")
    print(f"     Point win on serve: {tracker.p_point_p2_serve:.1%}")
    print(f"     Game hold %: {p_game_p2:.1%}")
    print(f"     Expected breaks received per set: {(1-p_game_p1)*6:.2f}")
    
    # Match probability
    p_match = tracker.prob_match_from_score()
    
    print(f"\n🏆 Match Win Probability:")
    print(f"   {player1}: {p_match:.1%}")
    print(f"   {player2}: {1-p_match:.1%}")
    
    # Key thresholds for in-play betting
    print(f"\n💡 In-Play Betting Thresholds:")
    print(f"   Break point probability (40-30): ~{1-tracker.prob_game_from_score(2, 3, tracker.p_point_p1_serve):.1%}")
    print(f"   Break point probability (30-40): ~{1-tracker.prob_game_from_score(2, 3, tracker.p_point_p2_serve):.1%}")
    
    tracker.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick analysis mode
        if len(sys.argv) >= 4:
            quick_analysis(sys.argv[2], sys.argv[3], 
                          sys.argv[4] if len(sys.argv) > 4 else "Hard",
                          int(sys.argv[5]) if len(sys.argv) > 5 else 3)
        else:
            print("Usage: python point_tracker.py --quick 'Player1' 'Player2' [Surface] [BestOf]")
    else:
        # Interactive mode
        interactive_tracker()
