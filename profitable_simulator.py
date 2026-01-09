"""
🎾 PROFITABLE LIVE TENNIS BETTING SIMULATOR
===========================================

Simulates live betting with:
- Real player data from database (Djokovic, Nadal, Alcaraz, etc.)
- Markov chain probabilities 
- Kelly criterion bet sizing
- Target: $1000 → $5000

Run: python profitable_simulator.py
"""

import numpy as np
import sqlite3
from datetime import datetime
from hierarchical_model import HierarchicalTennisModel
from scipy.special import comb
import sys


class ProfitableSimulator:
    """Simulate live tennis betting with profitable opportunities"""
    
    def __init__(self, bankroll=1000, target=5000):
        self.conn = sqlite3.connect('tennis_data.db')
        self.markov = HierarchicalTennisModel('tennis_data.db')
        
        # Bankroll
        self.starting_bankroll = bankroll
        self.bankroll = bankroll
        self.target = target
        self.kelly_fraction = 0.25
        self.min_edge = 0.025  # 2.5% minimum edge
        
        # Match state
        self.p1_name = ""
        self.p2_name = ""
        self.p1_id = None
        self.p2_id = None
        
        # True probabilities
        self.p_p1_serve = 0.65
        self.p_p2_serve = 0.62
        
        # Score
        self.sets = [0, 0]
        self.games = [0, 0]
        self.points = [0, 0]
        self.server = 0  # 0 = p1, 1 = p2
        
        # Bets
        self.bets = []
        self.active_bets = []
        
    def get_top_players(self):
        """Get list of well-known players from database"""
        query = """
        SELECT DISTINCT player_name 
        FROM players 
        WHERE player_name IN (
            'Novak Djokovic', 'Rafael Nadal', 'Roger Federer',
            'Carlos Alcaraz', 'Jannik Sinner', 'Daniil Medvedev',
            'Alexander Zverev', 'Stefanos Tsitsipas', 'Andrey Rublev',
            'Taylor Fritz', 'Casper Ruud', 'Holger Rune',
            'Grigor Dimitrov', 'Hubert Hurkacz', 'Tommy Paul'
        )
        ORDER BY player_name
        """
        cursor = self.conn.execute(query)
        return [row[0] for row in cursor.fetchall()]
    
    def setup_match(self, p1_name, p2_name, surface='Hard', best_of=3):
        """Setup match with real player data"""
        self.p1_name = p1_name
        self.p2_name = p2_name
        self.surface = surface
        self.best_of = best_of
        
        # Find players
        cursor = self.conn.execute(
            "SELECT player_id FROM players WHERE player_name = ?", (p1_name,)
        )
        result = cursor.fetchone()
        self.p1_id = result[0] if result else None
        
        cursor = self.conn.execute(
            "SELECT player_id FROM players WHERE player_name = ?", (p2_name,)
        )
        result = cursor.fetchone()
        self.p2_id = result[0] if result else None
        
        if self.p1_id and self.p2_id:
            # Get real probabilities
            match_pred = self.markov.predict_match(
                self.p1_id, self.p2_id, surface, num_sets=best_of
            )
            self.p_p1_serve = match_pred['p_point_1_serve']
            self.p_p2_serve = match_pred['p_point_2_serve']
            
            print(f"\n📊 PLAYER STATS FROM DATABASE:")
            print(f"  {p1_name}: {self.p_p1_serve:.1%} serve win rate")
            print(f"  {p2_name}: {self.p_p2_serve:.1%} serve win rate")
        else:
            print(f"\n⚠️  Using default probabilities")
    
    def p_game(self, p_point, sa, sb):
        """P(server wins game) from score (sa, sb)"""
        if sa >= 4 and sa >= sb + 2:
            return 1.0
        if sb >= 4 and sb >= sa + 2:
            return 0.0
        
        if sa >= 3 and sb >= 3:
            if sa == sb:  # Deuce
                q = 1 - p_point
                return (p_point ** 2) / (1 - 2 * p_point * q)
            elif sa > sb:  # Ad server
                q = 1 - p_point
                p_deuce = (p_point ** 2) / (1 - 2 * p_point * q)
                return p_point + (1 - p_point) * p_deuce
            else:  # Ad returner
                q = 1 - p_point
                p_deuce = (p_point ** 2) / (1 - 2 * p_point * q)
                return p_point * p_deuce
        
        # Recursive
        return (p_point * self.p_game(p_point, sa + 1, sb) + 
                (1 - p_point) * self.p_game(p_point, sa, sb + 1))
    
    def p_set_simple(self, p_game, ga, gb):
        """Simplified set probability"""
        if ga >= 6 and ga >= gb + 2:
            return 1.0
        if gb >= 6 and gb >= ga + 2:
            return 0.0
        if ga == 6 and gb == 6:
            return 0.5  # Tiebreak
        
        # Approximate
        games_ahead = ga - gb
        if games_ahead >= 2:
            return 0.75
        elif games_ahead <= -2:
            return 0.25
        else:
            return 0.5
    
    def p_match(self, p_set, sa, sb):
        """P(player wins match) from set score"""
        sets_needed = (self.best_of + 1) // 2
        if sa >= sets_needed:
            return 1.0
        if sb >= sets_needed:
            return 0.0
        
        # Remaining sets
        need_a = sets_needed - sa
        need_b = sets_needed - sb
        
        # Binomial calculation
        prob = 0.0
        for wins in range(need_a, need_a + need_b):
            if wins + need_b - 1 >= wins:
                ways = comb(wins + need_b - 1, wins, exact=True)
                prob += ways * (p_set ** wins) * ((1 - p_set) ** (need_b - 1)) * p_set
        return prob
    
    def get_probabilities(self):
        """Calculate all current probabilities"""
        # Game probability
        if self.server == 0:
            p_server = self.p_p1_serve
            p_game_hold = self.p_game(p_server, self.points[0], self.points[1])
        else:
            p_server = self.p_p2_serve
            p_game_hold = self.p_game(p_server, self.points[1], self.points[0])
        
        p_p1_game = p_game_hold if self.server == 0 else (1 - p_game_hold)
        
        # Set probability
        # Average game win prob for p1
        p_p1_game_avg = (self.p_game(self.p_p1_serve, 0, 0) + 
                         (1 - self.p_game(self.p_p2_serve, 0, 0))) / 2
        p_p1_set = self.p_set_simple(p_p1_game_avg, self.games[0], self.games[1])
        
        # Match probability
        p_p1_match = self.p_match(p_p1_set, self.sets[0], self.sets[1])
        
        return {
            'p1_game': p_p1_game,
            'p2_game': 1 - p_p1_game,
            'p1_set': p_p1_set,
            'p2_set': 1 - p_p1_set,
            'p1_match': p_p1_match,
            'p2_match': 1 - p_p1_match
        }
    
    def generate_odds(self, true_prob, margin=0.05):
        """Generate bookmaker odds with margin and noise"""
        noise = np.random.normal(0, 0.03)
        implied = true_prob * (1 + margin) + noise
        implied = np.clip(implied, 0.01, 0.99)
        return 1.0 / implied
    
    def edge(self, true_prob, odds):
        """Calculate edge"""
        return (true_prob * odds) - 1.0
    
    def kelly_size(self, edge, odds):
        """Kelly bet size as fraction of bankroll"""
        if edge <= 0 or odds <= 1.0:
            return 0.0
        kelly = edge / (odds - 1.0)
        return min(kelly * self.kelly_fraction, 0.10)  # Max 10% per bet
    
    def find_opportunities(self):
        """Find profitable betting opportunities"""
        probs = self.get_probabilities()
        opps = []
        
        # Match winner
        for player, p_win in [(0, probs['p1_match']), (1, probs['p2_match'])]:
            odds = self.generate_odds(p_win)
            e = self.edge(p_win, odds)
            
            if e >= self.min_edge:
                size = self.kelly_size(e, odds)
                stake = self.bankroll * size
                
                if stake >= 10:
                    opps.append({
                        'market': 'match_winner',
                        'player': player,
                        'odds': odds,
                        'edge': e,
                        'stake': stake,
                        'true_prob': p_win
                    })
        
        # Current game winner (in-play)
        for player, p_win in [(0, probs['p1_game']), (1, probs['p2_game'])]:
            odds = self.generate_odds(p_win, margin=0.08)  # Higher margin in-play
            e = self.edge(p_win, odds)
            
            if e >= self.min_edge * 1.5:  # Higher threshold for in-play
                size = self.kelly_size(e, odds)
                stake = self.bankroll * size
                
                if stake >= 10:
                    opps.append({
                        'market': 'game_winner',
                        'player': player,
                        'odds': odds,
                        'edge': e,
                        'stake': stake,
                        'true_prob': p_win
                    })
        
        return opps
    
    def place_bet(self, opp):
        """Place a bet"""
        bet = {
            'market': opp['market'],
            'player': opp['player'],
            'odds': opp['odds'],
            'stake': opp['stake'],
            'edge': opp['edge'],
            'score': self.score_str(),
            'settled': False
        }
        
        self.bets.append(bet)
        self.active_bets.append(bet)
        self.bankroll -= bet['stake']
        
        player_name = self.p1_name if opp['player'] == 0 else self.p2_name
        market_desc = "to win match" if opp['market'] == 'match_winner' else "to win game"
        
        print(f"\n💸 BET: {player_name} {market_desc}")
        print(f"   Odds: {bet['odds']:.2f} | Stake: ${bet['stake']:.0f} | Edge: {bet['edge']*100:.1f}%")
        print(f"   Bankroll: ${self.bankroll:,.0f}")
    
    def simulate_point(self):
        """Simulate one point"""
        p_server = self.p_p1_serve if self.server == 0 else self.p_p2_serve
        server_wins = np.random.random() < p_server
        
        if server_wins:
            self.points[self.server] += 1
        else:
            self.points[1 - self.server] += 1
        
        # Check game end
        if self.points[0] >= 4 and self.points[0] >= self.points[1] + 2:
            self.games[0] += 1
            self.points = [0, 0]
            self.server = 1 - self.server
            self.check_game_bets(winner=0)
            self.check_set()
        elif self.points[1] >= 4 and self.points[1] >= self.points[0] + 2:
            self.games[1] += 1
            self.points = [0, 0]
            self.server = 1 - self.server
            self.check_game_bets(winner=1)
            self.check_set()
    
    def check_game_bets(self, winner):
        """Settle game bets"""
        for bet in self.active_bets[:]:
            if bet['market'] == 'game_winner' and not bet['settled']:
                if bet['player'] == winner:
                    winnings = bet['stake'] * bet['odds']
                    self.bankroll += winnings
                    print(f"   ✅ Bet WON: +${winnings - bet['stake']:.0f}")
                else:
                    print(f"   ❌ Bet LOST: -${bet['stake']:.0f}")
                
                bet['settled'] = True
                self.active_bets.remove(bet)
    
    def check_set(self):
        """Check if set is won"""
        if self.games[0] >= 6 and self.games[0] >= self.games[1] + 2:
            self.sets[0] += 1
            self.games = [0, 0]
            print(f"\n🎾 SET WON BY {self.p1_name}!")
        elif self.games[1] >= 6 and self.games[1] >= self.games[0] + 2:
            self.sets[1] += 1
            self.games = [0, 0]
            print(f"\n🎾 SET WON BY {self.p2_name}!")
    
    def is_match_over(self):
        """Check if match is over"""
        sets_needed = (self.best_of + 1) // 2
        if self.sets[0] >= sets_needed:
            self.settle_match_bets(winner=0)
            return True
        if self.sets[1] >= sets_needed:
            self.settle_match_bets(winner=1)
            return True
        return False
    
    def settle_match_bets(self, winner):
        """Settle match bets"""
        for bet in self.active_bets[:]:
            if bet['market'] == 'match_winner' and not bet['settled']:
                if bet['player'] == winner:
                    winnings = bet['stake'] * bet['odds']
                    self.bankroll += winnings
                    print(f"   ✅ Match bet WON: +${winnings - bet['stake']:.0f}")
                else:
                    print(f"   ❌ Match bet LOST: -${bet['stake']:.0f}")
                
                bet['settled'] = True
                self.active_bets.remove(bet)
    
    def score_str(self):
        """Score as string"""
        pts = {0: '0', 1: '15', 2: '30', 3: '40'}
        if self.points[0] >= 3 and self.points[1] >= 3:
            if self.points[0] == self.points[1]:
                pt_str = "40-40"
            elif self.points[0] > self.points[1]:
                pt_str = "AD-40"
            else:
                pt_str = "40-AD"
        else:
            pt_str = f"{pts.get(self.points[0], '40')}-{pts.get(self.points[1], '40')}"
        
        return f"{self.sets[0]}-{self.sets[1]} | {self.games[0]}-{self.games[1]} | {pt_str}"
    
    def print_status(self):
        """Print status"""
        probs = self.get_probabilities()
        
        print(f"\n{self.score_str()}")
        print(f"  {self.p1_name}: {probs['p1_match']*100:.0f}% match | {probs['p1_game']*100:.0f}% game")
        print(f"  {self.p2_name}: {probs['p2_match']*100:.0f}% match | {probs['p2_game']*100:.0f}% game")
        print(f"  💰 ${self.bankroll:,.0f} | P&L: ${self.bankroll - self.starting_bankroll:+,.0f}")
        print(f"  🎯 {self.bankroll/self.target*100:.0f}% to target")
    
    def run(self):
        """Run simulation"""
        point_count = 0
        
        while not self.is_match_over():
            point_count += 1
            
            # Check for bets every few points
            if point_count % 4 == 0:
                opps = self.find_opportunities()
                if opps:
                    best = max(opps, key=lambda x: x['edge'])
                    self.place_bet(best)
            
            # Simulate point
            self.simulate_point()
            
            # Print status every game
            if self.points[0] == 0 and self.points[1] == 0:
                self.print_status()
            
            # Check target
            if self.bankroll >= self.target:
                print(f"\n🎉 TARGET REACHED: ${self.bankroll:,.0f}!")
                break
            
            if self.bankroll < 10:
                print(f"\n💔 BANKRUPT: ${self.bankroll:,.0f}")
                break
        
        self.print_final()
    
    def print_final(self):
        """Print final report"""
        winner = self.p1_name if self.sets[0] > self.sets[1] else self.p2_name
        
        print(f"\n{'='*70}")
        print(f"🏆 FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Winner: {winner}")
        print(f"Score: {self.sets[0]}-{self.sets[1]}")
        print(f"\nBankroll:")
        print(f"  Start: ${self.starting_bankroll:,.0f}")
        print(f"  Final: ${self.bankroll:,.0f}")
        print(f"  P&L: ${self.bankroll - self.starting_bankroll:+,.0f}")
        print(f"  ROI: {(self.bankroll/self.starting_bankroll - 1)*100:+.1f}%")
        print(f"\nBets: {len(self.bets)} placed")
        print(f"{'='*70}\n")


def main():
    """Main"""
    sim = ProfitableSimulator(bankroll=1000, target=5000)
    
    print("\n🎾 PROFITABLE TENNIS BETTING SIMULATOR")
    print("="*70)
    print("\nAvailable players in database:")
    
    players = sim.get_top_players()
    for i, p in enumerate(players, 1):
        print(f"  {i:2}. {p}")
    
    print("\n" + "="*70)
    p1 = input("\nPlayer 1: ").strip()
    p2 = input("Player 2: ").strip()
    
    sim.setup_match(p1, p2, surface='Hard', best_of=3)
    
    print(f"\n{'='*70}")
    print(f"MATCH: {p1} vs {p2}")
    print(f"Bankroll: ${sim.bankroll:,.0f} → Target: ${sim.target:,.0f}")
    print(f"{'='*70}")
    
    input("\nPress Enter to start...")
    sim.run()


if __name__ == "__main__":
    main()
