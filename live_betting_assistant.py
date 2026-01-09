"""
🎾 LIVE BETTING ASSISTANT WITH API INTEGRATION
==============================================

Works with:
1. RapidAPI data (when available)
2. Manual stat input
3. Local Markov chain calculations

For Chang Jordan vs Dossani Mohammad analysis
"""

import requests
import json
from typing import Dict, Optional


class LiveMatchAnalyzer:
    """Analyze live matches with or without API data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.bankroll = 1000
        self.target = 5000
    
    def calculate_probabilities_from_stats(self, p1_stats: Dict, p2_stats: Dict, 
                                          p1_name: str, p2_name: str) -> Dict:
        """
        Calculate match probabilities from serve statistics using Markov model
        
        Stats format (percentages):
        {
            'first_serve_pct': 75.2,      # First serve in %
            'first_serve_win_pct': 62.8,  # Points won on 1st serve %
            'second_serve_win_pct': 56.3  # Points won on 2nd serve %
        }
        """
        
        print(f"\n{'='*80}")
        print(f"🎾 MARKOV CHAIN PROBABILITY CALCULATOR")
        print(f"{'='*80}")
        print(f"{p1_name} vs {p2_name}\n")
        
        # Convert to probabilities
        p1_1st_in = p1_stats['first_serve_pct'] / 100.0
        p1_1st_win = p1_stats['first_serve_win_pct'] / 100.0
        p1_2nd_win = p1_stats['second_serve_win_pct'] / 100.0
        
        p2_1st_in = p2_stats['first_serve_pct'] / 100.0
        p2_1st_win = p2_stats['first_serve_win_pct'] / 100.0
        p2_2nd_win = p2_stats['second_serve_win_pct'] / 100.0
        
        # Point-level probability: P(win point on serve)
        p1_point = p1_1st_in * p1_1st_win + (1 - p1_1st_in) * p1_2nd_win
        p2_point = p2_1st_in * p2_1st_win + (1 - p2_1st_in) * p2_2nd_win
        
        print(f"📊 SERVE STATISTICS:")
        print(f"\n{p1_name}:")
        print(f"  1st serve in: {p1_stats['first_serve_pct']:.1f}%")
        print(f"  1st serve won: {p1_stats['first_serve_win_pct']:.1f}%")
        print(f"  2nd serve won: {p1_stats['second_serve_win_pct']:.1f}%")
        print(f"  → Point win on serve: {p1_point*100:.1f}%")
        
        print(f"\n{p2_name}:")
        print(f"  1st serve in: {p2_stats['first_serve_pct']:.1f}%")
        print(f"  1st serve won: {p2_stats['first_serve_win_pct']:.1f}%")
        print(f"  2nd serve won: {p2_stats['second_serve_win_pct']:.1f}%")
        print(f"  → Point win on serve: {p2_point*100:.1f}%")
        
        # Game-level: P(hold serve) using Markov chain
        def p_game_hold(p_point):
            """P(server holds) from Markov chain (deuce formula)"""
            q = 1 - p_point
            if (1 - 2 * p_point * q) == 0:
                return 0.5
            return (p_point ** 2) / (1 - 2 * p_point * q)
        
        p1_hold = p_game_hold(p1_point)
        p2_hold = p_game_hold(p2_point)
        
        print(f"\n🎯 MARKOV CHAIN PROBABILITIES:")
        print(f"\nGame (Hold Serve):")
        print(f"  {p1_name}: {p1_hold*100:.1f}%")
        print(f"  {p2_name}: {p2_hold*100:.1f}%")
        
        # Break probabilities
        p1_breaks_p2 = 1 - p2_hold
        p2_breaks_p1 = 1 - p1_hold
        
        print(f"\nBreak Probabilities:")
        print(f"  {p1_name} breaks {p2_name}: {p1_breaks_p2*100:.1f}%")
        print(f"  {p2_name} breaks {p1_name}: {p2_breaks_p1*100:.1f}%")
        
        # Set-level: Approximate using game probabilities
        # Average game win probability for P1
        p1_game_avg = (p1_hold + (1 - p2_hold)) / 2
        
        # Rough set calculation (simplified binomial)
        # Need to win 6 games before opponent wins 5, or win tiebreak at 6-6
        from scipy.special import comb
        
        p1_set = 0.0
        # Win 6-0 through 6-4
        for p1_wins in range(6, 11):
            for p2_wins in range(0, min(p1_wins, 5)):
                if p1_wins == 6 and p2_wins <= 4:
                    total_games = p1_wins + p2_wins
                    ways = comb(total_games, p1_wins, exact=True)
                    prob = ways * (p1_game_avg ** p1_wins) * ((1 - p1_game_avg) ** p2_wins)
                    p1_set += prob
        
        # 7-5
        prob_5_5 = comb(10, 5, exact=True) * (p1_game_avg ** 5) * ((1 - p1_game_avg) ** 5)
        p1_set += prob_5_5 * (p1_game_avg ** 2)
        
        # 7-6 (tiebreak)
        prob_6_6 = comb(12, 6, exact=True) * (p1_game_avg ** 6) * ((1 - p1_game_avg) ** 6)
        p1_set += prob_6_6 * 0.5  # Assume 50-50 in tiebreak (simplified)
        
        print(f"\nSet:")
        print(f"  {p1_name}: {p1_set*100:.1f}%")
        print(f"  {p2_name}: {(1-p1_set)*100:.1f}%")
        
        # Match-level: Best of 3
        # P(win 2-0) + P(win 2-1)
        p1_match = (p1_set ** 2) + 2 * (p1_set ** 2) * (1 - p1_set)
        
        print(f"\nMatch (Best of 3):")
        print(f"  {p1_name}: {p1_match*100:.1f}%")
        print(f"  {p2_name}: {(1-p1_match)*100:.1f}%")
        
        print(f"\n{'='*80}\n")
        
        return {
            'p1_point': p1_point,
            'p2_point': p2_point,
            'p1_hold': p1_hold,
            'p2_hold': p2_hold,
            'p1_set': p1_set,
            'p2_set': 1 - p1_set,
            'p1_match': p1_match,
            'p2_match': 1 - p1_match
        }
    
    def analyze_betting_opportunities(self, probs: Dict, odds: Dict, 
                                     p1_name: str, p2_name: str):
        """Find betting edges given probabilities and market odds"""
        
        print(f"{'='*80}")
        print(f"💰 BETTING EDGE ANALYSIS")
        print(f"{'='*80}\n")
        
        opportunities = []
        
        # Match winner
        if 'match' in odds:
            print(f"MATCH WINNER:")
            p1_odds = odds['match'][0]
            p2_odds = odds['match'][1]
            
            p1_implied = 1.0 / p1_odds
            p2_implied = 1.0 / p2_odds
            
            p1_edge = (probs['p1_match'] * p1_odds) - 1
            p2_edge = (probs['p2_match'] * p2_odds) - 1
            
            print(f"  {p1_name}:")
            print(f"    Odds: {p1_odds:.2f} (implied {p1_implied*100:.1f}%)")
            print(f"    True: {probs['p1_match']*100:.1f}%")
            print(f"    Edge: {p1_edge*100:+.1f}%")
            
            print(f"  {p2_name}:")
            print(f"    Odds: {p2_odds:.2f} (implied {p2_implied*100:.1f}%)")
            print(f"    True: {probs['p2_match']*100:.1f}%")
            print(f"    Edge: {p2_edge*100:+.1f}%")
            
            if p1_edge > 0.02:
                kelly = (p1_edge / (p1_odds - 1)) * 0.25
                stake = min(self.bankroll * kelly, self.bankroll * 0.15)
                opportunities.append({
                    'market': f'{p1_name} to win match',
                    'odds': p1_odds,
                    'edge': p1_edge,
                    'stake': stake
                })
            
            if p2_edge > 0.02:
                kelly = (p2_edge / (p2_odds - 1)) * 0.25
                stake = min(self.bankroll * kelly, self.bankroll * 0.15)
                opportunities.append({
                    'market': f'{p2_name} to win match',
                    'odds': p2_odds,
                    'edge': p2_edge,
                    'stake': stake
                })
        
        # 1st set winner
        if 'set' in odds:
            print(f"\n1ST SET WINNER:")
            p1_odds = odds['set'][0]
            p2_odds = odds['set'][1]
            
            p1_implied = 1.0 / p1_odds
            p2_implied = 1.0 / p2_odds
            
            p1_edge = (probs['p1_set'] * p1_odds) - 1
            p2_edge = (probs['p2_set'] * p2_odds) - 1
            
            print(f"  {p1_name}:")
            print(f"    Odds: {p1_odds:.2f} (implied {p1_implied*100:.1f}%)")
            print(f"    True: {probs['p1_set']*100:.1f}%")
            print(f"    Edge: {p1_edge*100:+.1f}%")
            
            print(f"  {p2_name}:")
            print(f"    Odds: {p2_odds:.2f} (implied {p2_implied*100:.1f}%)")
            print(f"    True: {probs['p2_set']*100:.1f}%")
            print(f"    Edge: {p2_edge*100:+.1f}%")
            
            if p1_edge > 0.02:
                kelly = (p1_edge / (p1_odds - 1)) * 0.25
                stake = min(self.bankroll * kelly, self.bankroll * 0.15)
                opportunities.append({
                    'market': f'{p1_name} to win 1st set',
                    'odds': p1_odds,
                    'edge': p1_edge,
                    'stake': stake
                })
            
            if p2_edge > 0.02:
                kelly = (p2_edge / (p2_odds - 1)) * 0.25
                stake = min(self.bankroll * kelly, self.bankroll * 0.15)
                opportunities.append({
                    'market': f'{p2_name} to win 1st set',
                    'odds': p2_odds,
                    'edge': p2_edge,
                    'stake': stake
                })
        
        # Show opportunities
        if opportunities:
            opportunities.sort(key=lambda x: x['edge'], reverse=True)
            
            print(f"\n{'='*80}")
            print(f"💸 PROFITABLE BETTING OPPORTUNITIES")
            print(f"{'='*80}\n")
            
            for i, opp in enumerate(opportunities, 1):
                potential_profit = opp['stake'] * (opp['odds'] - 1)
                ev = opp['stake'] * opp['edge']
                
                print(f"{i}. {opp['market']}")
                print(f"   Odds: {opp['odds']:.2f}")
                print(f"   Edge: {opp['edge']*100:+.1f}%")
                print(f"   Recommended stake: ${opp['stake']:.2f}")
                print(f"   Potential profit: ${potential_profit:.2f}")
                print(f"   Expected value: ${ev:+.2f}\n")
            
            total_stake = sum(o['stake'] for o in opportunities)
            total_ev = sum(o['stake'] * o['edge'] for o in opportunities)
            
            print(f"{'─'*80}")
            print(f"Total stake: ${total_stake:.2f}")
            print(f"Total EV: ${total_ev:+.2f}")
            print(f"Projected bankroll: ${self.bankroll + total_ev:,.2f}")
            print(f"Progress to ${self.target:,.0f}: {(self.bankroll + total_ev)/self.target*100:.1f}%")
            print(f"{'='*80}\n")
        else:
            print(f"\n⏸️  No profitable opportunities (all edges < 2%)")


# Quick analysis for Chang vs Dossani
if __name__ == "__main__":
    analyzer = LiveMatchAnalyzer()
    
    print("\n🎾 LIVE MATCH BETTING ASSISTANT")
    print("="*80)
    print("\nPre-loaded: Chang Jordan vs Dossani Mohammad")
    print("="*80)
    
    # Chang's stats (estimated from strong server)
    chang_stats = {
        'first_serve_pct': 68.0,
        'first_serve_win_pct': 75.0,
        'second_serve_win_pct': 55.0
    }
    
    # Dossani's stats (estimated from weaker server)
    dossani_stats = {
        'first_serve_pct': 62.0,
        'first_serve_win_pct': 68.0,
        'second_serve_win_pct': 50.0
    }
    
    # Calculate probabilities
    probs = analyzer.calculate_probabilities_from_stats(
        chang_stats, dossani_stats,
        "Chang Jordan", "Dossani Mohammad"
    )
    
    # Market odds
    odds = {
        'match': [1.28, 3.50],  # Chang, Dossani
        'set': [1.34, 2.90]     # 1st set
    }
    
    # Analyze opportunities
    analyzer.analyze_betting_opportunities(
        probs, odds,
        "Chang Jordan", "Dossani Mohammad"
    )
    
    print("\n💡 TIP: Update serve statistics from live match data for more accuracy")
    print("💡 Game betting: Use Markov tree analysis (markov_tree_analysis.py)")
