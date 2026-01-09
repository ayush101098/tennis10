"""
🎾 LIVE MATCH ANALYSIS
Chang Jordan vs Dossani Mohammad
UTR Men Athens
"""

import numpy as np
from scipy.special import comb

class LiveMatchAnalyzer:
    """Analyze live match for profitable betting opportunities"""
    
    def __init__(self, bankroll=1000, target=5000):
        self.bankroll = bankroll
        self.target = target
        self.kelly_fraction = 0.25
        
    def p_game(self, p_point, sa, sb):
        """P(server wins game) from score"""
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
        
        return (p_point * self.p_game(p_point, sa + 1, sb) + 
                (1 - p_point) * self.p_game(p_point, sa, sb + 1))
    
    def p_set_markov(self, p_hold_a, p_hold_b):
        """
        Calculate P(player A wins set) using Markov chain.
        Assumes alternating serve.
        """
        # Simplified: average hold probability
        p_game_a = (p_hold_a + (1 - p_hold_b)) / 2
        
        # Probability to win set from 0-0
        # Using binomial approximation: need 6 games before opponent
        total = 0.0
        for wins_a in range(6, 13):  # A wins in 6-12 games
            for wins_b in range(0, min(wins_a, 6)):
                if wins_a == 6 and wins_b <= 4:
                    # 6-0, 6-1, 6-2, 6-3, 6-4
                    games = wins_a + wins_b
                    ways = comb(games, wins_a, exact=True)
                    prob = ways * (p_game_a ** wins_a) * ((1 - p_game_a) ** wins_b)
                    total += prob
                elif wins_a == 7 and wins_b == 5:
                    # 7-5
                    # Must be 5-5, then A wins 2 straight
                    prob_to_5_5 = comb(10, 5, exact=True) * (p_game_a ** 5) * ((1 - p_game_a) ** 5)
                    total += prob_to_5_5 * (p_game_a ** 2)
        
        # Tiebreak scenarios (6-6)
        prob_to_6_6 = comb(12, 6, exact=True) * (p_game_a ** 6) * ((1 - p_game_a) ** 6)
        total += prob_to_6_6 * 0.5  # Assume 50% in tiebreak (simplification)
        
        return total
    
    def calculate_true_probabilities(self, p1_serve, p2_serve):
        """Calculate true match probabilities using Markov chain"""
        
        # Game probabilities
        p1_hold = self.p_game(p1_serve, 0, 0)
        p2_hold = self.p_game(p2_serve, 0, 0)
        
        # Set probability
        p1_wins_set = self.p_set_markov(p1_hold, p2_hold)
        
        # Match probability (best of 3)
        # P(win match) = P(win 2-0) + P(win 2-1)
        p_set = p1_wins_set
        p1_wins_match = (p_set ** 2) + 2 * (p_set ** 2) * (1 - p_set)
        
        return {
            'p1_hold': p1_hold,
            'p2_hold': p2_hold,
            'p1_set': p1_wins_set,
            'p2_set': 1 - p1_wins_set,
            'p1_match': p1_wins_match,
            'p2_match': 1 - p1_wins_match
        }
    
    def odds_to_prob(self, odds):
        """Convert decimal odds to implied probability"""
        return 1.0 / odds
    
    def calculate_edge(self, true_prob, odds):
        """Calculate edge: EV% = (true_prob * odds) - 1"""
        return (true_prob * odds) - 1.0
    
    def kelly_bet_size(self, edge, odds):
        """Kelly criterion bet size"""
        if edge <= 0 or odds <= 1.0:
            return 0.0
        kelly = edge / (odds - 1.0)
        return min(kelly * self.kelly_fraction, 0.15)  # Max 15% per bet
    
    def analyze_match(self, p1_name, p2_name, p1_serve, p2_serve, 
                     match_odds, set_odds):
        """Analyze match for profitable opportunities"""
        
        print(f"\n{'='*70}")
        print(f"🎾 LIVE MATCH ANALYSIS")
        print(f"{'='*70}")
        print(f"{p1_name} vs {p2_name}")
        print(f"{'='*70}\n")
        
        # Calculate true probabilities
        print(f"📊 PLAYER SERVE STRENGTHS:")
        print(f"  {p1_name}: {p1_serve:.1%} point win on serve")
        print(f"  {p2_name}: {p2_serve:.1%} point win on serve\n")
        
        probs = self.calculate_true_probabilities(p1_serve, p2_serve)
        
        print(f"🎯 MARKOV CHAIN PROBABILITIES:")
        print(f"  Game hold rates:")
        print(f"    {p1_name}: {probs['p1_hold']*100:.1f}%")
        print(f"    {p2_name}: {probs['p2_hold']*100:.1f}%")
        print(f"  Set win probabilities:")
        print(f"    {p1_name}: {probs['p1_set']*100:.1f}%")
        print(f"    {p2_name}: {probs['p2_set']*100:.1f}%")
        print(f"  Match win probabilities:")
        print(f"    {p1_name}: {probs['p1_match']*100:.1f}%")
        print(f"    {p2_name}: {probs['p2_match']*100:.1f}%\n")
        
        # Market odds analysis
        print(f"{'─'*70}")
        print(f"💰 MARKET ODDS & EDGE ANALYSIS")
        print(f"{'─'*70}\n")
        
        # Match winner market
        print(f"MATCH WINNER:")
        match_implied = {
            p1_name: self.odds_to_prob(match_odds[0]),
            p2_name: self.odds_to_prob(match_odds[1])
        }
        
        edge_p1_match = self.calculate_edge(probs['p1_match'], match_odds[0])
        edge_p2_match = self.calculate_edge(probs['p2_match'], match_odds[1])
        
        print(f"  {p1_name}:")
        print(f"    Odds: {match_odds[0]:.2f} (implied {match_implied[p1_name]*100:.1f}%)")
        print(f"    True prob: {probs['p1_match']*100:.1f}%")
        print(f"    Edge: {edge_p1_match*100:+.1f}%")
        
        print(f"  {p2_name}:")
        print(f"    Odds: {match_odds[1]:.2f} (implied {match_implied[p2_name]*100:.1f}%)")
        print(f"    True prob: {probs['p2_match']*100:.1f}%")
        print(f"    Edge: {edge_p2_match*100:+.1f}%\n")
        
        # 1st set market
        print(f"1ST SET WINNER:")
        set_implied = {
            p1_name: self.odds_to_prob(set_odds[0]),
            p2_name: self.odds_to_prob(set_odds[1])
        }
        
        edge_p1_set = self.calculate_edge(probs['p1_set'], set_odds[0])
        edge_p2_set = self.calculate_edge(probs['p2_set'], set_odds[1])
        
        print(f"  {p1_name}:")
        print(f"    Odds: {set_odds[0]:.2f} (implied {set_implied[p1_name]*100:.1f}%)")
        print(f"    True prob: {probs['p1_set']*100:.1f}%")
        print(f"    Edge: {edge_p1_set*100:+.1f}%")
        
        print(f"  {p2_name}:")
        print(f"    Odds: {set_odds[1]:.2f} (implied {set_implied[p2_name]*100:.1f}%)")
        print(f"    True prob: {probs['p2_set']*100:.1f}%")
        print(f"    Edge: {edge_p2_set*100:+.1f}%\n")
        
        # Profitable opportunities
        print(f"{'='*70}")
        print(f"💸 PROFITABLE BETTING OPPORTUNITIES")
        print(f"{'='*70}\n")
        
        opportunities = []
        
        # Check all markets
        if edge_p1_match > 0.02:
            size = self.kelly_bet_size(edge_p1_match, match_odds[0])
            stake = self.bankroll * size
            opportunities.append({
                'market': f'{p1_name} to win match',
                'odds': match_odds[0],
                'edge': edge_p1_match,
                'stake': stake,
                'ev': stake * edge_p1_match
            })
        
        if edge_p2_match > 0.02:
            size = self.kelly_bet_size(edge_p2_match, match_odds[1])
            stake = self.bankroll * size
            opportunities.append({
                'market': f'{p2_name} to win match',
                'odds': match_odds[1],
                'edge': edge_p2_match,
                'stake': stake,
                'ev': stake * edge_p2_match
            })
        
        if edge_p1_set > 0.02:
            size = self.kelly_bet_size(edge_p1_set, set_odds[0])
            stake = self.bankroll * size
            opportunities.append({
                'market': f'{p1_name} to win 1st set',
                'odds': set_odds[0],
                'edge': edge_p1_set,
                'stake': stake,
                'ev': stake * edge_p1_set
            })
        
        if edge_p2_set > 0.02:
            size = self.kelly_bet_size(edge_p2_set, set_odds[1])
            stake = self.bankroll * size
            opportunities.append({
                'market': f'{p2_name} to win 1st set',
                'odds': set_odds[1],
                'edge': edge_p2_set,
                'stake': stake,
                'ev': stake * edge_p2_set
            })
        
        if opportunities:
            # Sort by EV
            opportunities.sort(key=lambda x: x['ev'], reverse=True)
            
            print(f"Found {len(opportunities)} profitable bet(s):\n")
            
            for i, opp in enumerate(opportunities, 1):
                print(f"  {i}. {opp['market']}")
                print(f"     Odds: {opp['odds']:.2f}")
                print(f"     Edge: {opp['edge']*100:+.1f}%")
                print(f"     Recommended stake: ${opp['stake']:.2f}")
                print(f"     Expected value: ${opp['ev']:+.2f}")
                print()
            
            # Best bet
            best = opportunities[0]
            print(f"{'─'*70}")
            print(f"🏆 BEST BET:")
            print(f"{'─'*70}")
            print(f"  Market: {best['market']}")
            print(f"  Odds: {best['odds']:.2f}")
            print(f"  Edge: {best['edge']*100:+.1f}%")
            print(f"  Stake: ${best['stake']:.2f} ({best['stake']/self.bankroll*100:.1f}% of bankroll)")
            print(f"  Expected profit: ${best['ev']:+.2f}")
            print(f"  Potential win: ${best['stake'] * (best['odds'] - 1):.2f}")
            print(f"{'─'*70}\n")
            
        else:
            print("❌ No profitable opportunities found (all edges < 2%)\n")
            print("Market appears efficient. Wait for better odds or skip this match.\n")
        
        # Bankroll projection
        if opportunities:
            total_stake = sum(o['stake'] for o in opportunities)
            total_ev = sum(o['ev'] for o in opportunities)
            
            print(f"{'='*70}")
            print(f"📈 BANKROLL PROJECTION")
            print(f"{'='*70}")
            print(f"Current bankroll: ${self.bankroll:,.2f}")
            print(f"Total recommended stakes: ${total_stake:.2f}")
            print(f"Total expected value: ${total_ev:+.2f}")
            print(f"Projected bankroll: ${self.bankroll + total_ev:,.2f}")
            print(f"Progress to target: {(self.bankroll + total_ev)/self.target*100:.1f}%")
            print(f"{'='*70}\n")


# Run analysis for Chang vs Dossani
if __name__ == "__main__":
    analyzer = LiveMatchAnalyzer(bankroll=1000, target=5000)
    
    # Estimate serve strengths based on odds
    # Chang is favorite (1.28) -> stronger player
    # Dossani is underdog (3.5)
    
    print("\n🔍 Estimating player strengths...")
    print("(No database stats available - using odds-based estimation)")
    
    # Rough estimation from odds
    # Chang @ 1.28 implies ~78% win rate -> strong server
    # Dossani @ 3.5 implies ~29% win rate -> weaker server
    
    p_chang_serve = 0.66  # Strong server (ATP average ~0.65)
    p_dossani_serve = 0.60  # Below average
    
    analyzer.analyze_match(
        p1_name="Chang Jordan",
        p2_name="Dossani Mohammad",
        p1_serve=p_chang_serve,
        p2_serve=p_dossani_serve,
        match_odds=[1.28, 3.5],
        set_odds=[1.34, 2.90]
    )
    
    print("\n💡 TIP: These probabilities are estimated based on odds.")
    print("For better accuracy, use actual serve statistics from database or live stats.")
