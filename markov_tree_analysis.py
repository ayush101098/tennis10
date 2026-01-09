"""
🎾 MARKOV CHAIN GAME TREE ANALYSIS
Chang Jordan vs Dossani Mohammad

Shows full probability tree for game winner at every score
"""

import numpy as np
from typing import Dict, Tuple

class MarkovGameTree:
    """Calculate and display full Markov chain game tree"""
    
    def __init__(self):
        self.tree_cache = {}
    
    def p_game_recursive(self, p_point: float, sa: int, sb: int, depth: int = 0, 
                        show_tree: bool = False, prefix: str = "") -> float:
        """
        Calculate P(server wins game) from score (sa, sb) with tree visualization
        
        Args:
            p_point: Probability server wins a point
            sa: Server's score (0-4+)
            sb: Returner's score (0-4+)
            depth: Current depth in tree
            show_tree: Whether to print the tree
            prefix: Prefix for tree printing
        """
        # Terminal states
        if sa >= 4 and sa >= sb + 2:
            if show_tree:
                score_str = self.score_to_string(sa, sb)
                print(f"{prefix}└─ {score_str} → SERVER WINS (100.0%)")
            return 1.0
            
        if sb >= 4 and sb >= sa + 2:
            if show_tree:
                score_str = self.score_to_string(sa, sb)
                print(f"{prefix}└─ {score_str} → RETURNER WINS (0.0%)")
            return 0.0
        
        # Deuce situations (analytical formula)
        if sa >= 3 and sb >= 3:
            q = 1 - p_point
            if sa == sb:  # Deuce
                p_win = (p_point ** 2) / (1 - 2 * p_point * q)
                if show_tree:
                    score_str = self.score_to_string(sa, sb)
                    print(f"{prefix}└─ {score_str} (DEUCE) → {p_win*100:.1f}%")
                return p_win
            elif sa == sb + 1:  # Advantage server
                p_deuce = (p_point ** 2) / (1 - 2 * p_point * q)
                p_win = p_point + (1 - p_point) * p_deuce
                if show_tree:
                    score_str = self.score_to_string(sa, sb)
                    print(f"{prefix}└─ {score_str} (AD-IN) → {p_win*100:.1f}%")
                return p_win
            else:  # Advantage returner
                p_deuce = (p_point ** 2) / (1 - 2 * p_point * q)
                p_win = p_point * p_deuce
                if show_tree:
                    score_str = self.score_to_string(sa, sb)
                    print(f"{prefix}└─ {score_str} (AD-OUT) → {p_win*100:.1f}%")
                return p_win
        
        # Recursive case - before deuce
        # Server wins point
        p_win_point = self.p_game_recursive(p_point, sa + 1, sb, depth + 1, False, prefix + "  ")
        # Server loses point
        p_lose_point = self.p_game_recursive(p_point, sa, sb + 1, depth + 1, False, prefix + "  ")
        
        p_total = p_point * p_win_point + (1 - p_point) * p_lose_point
        
        if show_tree and depth < 8:  # Limit depth to avoid clutter
            score_str = self.score_to_string(sa, sb)
            print(f"{prefix}├─ {score_str} → {p_total*100:.1f}%")
            print(f"{prefix}│  ├─ Server wins point ({p_point*100:.0f}%) → {self.score_to_string(sa+1, sb)}")
            print(f"{prefix}│  └─ Server loses point ({(1-p_point)*100:.0f}%) → {self.score_to_string(sa, sb+1)}")
        
        return p_total
    
    def score_to_string(self, sa: int, sb: int) -> str:
        """Convert score to tennis notation"""
        points = {0: '0', 1: '15', 2: '30', 3: '40'}
        
        if sa >= 3 and sb >= 3:
            if sa == sb:
                return "DEUCE"
            elif sa > sb:
                return "ADV-SERVER"
            else:
                return "ADV-RETURNER"
        
        return f"{points.get(sa, '40')}-{points.get(sb, '40')}"
    
    def generate_full_tree(self, p_point: float, server_name: str):
        """Generate complete game tree with all probabilities"""
        
        print(f"\n{'='*80}")
        print(f"MARKOV CHAIN GAME TREE: {server_name} SERVING")
        print(f"{'='*80}")
        print(f"Point win probability: {p_point*100:.1f}%")
        print(f"{'='*80}\n")
        
        # Calculate all score probabilities
        scores = []
        
        print("COMPLETE PROBABILITY TABLE:")
        print(f"{'Score':<15} {'P(Server Wins Game)':<25} {'P(Returner Wins Game)':<25}")
        print("─" * 80)
        
        # Generate all relevant scores
        for sa in range(5):
            for sb in range(5):
                if sa == 4 and sb <= 2:
                    continue  # Game already over
                if sb == 4 and sa <= 2:
                    continue  # Game already over
                
                p_win = self.p_game_recursive(p_point, sa, sb, show_tree=False)
                score_str = self.score_to_string(sa, sb)
                
                scores.append({
                    'score': score_str,
                    'sa': sa,
                    'sb': sb,
                    'p_server': p_win,
                    'p_returner': 1 - p_win
                })
                
                # Color coding for printing
                if p_win >= 0.7:
                    indicator = "✓✓"
                elif p_win >= 0.5:
                    indicator = "✓"
                elif p_win >= 0.3:
                    indicator = "~"
                else:
                    indicator = "✗"
                
                print(f"{score_str:<15} {p_win*100:>6.2f}% {indicator:<17} {(1-p_win)*100:>6.2f}%")
        
        # Deuce analysis
        print("\n" + "─" * 80)
        print("DEUCE SITUATIONS:")
        print("─" * 80)
        
        q = 1 - p_point
        p_from_deuce = (p_point ** 2) / (1 - 2 * p_point * q)
        p_from_ad_in = p_point + (1 - p_point) * p_from_deuce
        p_from_ad_out = p_point * p_from_deuce
        
        print(f"{'DEUCE (40-40)':<15} {p_from_deuce*100:>6.2f}%            {(1-p_from_deuce)*100:>6.2f}%")
        print(f"{'ADV-IN':<15} {p_from_ad_in*100:>6.2f}%            {(1-p_from_ad_in)*100:>6.2f}%")
        print(f"{'ADV-OUT':<15} {p_from_ad_out*100:>6.2f}%            {(1-p_from_ad_out)*100:>6.2f}%")
        
        # Critical scores analysis
        print("\n" + "─" * 80)
        print("KEY SCORING OPPORTUNITIES (BREAK POINTS / GAME POINTS):")
        print("─" * 80)
        
        critical_scores = [
            (0, 3, "0-40 (3 Break Points)"),
            (1, 3, "15-40 (2 Break Points)"),
            (2, 3, "30-40 (1 Break Point)"),
            (3, 0, "40-0 (3 Game Points)"),
            (3, 1, "40-15 (2 Game Points)"),
            (3, 2, "40-30 (1 Game Point)"),
        ]
        
        for sa, sb, desc in critical_scores:
            p_win = self.p_game_recursive(p_point, sa, sb, show_tree=False)
            print(f"{desc:<30} Server: {p_win*100:>6.2f}%  Returner: {(1-p_win)*100:>6.2f}%")
        
        # Overall game probability
        print("\n" + "=" * 80)
        p_game_from_0_0 = self.p_game_recursive(p_point, 0, 0, show_tree=False)
        print(f"OVERALL: P({server_name} holds serve) = {p_game_from_0_0*100:.2f}%")
        print(f"         P(Break) = {(1-p_game_from_0_0)*100:.2f}%")
        print("=" * 80 + "\n")
        
        return p_game_from_0_0


class GameBettingAnalyzer:
    """Analyze game betting opportunities"""
    
    def __init__(self, bankroll=1000):
        self.bankroll = bankroll
        self.kelly_fraction = 0.20  # More aggressive for in-play
    
    def analyze_game_bets(self, p_server_holds: float, server_name: str, returner_name: str):
        """Analyze game winner betting opportunities"""
        
        print(f"\n{'='*80}")
        print(f"💸 GAME WINNER BETTING MARKETS")
        print(f"{'='*80}\n")
        
        print("WHERE TO BET ON GAME WINNER:")
        print("─" * 80)
        print("1. ✅ Bet365 - Live 'Next Game Winner' market")
        print("2. ✅ Pinnacle - In-play 'Game Betting' market")
        print("3. ✅ Betfair Exchange - 'To Win Game' market (live)")
        print("4. ✅ BetOnline - Live tennis 'Game Lines'")
        print("5. ✅ Bovada - In-play 'Next Game' market")
        print("6. ✅ DraftKings - Live 'Game Winner' (select matches)")
        print("\n⚠️  Note: Game markets typically available during the game being played")
        print("         Check 'In-Play' or 'Live Betting' sections\n")
        
        print("─" * 80)
        print("SAMPLE GAME ODDS SCENARIOS:")
        print("─" * 80 + "\n")
        
        # Scenario 1: Start of game (0-0)
        print(f"📍 SCORE: 0-0 (Start of game)")
        print(f"   True probability: {server_name} {p_server_holds*100:.1f}% | {returner_name} {(1-p_server_holds)*100:.1f}%")
        
        # Simulate bookmaker odds
        scenarios = [
            (0.05, "Efficient market"),
            (0.10, "Typical market"),
            (0.15, "Inefficient market")
        ]
        
        for margin, desc in scenarios:
            # Bookmaker inflates favorite, deflates underdog
            if p_server_holds > 0.5:
                implied_server = p_server_holds * (1 + margin)
                implied_returner = (1 - p_server_holds) * (1 - margin)
            else:
                implied_server = p_server_holds * (1 - margin)
                implied_returner = (1 - p_server_holds) * (1 + margin)
            
            implied_server = np.clip(implied_server, 0.05, 0.95)
            implied_returner = np.clip(implied_returner, 0.05, 0.95)
            
            odds_server = 1.0 / implied_server
            odds_returner = 1.0 / implied_returner
            
            edge_server = (p_server_holds * odds_server) - 1
            edge_returner = ((1 - p_server_holds) * odds_returner) - 1
            
            print(f"\n   {desc} ({margin*100:.0f}% margin):")
            print(f"   {server_name}: {odds_server:.2f} (Edge: {edge_server*100:+.1f}%)")
            print(f"   {returner_name}: {odds_returner:.2f} (Edge: {edge_returner*100:+.1f}%)")
            
            if edge_server > 0.03:
                kelly = (edge_server / (odds_server - 1)) * self.kelly_fraction
                stake = min(self.bankroll * kelly, self.bankroll * 0.10)
                print(f"   💰 BET: ${stake:.2f} on {server_name} @ {odds_server:.2f}")
            elif edge_returner > 0.03:
                kelly = (edge_returner / (odds_returner - 1)) * self.kelly_fraction
                stake = min(self.bankroll * kelly, self.bankroll * 0.10)
                print(f"   💰 BET: ${stake:.2f} on {returner_name} @ {odds_returner:.2f}")
            else:
                print(f"   ⏸️  No bet (insufficient edge)")
        
        print("\n" + "─" * 80)
        print("📍 SCORE: 30-30 (Critical point)")
        
        # At 30-30, calculate new probability
        tree = MarkovGameTree()
        p_from_30_30 = tree.p_game_recursive(0.66, 2, 2, show_tree=False)
        
        print(f"   True probability: {server_name} {p_from_30_30*100:.1f}% | {returner_name} {(1-p_from_30_30)*100:.1f}%")
        print(f"   This is a 🔥 KEY MOMENT - odds will shift significantly")
        print(f"   Market may be slow to adjust → EDGE OPPORTUNITY")
        
        print("\n" + "─" * 80)
        print("📍 SCORE: 40-30 (Game point for server)")
        
        p_from_40_30 = tree.p_game_recursive(0.66, 3, 2, show_tree=False)
        print(f"   True probability: {server_name} {p_from_40_30*100:.1f}% | {returner_name} {(1-p_from_40_30)*100:.1f}%")
        print(f"   Server heavily favored - odds very short")
        
        print("\n" + "─" * 80)
        print("📍 SCORE: 30-40 (Break point)")
        
        p_from_30_40 = tree.p_game_recursive(0.66, 2, 3, show_tree=False)
        print(f"   True probability: {server_name} {p_from_30_40*100:.1f}% | {returner_name} {(1-p_from_30_40)*100:.1f}%")
        print(f"   Returner favored - odds may overreact")
        
        print("\n" + "=" * 80)
        print("STRATEGY FOR GAME BETTING:")
        print("=" * 80)
        print("✓ Best edges found at 0-0, 15-15, 30-30 (before critical points)")
        print("✓ Markets slow to adjust after big points")
        print("✓ Use smaller stakes (5-10% max) - games are volatile")
        print("✓ Strong servers (65%+ point win) have 75-85% hold rates")
        print("✓ Look for discrepancies between true probability and odds")
        print("=" * 80 + "\n")


# Run analysis
if __name__ == "__main__":
    print("\n" + "🎾" * 40)
    print("COMPLETE MARKOV CHAIN ANALYSIS")
    print("Chang Jordan vs Dossani Mohammad - UTR Men Athens")
    print("🎾" * 40)
    
    tree = MarkovGameTree()
    
    # Chang serving (66% point win rate)
    print("\n\n" + "█" * 80)
    print("SCENARIO 1: CHANG JORDAN SERVING")
    print("█" * 80)
    p_chang_serve = 0.66
    p_chang_holds = tree.generate_full_tree(p_chang_serve, "Chang Jordan")
    
    # Dossani serving (60% point win rate)
    print("\n\n" + "█" * 80)
    print("SCENARIO 2: DOSSANI MOHAMMAD SERVING")
    print("█" * 80)
    p_dossani_serve = 0.60
    p_dossani_holds = tree.generate_full_tree(p_dossani_serve, "Dossani Mohammad")
    
    # Break probability analysis
    print("\n" + "=" * 80)
    print("BREAK PROBABILITY COMPARISON")
    print("=" * 80)
    print(f"P(Chang gets broken) = {(1-p_chang_holds)*100:.2f}%")
    print(f"P(Chang breaks Dossani) = {(1-p_dossani_holds)*100:.2f}%")
    print(f"\nExpected breaks per set:")
    print(f"  Chang: {(1-p_chang_holds) * 6:.2f} games")
    print(f"  Dossani: {(1-p_dossani_holds) * 6:.2f} games")
    print(f"\nNet advantage: Chang +{((1-p_dossani_holds) - (1-p_chang_holds)) * 6:.2f} breaks/set")
    print("=" * 80)
    
    # Game betting analysis
    analyzer = GameBettingAnalyzer(bankroll=1000)
    analyzer.analyze_game_bets(p_chang_holds, "Chang Jordan", "Dossani Mohammad")
