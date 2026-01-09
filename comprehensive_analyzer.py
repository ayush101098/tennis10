"""
🎾 COMPREHENSIVE TENNIS DATA AGGREGATOR
======================================

Aggregates tennis data from multiple sources:
1. TennisExplorer.com (live scores, odds, stats)
2. Local database (historical data)
3. Manual input (for specific matches)

Integrates with Markov model for probability calculation and edge detection.
"""

import requests
from bs4 import BeautifulSoup
import sqlite3
from typing import Dict, List, Optional
import re
from datetime import datetime
import time


class ComprehensiveTennisData:
    """Aggregate tennis data from all available sources"""
    
    def __init__(self, db_path='tennis_data.db'):
        self.db_path = db_path
        self.bankroll = 1000
        self.target = 5000
        
        # Session for web scraping
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def get_player_historical_stats(self, player_name: str, surface: str = 'Hard') -> Optional[Dict]:
        """Get player stats from local database"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Find player
        query = """
        SELECT player_id, player_name 
        FROM players 
        WHERE LOWER(player_name) LIKE ? 
        LIMIT 1
        """
        cursor = conn.execute(query, (f'%{player_name.lower()}%',))
        result = cursor.fetchone()
        
        if not result:
            return None
        
        player_id = result[0]
        
        # Get serve statistics
        stats_query = """
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
            AND m.tournament_date >= date('now', '-365 days')
            AND s.first_serve_pct IS NOT NULL
        """
        
        cursor = conn.execute(stats_query, (player_id, surface))
        stats = cursor.fetchone()
        
        conn.close()
        
        if stats and stats[5] > 0:  # match_count > 0
            return {
                'player_name': result[1],
                'player_id': player_id,
                'first_serve_pct': stats[0],
                'first_serve_win_pct': stats[1],
                'second_serve_win_pct': stats[2],
                'aces_per_game': stats[3],
                'df_per_game': stats[4],
                'match_count': stats[5],
                'surface': surface,
                'source': 'database'
            }
        
        return None
    
    def analyze_match_with_odds(self, player1_name: str, player2_name: str,
                               odds_p1: float, odds_p2: float,
                               surface: str = 'Hard') -> Dict:
        """
        Comprehensive match analysis combining database stats and current odds
        
        Args:
            player1_name: First player name
            player2_name: Second player name  
            odds_p1: Decimal odds for player 1
            odds_p2: Decimal odds for player 2
            surface: Court surface
        """
        
        print(f"\n{'='*80}")
        print(f"🎾 COMPREHENSIVE MATCH ANALYSIS")
        print(f"{'='*80}")
        print(f"{player1_name} vs {player2_name}")
        print(f"Surface: {surface}")
        print(f"{'='*80}\n")
        
        # Try to get historical stats from database
        print("📊 Fetching historical statistics from database...")
        
        p1_stats = self.get_player_historical_stats(player1_name, surface)
        p2_stats = self.get_player_historical_stats(player2_name, surface)
        
        if p1_stats:
            print(f"\n✅ Found {p1_stats['player_name']} ({p1_stats['match_count']} matches)")
            print(f"   1st serve: {p1_stats['first_serve_pct']:.1f}%")
            print(f"   1st serve win: {p1_stats['first_serve_win_pct']:.1f}%")
            print(f"   2nd serve win: {p1_stats['second_serve_win_pct']:.1f}%")
        else:
            print(f"⚠️  {player1_name} not found in database - using estimates")
            p1_stats = self._estimate_stats_from_odds(odds_p1, 1)
        
        if p2_stats:
            print(f"\n✅ Found {p2_stats['player_name']} ({p2_stats['match_count']} matches)")
            print(f"   1st serve: {p2_stats['first_serve_pct']:.1f}%")
            print(f"   1st serve win: {p2_stats['first_serve_win_pct']:.1f}%")
            print(f"   2nd serve win: {p2_stats['second_serve_win_pct']:.1f}%")
        else:
            print(f"⚠️  {player2_name} not found in database - using estimates")
            p2_stats = self._estimate_stats_from_odds(odds_p2, 2)
        
        # Calculate Markov probabilities
        print(f"\n🔢 Calculating Markov chain probabilities...")
        
        from live_betting_assistant import LiveMatchAnalyzer
        analyzer = LiveMatchAnalyzer()
        analyzer.bankroll = self.bankroll
        
        probs = analyzer.calculate_probabilities_from_stats(
            {
                'first_serve_pct': p1_stats['first_serve_pct'],
                'first_serve_win_pct': p1_stats['first_serve_win_pct'],
                'second_serve_win_pct': p1_stats['second_serve_win_pct']
            },
            {
                'first_serve_pct': p2_stats['first_serve_pct'],
                'first_serve_win_pct': p2_stats['first_serve_win_pct'],
                'second_serve_win_pct': p2_stats['second_serve_win_pct']
            },
            player1_name,
            player2_name
        )
        
        # Analyze betting edges
        print(f"💰 Analyzing betting opportunities with current odds...")
        
        odds_dict = {
            'match': [odds_p1, odds_p2]
        }
        
        analyzer.analyze_betting_opportunities(
            probs, odds_dict,
            player1_name, player2_name
        )
        
        return {
            'probabilities': probs,
            'odds': odds_dict,
            'p1_stats_source': p1_stats.get('source', 'estimated'),
            'p2_stats_source': p2_stats.get('source', 'estimated')
        }
    
    def _estimate_stats_from_odds(self, odds: float, player_num: int) -> Dict:
        """Estimate serve statistics from betting odds"""
        
        # Convert odds to implied probability
        implied_prob = 1.0 / odds
        
        # Estimate serve strength based on implied win probability
        # Rough mapping: 
        # - 80%+ win prob → elite server (68%+ point win)
        # - 60-80% → strong (64-67%)
        # - 40-60% → average (60-63%)
        # - <40% → weak (<60%)
        
        if implied_prob >= 0.80:
            point_serve = 0.68
        elif implied_prob >= 0.60:
            point_serve = 0.64 + (implied_prob - 0.60) * 0.20
        elif implied_prob >= 0.40:
            point_serve = 0.60 + (implied_prob - 0.40) * 0.20
        else:
            point_serve = 0.56 + implied_prob * 0.10
        
        # Estimate component stats
        # Typical ATP: 62% 1st in, 72% 1st win, 52% 2nd win
        # Adjust based on overall strength
        
        strength_factor = (point_serve - 0.60) / 0.08  # -0.5 to +1.0
        
        first_in = 62.0 + strength_factor * 6.0
        first_win = 72.0 + strength_factor * 5.0
        second_win = 52.0 + strength_factor * 6.0
        
        return {
            'first_serve_pct': max(50, min(75, first_in)),
            'first_serve_win_pct': max(60, min(80, first_win)),
            'second_serve_win_pct': max(40, min(65, second_win)),
            'source': 'estimated'
        }
    
    def batch_analyze_matches(self, matches: List[Dict]):
        """Analyze multiple matches at once"""
        
        print(f"\n{'='*80}")
        print(f"📊 BATCH ANALYSIS: {len(matches)} MATCHES")
        print(f"{'='*80}\n")
        
        all_opportunities = []
        
        for i, match in enumerate(matches, 1):
            print(f"\n{'-'*80}")
            print(f"MATCH {i}/{len(matches)}")
            print(f"{'-'*80}")
            
            try:
                result = self.analyze_match_with_odds(
                    match['player1'],
                    match['player2'],
                    match['odds_p1'],
                    match['odds_p2'],
                    match.get('surface', 'Hard')
                )
                
                # Check for edges
                probs = result['probabilities']
                odds = match['odds_p1']
                
                edge = (probs['p1_match'] * odds) - 1
                
                if edge > 0.02:
                    all_opportunities.append({
                        'match': f"{match['player1']} vs {match['player2']}",
                        'bet': match['player1'],
                        'odds': odds,
                        'edge': edge,
                        'tournament': match.get('tournament', 'Unknown')
                    })
                
                # Check player 2
                odds2 = match['odds_p2']
                edge2 = (probs['p2_match'] * odds2) - 1
                
                if edge2 > 0.02:
                    all_opportunities.append({
                        'match': f"{match['player1']} vs {match['player2']}",
                        'bet': match['player2'],
                        'odds': odds2,
                        'edge': edge2,
                        'tournament': match.get('tournament', 'Unknown')
                    })
                
            except Exception as e:
                print(f"❌ Error analyzing match: {e}")
                continue
            
            time.sleep(0.5)  # Brief pause between analyses
        
        # Summary
        if all_opportunities:
            print(f"\n{'='*80}")
            print(f"💸 SUMMARY: {len(all_opportunities)} PROFITABLE OPPORTUNITIES")
            print(f"{'='*80}\n")
            
            all_opportunities.sort(key=lambda x: x['edge'], reverse=True)
            
            total_stake = 0
            total_ev = 0
            
            for i, opp in enumerate(all_opportunities, 1):
                kelly = (opp['edge'] / (opp['odds'] - 1)) * 0.25
                stake = min(self.bankroll * kelly, self.bankroll * 0.10)
                ev = stake * opp['edge']
                
                total_stake += stake
                total_ev += ev
                
                print(f"{i}. {opp['match']}")
                print(f"   Bet: {opp['bet']} @ {opp['odds']:.2f}")
                print(f"   Edge: {opp['edge']*100:+.1f}% | Stake: ${stake:.2f} | EV: ${ev:+.2f}\n")
            
            print(f"{'─'*80}")
            print(f"Total stakes: ${total_stake:.2f}")
            print(f"Total expected value: ${total_ev:+.2f}")
            print(f"Projected bankroll: ${self.bankroll + total_ev:,.2f}")
            print(f"Progress to ${self.target:,.0f}: {(self.bankroll + total_ev)/self.target*100:.1f}%")
            print(f"{'='*80}\n")
        else:
            print(f"\n⏸️  No profitable opportunities found in this batch")


def main():
    """Main entry point"""
    
    print("\n🎾 COMPREHENSIVE TENNIS BETTING ANALYZER")
    print("="*80)
    
    analyzer = ComprehensiveTennisData()
    
    print("\nOptions:")
    print("1. Analyze single match with odds")
    print("2. Analyze Chang Jordan vs Dossani Mohammad (UTR Athens)")
    print("3. Batch analyze multiple matches")
    print("4. Check player in database")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        p1 = input("\nPlayer 1 name: ").strip()
        p2 = input("Player 2 name: ").strip()
        odds1 = float(input(f"{p1} odds: "))
        odds2 = float(input(f"{p2} odds: "))
        surface = input("Surface (Hard/Clay/Grass) [Hard]: ").strip() or "Hard"
        
        analyzer.analyze_match_with_odds(p1, p2, odds1, odds2, surface)
    
    elif choice == "2":
        # Chang vs Dossani with real odds
        analyzer.analyze_match_with_odds(
            "Chang Jordan",
            "Dossani Mohammad",
            1.28,  # Chang odds
            3.50,  # Dossani odds
            "Hard"
        )
    
    elif choice == "3":
        # Example batch
        matches = [
            {
                'player1': 'Chang Jordan',
                'player2': 'Dossani Mohammad',
                'odds_p1': 1.28,
                'odds_p2': 3.50,
                'surface': 'Hard',
                'tournament': 'UTR Men Athens'
            },
            # Add more matches here
        ]
        
        print("\n💡 Add more matches to the list in the code for batch analysis")
        print(f"Currently analyzing {len(matches)} match(es)...\n")
        
        analyzer.batch_analyze_matches(matches)
    
    elif choice == "4":
        player = input("\nPlayer name to search: ").strip()
        stats = analyzer.get_player_historical_stats(player)
        
        if stats:
            print(f"\n✅ Found: {stats['player_name']}")
            print(f"   Matches in database: {stats['match_count']}")
            print(f"   First serve: {stats['first_serve_pct']:.1f}%")
            print(f"   First serve win: {stats['first_serve_win_pct']:.1f}%")
            print(f"   Second serve win: {stats['second_serve_win_pct']:.1f}%")
            print(f"   Aces/game: {stats['aces_per_game']:.2f}")
            print(f"   DF/game: {stats['df_per_game']:.2f}")
        else:
            print(f"\n❌ '{player}' not found in database")


if __name__ == "__main__":
    main()
