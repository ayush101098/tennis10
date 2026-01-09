"""
🎾 RapidAPI Tennis Live Data Integration
========================================

Integrates live tennis data and statistics from RapidAPI:
1. Tennis Statistics IQ API - Calculate match probabilities from serve stats
2. Tennis API4 - Live match data, scores, and odds

Usage:
    python rapid_api_integration.py
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time


class TennisStatisticsAPI:
    """Interface to Tennis Statistics IQ API for probability calculations"""
    
    BASE_URL = "https://tennis-statistics-iq.p.rapidapi.com"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            'x-rapidapi-host': 'tennis-statistics-iq.p.rapidapi.com',
            'x-rapidapi-key': api_key
        }
    
    def calculate_match_probability(self, player1_stats: Dict, player2_stats: Dict) -> Dict:
        """
        Calculate match probability from serve statistics
        
        Args:
            player1_stats: {
                'first_serve_pct': float,  # % (e.g., 75.2)
                'first_serve_win_pct': float,  # % (e.g., 62.8)
                'second_serve_win_pct': float  # % (e.g., 56.3)
            }
            player2_stats: Similar dict for player 2
        
        Returns:
            API response with match probabilities
        """
        url = f"{self.BASE_URL}/calculate"
        
        params = {
            'FirstIn1_percentual': player1_stats['first_serve_pct'],
            'FirstWin1_percentual': player1_stats['first_serve_win_pct'],
            'SecondWin1_percentual': player1_stats['second_serve_win_pct'],
            'FirstIn2_percentual': player2_stats['first_serve_pct'],
            'FirstWin2_percentual': player2_stats['first_serve_win_pct'],
            'SecondWin2_percentual': player2_stats['second_serve_win_pct']
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching statistics: {e}")
            return None


class TennisLiveAPI:
    """Interface to Tennis API4 for live match data and odds"""
    
    BASE_URL = "https://tennis-api4.p.rapidapi.com"
    
    def __init__(self, api_key: str, proxy_secret: str):
        self.api_key = api_key
        self.proxy_secret = proxy_secret
        self.headers = {
            'x-rapidapi-host': 'tennis-api4.p.rapidapi.com',
            'x-rapidapi-key': api_key,
            'X-RapidAPI-Proxy-Secret': proxy_secret
        }
    
    def get_live_matches(self) -> Optional[List[Dict]]:
        """Get all live matches"""
        url = f"{self.BASE_URL}/live"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching live matches: {e}")
            return None
    
    def get_match_details(self, event_id: str) -> Optional[Dict]:
        """Get detailed match information including live score"""
        url = f"{self.BASE_URL}/match/{event_id}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching match details: {e}")
            return None
    
    def get_match_markets(self, event_id: str, market_type: int = 2) -> Optional[Dict]:
        """
        Get betting markets for a match
        
        Args:
            event_id: Match event ID
            market_type: 
                1 = Match Winner
                2 = Set Winner
                3 = Game Winner
                etc.
        """
        url = f"{self.BASE_URL}/markets/{market_type}/{event_id}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching markets: {e}")
            return None
    
    def get_match_statistics(self, event_id: str) -> Optional[Dict]:
        """Get live match statistics (aces, double faults, etc.)"""
        url = f"{self.BASE_URL}/statistics/{event_id}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching statistics: {e}")
            return None


class LiveBettingAnalyzer:
    """Analyze live matches for betting opportunities using RapidAPI data"""
    
    def __init__(self, api_key: str, proxy_secret: str):
        self.stats_api = TennisStatisticsAPI(api_key)
        self.live_api = TennisLiveAPI(api_key, proxy_secret)
        self.bankroll = 1000
        self.min_edge = 0.025
    
    def analyze_live_match(self, event_id: str, player1_stats: Dict, player2_stats: Dict):
        """
        Analyze a live match for betting opportunities
        
        Args:
            event_id: Match ID from Tennis API4
            player1_stats: Serve statistics for player 1
            player2_stats: Serve statistics for player 2
        """
        print(f"\n{'='*80}")
        print(f"🎾 LIVE MATCH ANALYSIS")
        print(f"{'='*80}\n")
        
        # Get match details
        match_details = self.live_api.get_match_details(event_id)
        if not match_details:
            print("❌ Could not fetch match details")
            return
        
        print(f"📊 Fetching live match data...")
        
        # Get live statistics
        live_stats = self.live_api.get_match_statistics(event_id)
        
        # Calculate probabilities from serve stats
        print(f"\n🔢 Calculating Markov probabilities from serve statistics...")
        prob_result = self.stats_api.calculate_match_probability(player1_stats, player2_stats)
        
        if prob_result:
            print(f"✅ Got probability calculations from API")
            print(json.dumps(prob_result, indent=2))
        else:
            print("⚠️  Using local Markov calculation")
            prob_result = self._calculate_local_probability(player1_stats, player2_stats)
        
        # Get betting markets
        print(f"\n💰 Fetching betting markets...")
        
        # Match winner market
        match_markets = self.live_api.get_match_markets(event_id, market_type=1)
        if match_markets:
            print(f"\n📈 MATCH WINNER ODDS:")
            print(json.dumps(match_markets, indent=2))
        
        # Set winner market
        set_markets = self.live_api.get_match_markets(event_id, market_type=2)
        if set_markets:
            print(f"\n📈 SET WINNER ODDS:")
            print(json.dumps(set_markets, indent=2))
        
        # Game winner market
        game_markets = self.live_api.get_match_markets(event_id, market_type=3)
        if game_markets:
            print(f"\n📈 GAME WINNER ODDS:")
            print(json.dumps(game_markets, indent=2))
        
        # Identify opportunities
        self._find_betting_edges(prob_result, match_markets, set_markets, game_markets)
    
    def _calculate_local_probability(self, p1_stats: Dict, p2_stats: Dict) -> Dict:
        """Calculate probabilities locally using Markov model"""
        from hierarchical_model import HierarchicalTennisModel
        
        # Convert percentage stats to point-level probabilities
        # P(win point on serve) = first_in% * first_win% + (1-first_in%) * second_win%
        
        p1_first_in = p1_stats['first_serve_pct'] / 100.0
        p1_first_win = p1_stats['first_serve_win_pct'] / 100.0
        p1_second_win = p1_stats['second_serve_win_pct'] / 100.0
        
        p2_first_in = p2_stats['first_serve_pct'] / 100.0
        p2_first_win = p2_stats['first_serve_win_pct'] / 100.0
        p2_second_win = p2_stats['second_serve_win_pct'] / 100.0
        
        p1_point = p1_first_in * p1_first_win + (1 - p1_first_in) * p1_second_win
        p2_point = p2_first_in * p2_first_win + (1 - p2_first_in) * p2_second_win
        
        # Calculate game hold probabilities
        def p_game(p_point):
            """Simplified game probability from deuce formula"""
            q = 1 - p_point
            return (p_point ** 2) / (1 - 2 * p_point * q) if (1 - 2 * p_point * q) != 0 else 0.5
        
        p1_hold = p_game(p1_point)
        p2_hold = p_game(p2_point)
        
        # Average game win probability
        p1_game_avg = (p1_hold + (1 - p2_hold)) / 2
        
        # Rough set probability (simplified)
        p1_set = p1_game_avg ** 6 / (p1_game_avg ** 6 + (1 - p1_game_avg) ** 6)
        
        # Match probability (best of 3)
        p1_match = p1_set ** 2 + 2 * p1_set ** 2 * (1 - p1_set)
        
        return {
            'p1_point_on_serve': p1_point,
            'p2_point_on_serve': p2_point,
            'p1_hold': p1_hold,
            'p2_hold': p2_hold,
            'p1_match': p1_match,
            'p2_match': 1 - p1_match
        }
    
    def _find_betting_edges(self, probabilities: Dict, match_odds: Dict, 
                           set_odds: Dict, game_odds: Dict):
        """Identify profitable betting opportunities"""
        
        print(f"\n{'='*80}")
        print(f"💸 BETTING EDGE ANALYSIS")
        print(f"{'='*80}\n")
        
        if not probabilities:
            print("❌ No probability data available")
            return
        
        opportunities = []
        
        # Analyze match winner market
        if match_odds and 'p1_match' in probabilities:
            # Parse odds from API response
            # This will depend on the actual API response format
            # For now, show what we have
            print("📊 True probabilities:")
            print(f"   Player 1 win: {probabilities.get('p1_match', 0)*100:.1f}%")
            print(f"   Player 2 win: {probabilities.get('p2_match', 0)*100:.1f}%")
        
        # Note: Actual edge calculation depends on API response format
        # You'll need to extract the odds from the response and compare
        
        if not opportunities:
            print("\n⏸️  No clear edges identified")
            print("💡 TIP: Wait for live odds to update or check game-level markets")
    
    def scan_all_live_matches(self):
        """Scan all live matches for opportunities"""
        
        print(f"\n{'='*80}")
        print(f"🔍 SCANNING LIVE MATCHES")
        print(f"{'='*80}\n")
        
        live_matches = self.live_api.get_live_matches()
        
        if not live_matches:
            print("❌ No live matches found or API error")
            return
        
        print(f"✅ Found {len(live_matches) if isinstance(live_matches, list) else 'multiple'} live match(es)\n")
        print(json.dumps(live_matches, indent=2))
        
        # Process each match
        # Note: You'll need to extract event IDs from the response
        print("\n💡 Use the event IDs above to analyze specific matches")


def main():
    """Main entry point"""
    
    # API credentials
    API_KEY = "5d2522c01bmsh39e88183e2d0d38p18935ajsn87359504bb69"
    PROXY_SECRET = "17ad9d30-62d7-11f0-adbd-55af01a70c15"
    
    analyzer = LiveBettingAnalyzer(API_KEY, PROXY_SECRET)
    
    print("\n🎾 RAPIDAPI TENNIS LIVE BETTING ANALYZER")
    print("="*80)
    print("\nOptions:")
    print("1. Scan all live matches")
    print("2. Analyze specific match")
    print("3. Test API with sample data")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        analyzer.scan_all_live_matches()
    
    elif choice == "2":
        event_id = input("\nEnter event ID: ").strip()
        
        print("\nEnter Player 1 serve statistics:")
        p1_first_in = float(input("  First serve % (e.g., 75.2): "))
        p1_first_win = float(input("  First serve win % (e.g., 62.8): "))
        p1_second_win = float(input("  Second serve win % (e.g., 56.3): "))
        
        print("\nEnter Player 2 serve statistics:")
        p2_first_in = float(input("  First serve % (e.g., 69.8): "))
        p2_first_win = float(input("  First serve win % (e.g., 70.3): "))
        p2_second_win = float(input("  Second serve win % (e.g., 58.2): "))
        
        p1_stats = {
            'first_serve_pct': p1_first_in,
            'first_serve_win_pct': p1_first_win,
            'second_serve_win_pct': p1_second_win
        }
        
        p2_stats = {
            'first_serve_pct': p2_first_in,
            'first_serve_win_pct': p2_first_win,
            'second_serve_win_pct': p2_second_win
        }
        
        analyzer.analyze_live_match(event_id, p1_stats, p2_stats)
    
    elif choice == "3":
        # Test with sample data
        print("\n🧪 Testing APIs with sample data...")
        
        # Test Statistics API
        print("\n1️⃣  Testing Tennis Statistics IQ API...")
        p1_stats = {
            'first_serve_pct': 75.2,
            'first_serve_win_pct': 62.8,
            'second_serve_win_pct': 56.3
        }
        p2_stats = {
            'first_serve_pct': 69.8,
            'first_serve_win_pct': 70.3,
            'second_serve_win_pct': 58.2
        }
        
        result = analyzer.stats_api.calculate_match_probability(p1_stats, p2_stats)
        if result:
            print("✅ Statistics API working!")
            print(json.dumps(result, indent=2))
        else:
            print("❌ Statistics API failed")
        
        # Test Live API
        print("\n2️⃣  Testing Tennis Live API...")
        live_matches = analyzer.live_api.get_live_matches()
        if live_matches:
            print("✅ Live API working!")
            print(f"Found matches: {json.dumps(live_matches, indent=2)[:500]}...")
        else:
            print("❌ Live API failed")


if __name__ == "__main__":
    main()
