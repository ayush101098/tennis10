"""
🎾 LIVE TENNIS BETTING ANALYZER
================================

Uses The Odds API (FREE) to get:
- Live matches with real player names
- Bookmaker odds from 30+ bookmakers
- Start times and match status

Then calculates Markov probabilities for edge detection.

Setup:
1. Get FREE API key at: https://the-odds-api.com/
2. Free tier: 500 requests/month
3. Paste key below
"""

import requests
from datetime import datetime
from typing import List, Dict
import os


# ============================================================================
# GET YOUR FREE API KEY: https://the-odds-api.com/
# ============================================================================
ODDS_API_KEY = os.environ.get('ODDS_API_KEY', 'YOUR_KEY_HERE')


class LiveTennisBetting:
    """Fetch live tennis matches with bookmaker odds"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.session = requests.Session()
    
    def get_live_matches(self) -> List[Dict]:
        """Get all live and upcoming tennis matches with odds"""
        
        print("\n" + "="*80)
        print("🎾 FETCHING LIVE TENNIS MATCHES")
        print("="*80)
        
        # The Odds API supports multiple tennis leagues
        sports = [
            'tennis_atp',  # ATP Tour
            'tennis_wta',  # WTA Tour
            'tennis',      # General tennis
        ]
        
        all_matches = []
        
        for sport in sports:
            try:
                url = f"{self.base_url}/{sport}/odds/"
                
                params = {
                    'apiKey': self.api_key,
                    'regions': 'us,uk,eu,au',  # All major regions
                    'markets': 'h2h,spreads,totals',  # Match winner + handicaps
                    'oddsFormat': 'decimal',
                    'dateFormat': 'iso'
                }
                
                print(f"\n📡 Fetching {sport}...")
                
                response = self.session.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    print(f"✅ Found {len(data)} matches")
                    
                    for event in data:
                        match = self._parse_match(event, sport)
                        if match:
                            all_matches.append(match)
                
                elif response.status_code == 401:
                    print("❌ Invalid API key!")
                    print("Get free key at: https://the-odds-api.com/")
                    return []
                
                elif response.status_code == 429:
                    print("⚠️  Rate limit reached (500 requests/month on free tier)")
                    break
                
                else:
                    print(f"⚠️  Status {response.status_code}")
            
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n✅ Total: {len(all_matches)} matches with odds")
        return all_matches
    
    def _parse_match(self, event: Dict, sport: str) -> Dict:
        """Parse match data from API response"""
        
        try:
            # Get best odds from all bookmakers
            bookmaker_odds = []
            
            for bookmaker in event.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'h2h':  # Match winner
                        outcomes = market['outcomes']
                        
                        if len(outcomes) >= 2:
                            # Find player 1 and player 2
                            p1_outcome = next((o for o in outcomes if o['name'] == event['home_team']), None)
                            p2_outcome = next((o for o in outcomes if o['name'] == event['away_team']), None)
                            
                            if p1_outcome and p2_outcome:
                                bookmaker_odds.append({
                                    'bookmaker': bookmaker['title'],
                                    'player1': p1_outcome['price'],
                                    'player2': p2_outcome['price'],
                                    'last_update': bookmaker.get('last_update', '')
                                })
            
            if not bookmaker_odds:
                return None
            
            # Get best odds (highest for each player)
            best_p1 = max(bookmaker_odds, key=lambda x: x['player1'])
            best_p2 = max(bookmaker_odds, key=lambda x: x['player2'])
            
            # Calculate average odds
            avg_p1 = sum(b['player1'] for b in bookmaker_odds) / len(bookmaker_odds)
            avg_p2 = sum(b['player2'] for b in bookmaker_odds) / len(bookmaker_odds)
            
            match = {
                'id': event['id'],
                'sport': sport,
                'player1': event['home_team'],
                'player2': event['away_team'],
                'commence_time': event['commence_time'],
                'bookmakers': bookmaker_odds,
                'num_bookmakers': len(bookmaker_odds),
                'best_odds': {
                    'player1': {
                        'odds': best_p1['player1'],
                        'bookmaker': best_p1['bookmaker']
                    },
                    'player2': {
                        'odds': best_p2['player2'],
                        'bookmaker': best_p2['bookmaker']
                    }
                },
                'average_odds': {
                    'player1': avg_p1,
                    'player2': avg_p2
                }
            }
            
            return match
        
        except Exception as e:
            print(f"⚠️  Parse error: {e}")
            return None
    
    def display_matches(self, matches: List[Dict]):
        """Display matches nicely"""
        
        if not matches:
            print("\n⚠️  No matches with odds available")
            return
        
        print("\n" + "="*80)
        print(f"💰 {len(matches)} MATCHES WITH BOOKMAKER ODDS")
        print("="*80 + "\n")
        
        for i, match in enumerate(matches, 1):
            # Parse start time
            try:
                start_time = datetime.fromisoformat(match['commence_time'].replace('Z', '+00:00'))
                time_str = start_time.strftime('%Y-%m-%d %H:%M UTC')
            except:
                time_str = match['commence_time']
            
            print(f"{i}. {match['player1']} vs {match['player2']}")
            print(f"   Start: {time_str}")
            print(f"   League: {match['sport'].upper()}")
            print(f"   Bookmakers: {match['num_bookmakers']}")
            print(f"   Best Odds:")
            print(f"     {match['player1']}: {match['best_odds']['player1']['odds']:.2f} ({match['best_odds']['player1']['bookmaker']})")
            print(f"     {match['player2']}: {match['best_odds']['player2']['odds']:.2f} ({match['best_odds']['player2']['bookmaker']})")
            print(f"   Average Odds: {match['average_odds']['player1']:.2f} / {match['average_odds']['player2']:.2f}")
            print()
    
    def analyze_match(self, match: Dict):
        """Analyze a match with Markov probabilities"""
        
        print("\n" + "="*80)
        print(f"🎯 ANALYZING: {match['player1']} vs {match['player2']}")
        print("="*80)
        
        # Show all bookmaker odds
        print(f"\n💰 ODDS FROM {match['num_bookmakers']} BOOKMAKERS:\n")
        
        for bm in match['bookmakers']:
            print(f"  {bm['bookmaker']:20s} {bm['player1']:6.2f}  /  {bm['player2']:6.2f}")
        
        print(f"\n  {'BEST ODDS':20s} {match['best_odds']['player1']['odds']:6.2f}  /  {match['best_odds']['player2']['odds']:6.2f}")
        print(f"  {'AVERAGE':20s} {match['average_odds']['player1']:6.2f}  /  {match['average_odds']['player2']:6.2f}")
        
        # Run Markov analysis with BEST odds
        print("\n🔄 Running Markov probability analysis...")
        
        try:
            from comprehensive_analyzer import ComprehensiveTennisData
            
            analyzer = ComprehensiveTennisData()
            
            # Use BEST odds for maximum edge
            analyzer.analyze_match_with_odds(
                match['player1'],
                match['player2'],
                match['best_odds']['player1']['odds'],
                match['best_odds']['player2']['odds']
            )
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    
    print("\n" + "🎾"*40)
    print("LIVE TENNIS BETTING ANALYZER")
    print("Real-time matches + Bookmaker odds + Markov probabilities")
    print("🎾"*40)
    print(f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    if ODDS_API_KEY == 'YOUR_KEY_HERE':
        print("\n" + "="*80)
        print("⚠️  NO API KEY CONFIGURED")
        print("="*80)
        print("\nTo get live matches with real bookmaker odds:")
        print("1. Go to: https://the-odds-api.com/")
        print("2. Sign up for FREE (500 requests/month)")
        print("3. Copy your API key")
        print("4. Run: export ODDS_API_KEY='your_key_here'")
        print("5. Or edit this file and paste key at top")
        print("\nFREE tier includes:")
        print("  ✓ Live matches from 30+ bookmakers")
        print("  ✓ ATP & WTA tournaments")
        print("  ✓ Real-time odds updates")
        print("  ✓ 500 requests/month")
        print("="*80 + "\n")
        return
    
    # Initialize
    betting = LiveTennisBetting(ODDS_API_KEY)
    
    # Fetch matches
    matches = betting.get_live_matches()
    
    # Display
    betting.display_matches(matches)
    
    # Let user analyze a match
    if matches:
        print("\n" + "="*80)
        print("🎯 SELECT MATCH TO ANALYZE")
        print("="*80)
        
        choice = input(f"\nEnter number (1-{len(matches)}) or press Enter to skip: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(matches):
            match = matches[int(choice) - 1]
            betting.analyze_match(match)
    
    print("\n✅ Complete! Use this tool before every bet for maximum edge.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
