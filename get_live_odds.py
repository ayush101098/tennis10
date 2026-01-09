"""
🎾 GET LIVE ODDS & UPCOMING MATCHES
====================================

Uses The Odds API to get:
- All upcoming tennis matches
- Real odds from 30+ bookmakers
- Best odds for each match

SETUP:
1. Get FREE API key: https://the-odds-api.com/
2. Run: export ODDS_API_KEY='your_key'
3. Run this script
"""

import os
import requests
from datetime import datetime
from typing import List, Dict
import json


ODDS_API_KEY = os.environ.get('ODDS_API_KEY', 'YOUR_KEY_HERE')


def get_live_tennis_matches_and_odds() -> List[Dict]:
    """
    Get all upcoming tennis matches with real bookmaker odds.
    """
    
    print("\n" + "="*80)
    print("🎾 FETCHING LIVE TENNIS MATCHES & ODDS")
    print("="*80)
    
    if ODDS_API_KEY == 'YOUR_KEY_HERE':
        print("\n❌ NO API KEY SET")
        print("\nQuick setup:")
        print("1. Visit: https://the-odds-api.com/")
        print("2. Sign up (FREE - 500 requests/month)")
        print("3. Copy your API key")
        print("4. Run: export ODDS_API_KEY='your_key_here'")
        print("5. Run this script again")
        return []
    
    base_url = "https://api.the-odds-api.com/v4/sports"
    
    # Tennis leagues
    sports = [
        ('tennis_atp', 'ATP'),
        ('tennis_wta', 'WTA'),
        ('tennis_atp_us_open', 'US Open'),
        ('tennis_atp_french_open', 'French Open'),
        ('tennis_atp_wimbledon', 'Wimbledon'),
        ('tennis_atp_australian_open', 'Australian Open'),
    ]
    
    all_matches = []
    
    for sport_key, sport_name in sports:
        print(f"\n📡 Fetching {sport_name}...")
        
        url = f"{base_url}/{sport_key}/odds/"
        
        params = {
            'apiKey': ODDS_API_KEY,
            'regions': 'us,uk,eu,au',  # All regions
            'markets': 'h2h',  # Match winner
            'oddsFormat': 'decimal',
            'dateFormat': 'iso'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data:
                    print(f"   ✅ {len(data)} matches found")
                    
                    for event in data:
                        match = parse_match(event, sport_name)
                        if match:
                            all_matches.append(match)
                else:
                    print(f"   ⚠️  No matches scheduled")
            
            elif response.status_code == 401:
                print(f"   ❌ Invalid API key!")
                return []
            
            elif response.status_code == 429:
                print(f"   ⚠️  Rate limit reached")
                break
            
            else:
                print(f"   ⚠️  Status {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ TOTAL: {len(all_matches)} matches with odds")
    print(f"{'='*80}\n")
    
    return all_matches


def parse_match(event: Dict, league: str) -> Dict:
    """Parse match data from API response"""
    
    try:
        player1 = event['home_team']
        player2 = event['away_team']
        start_time = event['commence_time']
        
        # Parse bookmaker odds
        all_bookmakers = []
        
        for bookmaker in event.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':
                    outcomes = market['outcomes']
                    
                    if len(outcomes) >= 2:
                        # Find odds for each player
                        p1_odds = next((o['price'] for o in outcomes if o['name'] == player1), None)
                        p2_odds = next((o['price'] for o in outcomes if o['name'] == player2), None)
                        
                        if p1_odds and p2_odds:
                            all_bookmakers.append({
                                'name': bookmaker['title'],
                                'player1_odds': p1_odds,
                                'player2_odds': p2_odds,
                                'last_update': bookmaker.get('last_update', '')
                            })
        
        if not all_bookmakers:
            return None
        
        # Calculate best and average odds
        best_p1 = max(all_bookmakers, key=lambda x: x['player1_odds'])
        best_p2 = max(all_bookmakers, key=lambda x: x['player2_odds'])
        
        avg_p1 = sum(b['player1_odds'] for b in all_bookmakers) / len(all_bookmakers)
        avg_p2 = sum(b['player2_odds'] for b in all_bookmakers) / len(all_bookmakers)
        
        return {
            'player1': player1,
            'player2': player2,
            'league': league,
            'start_time': start_time,
            'num_bookmakers': len(all_bookmakers),
            'best_odds': {
                'player1': best_p1['player1_odds'],
                'player2': best_p2['player2_odds'],
                'bookmaker_p1': best_p1['name'],
                'bookmaker_p2': best_p2['name']
            },
            'average_odds': {
                'player1': avg_p1,
                'player2': avg_p2
            },
            'all_bookmakers': all_bookmakers
        }
    
    except Exception as e:
        return None


def display_matches(matches: List[Dict]):
    """Display matches in a nice format"""
    
    if not matches:
        print("\n⚠️  No upcoming matches found")
        return
    
    print("\n" + "="*80)
    print(f"📋 {len(matches)} UPCOMING MATCHES")
    print("="*80 + "\n")
    
    for i, match in enumerate(matches, 1):
        # Parse start time
        try:
            start_dt = datetime.fromisoformat(match['start_time'].replace('Z', '+00:00'))
            start_str = start_dt.strftime('%Y-%m-%d %H:%M UTC')
        except:
            start_str = match['start_time']
        
        print(f"{i}. {match['player1']} vs {match['player2']}")
        print(f"   League: {match['league']}")
        print(f"   Start: {start_str}")
        print(f"   Bookmakers: {match['num_bookmakers']}")
        
        print(f"\n   BEST ODDS:")
        print(f"     {match['player1']}: {match['best_odds']['player1']:.2f} @ {match['best_odds']['bookmaker_p1']}")
        print(f"     {match['player2']}: {match['best_odds']['player2']:.2f} @ {match['best_odds']['bookmaker_p2']}")
        
        print(f"\n   AVERAGE ODDS:")
        print(f"     {match['player1']}: {match['average_odds']['player1']:.2f}")
        print(f"     {match['player2']}: {match['average_odds']['player2']:.2f}")
        
        # Show all bookmakers
        if match['num_bookmakers'] > 1:
            print(f"\n   ALL BOOKMAKERS:")
            for bm in match['all_bookmakers'][:5]:  # First 5
                print(f"     {bm['name']:20s} {bm['player1_odds']:6.2f} / {bm['player2_odds']:6.2f}")
            
            if match['num_bookmakers'] > 5:
                print(f"     ... and {match['num_bookmakers'] - 5} more")
        
        print()


def save_to_file(matches: List[Dict]):
    """Save matches to JSON file"""
    
    if not matches:
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'upcoming_matches_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(matches, f, indent=2)
    
    print(f"💾 Saved to: {filename}")


def main():
    """Main function"""
    
    print("\n" + "🎾"*40)
    print("LIVE TENNIS MATCHES & BOOKMAKER ODDS")
    print("🎾"*40)
    print(f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch matches
    matches = get_live_tennis_matches_and_odds()
    
    # Display
    display_matches(matches)
    
    # Save
    if matches:
        save_to_file(matches)
        
        print("\n" + "="*80)
        print("📊 QUICK STATS")
        print("="*80)
        print(f"Total matches: {len(matches)}")
        print(f"Total bookmakers: {sum(m['num_bookmakers'] for m in matches)}")
        print(f"Average bookmakers per match: {sum(m['num_bookmakers'] for m in matches) / len(matches):.1f}")
        
        # Show which bookmakers are available
        all_bookmaker_names = set()
        for match in matches:
            for bm in match['all_bookmakers']:
                all_bookmaker_names.add(bm['name'])
        
        print(f"\nBookmakers available ({len(all_bookmaker_names)}):")
        for name in sorted(all_bookmaker_names):
            print(f"  • {name}")
        
        print("\n" + "="*80)
        print("\n✅ Ready to analyze for edges!")
        print("\nNext step:")
        print("  python find_real_edges.py")
        print("\nThis will analyze all matches and find profitable bets.")
    else:
        print("\n⚠️  No matches available")
        print("\nPossible reasons:")
        print("1. No tennis events scheduled right now")
        print("2. API key not set (run: export ODDS_API_KEY='your_key')")
        print("3. Rate limit reached (500 requests/month on free tier)")


if __name__ == "__main__":
    main()
