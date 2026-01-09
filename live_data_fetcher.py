"""
🎾 REAL-TIME LIVE TENNIS DATA FETCHER
=====================================

Fetches:
1. Live matches with current scores
2. Player serve statistics from live matches
3. Bookmaker odds directly from betting sites
4. Match state (game, set scores)
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
from typing import Dict, List, Optional


class LiveTennisData:
    """Fetch live tennis data from multiple sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def get_flashscore_live_matches(self) -> List[Dict]:
        """Fetch live matches from Flashscore"""
        
        print("\n" + "="*80)
        print("🔴 FETCHING LIVE MATCHES FROM FLASHSCORE")
        print("="*80)
        
        url = "https://www.flashscore.com/tennis/"
        
        try:
            response = self.session.get(url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            matches = []
            
            # Look for live match containers
            live_events = soup.find_all('div', class_=re.compile(r'event.*', re.I))
            
            print(f"Found {len(live_events)} potential live events")
            
            for event in live_events[:20]:
                try:
                    # Extract player names
                    participants = event.find_all(class_=re.compile(r'participant', re.I))
                    
                    if len(participants) >= 2:
                        player1 = participants[0].get_text(strip=True)
                        player2 = participants[1].get_text(strip=True)
                        
                        # Extract scores
                        scores = event.find_all(class_=re.compile(r'score', re.I))
                        score_text = ' '.join([s.get_text(strip=True) for s in scores])
                        
                        # Check if live
                        is_live = event.find(class_=re.compile(r'live|inplay', re.I)) is not None
                        
                        match = {
                            'player1': player1,
                            'player2': player2,
                            'score': score_text if score_text else 'Not started',
                            'is_live': is_live,
                            'source': 'flashscore'
                        }
                        
                        matches.append(match)
                        
                except Exception as e:
                    continue
            
            # If structured data didn't work, try JSON
            if not matches:
                # Look for embedded JSON data
                scripts = soup.find_all('script')
                for script in scripts:
                    if 'INITIAL_STATE' in str(script) or 'window.dataLayer' in str(script):
                        text = script.string
                        if text:
                            # Try to extract match data from JSON
                            json_matches = self._extract_json_matches(text)
                            matches.extend(json_matches)
            
            print(f"✅ Extracted {len(matches)} matches from Flashscore\n")
            return matches
            
        except Exception as e:
            print(f"❌ Flashscore error: {e}")
            return []
    
    def get_sofascore_live_matches(self) -> List[Dict]:
        """Fetch live matches from Sofascore API"""
        
        print("\n" + "="*80)
        print("🔴 FETCHING LIVE MATCHES FROM SOFASCORE")
        print("="*80)
        
        # Sofascore has a public API
        url = "https://api.sofascore.com/api/v1/sport/tennis/events/live"
        
        try:
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                matches = []
                events = data.get('events', [])
                
                print(f"Found {len(events)} live matches")
                
                for event in events:
                    try:
                        home = event.get('homeTeam', {}).get('name', 'Unknown')
                        away = event.get('awayTeam', {}).get('name', 'Unknown')
                        
                        home_score = event.get('homeScore', {})
                        away_score = event.get('awayScore', {})
                        
                        # Get current game score
                        current = home_score.get('current', 0)
                        current_away = away_score.get('current', 0)
                        
                        # Get set scores
                        sets = []
                        for i in range(1, 6):
                            set_home = home_score.get(f'period{i}', None)
                            set_away = away_score.get(f'period{i}', None)
                            if set_home is not None:
                                sets.append(f"{set_home}-{set_away}")
                        
                        score_text = ' '.join(sets) + f" ({current}-{current_away})" if sets else f"{current}-{current_away}"
                        
                        match = {
                            'player1': home,
                            'player2': away,
                            'score': score_text,
                            'is_live': True,
                            'tournament': event.get('tournament', {}).get('name', 'Unknown'),
                            'source': 'sofascore',
                            'event_id': event.get('id')
                        }
                        
                        matches.append(match)
                        
                    except Exception as e:
                        continue
                
                print(f"✅ Extracted {len(matches)} live matches\n")
                return matches
            else:
                print(f"❌ Status code: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Sofascore error: {e}")
            return []
    
    def get_match_statistics(self, match: Dict) -> Optional[Dict]:
        """Get live statistics for a match from Sofascore"""
        
        if match['source'] != 'sofascore' or 'event_id' not in match:
            return None
        
        print(f"\n📊 Fetching statistics for {match['player1']} vs {match['player2']}...")
        
        event_id = match['event_id']
        url = f"https://api.sofascore.com/api/v1/event/{event_id}/statistics"
        
        try:
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                stats = {
                    'player1': match['player1'],
                    'player2': match['player2'],
                    'serve_stats': {}
                }
                
                # Extract serve statistics
                periods = data.get('statistics', [])
                
                for period in periods:
                    groups = period.get('groups', [])
                    
                    for group in groups:
                        if 'serve' in group.get('groupName', '').lower():
                            stats_items = group.get('statisticsItems', [])
                            
                            for item in stats_items:
                                name = item.get('name', '')
                                home_val = item.get('home', 0)
                                away_val = item.get('away', 0)
                                
                                if 'first serve' in name.lower():
                                    stats['serve_stats']['first_serve_pct'] = {
                                        'player1': home_val,
                                        'player2': away_val
                                    }
                                elif 'aces' in name.lower():
                                    stats['serve_stats']['aces'] = {
                                        'player1': home_val,
                                        'player2': away_val
                                    }
                                elif 'double faults' in name.lower():
                                    stats['serve_stats']['double_faults'] = {
                                        'player1': home_val,
                                        'player2': away_val
                                    }
                
                print(f"✅ Retrieved statistics")
                return stats
            else:
                print(f"❌ Stats not available (status {response.status_code})")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching stats: {e}")
            return None
    
    def get_oddsportal_odds(self, player1: str, player2: str) -> Optional[Dict]:
        """Fetch bookmaker odds from OddsPortal"""
        
        print(f"\n💰 Fetching odds for {player1} vs {player2}...")
        
        # Clean player names for URL
        p1_clean = player1.replace(' ', '-').lower()
        p2_clean = player2.replace(' ', '-').lower()
        
        # Try to find match on OddsPortal
        search_url = f"https://www.oddsportal.com/search/{p1_clean}/"
        
        try:
            response = self.session.get(search_url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for odds
            odds_elements = soup.find_all(class_=re.compile(r'odds', re.I))
            
            if odds_elements:
                odds = {
                    'player1_odds': [],
                    'player2_odds': [],
                    'bookmakers': []
                }
                
                for elem in odds_elements[:5]:
                    text = elem.get_text(strip=True)
                    # Try to parse decimal odds
                    match_odds = re.findall(r'\d+\.\d+', text)
                    if len(match_odds) >= 2:
                        odds['player1_odds'].append(float(match_odds[0]))
                        odds['player2_odds'].append(float(match_odds[1]))
                
                if odds['player1_odds']:
                    # Average odds
                    avg_p1 = sum(odds['player1_odds']) / len(odds['player1_odds'])
                    avg_p2 = sum(odds['player2_odds']) / len(odds['player2_odds'])
                    
                    print(f"✅ Found odds: {avg_p1:.2f} / {avg_p2:.2f}")
                    
                    return {
                        'player1': avg_p1,
                        'player2': avg_p2,
                        'source': 'oddsportal'
                    }
            
            print("❌ Odds not found")
            return None
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def get_bet365_odds_api(self) -> List[Dict]:
        """Try to fetch odds from betting API aggregator"""
        
        print("\n💰 Fetching odds from The Odds API...")
        
        # The Odds API (free tier available)
        # You need to get a free API key from: https://the-odds-api.com/
        
        api_key = "YOUR_API_KEY_HERE"  # User needs to get this
        
        url = f"https://api.the-odds-api.com/v4/sports/tennis/odds/"
        params = {
            'apiKey': api_key,
            'regions': 'us,uk,eu',
            'markets': 'h2h',
            'oddsFormat': 'decimal'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                matches_with_odds = []
                
                for event in data:
                    match = {
                        'player1': event['home_team'],
                        'player2': event['away_team'],
                        'commence_time': event['commence_time'],
                        'bookmaker_odds': []
                    }
                    
                    for bookmaker in event.get('bookmakers', []):
                        for market in bookmaker.get('markets', []):
                            if market['key'] == 'h2h':
                                outcomes = market['outcomes']
                                if len(outcomes) >= 2:
                                    match['bookmaker_odds'].append({
                                        'bookmaker': bookmaker['title'],
                                        'player1': outcomes[0]['price'],
                                        'player2': outcomes[1]['price']
                                    })
                    
                    matches_with_odds.append(match)
                
                print(f"✅ Found {len(matches_with_odds)} matches with odds")
                return matches_with_odds
            else:
                print(f"❌ API returned status {response.status_code}")
                if 'Invalid API key' in response.text:
                    print("⚠️  Get a free API key at: https://the-odds-api.com/")
                return []
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def _extract_json_matches(self, json_text: str) -> List[Dict]:
        """Extract matches from embedded JSON"""
        matches = []
        
        try:
            # Try to find JSON objects
            json_objects = re.findall(r'\{[^{}]*"player"[^{}]*\}', json_text)
            
            for obj_str in json_objects:
                try:
                    obj = json.loads(obj_str)
                    # Extract relevant data
                    # This is highly site-specific
                    pass
                except:
                    continue
        except:
            pass
        
        return matches


def display_live_matches(matches: List[Dict]):
    """Display live matches nicely"""
    
    if not matches:
        print("\n⚠️  No live matches found")
        return
    
    print("\n" + "="*80)
    print(f"🔴 {len(matches)} LIVE MATCHES FOUND")
    print("="*80 + "\n")
    
    for i, match in enumerate(matches, 1):
        status = "🔴 LIVE" if match.get('is_live') else "⏰ Upcoming"
        
        print(f"{i}. {status} {match['player1']} vs {match['player2']}")
        print(f"   Score: {match.get('score', 'N/A')}")
        if 'tournament' in match:
            print(f"   Tournament: {match['tournament']}")
        print(f"   Source: {match['source']}")
        print()


def analyze_live_match(match: Dict, fetcher: LiveTennisData):
    """Analyze a live match with statistics and odds"""
    
    print("\n" + "="*80)
    print(f"🎯 ANALYZING: {match['player1']} vs {match['player2']}")
    print("="*80)
    
    # Get statistics
    stats = fetcher.get_match_statistics(match)
    
    if stats:
        print("\n📊 LIVE STATISTICS:")
        print(json.dumps(stats['serve_stats'], indent=2))
    
    # Get odds
    odds = fetcher.get_oddsportal_odds(match['player1'], match['player2'])
    
    if odds:
        print(f"\n💰 BOOKMAKER ODDS:")
        print(f"  {match['player1']}: {odds['player1']:.2f}")
        print(f"  {match['player2']}: {odds['player2']:.2f}")
        
        # Run Markov analysis
        print("\n🔄 Running Markov analysis...")
        
        try:
            from comprehensive_analyzer import ComprehensiveTennisData
            analyzer = ComprehensiveTennisData()
            
            analyzer.analyze_match_with_odds(
                match['player1'],
                match['player2'],
                odds['player1'],
                odds['player2']
            )
        except Exception as e:
            print(f"❌ Analysis error: {e}")
    else:
        print("\n⚠️  No odds available - try manual entry")


def main():
    """Main function"""
    
    print("\n" + "🎾"*40)
    print("REAL-TIME LIVE TENNIS DATA SYSTEM")
    print("🎾"*40)
    print(f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    fetcher = LiveTennisData()
    
    all_matches = []
    
    # Try Sofascore first (best API)
    sofascore_matches = fetcher.get_sofascore_live_matches()
    all_matches.extend(sofascore_matches)
    
    # Try Flashscore
    if not all_matches:
        flashscore_matches = fetcher.get_flashscore_live_matches()
        all_matches.extend(flashscore_matches)
    
    # Display all matches
    display_live_matches(all_matches)
    
    # Let user select a match to analyze
    if all_matches:
        print("\n" + "="*80)
        print("Select a match to analyze (or press Enter to skip):")
        
        choice = input(f"\nEnter number (1-{len(all_matches)}): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(all_matches):
            match = all_matches[int(choice) - 1]
            analyze_live_match(match, fetcher)
    
    print("\n" + "="*80)
    print("📝 TO GET BETTING ODDS:")
    print("="*80)
    print("1. Get free API key: https://the-odds-api.com/")
    print("2. Update api_key in this file")
    print("3. Re-run for automatic odds fetching")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
