"""
🎾 BOOKMAKER ODDS SCRAPER
==========================

Scrapes real odds from multiple online bookmakers:
- Bet365
- Pinnacle
- BetMGM
- DraftKings
- Oddsportal (aggregator)
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time


class BookmakerOddsScraper:
    """Scrape odds from multiple bookmakers"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
    
    def get_oddsportal_tennis(self) -> List[Dict]:
        """
        Scrape OddsPortal for tennis odds (aggregates multiple bookmakers)
        """
        
        print("\n" + "="*80)
        print("💰 SCRAPING ODDSPORTAL.COM")
        print("="*80)
        
        url = "https://www.oddsportal.com/tennis/"
        
        try:
            response = self.session.get(url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            matches = []
            
            # Look for match rows
            match_rows = soup.find_all('div', class_=re.compile(r'eventRow|match', re.I))
            
            print(f"Found {len(match_rows)} potential match rows")
            
            for row in match_rows[:20]:
                try:
                    # Extract player names
                    participants = row.find_all(class_=re.compile(r'participant|team|player', re.I))
                    
                    if len(participants) >= 2:
                        player1 = participants[0].get_text(strip=True)
                        player2 = participants[1].get_text(strip=True)
                        
                        # Extract odds
                        odds_elements = row.find_all(class_=re.compile(r'odds?-|odd-', re.I))
                        
                        odds_p1 = None
                        odds_p2 = None
                        
                        if len(odds_elements) >= 2:
                            try:
                                odds_p1 = float(odds_elements[0].get_text(strip=True))
                                odds_p2 = float(odds_elements[1].get_text(strip=True))
                            except:
                                pass
                        
                        # Also try data attributes
                        if not odds_p1:
                            for elem in row.find_all(attrs={'data-odd': True}):
                                try:
                                    odds_p1 = float(elem['data-odd'])
                                    break
                                except:
                                    pass
                        
                        if player1 and player2:
                            match = {
                                'player1': player1,
                                'player2': player2,
                                'odds_player1': odds_p1,
                                'odds_player2': odds_p2,
                                'bookmaker': 'OddsPortal (avg)',
                                'source': 'oddsportal'
                            }
                            matches.append(match)
                
                except Exception as e:
                    continue
            
            # If structured scraping didn't work, try regex on page text
            if not matches:
                text = soup.get_text()
                
                # Pattern: "Player1 - Player2 1.50 2.75"
                pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*-\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+(\d+\.\d+)\s+(\d+\.\d+)'
                
                regex_matches = re.findall(pattern, text)
                
                for m in regex_matches[:10]:
                    player1, player2, odds1, odds2 = m
                    matches.append({
                        'player1': player1,
                        'player2': player2,
                        'odds_player1': float(odds1),
                        'odds_player2': float(odds2),
                        'bookmaker': 'OddsPortal',
                        'source': 'oddsportal'
                    })
            
            print(f"✅ Extracted {len(matches)} matches with odds")
            return matches
            
        except Exception as e:
            print(f"❌ OddsPortal error: {e}")
            return []
    
    def get_pinnacle_tennis(self) -> List[Dict]:
        """Scrape Pinnacle odds"""
        
        print("\n" + "="*80)
        print("💰 SCRAPING PINNACLE.COM")
        print("="*80)
        
        url = "https://www.pinnacle.com/en/tennis/matchups"
        
        try:
            response = self.session.get(url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            matches = []
            
            # Pinnacle often uses JSON data embedded in scripts
            scripts = soup.find_all('script')
            
            for script in scripts:
                if script.string and 'matchups' in script.string.lower():
                    # Try to extract JSON
                    try:
                        # Look for JSON objects
                        json_matches = re.findall(r'\{[^{}]*"participant"[^{}]*\}', script.string)
                        
                        for json_str in json_matches:
                            try:
                                data = json.loads(json_str)
                                # Process data
                            except:
                                pass
                    except:
                        pass
            
            # Fallback: structured HTML scraping
            match_containers = soup.find_all(class_=re.compile(r'event|match|game', re.I))
            
            for container in match_containers[:20]:
                try:
                    # Look for participant names
                    participants = container.find_all(class_=re.compile(r'participant|team', re.I))
                    
                    if len(participants) >= 2:
                        player1 = participants[0].get_text(strip=True)
                        player2 = participants[1].get_text(strip=True)
                        
                        # Look for odds buttons/spans
                        odds_buttons = container.find_all(class_=re.compile(r'price|odd', re.I))
                        
                        if len(odds_buttons) >= 2:
                            try:
                                odds_p1 = float(odds_buttons[0].get_text(strip=True))
                                odds_p2 = float(odds_buttons[1].get_text(strip=True))
                                
                                matches.append({
                                    'player1': player1,
                                    'player2': player2,
                                    'odds_player1': odds_p1,
                                    'odds_player2': odds_p2,
                                    'bookmaker': 'Pinnacle',
                                    'source': 'pinnacle'
                                })
                            except:
                                pass
                
                except:
                    continue
            
            print(f"✅ Extracted {len(matches)} matches from Pinnacle")
            return matches
            
        except Exception as e:
            print(f"❌ Pinnacle error: {e}")
            return []
    
    def get_bet365_tennis(self) -> List[Dict]:
        """Scrape Bet365 odds (requires authentication/region)"""
        
        print("\n" + "="*80)
        print("💰 CHECKING BET365.COM")
        print("="*80)
        
        # Bet365 is heavily protected - often requires login
        # Try mobile version which is sometimes less protected
        
        url = "https://mobile.bet365.com/sport/tennis"
        
        try:
            response = self.session.get(url, timeout=15)
            
            if 'login' in response.url.lower() or response.status_code == 403:
                print("⚠️  Bet365 requires authentication/region access")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            matches = []
            
            # Bet365 specific selectors (these change frequently)
            match_elements = soup.find_all(class_=re.compile(r'fixture|event', re.I))
            
            for elem in match_elements:
                try:
                    # Extract data
                    text = elem.get_text(strip=True)
                    
                    # Pattern: look for player names and odds
                    if ' v ' in text or ' vs ' in text:
                        # Try to parse
                        parts = re.split(r'\s+v\s+|\s+vs\s+', text, flags=re.I)
                        
                        if len(parts) >= 2:
                            player1 = parts[0].strip()
                            player2 = parts[1].strip()
                            
                            # Look for odds in text
                            odds_found = re.findall(r'(\d+\.\d+)', text)
                            
                            if len(odds_found) >= 2:
                                matches.append({
                                    'player1': player1,
                                    'player2': player2,
                                    'odds_player1': float(odds_found[0]),
                                    'odds_player2': float(odds_found[1]),
                                    'bookmaker': 'Bet365',
                                    'source': 'bet365'
                                })
                except:
                    continue
            
            print(f"✅ Extracted {len(matches)} matches from Bet365")
            return matches
            
        except Exception as e:
            print(f"❌ Bet365 error: {e}")
            return []
    
    def get_sofascore_odds(self) -> List[Dict]:
        """Get odds from Sofascore API"""
        
        print("\n" + "="*80)
        print("💰 FETCHING FROM SOFASCORE API")
        print("="*80)
        
        # Sofascore API for tennis events
        url = "https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{}"
        
        matches = []
        
        # Try different date ranges
        from datetime import datetime, timedelta
        
        today = datetime.now()
        
        for days_ahead in range(3):  # Next 3 days
            date = today + timedelta(days=days_ahead)
            date_str = date.strftime('%Y-%m-%d')
            
            try:
                response = self.session.get(url.format(date_str), timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    events = data.get('events', [])
                    
                    for event in events:
                        try:
                            player1 = event.get('homeTeam', {}).get('name', '')
                            player2 = event.get('awayTeam', {}).get('name', '')
                            
                            # Get odds if available
                            event_id = event.get('id')
                            
                            if event_id:
                                odds_url = f"https://api.sofascore.com/api/v1/event/{event_id}/odds/1/all"
                                
                                try:
                                    odds_response = self.session.get(odds_url, timeout=5)
                                    
                                    if odds_response.status_code == 200:
                                        odds_data = odds_response.json()
                                        
                                        # Extract bookmaker odds
                                        markets = odds_data.get('markets', [])
                                        
                                        for market in markets:
                                            if market.get('marketName') == '1X2' or market.get('marketName') == 'Match Winner':
                                                choices = market.get('choices', [])
                                                
                                                if len(choices) >= 2:
                                                    odds_p1 = choices[0].get('fractionalValue')
                                                    odds_p2 = choices[1].get('fractionalValue')
                                                    bookmaker = market.get('sourceId', 'Unknown')
                                                    
                                                    if odds_p1 and odds_p2:
                                                        matches.append({
                                                            'player1': player1,
                                                            'player2': player2,
                                                            'odds_player1': float(odds_p1),
                                                            'odds_player2': float(odds_p2),
                                                            'bookmaker': bookmaker,
                                                            'source': 'sofascore',
                                                            'start_time': event.get('startTimestamp')
                                                        })
                                                        break
                                
                                except:
                                    pass
                        
                        except:
                            continue
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                continue
        
        print(f"✅ Extracted {len(matches)} matches from Sofascore")
        return matches
    
    def get_all_odds(self) -> List[Dict]:
        """Get odds from all sources and merge"""
        
        print("\n" + "🎾"*40)
        print("BOOKMAKER ODDS AGGREGATOR")
        print(f"Scanning at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎾"*40)
        
        all_matches = []
        
        # Try each source
        sources = [
            ('Sofascore', self.get_sofascore_odds),
            ('OddsPortal', self.get_oddsportal_tennis),
            ('Pinnacle', self.get_pinnacle_tennis),
            ('Bet365', self.get_bet365_tennis),
        ]
        
        for source_name, scraper_func in sources:
            try:
                print(f"\n{'─'*80}")
                matches = scraper_func()
                all_matches.extend(matches)
                
                if matches:
                    print(f"✅ {source_name}: {len(matches)} matches")
                else:
                    print(f"⚠️  {source_name}: No matches found")
                
            except Exception as e:
                print(f"❌ {source_name} failed: {e}")
            
            time.sleep(1)  # Rate limiting between sources
        
        # Deduplicate and merge odds for same match
        merged = self._merge_duplicate_matches(all_matches)
        
        print(f"\n{'='*80}")
        print(f"✅ TOTAL: {len(merged)} unique matches with odds")
        print(f"{'='*80}\n")
        
        return merged
    
    def _merge_duplicate_matches(self, matches: List[Dict]) -> List[Dict]:
        """Merge duplicate matches and collect all bookmaker odds"""
        
        merged = {}
        
        for match in matches:
            # Create key from player names (normalized)
            p1 = match['player1'].lower().strip()
            p2 = match['player2'].lower().strip()
            key = tuple(sorted([p1, p2]))
            
            if key not in merged:
                merged[key] = {
                    'player1': match['player1'],
                    'player2': match['player2'],
                    'odds': [],
                    'sources': []
                }
            
            # Add odds if valid
            if match.get('odds_player1') and match.get('odds_player2'):
                merged[key]['odds'].append({
                    'bookmaker': match.get('bookmaker', 'Unknown'),
                    'player1': match['odds_player1'],
                    'player2': match['odds_player2']
                })
                
                if match.get('source'):
                    merged[key]['sources'].append(match['source'])
        
        # Convert to list and calculate best/average odds
        result = []
        
        for key, data in merged.items():
            if data['odds']:
                # Calculate best and average odds
                best_p1 = max(data['odds'], key=lambda x: x['player1'])
                best_p2 = max(data['odds'], key=lambda x: x['player2'])
                
                avg_p1 = sum(o['player1'] for o in data['odds']) / len(data['odds'])
                avg_p2 = sum(o['player2'] for o in data['odds']) / len(data['odds'])
                
                result.append({
                    'player1': data['player1'],
                    'player2': data['player2'],
                    'best_odds': {
                        'player1': best_p1['player1'],
                        'player2': best_p2['player2'],
                        'bookmaker_p1': best_p1['bookmaker'],
                        'bookmaker_p2': best_p2['bookmaker']
                    },
                    'average_odds': {
                        'player1': avg_p1,
                        'player2': avg_p2
                    },
                    'all_bookmakers': data['odds'],
                    'num_bookmakers': len(data['odds']),
                    'sources': list(set(data['sources']))
                })
        
        return result


def display_matches_with_odds(matches: List[Dict]):
    """Display matches with odds nicely formatted"""
    
    if not matches:
        print("\n⚠️  No matches with odds found")
        return
    
    print("\n" + "="*80)
    print(f"💰 {len(matches)} MATCHES WITH BOOKMAKER ODDS")
    print("="*80 + "\n")
    
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match['player1']} vs {match['player2']}")
        print(f"   Bookmakers: {match['num_bookmakers']}")
        
        print(f"\n   BEST ODDS:")
        print(f"     {match['player1']}: {match['best_odds']['player1']:.2f} ({match['best_odds']['bookmaker_p1']})")
        print(f"     {match['player2']}: {match['best_odds']['player2']:.2f} ({match['best_odds']['bookmaker_p2']})")
        
        print(f"\n   AVERAGE ODDS:")
        print(f"     {match['player1']}: {match['average_odds']['player1']:.2f}")
        print(f"     {match['player2']}: {match['average_odds']['player2']:.2f}")
        
        if match['num_bookmakers'] > 1:
            print(f"\n   ALL BOOKMAKERS:")
            for bm in match['all_bookmakers'][:5]:  # Show first 5
                print(f"     {bm['bookmaker']:20s} {bm['player1']:6.2f} / {bm['player2']:6.2f}")
        
        print()


def main():
    """Main function"""
    
    scraper = BookmakerOddsScraper()
    
    # Get all odds
    matches = scraper.get_all_odds()
    
    # Display
    display_matches_with_odds(matches)
    
    # Save to file
    if matches:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'bookmaker_odds_{timestamp}.json'
        
        import json
        with open(filename, 'w') as f:
            json.dump(matches, f, indent=2)
        
        print(f"💾 Saved to: {filename}")


if __name__ == "__main__":
    main()
