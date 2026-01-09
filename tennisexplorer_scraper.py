"""
🎾 TennisExplorer Live Data Integration
=======================================

Scrapes live and upcoming matches from tennisexplorer.com including:
- Live scores and match status
- Player statistics
- Betting odds from multiple bookmakers
- Both professional (ATP/WTA) and lower-level tournaments

Usage:
    python tennisexplorer_scraper.py
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
import re
import time
from datetime import datetime
import json


class TennisExplorerScraper:
    """Scrape live tennis data from tennisexplorer.com"""
    
    BASE_URL = "https://www.tennisexplorer.com"
    MATCHES_URL = f"{BASE_URL}/matches/"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
    
    def get_live_matches(self) -> List[Dict]:
        """Get all live matches"""
        
        print(f"\n{'='*80}")
        print(f"🔍 FETCHING LIVE MATCHES FROM TENNISEXPLORER.COM")
        print(f"{'='*80}\n")
        
        try:
            response = self.session.get(self.MATCHES_URL, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            matches = []
            
            # Find live matches section
            live_section = soup.find('div', {'id': 'live-matches'}) or soup.find('div', class_=re.compile(r'live'))
            
            if not live_section:
                # Try to find any match rows
                match_rows = soup.find_all('tr', class_=re.compile(r'(match|live)', re.I))
            else:
                match_rows = live_section.find_all('tr')
            
            print(f"Found {len(match_rows)} match row(s)\n")
            
            for row in match_rows:
                match_data = self._parse_match_row(row)
                if match_data:
                    matches.append(match_data)
            
            return matches
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching matches: {e}")
            return []
    
    def _parse_match_row(self, row) -> Optional[Dict]:
        """Parse a single match row"""
        try:
            # Extract player names
            players = row.find_all('a', class_=re.compile(r'player', re.I))
            if len(players) < 2:
                players = row.find_all('td', class_=re.compile(r'(player|name)', re.I))
            
            if len(players) >= 2:
                player1 = players[0].get_text(strip=True)
                player2 = players[1].get_text(strip=True)
            else:
                return None
            
            # Extract score
            score_elem = row.find('td', class_=re.compile(r'score', re.I)) or row.find('span', class_=re.compile(r'score', re.I))
            score = score_elem.get_text(strip=True) if score_elem else "0-0"
            
            # Extract tournament
            tournament_elem = row.find('td', class_=re.compile(r'(tournament|event)', re.I))
            tournament = tournament_elem.get_text(strip=True) if tournament_elem else "Unknown"
            
            # Extract match link
            match_link = row.find('a', href=re.compile(r'/match-detail/'))
            match_url = f"{self.BASE_URL}{match_link['href']}" if match_link else None
            
            # Status (live, upcoming, finished)
            status_elem = row.find('span', class_=re.compile(r'(status|live)', re.I))
            is_live = bool(status_elem and 'live' in status_elem.get_text(strip=True).lower())
            
            return {
                'player1': player1,
                'player2': player2,
                'score': score,
                'tournament': tournament,
                'match_url': match_url,
                'is_live': is_live,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return None
    
    def get_match_details(self, match_url: str) -> Optional[Dict]:
        """Get detailed match information including statistics and odds"""
        
        print(f"\n🔍 Fetching match details from: {match_url}")
        
        try:
            response = self.session.get(match_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            details = {
                'url': match_url,
                'statistics': self._extract_statistics(soup),
                'odds': self._extract_odds(soup),
                'score_detail': self._extract_score_detail(soup)
            }
            
            return details
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching match details: {e}")
            return None
    
    def _extract_statistics(self, soup) -> Dict:
        """Extract match statistics (aces, double faults, etc.)"""
        stats = {}
        
        # Look for statistics table
        stats_table = soup.find('table', class_=re.compile(r'stat', re.I))
        
        if stats_table:
            rows = stats_table.find_all('tr')
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    stat_name = cols[1].get_text(strip=True)
                    p1_value = cols[0].get_text(strip=True)
                    p2_value = cols[2].get_text(strip=True)
                    
                    stats[stat_name] = {
                        'player1': p1_value,
                        'player2': p2_value
                    }
        
        return stats
    
    def _extract_odds(self, soup) -> Dict:
        """Extract betting odds from various bookmakers"""
        odds = {}
        
        # Look for odds section
        odds_section = soup.find('div', {'id': 'odds'}) or soup.find('div', class_=re.compile(r'odds', re.I))
        
        if odds_section:
            # Find bookmaker odds
            odds_rows = odds_section.find_all('tr')
            
            for row in odds_rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    bookmaker = cols[0].get_text(strip=True)
                    p1_odds = cols[1].get_text(strip=True)
                    p2_odds = cols[2].get_text(strip=True)
                    
                    try:
                        odds[bookmaker] = {
                            'player1': float(p1_odds),
                            'player2': float(p2_odds)
                        }
                    except ValueError:
                        continue
        
        return odds
    
    def _extract_score_detail(self, soup) -> Dict:
        """Extract detailed score (sets, games, points)"""
        score_detail = {
            'sets': [],
            'current_set': None,
            'current_game': None
        }
        
        # Look for score detail
        score_elem = soup.find('div', class_=re.compile(r'score', re.I))
        
        if score_elem:
            # Extract set scores
            sets = score_elem.find_all('span', class_=re.compile(r'set', re.I))
            for s in sets:
                score_detail['sets'].append(s.get_text(strip=True))
        
        return score_detail
    
    def search_player_matches(self, player_name: str) -> List[Dict]:
        """Search for specific player's matches"""
        
        search_url = f"{self.BASE_URL}/search/"
        
        try:
            response = self.session.get(search_url, params={'q': player_name}, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse search results
            matches = []
            match_links = soup.find_all('a', href=re.compile(r'/match-detail/'))
            
            for link in match_links:
                match_url = f"{self.BASE_URL}{link['href']}"
                match_text = link.get_text(strip=True)
                
                matches.append({
                    'text': match_text,
                    'url': match_url
                })
            
            return matches
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error searching for player: {e}")
            return []


class LiveBettingIntegration:
    """Integrate TennisExplorer data with Markov model"""
    
    def __init__(self):
        self.scraper = TennisExplorerScraper()
        self.bankroll = 1000
        self.target = 5000
    
    def analyze_all_live_matches(self):
        """Analyze all live matches for betting opportunities"""
        
        matches = self.scraper.get_live_matches()
        
        if not matches:
            print("\n⚠️  No live matches found")
            print("💡 This could mean:")
            print("   - No matches currently in progress")
            print("   - Website structure changed (scraper needs update)")
            print("   - Connection/blocking issues")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 ANALYZING {len(matches)} LIVE MATCH(ES)")
        print(f"{'='*80}\n")
        
        opportunities = []
        
        for i, match in enumerate(matches, 1):
            print(f"\n{i}. {match['player1']} vs {match['player2']}")
            print(f"   Tournament: {match['tournament']}")
            print(f"   Score: {match['score']}")
            print(f"   Status: {'🔴 LIVE' if match['is_live'] else '⏸️  Upcoming'}")
            
            if match['match_url']:
                # Get detailed stats
                details = self.scraper.get_match_details(match['match_url'])
                
                if details and details['statistics']:
                    print(f"\n   📊 Statistics available:")
                    
                    # Extract serve stats if available
                    stats = details['statistics']
                    
                    # Try to find serve statistics
                    serve_stats = self._extract_serve_stats(stats)
                    
                    if serve_stats:
                        print(f"   ✅ Serve data found - calculating probabilities...")
                        
                        # Calculate with Markov model
                        from live_betting_assistant import LiveMatchAnalyzer
                        analyzer = LiveMatchAnalyzer()
                        
                        probs = analyzer.calculate_probabilities_from_stats(
                            serve_stats['player1'],
                            serve_stats['player2'],
                            match['player1'],
                            match['player2']
                        )
                        
                        # Check odds
                        if details['odds']:
                            best_odds = self._get_best_odds(details['odds'])
                            print(f"\n   💰 Best available odds:")
                            print(f"      {match['player1']}: {best_odds['player1']:.2f}")
                            print(f"      {match['player2']}: {best_odds['player2']:.2f}")
                            
                            # Calculate edge
                            p1_edge = (probs['p1_match'] * best_odds['player1']) - 1
                            p2_edge = (probs['p2_match'] * best_odds['player2']) - 1
                            
                            if p1_edge > 0.02:
                                opportunities.append({
                                    'match': f"{match['player1']} vs {match['player2']}",
                                    'bet': match['player1'],
                                    'odds': best_odds['player1'],
                                    'edge': p1_edge,
                                    'tournament': match['tournament']
                                })
                            
                            if p2_edge > 0.02:
                                opportunities.append({
                                    'match': f"{match['player1']} vs {match['player2']}",
                                    'bet': match['player2'],
                                    'odds': best_odds['player2'],
                                    'edge': p2_edge,
                                    'tournament': match['tournament']
                                })
                
                # Rate limiting
                time.sleep(2)
        
        # Show all opportunities
        if opportunities:
            self._display_opportunities(opportunities)
        else:
            print(f"\n{'='*80}")
            print("⏸️  No profitable opportunities found at current odds")
            print(f"{'='*80}\n")
    
    def _extract_serve_stats(self, statistics: Dict) -> Optional[Dict]:
        """Extract serve statistics from scraped data"""
        
        # Common stat names to look for
        serve_patterns = {
            'first_serve': ['1st Serve', 'First Serve', '1st serve %', 'Serve 1st in'],
            'first_win': ['1st Serve Points Won', '1st serve won', 'Win on 1st'],
            'second_win': ['2nd Serve Points Won', '2nd serve won', 'Win on 2nd']
        }
        
        serve_stats = {
            'player1': {},
            'player2': {}
        }
        
        for stat_name, stat_data in statistics.items():
            # Check each pattern
            for key, patterns in serve_patterns.items():
                if any(pattern.lower() in stat_name.lower() for pattern in patterns):
                    try:
                        # Extract percentage
                        p1_val = stat_data['player1']
                        p2_val = stat_data['player2']
                        
                        # Convert to float
                        p1_pct = float(re.findall(r'(\d+(?:\.\d+)?)', p1_val)[0]) if re.findall(r'(\d+(?:\.\d+)?)', p1_val) else None
                        p2_pct = float(re.findall(r'(\d+(?:\.\d+)?)', p2_val)[0]) if re.findall(r'(\d+(?:\.\d+)?)', p2_val) else None
                        
                        if p1_pct and p2_pct:
                            serve_stats['player1'][key] = p1_pct
                            serve_stats['player2'][key] = p2_pct
                    except:
                        continue
        
        # Check if we have minimum required stats
        required = ['first_serve', 'first_win', 'second_win']
        if all(k in serve_stats['player1'] for k in required):
            return {
                'player1': {
                    'first_serve_pct': serve_stats['player1']['first_serve'],
                    'first_serve_win_pct': serve_stats['player1']['first_win'],
                    'second_serve_win_pct': serve_stats['player1']['second_win']
                },
                'player2': {
                    'first_serve_pct': serve_stats['player2']['first_serve'],
                    'first_serve_win_pct': serve_stats['player2']['first_win'],
                    'second_serve_win_pct': serve_stats['player2']['second_win']
                }
            }
        
        return None
    
    def _get_best_odds(self, odds_dict: Dict) -> Dict:
        """Get best available odds from multiple bookmakers"""
        best = {'player1': 0, 'player2': 0}
        
        for bookmaker, odds in odds_dict.items():
            if odds['player1'] > best['player1']:
                best['player1'] = odds['player1']
            if odds['player2'] > best['player2']:
                best['player2'] = odds['player2']
        
        return best
    
    def _display_opportunities(self, opportunities: List[Dict]):
        """Display all betting opportunities"""
        
        print(f"\n{'='*80}")
        print(f"💸 PROFITABLE BETTING OPPORTUNITIES FOUND")
        print(f"{'='*80}\n")
        
        opportunities.sort(key=lambda x: x['edge'], reverse=True)
        
        for i, opp in enumerate(opportunities, 1):
            kelly = (opp['edge'] / (opp['odds'] - 1)) * 0.25
            stake = min(self.bankroll * kelly, self.bankroll * 0.15)
            potential = stake * (opp['odds'] - 1)
            
            print(f"{i}. {opp['match']}")
            print(f"   Tournament: {opp['tournament']}")
            print(f"   Bet: {opp['bet']}")
            print(f"   Odds: {opp['odds']:.2f}")
            print(f"   Edge: {opp['edge']*100:+.1f}%")
            print(f"   Recommended stake: ${stake:.2f}")
            print(f"   Potential profit: ${potential:.2f}\n")
        
        total_stakes = sum(min(self.bankroll * ((o['edge'] / (o['odds'] - 1)) * 0.25), self.bankroll * 0.15) for o in opportunities)
        
        print(f"{'─'*80}")
        print(f"Total recommended stakes: ${total_stakes:.2f}")
        print(f"{'='*80}\n")


def main():
    """Main entry point"""
    
    print("\n🎾 TENNISEXPLORER.COM LIVE DATA INTEGRATION")
    print("="*80)
    
    integrator = LiveBettingIntegration()
    
    print("\nOptions:")
    print("1. Scan all live matches for opportunities")
    print("2. Search specific player")
    print("3. Test scraper")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        integrator.analyze_all_live_matches()
    
    elif choice == "2":
        player_name = input("\nEnter player name: ").strip()
        
        scraper = TennisExplorerScraper()
        matches = scraper.search_player_matches(player_name)
        
        if matches:
            print(f"\n✅ Found {len(matches)} match(es):")
            for i, match in enumerate(matches, 1):
                print(f"\n{i}. {match['text']}")
                print(f"   URL: {match['url']}")
        else:
            print(f"\n❌ No matches found for '{player_name}'")
    
    elif choice == "3":
        print("\n🧪 Testing TennisExplorer scraper...")
        
        scraper = TennisExplorerScraper()
        matches = scraper.get_live_matches()
        
        print(f"\n✅ Scraper working! Found {len(matches)} match(es)")
        
        for match in matches[:5]:  # Show first 5
            print(f"\n{match['player1']} vs {match['player2']}")
            print(f"   Score: {match['score']}")
            print(f"   Tournament: {match['tournament']}")
            print(f"   Live: {match['is_live']}")


if __name__ == "__main__":
    main()
