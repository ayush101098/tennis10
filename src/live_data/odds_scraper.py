"""
Odds Scraper - Collect betting odds from multiple sources

Options:
1. The Odds API (Recommended - Paid but reliable)
2. OddsPortal scraping (Free but may be blocked)
3. Betfair API (Exchange odds)
"""

import os
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)


class OddsScraper:
    """Multi-source odds collection"""
    
    def __init__(self, odds_api_key: Optional[str] = None):
        self.odds_api_key = odds_api_key or os.environ.get('ODDS_API_KEY')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def get_all_odds(self, use_api: bool = True) -> pd.DataFrame:
        """
        Get odds from all available sources
        
        Args:
            use_api: If True, use The Odds API (requires API key)
                    If False, try web scraping
        
        Returns:
            DataFrame with all odds
        """
        
        all_odds = []
        
        if use_api and self.odds_api_key:
            # Use The Odds API (recommended)
            try:
                api_odds = self.get_odds_api()
                all_odds.extend(api_odds)
                logger.info(f"✅ The Odds API: {len(api_odds)} matches")
            except Exception as e:
                logger.error(f"❌ The Odds API failed: {e}")
        
        # Fallback to scraping if API not available
        if not all_odds:
            try:
                scraped_odds = self.scrape_oddsportal()
                all_odds.extend(scraped_odds)
                logger.info(f"✅ OddsPortal: {len(scraped_odds)} matches")
            except Exception as e:
                logger.error(f"❌ OddsPortal scraping failed: {e}")
        
        if not all_odds:
            logger.warning("No odds data collected from any source")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_odds)
        
        # Deduplicate and find best odds
        df = self._process_odds(df)
        
        return df
    
    def get_odds_api(self) -> List[Dict]:
        """
        Get odds from The Odds API
        
        Docs: https://the-odds-api.com/liveapi/guides/v4/
        
        Returns:
            List of odds dictionaries
        """
        
        if not self.odds_api_key:
            raise ValueError("The Odds API key not set. Set ODDS_API_KEY environment variable.")
        
        logger.info("Fetching odds from The Odds API...")
        
        base_url = "https://api.the-odds-api.com/v4/sports"
        
        # Tennis sports
        sports = [
            'tennis_atp',
            'tennis_wta',
            'tennis_atp_us_open',
            'tennis_atp_french_open',
            'tennis_atp_wimbledon',
            'tennis_atp_australian_open',
        ]
        
        all_odds = []
        
        for sport in sports:
            url = f"{base_url}/{sport}/odds/"
            
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us,uk,eu,au',
                'markets': 'h2h',
                'oddsFormat': 'decimal',
                'dateFormat': 'iso'
            }
            
            try:
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    for event in data:
                        odds_list = self._parse_odds_api_event(event)
                        all_odds.extend(odds_list)
                
                elif response.status_code == 401:
                    raise ValueError("Invalid API key")
                
                elif response.status_code == 429:
                    logger.warning("Rate limit reached")
                    break
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching {sport}: {e}")
        
        return all_odds
    
    def _parse_odds_api_event(self, event: Dict) -> List[Dict]:
        """Parse event from The Odds API"""
        
        match_id = f"odds_api_{event['id']}"
        player1 = event['home_team']
        player2 = event['away_team']
        commence_time = event['commence_time']
        
        odds_list = []
        
        for bookmaker in event.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':
                    outcomes = market['outcomes']
                    
                    if len(outcomes) >= 2:
                        p1_odds = next((o['price'] for o in outcomes if o['name'] == player1), None)
                        p2_odds = next((o['price'] for o in outcomes if o['name'] == player2), None)
                        
                        if p1_odds and p2_odds:
                            odds_list.append({
                                'match_id': match_id,
                                'player1_name': player1,
                                'player2_name': player2,
                                'bookmaker': bookmaker['title'],
                                'player1_odds': p1_odds,
                                'player2_odds': p2_odds,
                                'timestamp': datetime.fromisoformat(commence_time.replace('Z', '+00:00')),
                                'data_source': 'odds_api'
                            })
        
        return odds_list
    
    def scrape_oddsportal(self) -> List[Dict]:
        """
        Scrape OddsPortal for tennis odds
        
        Note: This is a simplified implementation
        Real OddsPortal scraping requires handling dynamic content
        
        Returns:
            List of odds dictionaries
        """
        
        logger.info("Scraping OddsPortal (may be slow/blocked)...")
        
        url = "https://www.oddsportal.com/tennis/"
        
        odds_list = []
        
        try:
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # Parse HTML (simplified - real implementation needs Selenium/Playwright)
                # OddsPortal has complex JS rendering
                
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find match rows
                matches = soup.find_all('div', class_='event-row')
                
                for match in matches[:10]:  # Limit to first 10
                    try:
                        # Extract data (this is approximate)
                        player_names = match.find_all('div', class_='participant-name')
                        odds_cells = match.find_all('div', class_='odds-cell')
                        
                        if len(player_names) >= 2 and len(odds_cells) >= 2:
                            odds_list.append({
                                'match_id': f"oddsportal_{len(odds_list)}",
                                'player1_name': player_names[0].text.strip(),
                                'player2_name': player_names[1].text.strip(),
                                'bookmaker': 'Pinnacle',  # Default
                                'player1_odds': float(odds_cells[0].text.strip()),
                                'player2_odds': float(odds_cells[1].text.strip()),
                                'timestamp': datetime.now(),
                                'data_source': 'oddsportal'
                            })
                    except:
                        continue
        
        except Exception as e:
            logger.error(f"OddsPortal scraping error: {e}")
        
        return odds_list
    
    def _process_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process odds DataFrame:
        - Deduplicate matches
        - Find best odds for each match
        - Calculate average odds
        
        Args:
            df: Raw odds DataFrame
        
        Returns:
            Processed DataFrame
        """
        
        if df.empty:
            return df
        
        # Group by match
        df['player_pair'] = df.apply(
            lambda row: tuple(sorted([row['player1_name'], row['player2_name']])),
            axis=1
        )
        
        # For each match, find best odds
        best_odds = []
        
        for player_pair, group in df.groupby('player_pair'):
            # Get best odds for each player across all bookmakers
            best_p1 = group['player1_odds'].max()
            best_p2 = group['player2_odds'].max()
            
            best_p1_bookie = group.loc[group['player1_odds'].idxmax(), 'bookmaker']
            best_p2_bookie = group.loc[group['player2_odds'].idxmax(), 'bookmaker']
            
            avg_p1 = group['player1_odds'].mean()
            avg_p2 = group['player2_odds'].mean()
            
            # Take first match info
            first = group.iloc[0]
            
            best_odds.append({
                'match_id': first['match_id'],
                'player1_name': first['player1_name'],
                'player2_name': first['player2_name'],
                'best_player1_odds': best_p1,
                'best_player2_odds': best_p2,
                'best_player1_bookmaker': best_p1_bookie,
                'best_player2_bookmaker': best_p2_bookie,
                'avg_player1_odds': avg_p1,
                'avg_player2_odds': avg_p2,
                'num_bookmakers': len(group),
                'timestamp': first['timestamp'],
                'data_source': first['data_source']
            })
        
        return pd.DataFrame(best_odds)


# Convenience functions

def get_tennis_odds(use_api: bool = True, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Get current tennis odds
    
    Args:
        use_api: Use The Odds API if True, else scrape
        api_key: The Odds API key (or set ODDS_API_KEY env var)
    
    Returns:
        DataFrame with best odds for each match
    """
    scraper = OddsScraper(odds_api_key=api_key)
    return scraper.get_all_odds(use_api=use_api)


def scrape_oddsportal() -> pd.DataFrame:
    """
    Scrape OddsPortal for odds
    
    Returns:
        DataFrame with odds
    """
    scraper = OddsScraper()
    odds = scraper.scrape_oddsportal()
    return pd.DataFrame(odds)


if __name__ == "__main__":
    print("🎾 Testing Odds Scraper\n")
    
    # Check if API key is set
    api_key = os.environ.get('ODDS_API_KEY')
    
    if api_key:
        print("✅ The Odds API key found")
        print("Fetching odds from The Odds API...\n")
        
        odds = get_tennis_odds(use_api=True)
        
        print(f"📊 Found {len(odds)} matches with odds\n")
        
        if not odds.empty:
            print(odds[['player1_name', 'player2_name', 'best_player1_odds', 
                       'best_player2_odds', 'num_bookmakers']].head(10))
            
            # Save to CSV
            odds.to_csv('current_odds.csv', index=False)
            print("\n💾 Saved to current_odds.csv")
    
    else:
        print("⚠️  No API key found")
        print("\nTo use The Odds API:")
        print("1. Visit: https://the-odds-api.com/")
        print("2. Sign up for free account")
        print("3. Copy your API key")
        print("4. Run: export ODDS_API_KEY='your_key'")
        print("\nAttempting OddsPortal scraping as fallback...")
        
        odds = get_tennis_odds(use_api=False)
        
        if not odds.empty:
            print(f"\n✅ Found {len(odds)} matches")
            print(odds.head())
        else:
            print("\n❌ No odds found. Please set up The Odds API.")
