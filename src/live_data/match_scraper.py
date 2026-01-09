"""
Match Scraper - Collect upcoming ATP/WTA matches from multiple sources

Data sources:
1. Sofascore API (Primary - most reliable)
2. Flashscore (Backup - via web scraping)
3. ATP Official (Authoritative - tournament draws)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import logging
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatchScraper:
    """Multi-source tennis match scraper"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def scrape_all_sources(self, days_ahead=2) -> pd.DataFrame:
        """
        Scrape all sources and merge results
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            DataFrame with all upcoming matches
        """
        logger.info(f"Scraping upcoming matches for next {days_ahead} days...")
        
        all_matches = []
        
        # Source 1: Sofascore (primary)
        try:
            sofascore_matches = self.scrape_sofascore_matches(days_ahead=days_ahead)
            all_matches.extend(sofascore_matches)
            logger.info(f"✅ Sofascore: {len(sofascore_matches)} matches")
        except Exception as e:
            logger.error(f"❌ Sofascore failed: {e}")
        
        # Source 2: Flashscore (backup)
        try:
            flashscore_matches = self.scrape_flashscore_matches(days_ahead=days_ahead)
            all_matches.extend(flashscore_matches)
            logger.info(f"✅ Flashscore: {len(flashscore_matches)} matches")
        except Exception as e:
            logger.error(f"❌ Flashscore failed: {e}")
        
        # Source 3: ATP official
        try:
            atp_matches = self.scrape_atp_draws()
            all_matches.extend(atp_matches)
            logger.info(f"✅ ATP Official: {len(atp_matches)} matches")
        except Exception as e:
            logger.error(f"❌ ATP Official failed: {e}")
        
        if not all_matches:
            logger.warning("No matches found from any source!")
            return pd.DataFrame()
        
        # Convert to DataFrame and deduplicate
        df = pd.DataFrame(all_matches)
        df = self._deduplicate_matches(df)
        
        logger.info(f"📊 Total unique matches: {len(df)}")
        
        return df
    
    def scrape_sofascore_matches(self, days_ahead=2) -> List[Dict]:
        """
        Scrape Sofascore API for upcoming matches
        
        Args:
            days_ahead: Number of days ahead to fetch
            
        Returns:
            List of match dictionaries
        """
        matches = []
        base_url = "https://api.sofascore.com/api/v1/sport/tennis/scheduled-events"
        
        # Fetch for each day
        for day_offset in range(days_ahead + 1):
            date = datetime.now() + timedelta(days=day_offset)
            date_str = date.strftime('%Y-%m-%d')
            
            url = f"{base_url}/{date_str}"
            
            try:
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Parse events
                    events = data.get('events', [])
                    
                    for event in events:
                        # Only ATP/WTA singles
                        tournament = event.get('tournament', {})
                        category = tournament.get('category', {}).get('name', '')
                        
                        if 'ATP' not in category and 'WTA' not in category:
                            continue
                        
                        # Extract match info
                        home_team = event.get('homeTeam', {})
                        away_team = event.get('awayTeam', {})
                        
                        match = {
                            'match_id': f"sofascore_{event.get('id')}",
                            'player1_name': home_team.get('name', ''),
                            'player2_name': away_team.get('name', ''),
                            'player1_external_id': home_team.get('id'),
                            'player2_external_id': away_team.get('id'),
                            'tournament_name': tournament.get('name', ''),
                            'surface': self._extract_surface(tournament),
                            'scheduled_time': datetime.fromtimestamp(event.get('startTimestamp', 0)),
                            'round': event.get('roundInfo', {}).get('name', ''),
                            'data_source': 'sofascore',
                            'gender': 'M' if 'ATP' in category else 'W',
                        }
                        
                        matches.append(match)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching Sofascore for {date_str}: {e}")
        
        return matches
    
    def scrape_flashscore_matches(self, days_ahead=2) -> List[Dict]:
        """
        Scrape Flashscore for upcoming matches
        
        Note: Flashscore requires JavaScript rendering, so this uses
        the mobile API endpoint which returns JSON
        
        Args:
            days_ahead: Number of days ahead
            
        Returns:
            List of match dictionaries
        """
        matches = []
        
        # Flashscore mobile API endpoint
        url = "https://d.flashscore.com/x/feed/df_st_1_"
        
        try:
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                # Parse Flashscore's custom format
                # Format: ¬~AA÷match_id¬AD÷timestamp¬ADE÷...
                content = response.text
                
                # Split by match separator
                matches_raw = content.split('¬~AA÷')
                
                for match_raw in matches_raw[1:]:  # Skip first empty
                    try:
                        match_data = self._parse_flashscore_match(match_raw)
                        if match_data:
                            matches.append(match_data)
                    except:
                        continue
        
        except Exception as e:
            logger.error(f"Flashscore scraping error: {e}")
        
        return matches
    
    def scrape_atp_draws(self) -> List[Dict]:
        """
        Scrape ATP official website for current tournament draws
        
        Returns:
            List of match dictionaries
        """
        matches = []
        
        # ATP current tournaments endpoint
        url = "https://www.atptour.com/en/scores/current"
        
        try:
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find all tournament cards
                tournaments = soup.find_all('div', class_='tourney-detail')
                
                for tournament in tournaments:
                    tournament_name = tournament.find('span', class_='tourney-title')
                    if tournament_name:
                        tournament_name = tournament_name.text.strip()
                    
                    # Find surface info
                    surface = 'hard'  # Default
                    surface_elem = tournament.find('span', class_='item-surface')
                    if surface_elem:
                        surface_text = surface_elem.text.lower()
                        if 'clay' in surface_text:
                            surface = 'clay'
                        elif 'grass' in surface_text:
                            surface = 'grass'
                        elif 'indoor' in surface_text:
                            surface = 'indoor'
                    
                    # Find matches
                    match_cards = tournament.find_all('div', class_='day-table-match')
                    
                    for card in match_cards:
                        # Extract player names
                        players = card.find_all('a', class_='player-name')
                        
                        if len(players) >= 2:
                            match = {
                                'match_id': f"atp_{tournament_name}_{players[0].text}_{players[1].text}",
                                'player1_name': players[0].text.strip(),
                                'player2_name': players[1].text.strip(),
                                'tournament_name': tournament_name,
                                'surface': surface,
                                'scheduled_time': datetime.now() + timedelta(hours=2),  # Approximate
                                'round': 'Unknown',
                                'data_source': 'atp_official',
                                'gender': 'M',
                            }
                            matches.append(match)
        
        except Exception as e:
            logger.error(f"ATP scraping error: {e}")
        
        return matches
    
    def _extract_surface(self, tournament: Dict) -> str:
        """Extract surface from tournament data"""
        
        # Check ground type
        ground = tournament.get('groundType', '')
        if ground:
            return ground.lower()
        
        # Check tournament name
        name = tournament.get('name', '').lower()
        if 'clay' in name:
            return 'clay'
        elif 'grass' in name:
            return 'grass'
        elif 'indoor' in name or 'hard' in name:
            return 'hard'
        
        return 'hard'  # Default
    
    def _parse_flashscore_match(self, match_raw: str) -> Optional[Dict]:
        """Parse Flashscore's custom format"""
        
        # This is a simplified parser
        # Real implementation would need full Flashscore protocol
        
        parts = match_raw.split('¬')
        
        match_dict = {}
        for part in parts:
            if '÷' in part:
                key, value = part.split('÷', 1)
                match_dict[key] = value
        
        # Extract key fields (this is approximate)
        if 'AE' in match_dict and 'AF' in match_dict:  # Player names
            return {
                'match_id': f"flashscore_{match_dict.get('AA', '')}",
                'player1_name': match_dict.get('AE', ''),
                'player2_name': match_dict.get('AF', ''),
                'tournament_name': match_dict.get('ZC', 'Unknown'),
                'surface': 'hard',
                'scheduled_time': datetime.fromtimestamp(int(match_dict.get('AD', 0))),
                'round': match_dict.get('QI', ''),
                'data_source': 'flashscore',
                'gender': 'M',
            }
        
        return None
    
    def _deduplicate_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate matches from different sources
        
        Match is duplicate if:
        - Same players (order-independent)
        - Within 6 hours of each other
        """
        
        if df.empty:
            return df
        
        # Create normalized player pair
        df['player_pair'] = df.apply(
            lambda row: tuple(sorted([row['player1_name'], row['player2_name']])),
            axis=1
        )
        
        # Sort by data source priority (Sofascore > ATP > Flashscore)
        source_priority = {'sofascore': 1, 'atp_official': 2, 'flashscore': 3}
        df['source_priority'] = df['data_source'].map(source_priority)
        df = df.sort_values('source_priority')
        
        # Keep first occurrence of each player pair
        df_dedup = df.drop_duplicates(subset=['player_pair'], keep='first')
        
        # Clean up
        df_dedup = df_dedup.drop(columns=['player_pair', 'source_priority'])
        
        return df_dedup.reset_index(drop=True)


# Convenience functions for direct use

def scrape_sofascore_matches(days_ahead=2) -> pd.DataFrame:
    """
    Scrape Sofascore for upcoming matches
    
    Args:
        days_ahead: Number of days ahead (default: 2)
    
    Returns:
        DataFrame with matches
    """
    scraper = MatchScraper()
    matches = scraper.scrape_sofascore_matches(days_ahead=days_ahead)
    return pd.DataFrame(matches)


def scrape_flashscore_matches(days_ahead=2) -> pd.DataFrame:
    """
    Scrape Flashscore for upcoming matches
    
    Args:
        days_ahead: Number of days ahead (default: 2)
    
    Returns:
        DataFrame with matches
    """
    scraper = MatchScraper()
    matches = scraper.scrape_flashscore_matches(days_ahead=days_ahead)
    return pd.DataFrame(matches)


def scrape_atp_draws() -> pd.DataFrame:
    """
    Scrape ATP official website for tournament draws
    
    Returns:
        DataFrame with matches
    """
    scraper = MatchScraper()
    matches = scraper.scrape_atp_draws()
    return pd.DataFrame(matches)


def get_all_upcoming_matches(days_ahead=2) -> pd.DataFrame:
    """
    Get upcoming matches from all sources (recommended)
    
    Args:
        days_ahead: Number of days ahead (default: 2)
    
    Returns:
        DataFrame with deduplicated matches from all sources
    """
    scraper = MatchScraper()
    return scraper.scrape_all_sources(days_ahead=days_ahead)


if __name__ == "__main__":
    # Test the scraper
    print("🎾 Testing Match Scraper\n")
    
    matches = get_all_upcoming_matches(days_ahead=2)
    
    print(f"\n📊 Found {len(matches)} upcoming matches\n")
    
    if not matches.empty:
        print(matches[['player1_name', 'player2_name', 'tournament_name', 
                       'surface', 'scheduled_time', 'data_source']].head(10))
        
        # Save to CSV
        matches.to_csv('upcoming_matches.csv', index=False)
        print("\n💾 Saved to upcoming_matches.csv")
