# Tennis APIs & Data Sources - Quick Integration Guide

## Summary of All Available Data Sources

| Source | Type | Coverage | Cost | Latency | Key Metrics |
|--------|------|----------|------|---------|------------|
| tennis-data.co.uk | Historical | ATP/WTA 2020+ | Free | N/A | Match results, odds, rankings |
| api-tennis.com | Live API | All tours | Paid | <1s | Live scores, odds, fixtures |
| TennisRatio.com | Web Scraping | All players | Free | 5-10s | Advanced stats, pressure points |
| SofaScore.com | Web Scraping | All matches | Free | 2-5s | Live match data, weather |
| ESPN.com | Web Scraping | Major tours | Free | 5-10s | Match info, player stats |
| Match Charting Project | Database | ATP/WTA sample | Free | N/A | Rally-level data |
| Betfair | API | In-play odds | Paid | <1s | Exchange odds, volume data |

---

## TIER 1: High-Impact Integrations (Start Here)

### 1. api-tennis.com - Live Odds & Scores

#### Setup
```python
# requirements.txt
requests>=2.31.0
python-dotenv>=1.0.0

# .env
API_TENNIS_KEY=your_api_key
```

#### Get Live Matches
```python
import requests
import os

api_key = os.environ.get("API_TENNIS_KEY")
base_url = "https://api.api-tennis.com/tennis/"

# Get all live matches
def get_live_matches():
    params = {
        "method": "events_live",
        "APIkey": api_key
    }
    response = requests.get(base_url, params=params, timeout=10)
    data = response.json()
    
    live_matches = []
    for event in data.get('result', []):
        if event['sport_event']['sport_name'] == 'Tennis':
            live_matches.append({
                'id': event['event_id'],
                'player1': event['sport_event']['competitors'][0]['name'],
                'player2': event['sport_event']['competitors'][1]['name'],
                'score': event['sport_event_status']['match_status'],
                'current_set': event['sport_event_status']['period'],
            })
    
    return live_matches

# Get fixtures for specific date
def get_fixtures(date_str):  # "2024-09-06"
    params = {
        "method": "fixtures",
        "APIkey": api_key,
        "date": date_str
    }
    response = requests.get(base_url, params=params)
    return response.json().get('result', [])

# Get odds for a match
def get_odds(event_id):
    params = {
        "method": "odds",
        "APIkey": api_key,
        "event_id": event_id
    }
    response = requests.get(base_url, params=params)
    odds_data = response.json().get('result', [])
    
    # Parse bookmaker odds
    odds_by_book = {}
    for odd_obj in odds_data:
        book = odd_obj['bookmaker_name']
        odds_by_book[book] = {
            'p1': odd_obj['odds1'],
            'p2': odd_obj['odds2'],
            'timestamp': odd_obj['timestamp']
        }
    
    return odds_by_book

# Get player profile
def get_player_profile(player_key):
    params = {
        "method": "players",
        "APIkey": api_key,
        "player_key": player_key
    }
    response = requests.get(base_url, params=params)
    return response.json().get('result', {})[0]
```

#### Integration with US Open Hub
```python
# In us_open_value_hub.py

from datetime import datetime
import requests

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_usopen_fixtures_live():
    """Fetch current US Open fixtures from API"""
    api_key = os.environ.get("API_TENNIS_KEY")
    
    if not api_key:
        st.warning("API_TENNIS_KEY not configured. Using sample data.")
        return get_usopen_fixtures()  # Fallback
    
    try:
        # Get all fixtures for US Open dates (September)
        fixtures = []
        
        # Usually Sept 1-7 for men's, Sept 8-14 for women's
        for day in range(1, 15):
            date_str = f"2024-09-{day:02d}"
            params = {
                "method": "fixtures",
                "APIkey": api_key,
                "date": date_str,
                "tournament_key": "us_open"  # if supported
            }
            
            response = requests.get(
                "https://api.api-tennis.com/tennis/",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                for event in data.get('result', []):
                    fixtures.append({
                        'id': event['event_id'],
                        'date': event['sport_event']['scheduled'][:10],
                        'player1': event['sport_event']['competitors'][0]['name'],
                        'player2': event['sport_event']['competitors'][1]['name'],
                        'odds1': event.get('odds', {}).get('odds1', 0),
                        'odds2': event.get('odds', {}).get('odds2', 0),
                        'round': parse_round(event.get('sport_event', {}).get('round', {})),
                        'surface': 'Hard',  # US Open is always hard court
                        'status': event['sport_event_status']['match_status'].lower()
                    })
        
        return fixtures
    
    except Exception as e:
        st.error(f"Error fetching fixtures: {e}")
        return get_usopen_fixtures()  # Fallback to sample
```

---

### 2. TennisRatio.com - Advanced Player Stats

#### Web Scraping with BeautifulSoup
```python
import requests
from bs4 import BeautifulSoup
import re

class TennisRatioScraper:
    BASE_URL = "https://www.tennisratio.com"
    
    @staticmethod
    def normalize_name(name):
        """Convert player name to URL format"""
        # "Jannik Sinner" -> "jannuksinner"
        name = name.lower().replace(' ', '')
        # Handle accents
        name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i')
        name = name.replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
        return name
    
    def get_h2h_data(self, player1, player2):
        """Scrape H2H comparison from TennisRatio"""
        try:
            p1_norm = self.normalize_name(player1)
            p2_norm = self.normalize_name(player2)
            
            # TennisRatio uses alphabetical order
            if p1_norm < p2_norm:
                url = f"{self.BASE_URL}/h2h-compare/{p1_norm}-vs-{p2_norm}.html"
            else:
                url = f"{self.BASE_URL}/h2h-compare/{p2_norm}-vs-{p1_norm}.html"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract key statistics
            data = {
                'url': url,
                'player1': player1,
                'player2': player2,
                'stats': {}
            }
            
            # Find stats tables
            tables = soup.find_all('table', {'class': 'stats-table'})
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 3:
                        stat_name = cols[0].text.strip()
                        p1_val = float(re.sub(r'[^0-9.]', '', cols[1].text))
                        p2_val = float(re.sub(r'[^0-9.]', '', cols[2].text))
                        
                        data['stats'][stat_name] = {
                            'player1': p1_val,
                            'player2': p2_val,
                            'advantage': 'P1' if p1_val > p2_val else 'P2'
                        }
            
            # Look for specific stats we care about
            pressure_pattern = r'Pressure.*?(\d+\.\d+)%.*?(\d+\.\d+)%'
            dominance_pattern = r'Dominance.*?(\d+\.\d+).*?(\d+\.\d+)'
            
            text = soup.get_text()
            
            pressure_match = re.search(pressure_pattern, text)
            if pressure_match:
                data['pressure_points'] = {
                    'player1': float(pressure_match.group(1)) / 100,
                    'player2': float(pressure_match.group(2)) / 100
                }
            
            dominance_match = re.search(dominance_pattern, text)
            if dominance_match:
                data['dominance_ratio'] = {
                    'player1': float(dominance_match.group(1)),
                    'player2': float(dominance_match.group(2))
                }
            
            return data
        
        except Exception as e:
            print(f"Error scraping TennisRatio: {e}")
            return None
    
    def get_player_profile(self, player_name):
        """Get player's profile page stats"""
        try:
            p_norm = self.normalize_name(player_name)
            url = f"{self.BASE_URL}/players/{p_norm}.html"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            profile = {
                'player': player_name,
                'stats': {}
            }
            
            # Extract stats from player profile
            # Look for key performance metrics
            stat_boxes = soup.find_all('div', {'class': 'stat-box'})
            
            for box in stat_boxes:
                label = box.find('span', {'class': 'stat-label'})
                value = box.find('span', {'class': 'stat-value'})
                
                if label and value:
                    profile['stats'][label.text.strip()] = float(
                        re.sub(r'[^0-9.]', '', value.text)
                    )
            
            return profile
        
        except Exception as e:
            print(f"Error getting player profile: {e}")
            return None

# Usage
scraper = TennisRatioScraper()

# Get H2H data
h2h = scraper.get_h2h_data("Jannik Sinner", "Taylor Fritz")
if h2h:
    print(f"Pressure Points - Sinner: {h2h['pressure_points']['player1']:.1%}")
    print(f"Pressure Points - Fritz: {h2h['pressure_points']['player2']:.1%}")

# Get player stats
stats = scraper.get_player_profile("Jannik Sinner")
print(stats['stats'])
```

---

### 3. Live Odds Comparison

#### Track Odds Movement Across Bookmakers
```python
import sqlite3
from datetime import datetime
import pandas as pd

class OddsTracker:
    def __init__(self, db_path='odds_history.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create database for tracking odds"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT,
                player1 TEXT,
                player2 TEXT,
                bookmaker TEXT,
                odds_p1 REAL,
                odds_p2 REAL,
                timestamp DATETIME,
                implied_p1 REAL
            )
        """)
        conn.commit()
        conn.close()
    
    def log_odds(self, match_id, player1, player2, bookmaker_odds):
        """Log odds from multiple bookmakers"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for bookmaker, odds in bookmaker_odds.items():
            implied_p1 = 1 / odds['p1'] if odds['p1'] > 0 else 0.5
            
            cursor.execute("""
                INSERT INTO odds_history 
                (match_id, player1, player2, bookmaker, odds_p1, odds_p2, timestamp, implied_p1)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (match_id, player1, player2, bookmaker, odds['p1'], odds['p2'], 
                  datetime.now(), implied_p1))
        
        conn.commit()
        conn.close()
    
    def get_odds_movement(self, match_id, player1):
        """Calculate odds movement (sharp money detection)"""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query("""
            SELECT bookmaker, odds_p1, timestamp
            FROM odds_history
            WHERE match_id = ?
            ORDER BY timestamp
        """, conn, params=(match_id,))
        
        conn.close()
        
        if df.empty:
            return None
        
        # Get opening and current odds
        first_odds = df.iloc[0]['odds_p1']
        latest_odds = df.iloc[-1]['odds_p1']
        
        movement = (latest_odds - first_odds) / first_odds if first_odds > 0 else 0
        
        # Calculate consensus (agreement across bookmakers)
        latest_odds_all = df[df['timestamp'] == df['timestamp'].max()]
        consensus_std = latest_odds_all['odds_p1'].std()
        
        return {
            'movement_pct': movement * 100,
            'opening_odds': first_odds,
            'current_odds': latest_odds,
            'consensus_std': consensus_std,
            'is_sharp': movement > 0.05,  # Odds moved up 5%+ = sharp money on player1
            'volume_by_book': latest_odds_all.groupby('bookmaker').size().to_dict()
        }

# Usage
tracker = OddsTracker()

# Log odds from multiple bookmakers
bookmaker_odds = {
    'bet365': {'p1': 1.85, 'p2': 2.00},
    'betfair': {'p1': 1.86, 'p2': 1.98},
    'bwin': {'p1': 1.84, 'p2': 2.01},
}

tracker.log_odds(
    match_id='usopen_2024_sf_1',
    player1='Jannik Sinner',
    player2='Taylor Fritz',
    bookmaker_odds=bookmaker_odds
)

# Check movement after 1 hour
movement = tracker.get_odds_movement('usopen_2024_sf_1', 'Jannik Sinner')
if movement['is_sharp']:
    print(f"⚠️ Sharp money detected! Odds moved {movement['movement_pct']:.1f}%")
```

---

## TIER 2: Advanced Integrations (Next Phase)

### 4. SofaScore - Weather & Live Match Data

```python
import requests
from datetime import datetime

class SofaScoreScraper:
    """Scrape SofaScore for live match data and weather"""
    
    BASE_URL = "https://www.sofascore.com/api/v1"
    
    def get_live_matches(self):
        """Get all live tennis matches"""
        try:
            url = f"{self.BASE_URL}/sport/tennis/events/live"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            live_matches = []
            for event in data.get('events', []):
                live_matches.append({
                    'id': event['id'],
                    'player1': event['homeTeam']['name'],
                    'player2': event['awayTeam']['name'],
                    'status': event['status'],
                    'current_set': event.get('tournament', {}).get('uniqueTournamentId'),
                    'score': {
                        'p1_sets': event['homeTeam']['score'],
                        'p2_sets': event['awayTeam']['score']
                    }
                })
            
            return live_matches
        
        except Exception as e:
            print(f"Error fetching live matches: {e}")
            return []
    
    def get_match_statistics(self, match_id):
        """Get detailed match statistics"""
        try:
            url = f"{self.BASE_URL}/event/{match_id}/statistics"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            stats = {}
            
            for stat_group in data.get('statistics', []):
                group_name = stat_group['groupName']
                stats[group_name] = {
                    'player1': {},
                    'player2': {}
                }
                
                for stat in stat_group['statisticsItems']:
                    stat_name = stat['name']
                    p1_val = stat['homeValue']
                    p2_val = stat['awayValue']
                    
                    stats[group_name]['player1'][stat_name] = p1_val
                    stats[group_name]['player2'][stat_name] = p2_val
            
            return stats
        
        except Exception as e:
            print(f"Error fetching match stats: {e}")
            return None
    
    def get_weather(self, match_id):
        """Get weather conditions for match"""
        try:
            url = f"{self.BASE_URL}/event/{match_id}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            event = data.get('event', {})
            weather = event.get('weather', {})
            
            return {
                'temperature': weather.get('tempCelsius'),
                'wind_speed': weather.get('windSpeedKmh'),
                'humidity': weather.get('humidityPercent'),
                'weather_condition': weather.get('description'),
                'court': event.get('venue', {}).get('courtName')
            }
        
        except Exception as e:
            print(f"Error fetching weather: {e}")
            return None

# Usage
sofa = SofaScoreScraper()

live = sofa.get_live_matches()
for match in live:
    print(f"{match['player1']} vs {match['player2']}")
    
    stats = sofa.get_match_statistics(match['id'])
    weather = sofa.get_weather(match['id'])
    
    if weather:
        print(f"  Weather: {weather['temperature']}°C, "
              f"Wind: {weather['wind_speed']}km/h, "
              f"Court: {weather['court']}")
```

---

### 5. Match Charting Project - Rally Analytics

```python
# If you have access to Match Charting Project data

class RallyAnalyzer:
    """Analyze rally-level data"""
    
    def calculate_rally_stats(self, match_rallies):
        """Analyze rally patterns"""
        
        short_rallies = []  # < 4 shots
        medium_rallies = []  # 4-8 shots
        long_rallies = []   # > 8 shots
        
        for rally in match_rallies:
            shot_count = len(rally['shots'])
            winner_id = rally['winner_id']
            
            if shot_count < 4:
                short_rallies.append(winner_id)
            elif shot_count <= 8:
                medium_rallies.append(winner_id)
            else:
                long_rallies.append(winner_id)
        
        # Calculate win percentages by rally type
        return {
            'short_rally_win_pct_p1': sum(1 for w in short_rallies if w == 1) / len(short_rallies) if short_rallies else 0.5,
            'medium_rally_win_pct_p1': sum(1 for w in medium_rallies if w == 1) / len(medium_rallies) if medium_rallies else 0.5,
            'long_rally_win_pct_p1': sum(1 for w in long_rallies if w == 1) / len(long_rallies) if long_rallies else 0.5,
            'avg_rally_length': np.mean([len(r['shots']) for r in match_rallies]),
            'first_strike_rate': self._calculate_first_strike_rate(match_rallies)
        }
    
    def _calculate_first_strike_rate(self, rallies):
        """% of rallies won with a winner on first/second shot"""
        first_strike_wins = 0
        for rally in rallies:
            if len(rally['shots']) <= 2 and rally['winner_id']:
                first_strike_wins += 1
        
        return first_strike_wins / len(rallies) if rallies else 0
```

---

## Best Practices

### Rate Limiting
```python
import time
from functools import wraps

def rate_limit(calls_per_second=1):
    """Decorator to rate limit API calls"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    
    return decorator

@rate_limit(calls_per_second=2)
def fetch_player_stats(player_name):
    """Fetch with rate limiting"""
    # ... API call ...
    pass
```

### Caching
```python
import json
from datetime import datetime, timedelta

class APICache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, datetime.now())
    
    def clear(self):
        self.cache.clear()

# Usage
cache = APICache(ttl_seconds=1800)  # 30 minute TTL

def get_player_stats(player_name):
    cached = cache.get(f"player_{player_name}")
    if cached:
        return cached
    
    stats = fetch_from_api(player_name)
    cache.set(f"player_{player_name}", stats)
    return stats
```

### Error Handling
```python
import logging

logger = logging.getLogger(__name__)

def fetch_with_retry(url, max_retries=3, backoff_factor=2):
    """Fetch with exponential backoff"""
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}")
        
        except requests.exceptions.HTTPError as e:
            if response.status_code >= 500:  # Server error
                logger.warning(f"Server error {response.status_code}")
            else:
                raise  # Don't retry client errors
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
        
        if attempt < max_retries - 1:
            wait_time = backoff_factor ** attempt
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    raise Exception(f"Failed after {max_retries} attempts")
```

---

## Summary

**Implement in this order:**
1. **Week 1**: api-tennis.com (live odds & fixtures)
2. **Week 1**: TennisRatio (advanced stats)
3. **Week 2**: OddsTracker (odds movement)
4. **Week 3**: SofaScore (weather & match stats)
5. **Week 4+**: Match Charting data (if available)

Each integration adds +1-3% accuracy to your model. Start with the top 2 for immediate +5% improvement.

