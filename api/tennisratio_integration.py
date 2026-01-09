"""
TennisRatio Data Integration Module
====================================

Fetches and integrates advanced statistics from tennisratio.com including:
- Player profiles and performance metrics
- Head-to-head comparisons with detailed stats
- Pressure points analysis
- Dominance and efficiency ratios
- Surface-specific breakdowns
- Live match data and odds performance

Author: Tennis Betting Intelligence Hub
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional, Tuple
import re
import json
from datetime import datetime


class TennisRatioAPI:
    """Interface to fetch data from TennisRatio.com"""
    
    BASE_URL = "https://www.tennisratio.com"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    @staticmethod
    def normalize_player_name(name: str) -> str:
        """Convert player name to TennisRatio URL format"""
        # Remove accents and special characters
        name = name.replace(' ', '')
        name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i')
        name = name.replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
        name = name.replace('ü', 'u').replace('ö', 'o').replace('ä', 'a')
        name = name.replace('š', 's').replace('č', 'c').replace('ž', 'z')
        name = name.replace('ć', 'c').replace('đ', 'd')
        # Remove hyphens and apostrophes
        name = name.replace('-', '').replace("'", '')
        return name
    
    def get_player_profile_url(self, player_name: str) -> str:
        """Get the player profile URL"""
        normalized = self.normalize_player_name(player_name)
        return f"{self.BASE_URL}/players/{normalized}.html"
    
    def get_h2h_url(self, player1: str, player2: str) -> str:
        """Get the H2H comparison URL"""
        p1 = self.normalize_player_name(player1).lower()
        p2 = self.normalize_player_name(player2).lower()
        # TennisRatio uses alphabetical order in URLs
        if p1 < p2:
            return f"{self.BASE_URL}/h2h-compare/{p1}-vs-{p2}.html"
        else:
            return f"{self.BASE_URL}/h2h-compare/{p2}-vs-{p1}.html"
    
    def fetch_h2h_data(self, player1: str, player2: str) -> Optional[Dict]:
        """
        Fetch head-to-head comparison data
        
        Returns:
            Dict with structure:
            {
                'player1_name': str,
                'player2_name': str,
                'h2h_record': {'player1_wins': int, 'player2_wins': int},
                'player1_stats': {...},
                'player2_stats': {...},
                'pressure_points': {...},
                'dominance_ratios': {...}
            }
        """
        try:
            url = self.get_h2h_url(player1, player2)
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse the data
            data = {
                'url': url,
                'player1_name': player1,
                'player2_name': player2,
                'fetched_at': datetime.now().isoformat(),
                'stats': {}
            }
            
            # Extract key stats from the page
            # This is a simplified version - full implementation would parse all tables
            stats_text = soup.get_text()
            
            # Extract aces per game
            aces_match = re.search(r'Aces per Game\s+([\d.]+)\s+([\d.]+)', stats_text)
            if aces_match:
                data['stats']['aces_per_game'] = {
                    'player1': float(aces_match.group(1)),
                    'player2': float(aces_match.group(2))
                }
            
            # Extract double faults
            df_match = re.search(r'Double Faults per Game\s+([\d.]+)\s+([\d.]+)', stats_text)
            if df_match:
                data['stats']['df_per_game'] = {
                    'player1': float(df_match.group(1)),
                    'player2': float(df_match.group(2))
                }
            
            # Extract break points
            bp_created_match = re.search(r'BPs Created per Game\s+([\d.]+)\s+([\d.]+)', stats_text)
            if bp_created_match:
                data['stats']['bp_created_per_game'] = {
                    'player1': float(bp_created_match.group(1)),
                    'player2': float(bp_created_match.group(2))
                }
            
            # Extract dominance ratio
            dom_match = re.search(r'Dominance Ratio\s+([\d.]+)\s+([\d.]+)', stats_text)
            if dom_match:
                data['stats']['dominance_ratio'] = {
                    'player1': float(dom_match.group(1)),
                    'player2': float(dom_match.group(2))
                }
            
            # Extract match efficiency
            eff_match = re.search(r'Match Efficiency\s+([\d.]+)\s+([\d.]+)', stats_text)
            if eff_match:
                data['stats']['match_efficiency'] = {
                    'player1': float(eff_match.group(1)),
                    'player2': float(eff_match.group(2))
                }
            
            # Extract pressure points won on serve
            pressure_serve_match = re.search(r'(\d+\.\d+)%\s+PRESSURE POINTS\s*WON ON SERVE.*?(\d+\.\d+)%\s+PRESSURE POINTS\s*WON ON SERVE', stats_text, re.DOTALL)
            if pressure_serve_match:
                data['stats']['pressure_points_serve'] = {
                    'player1': float(pressure_serve_match.group(1)),
                    'player2': float(pressure_serve_match.group(2))
                }
            
            # Extract win rates
            win_rate_match = re.search(r'(\d+)-(\d+)\s+([\d.]+)% wins.*?(\d+)-(\d+)\s+([\d.]+)% wins', stats_text, re.DOTALL)
            if win_rate_match:
                data['stats']['last_52_weeks'] = {
                    'player1': {
                        'wins': int(win_rate_match.group(1)),
                        'losses': int(win_rate_match.group(2)),
                        'win_pct': float(win_rate_match.group(3))
                    },
                    'player2': {
                        'wins': int(win_rate_match.group(4)),
                        'losses': int(win_rate_match.group(5)),
                        'win_pct': float(win_rate_match.group(6))
                    }
                }
            
            return data
            
        except Exception as e:
            print(f"Error fetching TennisRatio data: {e}")
            return None
    
    def get_enhanced_prediction(self, h2h_data: Dict) -> Dict:
        """
        Generate enhanced prediction based on TennisRatio data
        
        Uses:
        - Dominance ratio
        - Match efficiency
        - Pressure points performance
        - Recent form (last 52 weeks)
        """
        if not h2h_data or 'stats' not in h2h_data:
            return {'prediction': None, 'confidence': 'low'}
        
        stats = h2h_data['stats']
        score = 0.0
        factors = []
        
        # Factor 1: Dominance Ratio (weight: 0.25)
        if 'dominance_ratio' in stats:
            dom1 = stats['dominance_ratio']['player1']
            dom2 = stats['dominance_ratio']['player2']
            dom_diff = (dom1 - dom2) / (dom1 + dom2)  # Normalize to -0.5 to 0.5
            score += dom_diff * 0.25
            factors.append(f"Dominance: {dom1:.3f} vs {dom2:.3f}")
        
        # Factor 2: Match Efficiency (weight: 0.20)
        if 'match_efficiency' in stats:
            eff1 = stats['match_efficiency']['player1']
            eff2 = stats['match_efficiency']['player2']
            eff_diff = (eff1 - eff2) / (eff1 + eff2)
            score += eff_diff * 0.20
            factors.append(f"Efficiency: {eff1:.3f} vs {eff2:.3f}")
        
        # Factor 3: Pressure Points (weight: 0.30)
        if 'pressure_points_serve' in stats:
            pp1 = stats['pressure_points_serve']['player1']
            pp2 = stats['pressure_points_serve']['player2']
            pp_diff = (pp1 - pp2) / 100.0  # Convert to -1 to 1 range
            score += pp_diff * 0.30
            factors.append(f"Pressure Points: {pp1:.1f}% vs {pp2:.1f}%")
        
        # Factor 4: Recent Form (weight: 0.25)
        if 'last_52_weeks' in stats:
            form1 = stats['last_52_weeks']['player1']['win_pct']
            form2 = stats['last_52_weeks']['player2']['win_pct']
            form_diff = (form1 - form2) / 100.0
            score += form_diff * 0.25
            factors.append(f"Form: {form1:.1f}% vs {form2:.1f}%")
        
        # Convert score to probability (0 to 1)
        # score ranges from -1 to 1, convert to probability
        probability_p1 = 0.5 + (score * 0.5)  # Maps -1..1 to 0..1, but centered at 0.5
        probability_p1 = max(0.35, min(0.65, probability_p1))  # Clamp to reasonable range
        
        # Determine confidence
        if abs(score) > 0.15:
            confidence = 'high'
        elif abs(score) > 0.08:
            confidence = 'moderate'
        else:
            confidence = 'low'
        
        return {
            'probability_p1': probability_p1,
            'probability_p2': 1 - probability_p1,
            'confidence': confidence,
            'score': score,
            'factors': factors
        }


def get_tennisratio_insights(player1: str, player2: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Main function to get TennisRatio insights for a matchup
    
    Args:
        player1: Name of first player
        player2: Name of second player
    
    Returns:
        Tuple of (h2h_data, prediction)
    """
    api = TennisRatioAPI()
    
    # Fetch H2H data
    h2h_data = api.fetch_h2h_data(player1, player2)
    
    if not h2h_data:
        return None, None
    
    # Generate prediction
    prediction = api.get_enhanced_prediction(h2h_data)
    
    return h2h_data, prediction


# Example usage
if __name__ == "__main__":
    # Test with Munar vs Baez
    h2h, pred = get_tennisratio_insights("Jaume Munar", "Sebastian Baez")
    
    if h2h:
        print("=" * 70)
        print("TENNISRATIO.COM INTEGRATION - MUNAR VS BAEZ")
        print("=" * 70)
        print(f"\nURL: {h2h['url']}")
        print(f"\nData fetched: {h2h['fetched_at']}")
        
        if 'stats' in h2h and h2h['stats']:
            print("\n📊 KEY STATISTICS:")
            for stat, values in h2h['stats'].items():
                print(f"\n  {stat.replace('_', ' ').title()}:")
                print(f"    {values}")
        
        if pred:
            print("\n🤖 TENNISRATIO ENHANCED PREDICTION:")
            print(f"  Munar: {pred['probability_p1']:.1%}")
            print(f"  Baez: {pred['probability_p2']:.1%}")
            print(f"  Confidence: {pred['confidence'].upper()}")
            print(f"\n  Contributing Factors:")
            for factor in pred['factors']:
                print(f"    • {factor}")
    else:
        print("Failed to fetch data from TennisRatio")
