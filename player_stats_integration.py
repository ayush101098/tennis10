"""
Player Statistics Integration for Live Calculator
==================================================
Auto-fetch and populate player statistics from multiple sources:
1. Historical database (tennis_data.db)
2. Live ATP stats (scraped)
3. Manual override capability
"""

import sqlite3
import pandas as pd
import requests
from typing import Dict, Optional, Tuple
import time

class PlayerStatsProvider:
    """Fetch player statistics from multiple sources"""
    
    def __init__(self, db_path='tennis_betting.db'):
        self.db_path = db_path
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
    def get_player_stats(self, player_name: str, surface: str = 'Hard') -> Dict:
        """
        Get player statistics with fallback chain:
        1. Try database (historical stats)
        2. Try ATP website (recent form)
        3. Use intelligent defaults based on ranking
        
        Returns:
            Dict with serve_win_pct, return_win_pct, and other stats
        """
        # Check cache first
        cache_key = f"{player_name}_{surface}_{int(time.time() / self.cache_ttl)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try database first
        stats = self._get_from_database(player_name, surface)
        
        if not stats:
            # Try ATP website scraping
            stats = self._get_from_atp_website(player_name)
        
        if not stats:
            # Use intelligent defaults based on name patterns
            stats = self._get_intelligent_defaults(player_name)
        
        # Cache the result
        self.cache[cache_key] = stats
        return stats
    
    def _get_from_database(self, player_name: str, surface: str) -> Optional[Dict]:
        """Fetch historical stats from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query historical performance
            query = """
            SELECT 
                AVG(CASE 
                    WHEN m.winner_id = p.player_id THEN 1.0 
                    ELSE 0.0 
                END) as win_rate,
                COUNT(*) as matches_played,
                p.player_id
            FROM players p
            LEFT JOIN matches m ON (p.player_id = m.winner_id OR p.player_id = m.loser_id)
            WHERE LOWER(p.player_name) LIKE LOWER(?)
                AND (m.surface = ? OR ? = 'All')
                AND m.match_date >= date('now', '-365 days')
            GROUP BY p.player_id, p.player_name
            HAVING COUNT(*) >= 5
            ORDER BY matches_played DESC
            LIMIT 1
            """
            
            cursor = conn.execute(query, (f'%{player_name}%', surface, surface))
            row = cursor.fetchone()
            
            if row and row[1] >= 5:  # At least 5 matches
                win_rate = row[0] or 0.5
                
                # Estimate serve/return stats from win rate
                # Top players: ~68% serve, ~35% return
                # Average players: ~62% serve, ~30% return
                serve_pct = 0.60 + (win_rate - 0.5) * 0.16  # Maps 0.5->0.60, 0.7->0.63
                return_pct = 0.28 + (win_rate - 0.5) * 0.14  # Maps 0.5->0.28, 0.7->0.35
                
                conn.close()
                
                return {
                    'serve_win_pct': min(max(serve_pct, 0.55), 0.75),
                    'return_win_pct': min(max(return_pct, 0.25), 0.45),
                    'win_rate': win_rate,
                    'matches_played': row[1],
                    'source': 'database',
                    'surface': surface
                }
            
            conn.close()
            
        except Exception as e:
            print(f"Database error: {e}")
        
        return None
    
    def _get_from_atp_website(self, player_name: str) -> Optional[Dict]:
        """Scrape ATP website for recent statistics"""
        # This is a placeholder - actual implementation would scrape ATP stats
        # For now, return None to trigger defaults
        return None
    
    def _get_intelligent_defaults(self, player_name: str) -> Dict:
        """
        Provide intelligent defaults based on player name patterns
        Known top players get better stats
        """
        # Top 10 players (as of 2026)
        top_players = {
            'djokovic': {'serve': 0.68, 'return': 0.36},
            'alcaraz': {'serve': 0.66, 'return': 0.35},
            'sinner': {'serve': 0.67, 'return': 0.35},
            'medvedev': {'serve': 0.65, 'return': 0.36},
            'zverev': {'serve': 0.68, 'return': 0.33},
            'rublev': {'serve': 0.64, 'return': 0.34},
            'tsitsipas': {'serve': 0.66, 'return': 0.33},
            'rune': {'serve': 0.64, 'return': 0.34},
            'hurkacz': {'serve': 0.70, 'return': 0.31},
            'fritz': {'serve': 0.67, 'return': 0.32},
            'nadal': {'serve': 0.67, 'return': 0.38},  # Clay specialist
            'federer': {'serve': 0.69, 'return': 0.35},
        }
        
        # Check if known player
        name_lower = player_name.lower()
        for key, stats in top_players.items():
            if key in name_lower:
                return {
                    'serve_win_pct': stats['serve'],
                    'return_win_pct': stats['return'],
                    'win_rate': 0.65,
                    'matches_played': 0,
                    'source': 'known_player',
                    'surface': 'All'
                }
        
        # Default for unknown player (ATP average)
        return {
            'serve_win_pct': 0.63,
            'return_win_pct': 0.32,
            'win_rate': 0.50,
            'matches_played': 0,
            'source': 'default',
            'surface': 'All'
        }
    
    def get_head_to_head(self, player1: str, player2: str) -> Dict:
        """Get head-to-head statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                SUM(CASE WHEN LOWER(winner_name) LIKE LOWER(?) THEN 1 ELSE 0 END) as p1_wins,
                SUM(CASE WHEN LOWER(winner_name) LIKE LOWER(?) THEN 1 ELSE 0 END) as p2_wins,
                COUNT(*) as total_matches
            FROM matches m
            JOIN players w ON m.winner_id = w.player_id
            WHERE (LOWER(winner_name) LIKE LOWER(?) OR LOWER(winner_name) LIKE LOWER(?))
                AND match_id IN (
                    SELECT match_id FROM matches m2
                    JOIN players l ON m2.loser_id = l.player_id
                    WHERE LOWER(l.player_name) LIKE LOWER(?) OR LOWER(l.player_name) LIKE LOWER(?)
                )
            """
            
            cursor = conn.execute(query, (
                f'%{player1}%', f'%{player2}%',
                f'%{player1}%', f'%{player2}%',
                f'%{player1}%', f'%{player2}%'
            ))
            
            row = cursor.fetchone()
            conn.close()
            
            if row and row[2] > 0:
                return {
                    'p1_wins': row[0] or 0,
                    'p2_wins': row[1] or 0,
                    'total': row[2],
                    'p1_win_pct': (row[0] or 0) / row[2]
                }
            
        except Exception as e:
            print(f"H2H error: {e}")
        
        return {'p1_wins': 0, 'p2_wins': 0, 'total': 0, 'p1_win_pct': 0.5}
    
    def adjust_for_surface(self, stats: Dict, surface: str) -> Dict:
        """Adjust player stats based on surface"""
        adjustments = {
            'Clay': {
                # Clay slows down serve, improves returns
                'serve_multiplier': 0.97,
                'return_multiplier': 1.05
            },
            'Grass': {
                # Grass speeds up serve, hurts returns
                'serve_multiplier': 1.03,
                'return_multiplier': 0.95
            },
            'Hard': {
                # Neutral
                'serve_multiplier': 1.0,
                'return_multiplier': 1.0
            }
        }
        
        adj = adjustments.get(surface, adjustments['Hard'])
        
        return {
            **stats,
            'serve_win_pct': min(0.75, stats['serve_win_pct'] * adj['serve_multiplier']),
            'return_win_pct': min(0.45, stats['return_win_pct'] * adj['return_multiplier']),
            'surface': surface
        }


def get_player_stats_for_calculator(player1_name: str, player2_name: str, 
                                    surface: str = 'Hard') -> Tuple[Dict, Dict]:
    """
    Main function to get stats for both players
    Returns: (player1_stats, player2_stats)
    """
    provider = PlayerStatsProvider()
    
    # Get base stats
    p1_stats = provider.get_player_stats(player1_name, surface)
    p2_stats = provider.get_player_stats(player2_name, surface)
    
    # Adjust for surface if needed
    if p1_stats['surface'] != surface:
        p1_stats = provider.adjust_for_surface(p1_stats, surface)
    if p2_stats['surface'] != surface:
        p2_stats = provider.adjust_for_surface(p2_stats, surface)
    
    # Get H2H context
    h2h = provider.get_head_to_head(player1_name, player2_name)
    
    # Adjust based on H2H if significant history
    if h2h['total'] >= 5:
        h2h_factor = (h2h['p1_win_pct'] - 0.5) * 0.02  # Small adjustment
        p1_stats['serve_win_pct'] = min(0.75, max(0.55, p1_stats['serve_win_pct'] + h2h_factor))
        p2_stats['serve_win_pct'] = min(0.75, max(0.55, p2_stats['serve_win_pct'] - h2h_factor))
    
    # Add H2H info to stats
    p1_stats['h2h'] = h2h
    p2_stats['h2h'] = h2h
    
    return p1_stats, p2_stats


# Example usage
if __name__ == "__main__":
    print("🎾 Player Statistics Provider Test\n")
    
    # Test with known players
    p1_stats, p2_stats = get_player_stats_for_calculator(
        "Novak Djokovic", 
        "Carlos Alcaraz",
        surface="Hard"
    )
    
    print("Player 1 (Djokovic):")
    print(f"  Serve Win %: {p1_stats['serve_win_pct']:.1%}")
    print(f"  Return Win %: {p1_stats['return_win_pct']:.1%}")
    print(f"  Source: {p1_stats['source']}")
    print(f"  Surface: {p1_stats['surface']}")
    
    print("\nPlayer 2 (Alcaraz):")
    print(f"  Serve Win %: {p2_stats['serve_win_pct']:.1%}")
    print(f"  Return Win %: {p2_stats['return_win_pct']:.1%}")
    print(f"  Source: {p2_stats['source']}")
    print(f"  Surface: {p2_stats['surface']}")
    
    print(f"\nHead-to-Head:")
    print(f"  Total matches: {p1_stats['h2h']['total']}")
    print(f"  Djokovic wins: {p1_stats['h2h']['p1_wins']}")
    print(f"  Alcaraz wins: {p1_stats['h2h']['p2_wins']}")
