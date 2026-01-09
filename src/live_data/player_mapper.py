"""
Player Mapper - Match player names across different data sources

Different sources use different name formats:
- Sofascore: "Rafael Nadal"
- Flashscore: "R. Nadal"
- ATP: "Nadal R."
- Our DB: "Rafael Nadal"

Uses fuzzy string matching to handle variations.
"""

import sqlite3
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher
import re
import logging

logger = logging.getLogger(__name__)


class PlayerMapper:
    """Map external player names to internal database IDs"""
    
    def __init__(self, db_path='tennis_data.db'):
        self.db_path = db_path
        self._name_cache = {}  # Cache for faster lookups
        self._load_player_mappings()
    
    def _load_player_mappings(self):
        """Load existing player mappings from database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if player_mappings table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='player_mappings'
            """)
            
            if not cursor.fetchone():
                # Create table if it doesn't exist
                self._create_mappings_table(cursor)
                conn.commit()
            
            # Load mappings into cache
            cursor.execute("""
                SELECT player_id, player_name, sofascore_id, 
                       flashscore_id, atp_id, aliases
                FROM player_mappings
            """)
            
            for row in cursor.fetchall():
                player_id, name, sof_id, flash_id, atp_id, aliases = row
                
                self._name_cache[name.lower()] = {
                    'player_id': player_id,
                    'canonical_name': name,
                    'sofascore_id': sof_id,
                    'flashscore_id': flash_id,
                    'atp_id': atp_id,
                    'aliases': eval(aliases) if aliases else []
                }
            
            conn.close()
            
            logger.info(f"Loaded {len(self._name_cache)} player mappings")
            
        except Exception as e:
            logger.error(f"Error loading player mappings: {e}")
            self._name_cache = {}
    
    def _create_mappings_table(self, cursor):
        """Create player_mappings table"""
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_mappings (
                player_id INTEGER PRIMARY KEY,
                player_name TEXT NOT NULL,
                sofascore_id INTEGER,
                flashscore_id TEXT,
                atp_id TEXT,
                aliases TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_name)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_player_name 
            ON player_mappings(player_name)
        """)
    
    def match_player_name(self, external_name: str, source: str = 'sofascore',
                         external_id: Optional[str] = None,
                         min_confidence: float = 0.80) -> Optional[Dict]:
        """
        Match external player name to internal player ID
        
        Args:
            external_name: Player name from external source
            source: Data source ('sofascore', 'flashscore', 'atp_official')
            external_id: External player ID (if available)
            min_confidence: Minimum fuzzy match confidence (0-1)
        
        Returns:
            {
                'player_id': int,
                'canonical_name': str,
                'confidence': float,
                'match_method': 'exact' | 'alias' | 'fuzzy' | 'external_id'
            }
            or None if no match found
        """
        
        if not external_name or external_name.strip() == '':
            return None
        
        # Normalize name
        normalized = self._normalize_name(external_name)
        
        # 1. Try exact match on canonical name
        if normalized in self._name_cache:
            player = self._name_cache[normalized]
            return {
                'player_id': player['player_id'],
                'canonical_name': player['canonical_name'],
                'confidence': 1.0,
                'match_method': 'exact'
            }
        
        # 2. Try alias match
        for player_info in self._name_cache.values():
            aliases = [self._normalize_name(a) for a in player_info.get('aliases', [])]
            if normalized in aliases:
                return {
                    'player_id': player_info['player_id'],
                    'canonical_name': player_info['canonical_name'],
                    'confidence': 0.95,
                    'match_method': 'alias'
                }
        
        # 3. Try external ID match
        if external_id:
            for player_info in self._name_cache.values():
                if source == 'sofascore' and player_info.get('sofascore_id') == external_id:
                    return {
                        'player_id': player_info['player_id'],
                        'canonical_name': player_info['canonical_name'],
                        'confidence': 1.0,
                        'match_method': 'external_id'
                    }
                elif source == 'flashscore' and player_info.get('flashscore_id') == external_id:
                    return {
                        'player_id': player_info['player_id'],
                        'canonical_name': player_info['canonical_name'],
                        'confidence': 1.0,
                        'match_method': 'external_id'
                    }
                elif source == 'atp_official' and player_info.get('atp_id') == external_id:
                    return {
                        'player_id': player_info['player_id'],
                        'canonical_name': player_info['canonical_name'],
                        'confidence': 1.0,
                        'match_method': 'external_id'
                    }
        
        # 4. Try fuzzy matching
        best_match = self._fuzzy_match(normalized, min_confidence)
        
        if best_match:
            return best_match
        
        # 5. No match found
        logger.warning(f"Could not match player: {external_name} (source: {source})")
        return None
    
    def _fuzzy_match(self, normalized_name: str, min_confidence: float) -> Optional[Dict]:
        """
        Find best fuzzy match using Levenshtein-like similarity
        
        Args:
            normalized_name: Normalized player name to match
            min_confidence: Minimum confidence threshold
        
        Returns:
            Match dict or None
        """
        
        best_ratio = 0
        best_player = None
        
        for cached_name, player_info in self._name_cache.items():
            # Calculate similarity
            ratio = self._similarity_ratio(normalized_name, cached_name)
            
            # Also check against aliases
            for alias in player_info.get('aliases', []):
                alias_normalized = self._normalize_name(alias)
                alias_ratio = self._similarity_ratio(normalized_name, alias_normalized)
                ratio = max(ratio, alias_ratio)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_player = player_info
        
        if best_ratio >= min_confidence and best_player:
            return {
                'player_id': best_player['player_id'],
                'canonical_name': best_player['canonical_name'],
                'confidence': best_ratio,
                'match_method': 'fuzzy'
            }
        
        return None
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize player name for matching
        
        Examples:
        - "R. Nadal" → "r nadal"
        - "Nadal, Rafael" → "rafael nadal"
        - "FEDERER Roger" → "roger federer"
        """
        
        # Convert to lowercase
        name = name.lower()
        
        # Remove punctuation
        name = re.sub(r'[.,\-]', ' ', name)
        
        # Handle "Last, First" format
        if ',' in name:
            parts = name.split(',')
            name = f"{parts[1].strip()} {parts[0].strip()}"
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        return name
    
    def _similarity_ratio(self, name1: str, name2: str) -> float:
        """
        Calculate similarity ratio between two names
        
        Uses combination of:
        - SequenceMatcher (character-level similarity)
        - Token matching (word-level similarity)
        
        Returns:
            Similarity score 0-1
        """
        
        # Character-level similarity
        char_ratio = SequenceMatcher(None, name1, name2).ratio()
        
        # Token-level similarity (match words)
        tokens1 = set(name1.split())
        tokens2 = set(name2.split())
        
        if len(tokens1) == 0 or len(tokens2) == 0:
            token_ratio = 0
        else:
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            token_ratio = len(intersection) / len(union)
        
        # Weighted average (tokens more important for names)
        return 0.4 * char_ratio + 0.6 * token_ratio
    
    def add_mapping(self, player_name: str, external_id: str = None,
                   source: str = 'sofascore', aliases: List[str] = None) -> int:
        """
        Add new player mapping to database
        
        Args:
            player_name: Canonical player name
            external_id: External player ID
            source: Data source
            aliases: List of known aliases
        
        Returns:
            player_id
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if player already exists
            cursor.execute("""
                SELECT player_id FROM player_mappings
                WHERE player_name = ?
            """, (player_name,))
            
            existing = cursor.fetchone()
            
            if existing:
                player_id = existing[0]
                
                # Update external ID if provided
                if external_id:
                    if source == 'sofascore':
                        cursor.execute("""
                            UPDATE player_mappings
                            SET sofascore_id = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE player_id = ?
                        """, (external_id, player_id))
                    elif source == 'flashscore':
                        cursor.execute("""
                            UPDATE player_mappings
                            SET flashscore_id = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE player_id = ?
                        """, (external_id, player_id))
                    elif source == 'atp_official':
                        cursor.execute("""
                            UPDATE player_mappings
                            SET atp_id = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE player_id = ?
                        """, (external_id, player_id))
                
                conn.commit()
                
            else:
                # Insert new player
                aliases_str = str(aliases) if aliases else '[]'
                
                cursor.execute("""
                    INSERT INTO player_mappings 
                    (player_name, sofascore_id, flashscore_id, atp_id, aliases)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    player_name,
                    external_id if source == 'sofascore' else None,
                    external_id if source == 'flashscore' else None,
                    external_id if source == 'atp_official' else None,
                    aliases_str
                ))
                
                player_id = cursor.lastrowid
                conn.commit()
                
                # Update cache
                self._name_cache[player_name.lower()] = {
                    'player_id': player_id,
                    'canonical_name': player_name,
                    'sofascore_id': external_id if source == 'sofascore' else None,
                    'flashscore_id': external_id if source == 'flashscore' else None,
                    'atp_id': external_id if source == 'atp_official' else None,
                    'aliases': aliases or []
                }
            
            conn.close()
            
            return player_id
            
        except Exception as e:
            logger.error(f"Error adding player mapping: {e}")
            return None


# Convenience functions

def match_player_name(external_name: str, source: str = 'sofascore',
                     external_id: Optional[str] = None,
                     db_path: str = 'tennis_data.db') -> Optional[Dict]:
    """
    Match external player name to internal database ID
    
    Args:
        external_name: Player name from external source
        source: Data source ('sofascore', 'flashscore', 'atp_official')
        external_id: External player ID (optional)
        db_path: Path to database
    
    Returns:
        Match dict or None
    """
    mapper = PlayerMapper(db_path=db_path)
    return mapper.match_player_name(external_name, source, external_id)


def fuzzy_match_player(external_name: str, min_confidence: float = 0.75,
                      db_path: str = 'tennis_data.db') -> Optional[Dict]:
    """
    Find best fuzzy match for player name
    
    Args:
        external_name: Player name to match
        min_confidence: Minimum confidence (0-1)
        db_path: Path to database
    
    Returns:
        Match dict or None
    """
    mapper = PlayerMapper(db_path=db_path)
    normalized = mapper._normalize_name(external_name)
    return mapper._fuzzy_match(normalized, min_confidence)


if __name__ == "__main__":
    # Test the player mapper
    print("🎾 Testing Player Mapper\n")
    
    mapper = PlayerMapper()
    
    test_names = [
        ("Rafael Nadal", "sofascore"),
        ("R. Nadal", "flashscore"),
        ("Nadal R.", "atp_official"),
        ("Novak Djokovic", "sofascore"),
        ("N. Djokovic", "sofascore"),
        ("Roger Federer", "sofascore"),
    ]
    
    for name, source in test_names:
        result = mapper.match_player_name(name, source=source)
        
        if result:
            print(f"✅ {name:20s} → {result['canonical_name']:20s} "
                  f"({result['confidence']:.2%}, {result['match_method']})")
        else:
            print(f"❌ {name:20s} → No match found")
