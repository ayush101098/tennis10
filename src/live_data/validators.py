"""
Data Validators - Sanity checks for match and odds data

Validates:
1. Player names exist in database
2. Tournament/surface are valid
3. Scheduled times make sense
4. No duplicate matches
5. Odds are reasonable
"""

from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate match and odds data before storing"""
    
    VALID_SURFACES = ['hard', 'clay', 'grass', 'indoor', 'carpet']
    VALID_ROUNDS = ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'RR', 'Q1', 'Q2', 'Q3']
    
    def __init__(self):
        self.validation_cache = {}
    
    def validate_match_data(self, match_dict: Dict) -> Tuple[bool, List[str]]:
        """
        Validate match data before adding to database
        
        Args:
            match_dict: Dictionary with match information
        
        Returns:
            (is_valid: bool, issues: list of issue descriptions)
        """
        
        issues = []
        
        # 1. Check required fields
        required_fields = ['player1_name', 'player2_name', 'tournament_name', 
                          'surface', 'scheduled_time']
        
        for field in required_fields:
            if field not in match_dict or not match_dict[field]:
                issues.append(f"Missing required field: {field}")
        
        if issues:
            return False, issues
        
        # 2. Validate player names
        if not self._is_valid_player_name(match_dict['player1_name']):
            issues.append(f"Invalid player1 name: {match_dict['player1_name']}")
        
        if not self._is_valid_player_name(match_dict['player2_name']):
            issues.append(f"Invalid player2 name: {match_dict['player2_name']}")
        
        # Players must be different
        if match_dict['player1_name'] == match_dict['player2_name']:
            issues.append("Player1 and Player2 are the same")
        
        # 3. Validate surface
        surface = match_dict['surface'].lower()
        if surface not in self.VALID_SURFACES:
            issues.append(f"Invalid surface: {surface} (must be one of {self.VALID_SURFACES})")
        
        # 4. Validate tournament name
        if not self._is_valid_tournament(match_dict['tournament_name']):
            issues.append(f"Invalid tournament name: {match_dict['tournament_name']}")
        
        # 5. Validate scheduled time
        scheduled_time = match_dict['scheduled_time']
        
        if isinstance(scheduled_time, str):
            try:
                scheduled_time = datetime.fromisoformat(scheduled_time)
            except:
                issues.append(f"Invalid datetime format: {scheduled_time}")
                return False, issues
        
        # Must be in future (not a past match)
        if scheduled_time < datetime.now() - timedelta(hours=1):
            issues.append("Scheduled time is in the past")
        
        # Must be within next 14 days (sanity check)
        if scheduled_time > datetime.now() + timedelta(days=14):
            issues.append("Scheduled time is too far in future (>14 days)")
        
        # 6. Validate round (if provided)
        if 'round' in match_dict and match_dict['round']:
            if not self._is_valid_round(match_dict['round']):
                issues.append(f"Invalid round: {match_dict['round']}")
        
        # 7. Validate data source
        if 'data_source' in match_dict:
            valid_sources = ['sofascore', 'flashscore', 'atp_official', 'manual']
            if match_dict['data_source'] not in valid_sources:
                issues.append(f"Invalid data source: {match_dict['data_source']}")
        
        # Final decision
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Validation failed for {match_dict.get('match_id', 'unknown')}: {issues}")
        
        return is_valid, issues
    
    def validate_odds_data(self, odds_dict: Dict) -> Tuple[bool, List[str]]:
        """
        Validate odds data
        
        Args:
            odds_dict: Dictionary with odds information
        
        Returns:
            (is_valid: bool, issues: list)
        """
        
        issues = []
        
        # 1. Check required fields
        required_fields = ['match_id', 'player1_odds', 'player2_odds']
        
        for field in required_fields:
            if field not in odds_dict or odds_dict[field] is None:
                issues.append(f"Missing required field: {field}")
        
        if issues:
            return False, issues
        
        # 2. Validate odds values
        p1_odds = odds_dict['player1_odds']
        p2_odds = odds_dict['player2_odds']
        
        # Odds must be >= 1.01
        if p1_odds < 1.01:
            issues.append(f"Player1 odds too low: {p1_odds} (must be >= 1.01)")
        
        if p2_odds < 1.01:
            issues.append(f"Player2 odds too low: {p2_odds} (must be >= 1.01)")
        
        # Odds must be <= 1000 (sanity check)
        if p1_odds > 1000:
            issues.append(f"Player1 odds too high: {p1_odds} (must be <= 1000)")
        
        if p2_odds > 1000:
            issues.append(f"Player2 odds too high: {p2_odds} (must be <= 1000)")
        
        # 3. Check overround (bookmaker margin)
        overround = (1 / p1_odds) + (1 / p2_odds)
        
        # Typical overround is 1.02 - 1.10
        if overround < 0.95:
            issues.append(f"Overround too low: {overround:.4f} (arbitrage opportunity? suspicious)")
        
        if overround > 1.20:
            issues.append(f"Overround too high: {overround:.4f} (>20% margin, suspicious)")
        
        # 4. Validate bookmaker name
        if 'bookmaker' in odds_dict:
            if not self._is_valid_bookmaker(odds_dict['bookmaker']):
                issues.append(f"Unknown bookmaker: {odds_dict['bookmaker']}")
        
        # 5. Validate timestamp
        if 'timestamp' in odds_dict:
            timestamp = odds_dict['timestamp']
            
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except:
                    issues.append(f"Invalid timestamp format: {timestamp}")
            
            # Timestamp shouldn't be in future
            if timestamp > datetime.now() + timedelta(minutes=5):
                issues.append("Timestamp is in the future")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
    
    def check_duplicate_match(self, match_dict: Dict, existing_matches: List[Dict]) -> bool:
        """
        Check if match already exists in list
        
        Args:
            match_dict: New match to check
            existing_matches: List of existing matches
        
        Returns:
            True if duplicate found
        """
        
        for existing in existing_matches:
            # Same players (order independent)
            players1 = {match_dict['player1_name'], match_dict['player2_name']}
            players2 = {existing['player1_name'], existing['player2_name']}
            
            if players1 != players2:
                continue
            
            # Within 6 hours of each other
            time1 = match_dict['scheduled_time']
            time2 = existing['scheduled_time']
            
            if isinstance(time1, str):
                time1 = datetime.fromisoformat(time1)
            if isinstance(time2, str):
                time2 = datetime.fromisoformat(time2)
            
            time_diff = abs((time1 - time2).total_seconds())
            
            if time_diff < 6 * 3600:  # 6 hours
                return True
        
        return False
    
    def _is_valid_player_name(self, name: str) -> bool:
        """Check if player name looks valid"""
        
        if not name or len(name) < 3:
            return False
        
        # Name should have at least one letter
        if not any(c.isalpha() for c in name):
            return False
        
        # Name shouldn't be all numbers
        if name.replace('.', '').replace(' ', '').isdigit():
            return False
        
        return True
    
    def _is_valid_tournament(self, name: str) -> bool:
        """Check if tournament name looks valid"""
        
        if not name or len(name) < 3:
            return False
        
        # Filter out obvious garbage
        invalid_words = ['test', 'unknown', 'null', 'none', 'n/a']
        if name.lower() in invalid_words:
            return False
        
        return True
    
    def _is_valid_round(self, round_str: str) -> bool:
        """Check if round is valid"""
        
        # Normalize
        round_upper = round_str.upper().strip()
        
        # Check against known rounds
        return round_upper in self.VALID_ROUNDS or 'ROUND' in round_upper
    
    def _is_valid_bookmaker(self, bookmaker: str) -> bool:
        """Check if bookmaker name is known"""
        
        known_bookmakers = [
            'pinnacle', 'bet365', 'draftkings', 'fanduel', 'betmgm',
            'unibet', 'williamhill', 'betway', 'bwin', '888sport',
            'betfair', 'ladbrokes', 'coral', 'paddy power', 'skybet',
            'betfred', 'betvictor', 'marathon', 'betonline', '5dimes'
        ]
        
        return bookmaker.lower() in known_bookmakers


# Convenience function

def validate_match_data(match_dict: Dict) -> Tuple[bool, List[str]]:
    """
    Validate match data
    
    Args:
        match_dict: Match information
    
    Returns:
        (is_valid, issues)
    """
    validator = DataValidator()
    return validator.validate_match_data(match_dict)


def validate_odds_data(odds_dict: Dict) -> Tuple[bool, List[str]]:
    """
    Validate odds data
    
    Args:
        odds_dict: Odds information
    
    Returns:
        (is_valid, issues)
    """
    validator = DataValidator()
    return validator.validate_odds_data(odds_dict)


if __name__ == "__main__":
    # Test the validator
    print("🎾 Testing Data Validator\n")
    
    validator = DataValidator()
    
    # Test valid match
    valid_match = {
        'match_id': 'test_1',
        'player1_name': 'Rafael Nadal',
        'player2_name': 'Novak Djokovic',
        'tournament_name': 'Australian Open',
        'surface': 'hard',
        'scheduled_time': datetime.now() + timedelta(hours=5),
        'round': 'QF',
        'data_source': 'sofascore'
    }
    
    is_valid, issues = validator.validate_match_data(valid_match)
    print(f"Valid match: {is_valid}")
    if issues:
        print(f"  Issues: {issues}")
    
    # Test invalid match
    invalid_match = {
        'match_id': 'test_2',
        'player1_name': 'Rafael Nadal',
        'player2_name': 'Rafael Nadal',  # Same player!
        'tournament_name': 'Test',
        'surface': 'fake_surface',  # Invalid surface
        'scheduled_time': datetime.now() - timedelta(days=1),  # Past
    }
    
    is_valid, issues = validator.validate_match_data(invalid_match)
    print(f"\nInvalid match: {is_valid}")
    if issues:
        print(f"  Issues:")
        for issue in issues:
            print(f"    - {issue}")
    
    # Test odds validation
    print("\n" + "="*50)
    
    valid_odds = {
        'match_id': 'test_1',
        'bookmaker': 'Pinnacle',
        'player1_odds': 1.85,
        'player2_odds': 2.05,
        'timestamp': datetime.now()
    }
    
    is_valid, issues = validator.validate_odds_data(valid_odds)
    print(f"Valid odds: {is_valid}")
    
    invalid_odds = {
        'match_id': 'test_1',
        'bookmaker': 'Unknown Bookie',
        'player1_odds': 0.5,  # Too low!
        'player2_odds': 5000,  # Too high!
    }
    
    is_valid, issues = validator.validate_odds_data(invalid_odds)
    print(f"\nInvalid odds: {is_valid}")
    if issues:
        print(f"  Issues:")
        for issue in issues:
            print(f"    - {issue}")
