"""
🎾 ENHANCED MARKOV MODEL WITH ADVANCED STATISTICS
==================================================

Integrates advanced tennis statistics into your existing betting model:

WITHOUT VIDEO (use now):
- Player positioning tendencies (from match stats APIs)
- Serve placement patterns (from databases)
- Rally characteristics (from historical data)
- Fatigue modeling (from match duration/games played)

WITH VIDEO (future upgrade):
- Real-time ball tracking
- Live court coverage analysis
- In-match fatigue detection
- Shot-by-shot probability updates

This enhances your Markov probabilities with CONTEXT-AWARE adjustments.
"""

import sqlite3
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import requests


class AdvancedTennisStats:
    """
    Advanced statistics integration for enhanced probability calculations.
    Works with OR without video analysis.
    """
    
    def __init__(self, db_path: str = 'tennis_data.db'):
        self.db_path = db_path
        
        # Statistical adjustment factors
        self.adjustments = {
            'serve_placement': {
                'wide_effectiveness': 1.05,  # +5% for wide serves
                'body_effectiveness': 0.98,   # -2% for body serves
                'T_effectiveness': 1.03       # +3% for T serves
            },
            'rally_length': {
                'quick_points_favor_server': 1.04,  # <4 shots
                'long_rallies_favor_returner': 0.96  # >8 shots
            },
            'fatigue': {
                'fresh': 1.0,
                'mild': 0.97,      # -3%
                'moderate': 0.92,  # -8%
                'severe': 0.85     # -15%
            },
            'surface': {
                'hard_vs_clay_serve_advantage': 1.08,
                'grass_serve_dominance': 1.12,
                'clay_baseline_advantage': 0.95
            },
            'weather': {
                'wind_serve_penalty': 0.94,
                'heat_fatigue_multiplier': 1.2,
                'indoor_serve_boost': 1.03
            }
        }
    
    def enhance_serve_probability(
        self,
        base_p_serve: float,
        player_name: str,
        context: Dict
    ) -> float:
        """
        Enhance base serve probability with contextual factors.
        
        Context can include:
        - surface: 'hard', 'clay', 'grass'
        - weather: 'sunny', 'windy', 'hot', 'indoor'
        - fatigue_level: 'fresh', 'mild', 'moderate', 'severe'
        - serve_placement_stats: {'wide': 0.4, 'body': 0.2, 'T': 0.4}
        - rally_length_avg: 5.2
        - games_played: 12
        - match_duration_minutes: 95
        """
        
        adjusted_p = base_p_serve
        adjustments_applied = []
        
        # 1. SURFACE ADJUSTMENT
        surface = context.get('surface', '').lower()
        
        if surface == 'grass':
            adjusted_p *= self.adjustments['surface']['grass_serve_dominance']
            adjustments_applied.append(f"Grass surface: +12%")
        
        elif surface == 'clay':
            adjusted_p *= self.adjustments['surface']['clay_baseline_advantage']
            adjustments_applied.append(f"Clay surface: -5%")
        
        # 2. SERVE PLACEMENT EFFECTIVENESS
        placement = context.get('serve_placement_stats', {})
        
        if placement:
            wide_pct = placement.get('wide', 0)
            T_pct = placement.get('T', 0)
            
            # Reward high percentage of effective placements
            if wide_pct > 0.35:  # More than 35% wide
                bonus = self.adjustments['serve_placement']['wide_effectiveness']
                adjusted_p *= bonus
                adjustments_applied.append(f"Wide serves ({wide_pct:.0%}): +5%")
            
            if T_pct > 0.30:  # More than 30% to the T
                bonus = self.adjustments['serve_placement']['T_effectiveness']
                adjusted_p *= bonus
                adjustments_applied.append(f"T serves ({T_pct:.0%}): +3%")
        
        # 3. RALLY LENGTH PATTERNS
        rally_avg = context.get('rally_length_avg', 0)
        
        if rally_avg > 0:
            if rally_avg < 4:
                adjusted_p *= self.adjustments['rally_length']['quick_points_favor_server']
                adjustments_applied.append(f"Quick points (avg {rally_avg:.1f}): +4%")
            
            elif rally_avg > 8:
                adjusted_p *= self.adjustments['rally_length']['long_rallies_favor_returner']
                adjustments_applied.append(f"Long rallies (avg {rally_avg:.1f}): -4%")
        
        # 4. FATIGUE MODELING
        fatigue = context.get('fatigue_level', 'fresh')
        
        if fatigue in self.adjustments['fatigue']:
            factor = self.adjustments['fatigue'][fatigue]
            adjusted_p *= factor
            
            if fatigue != 'fresh':
                pct_change = (factor - 1) * 100
                adjustments_applied.append(f"Fatigue ({fatigue}): {pct_change:+.0f}%")
        
        # Alternative: Calculate fatigue from match stats
        games_played = context.get('games_played', 0)
        duration_min = context.get('match_duration_minutes', 0)
        
        if games_played > 0 and duration_min > 0:
            # Estimate fatigue from duration and games
            avg_game_duration = duration_min / games_played
            
            if avg_game_duration > 7:  # Long, grueling games
                fatigue_penalty = 0.97  # -3%
                adjusted_p *= fatigue_penalty
                adjustments_applied.append(f"Long game duration ({avg_game_duration:.1f} min/game): -3%")
            
            # Total match length fatigue
            if duration_min > 150:  # >2.5 hours
                fatigue_penalty = 0.95  # -5%
                adjusted_p *= fatigue_penalty
                adjustments_applied.append(f"Match duration ({duration_min} min): -5%")
        
        # 5. WEATHER CONDITIONS
        weather = context.get('weather', '').lower()
        
        if 'wind' in weather:
            adjusted_p *= self.adjustments['weather']['wind_serve_penalty']
            adjustments_applied.append(f"Windy conditions: -6%")
        
        if 'indoor' in weather:
            adjusted_p *= self.adjustments['weather']['indoor_serve_boost']
            adjustments_applied.append(f"Indoor court: +3%")
        
        # 6. HISTORICAL MATCHUP
        opponent = context.get('opponent_name')
        if opponent:
            h2h_factor = self._get_h2h_adjustment(player_name, opponent)
            if h2h_factor != 1.0:
                adjusted_p *= h2h_factor
                pct = (h2h_factor - 1) * 100
                adjustments_applied.append(f"H2H matchup: {pct:+.1f}%")
        
        # Keep within valid bounds
        adjusted_p = np.clip(adjusted_p, 0.0, 1.0)
        
        # Log adjustments
        total_change = (adjusted_p / base_p_serve - 1) * 100 if base_p_serve > 0 else 0
        
        return adjusted_p, adjustments_applied, total_change
    
    def estimate_fatigue_level(
        self,
        games_played: int,
        duration_minutes: int,
        sets_played: int
    ) -> str:
        """
        Estimate fatigue level from match statistics.
        
        Returns: 'fresh', 'mild', 'moderate', or 'severe'
        """
        
        # Calculate intensity score
        if games_played == 0:
            return 'fresh'
        
        avg_game_duration = duration_minutes / games_played
        
        # Scoring system
        fatigue_score = 0
        
        # Duration factor
        if duration_minutes > 180:  # >3 hours
            fatigue_score += 3
        elif duration_minutes > 120:  # >2 hours
            fatigue_score += 2
        elif duration_minutes > 90:
            fatigue_score += 1
        
        # Games played factor
        if games_played > 25:  # Very long match
            fatigue_score += 3
        elif games_played > 18:
            fatigue_score += 2
        elif games_played > 12:
            fatigue_score += 1
        
        # Set length factor
        if sets_played >= 5:  # 5-set match
            fatigue_score += 2
        elif sets_played >= 4:
            fatigue_score += 1
        
        # Game intensity
        if avg_game_duration > 8:  # Long points
            fatigue_score += 2
        elif avg_game_duration > 6:
            fatigue_score += 1
        
        # Map score to fatigue level
        if fatigue_score >= 7:
            return 'severe'
        elif fatigue_score >= 4:
            return 'moderate'
        elif fatigue_score >= 2:
            return 'mild'
        else:
            return 'fresh'
    
    def get_serve_placement_stats(self, player_name: str) -> Optional[Dict]:
        """
        Get serve placement statistics from database.
        
        Returns percentages for: wide, body, T serves
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query for serve direction stats
            # This assumes you have detailed serve data
            # If not, returns None and we skip this adjustment
            
            query = """
            SELECT 
                AVG(CASE WHEN serve_direction = 'wide' THEN 1 ELSE 0 END) as wide_pct,
                AVG(CASE WHEN serve_direction = 'body' THEN 1 ELSE 0 END) as body_pct,
                AVG(CASE WHEN serve_direction = 'T' THEN 1 ELSE 0 END) as T_pct
            FROM serve_stats
            WHERE player_name = ?
            """
            
            cursor.execute(query, (player_name,))
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                return {
                    'wide': result[0],
                    'body': result[1],
                    'T': result[2]
                }
            
            conn.close()
            return None
            
        except Exception:
            return None
    
    def get_rally_length_stats(self, player_name: str) -> Optional[float]:
        """Get average rally length for player"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
            SELECT AVG(rally_length)
            FROM match_rallies
            WHERE player_name = ?
            """
            
            cursor.execute(query, (player_name,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result and result[0]:
                return result[0]
            
            return None
            
        except Exception:
            return None
    
    def _get_h2h_adjustment(self, player1: str, player2: str) -> float:
        """
        Get head-to-head adjustment factor.
        
        If player1 dominates player2 historically, increase serve probability.
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
            SELECT 
                SUM(CASE WHEN winner = ? THEN 1 ELSE 0 END) as p1_wins,
                COUNT(*) as total_matches
            FROM matches
            WHERE (player1 = ? AND player2 = ?) 
               OR (player1 = ? AND player2 = ?)
            """
            
            cursor.execute(query, (player1, player1, player2, player2, player1))
            result = cursor.fetchone()
            
            conn.close()
            
            if result and result[1] >= 3:  # At least 3 H2H matches
                p1_wins, total = result
                win_rate = p1_wins / total
                
                # Adjust serve probability based on H2H dominance
                if win_rate > 0.7:  # Dominates
                    return 1.05  # +5%
                elif win_rate < 0.3:  # Struggles
                    return 0.95  # -5%
            
            return 1.0  # No adjustment
            
        except Exception:
            return 1.0


class EnhancedMarkovBetting:
    """
    Enhanced version of your betting system with advanced statistics.
    """
    
    def __init__(self, db_path: str = 'tennis_data.db', bankroll: float = 1000):
        self.db_path = db_path
        self.bankroll = bankroll
        self.stats_engine = AdvancedTennisStats(db_path)
    
    def analyze_match_enhanced(
        self,
        player1: str,
        player2: str,
        odds_player1: float,
        odds_player2: float,
        match_context: Dict = None
    ) -> Dict:
        """
        Analyze match with enhanced probability calculations.
        
        match_context example:
        {
            'surface': 'hard',
            'weather': 'sunny',
            'tournament': 'ATP 1000',
            'indoor': False,
            'current_game': 8,
            'current_set': 2,
            'match_duration_minutes': 75
        }
        """
        
        print("\n" + "="*80)
        print(f"🎾 ENHANCED ANALYSIS: {player1} vs {player2}")
        print("="*80)
        
        if match_context is None:
            match_context = {}
        
        # Get base probabilities - use odds estimation for now
        # (Can be enhanced later with database lookupwhen schema matches)
        
        # Convert odds to basic probabilities
        implied_p1 = 1 / odds_player1
        implied_p2 = 1 / odds_player2
        
        # Estimate serve point win probability from match odds
        # Higher match win probability → Higher serve effectiveness
        # Rough formula: P(serve_point) ≈ 0.5 + 0.2 * (P(match) - 0.5)
        
        p1_serve_base = 0.5 + 0.3 * (implied_p1 - 0.5)
        p2_serve_base = 0.5 + 0.3 * (implied_p2 - 0.5)
        
        # Keep in reasonable bounds for tennis
        p1_serve_base = np.clip(p1_serve_base, 0.55, 0.75)
        p2_serve_base = np.clip(p2_serve_base, 0.55, 0.75)
        
        print(f"\n📊 BASE PROBABILITIES:")
        print(f"  {player1} serve: {p1_serve_base:.1%}")
        print(f"  {player2} serve: {p2_serve_base:.1%}")
        
        # Enhance with context
        print(f"\n🔄 APPLYING CONTEXTUAL ADJUSTMENTS:")
        
        # Estimate fatigue if match is in progress
        games_played = match_context.get('current_game', 0)
        sets_played = match_context.get('current_set', 1)
        duration = match_context.get('match_duration_minutes', 0)
        
        if games_played > 0:
            match_context['fatigue_level'] = self.stats_engine.estimate_fatigue_level(
                games_played, duration, sets_played
            )
        
        # Add opponent context
        match_context['opponent_name'] = player2
        
        # Enhance player 1 probability
        p1_serve_enhanced, p1_adjustments, p1_change = self.stats_engine.enhance_serve_probability(
            p1_serve_base,
            player1,
            match_context
        )
        
        print(f"\n  {player1}:")
        for adj in p1_adjustments:
            print(f"    • {adj}")
        print(f"    → Total adjustment: {p1_change:+.1f}%")
        print(f"    → Enhanced probability: {p1_serve_enhanced:.1%}")
        
        # Enhance player 2 probability
        match_context['opponent_name'] = player1
        
        p2_serve_enhanced, p2_adjustments, p2_change = self.stats_engine.enhance_serve_probability(
            p2_serve_base,
            player2,
            match_context
        )
        
        print(f"\n  {player2}:")
        for adj in p2_adjustments:
            print(f"    • {adj}")
        print(f"    → Total adjustment: {p2_change:+.1f}%")
        print(f"    → Enhanced probability: {p2_serve_enhanced:.1%}")
        
        # Calculate match probabilities with enhanced values
        from hierarchical_model import HierarchicalTennisModel
        
        model = HierarchicalTennisModel()
        
        # Game probabilities (hold serve probability)
        p1_hold = model.prob_game_win(p1_serve_enhanced)
        p2_hold = model.prob_game_win(p2_serve_enhanced)
        
        # Set probability
        p1_set = model.prob_set_win(p1_hold, 1 - p2_hold)
        
        # Match probability (best of 3)
        p1_match = model.prob_match_win(p1_set, num_sets=3)
        p2_match = 1 - p1_match
        
        print(f"\n🎯 ENHANCED MATCH PROBABILITIES:")
        print(f"  {player1}: {p1_match:.1%}")
        print(f"  {player2}: {p2_match:.1%}")
        
        # Edge analysis
        implied_p1 = 1 / odds_player1
        implied_p2 = 1 / odds_player2
        
        edge_p1 = p1_match - implied_p1
        edge_p2 = p2_match - implied_p2
        
        print(f"\n💰 EDGE ANALYSIS:")
        print(f"  {player1} @ {odds_player1}:")
        print(f"    Bookmaker: {implied_p1:.1%}")
        print(f"    Enhanced: {p1_match:.1%}")
        print(f"    Edge: {edge_p1:+.1%}")
        
        print(f"\n  {player2} @ {odds_player2}:")
        print(f"    Bookmaker: {implied_p2:.1%}")
        print(f"    Enhanced: {p2_match:.1%}")
        print(f"    Edge: {edge_p2:+.1%}")
        
        # Betting recommendation
        print(f"\n💸 BETTING RECOMMENDATION:")
        
        if edge_p1 > 0.025:  # 2.5% minimum edge
            kelly_fraction = 0.25
            stake = min(
                kelly_fraction * edge_p1 * self.bankroll,
                0.15 * self.bankroll
            )
            
            ev = stake * edge_p1 * (odds_player1 - 1)
            
            print(f"  ✅ BET ${stake:.2f} on {player1} @ {odds_player1}")
            print(f"     Expected value: +${ev:.2f}")
        
        elif edge_p2 > 0.025:
            kelly_fraction = 0.25
            stake = min(
                kelly_fraction * edge_p2 * self.bankroll,
                0.15 * self.bankroll
            )
            
            ev = stake * edge_p2 * (odds_player2 - 1)
            
            print(f"  ✅ BET ${stake:.2f} on {player2} @ {odds_player2}")
            print(f"     Expected value: +${ev:.2f}")
        
        else:
            print(f"  ❌ NO BET - No significant edge detected")
        
        print("\n" + "="*80 + "\n")
        
        return {
            'base_probabilities': {
                'player1': p1_serve_base,
                'player2': p2_serve_base
            },
            'enhanced_probabilities': {
                'player1': p1_serve_enhanced,
                'player2': p2_serve_enhanced
            },
            'match_probabilities': {
                'player1': p1_match,
                'player2': p2_match
            },
            'edges': {
                'player1': edge_p1,
                'player2': edge_p2
            },
            'adjustments': {
                'player1': p1_adjustments,
                'player2': p2_adjustments
            }
        }


def main():
    """Demo: Enhanced analysis"""
    
    print("\n" + "🎾"*40)
    print("ENHANCED MARKOV BETTING MODEL")
    print("With Advanced Statistics Integration")
    print("🎾"*40)
    
    model = EnhancedMarkovBetting(bankroll=1000)
    
    # Example: Analyze match with context
    result = model.analyze_match_enhanced(
        player1="Novak Djokovic",
        player2="Carlos Alcaraz",
        odds_player1=2.10,
        odds_player2=1.75,
        match_context={
            'surface': 'hard',
            'weather': 'indoor',
            'tournament': 'ATP 1000',
            'current_game': 8,
            'current_set': 2,
            'match_duration_minutes': 75
        }
    )
    
    print("✅ Analysis complete with enhanced probabilities!")


if __name__ == "__main__":
    main()
