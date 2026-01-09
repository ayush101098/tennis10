"""
Comprehensive Player Intelligence Fetcher
Fetches real-time data from multiple sources for accurate match predictions
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json

class PlayerIntelligence:
    def __init__(self, db_path='tennis_betting.db'):
        self.db_path = db_path
        self.player_data = {}
        
    def get_player_id(self, player_name):
        """Get player ID from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Try exact match first
        cursor.execute("""
            SELECT player_id, player_name FROM players 
            WHERE player_name LIKE ?
        """, (f'%{player_name}%',))
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            return results[0][0], results[0][1]
        return None, None
    
    def fetch_database_intelligence(self, player_id, player_name):
        """Fetch comprehensive data from local database"""
        conn = sqlite3.connect(self.db_path)
        
        intelligence = {
            'player_id': player_id,
            'player_name': player_name,
            'database_stats': {},
            'recent_form': {},
            'surface_performance': {},
            'special_parameters': {}
        }
        
        # Get special parameters
        sp_query = """
            SELECT momentum_score, best_surface, surface_mastery, 
                   clutch_performance, bp_defense_rate, first_serve_win_pct,
                   consistency_rating, peak_rating, career_win_rate, total_matches
            FROM special_parameters 
            WHERE player_id = ?
        """
        sp_df = pd.read_sql_query(sp_query, conn, params=(player_id,))
        
        if len(sp_df) > 0:
            sp_row = sp_df.iloc[0]
            intelligence['special_parameters'] = {
                'momentum_score': float(sp_row['momentum_score']),
                'best_surface': sp_row['best_surface'],
                'surface_mastery': float(sp_row['surface_mastery']),
                'clutch_performance': float(sp_row['clutch_performance']),
                'bp_defense_rate': float(sp_row['bp_defense_rate']),
                'first_serve_win_pct': float(sp_row['first_serve_win_pct']),
                'consistency_rating': float(sp_row['consistency_rating']),
                'peak_rating': float(sp_row['peak_rating']),
                'career_win_rate': float(sp_row['career_win_rate']),
                'total_matches': int(sp_row['total_matches'])
            }
        
        # Get recent matches (last 20)
        wins_query = """
            SELECT tourney_date, tourney_name, surface, tourney_level,
                   w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_bpSaved, w_bpFaced,
                   winner_rank, loser_rank
            FROM matches 
            WHERE winner_id = ?
            ORDER BY tourney_date DESC
            LIMIT 20
        """
        wins_df = pd.read_sql_query(wins_query, conn, params=(player_id,))
        
        losses_query = """
            SELECT tourney_date, tourney_name, surface, tourney_level,
                   l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_bpSaved, l_bpFaced,
                   loser_rank, winner_rank
            FROM matches 
            WHERE loser_id = ?
            ORDER BY tourney_date DESC
            LIMIT 20
        """
        losses_df = pd.read_sql_query(losses_query, conn, params=(player_id,))
        
        # Calculate recent form (last 20 matches)
        total_recent = len(wins_df) + len(losses_df)
        recent_win_pct = len(wins_df) / total_recent if total_recent > 0 else 0
        
        # Calculate serve statistics from wins
        if len(wins_df) > 0 and wins_df['w_svpt'].sum() > 0:
            total_serve_pts = wins_df['w_svpt'].sum()
            total_1st_in = wins_df['w_1stIn'].sum()
            total_1st_won = wins_df['w_1stWon'].sum()
            total_2nd_won = wins_df['w_2ndWon'].sum()
            total_bp_saved = wins_df['w_bpSaved'].sum()
            total_bp_faced = wins_df['w_bpFaced'].sum()
            
            serve_win_pct = ((total_1st_won + total_2nd_won) / total_serve_pts) * 100
            first_serve_pct = (total_1st_in / total_serve_pts) * 100
            bp_save_pct = (total_bp_saved / total_bp_faced * 100) if total_bp_faced > 0 else 0
        else:
            serve_win_pct = 0
            first_serve_pct = 0
            bp_save_pct = 0
        
        intelligence['recent_form'] = {
            'record': f"{len(wins_df)}-{len(losses_df)}",
            'win_percentage': round(recent_win_pct * 100, 1),
            'serve_win_pct': round(serve_win_pct, 1),
            'first_serve_pct': round(first_serve_pct, 1),
            'bp_save_pct': round(bp_save_pct, 1),
            'last_match_date': wins_df['tourney_date'].max() if len(wins_df) > 0 else None
        }
        
        # Surface-specific performance
        for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
            surface_wins = wins_df[wins_df['surface'] == surface]
            surface_losses = losses_df[losses_df['surface'] == surface]
            surface_total = len(surface_wins) + len(surface_losses)
            
            if surface_total > 0:
                intelligence['surface_performance'][surface] = {
                    'matches': surface_total,
                    'win_pct': round((len(surface_wins) / surface_total) * 100, 1),
                    'record': f"{len(surface_wins)}-{len(surface_losses)}"
                }
        
        conn.close()
        return intelligence
    
    def calculate_return_stats(self, player_id):
        """Calculate return statistics (opponent's serve stats when player won)"""
        conn = sqlite3.connect(self.db_path)
        
        # When player wins, look at opponent's serve stats
        return_query = """
            SELECT l_svpt, l_1stIn, l_1stWon, l_2ndWon
            FROM matches
            WHERE winner_id = ?
            ORDER BY tourney_date DESC
            LIMIT 20
        """
        df = pd.read_sql_query(return_query, conn, params=(player_id,))
        
        if len(df) > 0 and df['l_svpt'].sum() > 0:
            opp_serve_pts = df['l_svpt'].sum()
            opp_won_pts = df['l_1stWon'].sum() + df['l_2ndWon'].sum()
            player_return_won = opp_serve_pts - opp_won_pts
            return_win_pct = (player_return_won / opp_serve_pts) * 100
        else:
            return_win_pct = 0
        
        conn.close()
        return round(return_win_pct, 1)
    
    def fetch_h2h(self, player1_id, player2_id):
        """Get head-to-head record"""
        conn = sqlite3.connect(self.db_path)
        
        h2h_query = """
            SELECT tourney_date, tourney_name, surface, score,
                   CASE 
                       WHEN winner_id = ? THEN 'W'
                       ELSE 'L'
                   END as result
            FROM matches
            WHERE (winner_id = ? AND loser_id = ?)
               OR (winner_id = ? AND loser_id = ?)
            ORDER BY tourney_date DESC
        """
        
        df = pd.read_sql_query(h2h_query, conn, params=(
            player1_id, player1_id, player2_id, player2_id, player1_id
        ))
        
        conn.close()
        
        if len(df) > 0:
            p1_wins = len(df[df['result'] == 'W'])
            p2_wins = len(df) - p1_wins
            return {
                'total_matches': len(df),
                'player1_wins': p1_wins,
                'player2_wins': p2_wins,
                'last_5': df['result'].head(5).tolist(),
                'last_meeting': df.iloc[0]['tourney_date'] if len(df) > 0 else None
            }
        
        return {'total_matches': 0, 'player1_wins': 0, 'player2_wins': 0}
    
    def generate_match_intelligence(self, player1_name, player2_name, surface='Hard'):
        """Generate comprehensive intelligence for a matchup"""
        print("="*80)
        print(f"MATCH INTELLIGENCE: {player1_name} vs {player2_name}")
        print(f"Surface: {surface}")
        print("="*80)
        
        # Get player IDs
        p1_id, p1_full_name = self.get_player_id(player1_name)
        p2_id, p2_full_name = self.get_player_id(player2_name)
        
        if not p1_id or not p2_id:
            print(f"❌ Players not found in database")
            print(f"   {player1_name}: {'✓' if p1_id else '✗'}")
            print(f"   {player2_name}: {'✓' if p2_id else '✗'}")
            return None
        
        print(f"\n✅ Players Found:")
        print(f"   Player 1: {p1_full_name} (ID: {p1_id})")
        print(f"   Player 2: {p2_full_name} (ID: {p2_id})")
        
        # Fetch intelligence for both players
        p1_intel = self.fetch_database_intelligence(p1_id, p1_full_name)
        p2_intel = self.fetch_database_intelligence(p2_id, p2_full_name)
        
        # Add return stats
        p1_intel['recent_form']['return_win_pct'] = self.calculate_return_stats(p1_id)
        p2_intel['recent_form']['return_win_pct'] = self.calculate_return_stats(p2_id)
        
        # Get H2H
        h2h = self.fetch_h2h(p1_id, p2_id)
        
        # Display results
        self._display_intelligence(p1_intel, p2_intel, h2h, surface)
        
        # Store for model usage
        matchup_data = {
            'player1': p1_intel,
            'player2': p2_intel,
            'h2h': h2h,
            'surface': surface,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save to JSON
        output_file = f'match_intelligence_{p1_full_name.replace(" ", "_")}_vs_{p2_full_name.replace(" ", "_")}.json'
        with open(output_file, 'w') as f:
            json.dump(matchup_data, f, indent=2, default=str)
        
        print(f"\n💾 Intelligence saved to: {output_file}")
        
        return matchup_data
    
    def _display_intelligence(self, p1, p2, h2h, surface):
        """Display formatted intelligence report"""
        print(f"\n{'='*80}")
        print(f"PLAYER 1: {p1['player_name']}")
        print(f"{'='*80}")
        
        if p1['special_parameters']:
            sp = p1['special_parameters']
            print(f"\n🎯 SPECIAL PARAMETERS:")
            print(f"   Momentum Score:     {sp['momentum_score']:.3f}")
            print(f"   Best Surface:       {sp['best_surface']} (Mastery: {sp['surface_mastery']:.1f}%)")
            print(f"   Clutch Performance: {sp['clutch_performance']:.1f}%")
            print(f"   BP Defense Rate:    {sp['bp_defense_rate']:.1f}%")
            print(f"   Career Win Rate:    {sp['career_win_rate']:.1f}%")
            print(f"   Total Matches:      {sp['total_matches']}")
        
        if p1['recent_form']:
            rf = p1['recent_form']
            print(f"\n📊 RECENT FORM (Last 20 matches):")
            print(f"   Record:            {rf['record']} ({rf['win_percentage']:.1f}%)")
            print(f"   Serve Win %:       {rf['serve_win_pct']:.1f}%")
            print(f"   1st Serve %:       {rf['first_serve_pct']:.1f}%")
            print(f"   Return Win %:      {rf['return_win_pct']:.1f}%")
            print(f"   BP Save %:         {rf['bp_save_pct']:.1f}%")
        
        if p1['surface_performance'].get(surface):
            sp = p1['surface_performance'][surface]
            print(f"\n🎾 {surface.upper()} COURT PERFORMANCE:")
            print(f"   Record:  {sp['record']} ({sp['win_pct']:.1f}%)")
            print(f"   Matches: {sp['matches']}")
        
        print(f"\n{'='*80}")
        print(f"PLAYER 2: {p2['player_name']}")
        print(f"{'='*80}")
        
        if p2['special_parameters']:
            sp = p2['special_parameters']
            print(f"\n🎯 SPECIAL PARAMETERS:")
            print(f"   Momentum Score:     {sp['momentum_score']:.3f}")
            print(f"   Best Surface:       {sp['best_surface']} (Mastery: {sp['surface_mastery']:.1f}%)")
            print(f"   Clutch Performance: {sp['clutch_performance']:.1f}%")
            print(f"   BP Defense Rate:    {sp['bp_defense_rate']:.1f}%")
            print(f"   Career Win Rate:    {sp['career_win_rate']:.1f}%")
            print(f"   Total Matches:      {sp['total_matches']}")
        
        if p2['recent_form']:
            rf = p2['recent_form']
            print(f"\n📊 RECENT FORM (Last 20 matches):")
            print(f"   Record:            {rf['record']} ({rf['win_percentage']:.1f}%)")
            print(f"   Serve Win %:       {rf['serve_win_pct']:.1f}%")
            print(f"   1st Serve %:       {rf['first_serve_pct']:.1f}%)")
            print(f"   Return Win %:      {rf['return_win_pct']:.1f}%")
            print(f"   BP Save %:         {rf['bp_save_pct']:.1f}%")
        
        if p2['surface_performance'].get(surface):
            sp = p2['surface_performance'][surface]
            print(f"\n🎾 {surface.upper()} COURT PERFORMANCE:")
            print(f"   Record:  {sp['record']} ({sp['win_pct']:.1f}%)")
            print(f"   Matches: {sp['matches']}")
        
        if h2h['total_matches'] > 0:
            print(f"\n{'='*80}")
            print(f"HEAD-TO-HEAD")
            print(f"{'='*80}")
            print(f"\n🥊 Total Meetings: {h2h['total_matches']}")
            print(f"   {p1['player_name']}: {h2h['player1_wins']} wins")
            print(f"   {p2['player_name']}: {h2h['player2_wins']} wins")
            if h2h.get('last_5'):
                print(f"   Last 5: {' '.join(h2h['last_5'])} (from P1 perspective)")
        else:
            print(f"\n{'='*80}")
            print(f"HEAD-TO-HEAD: No previous meetings")
            print(f"{'='*80}")


if __name__ == "__main__":
    # Initialize intelligence fetcher
    intel = PlayerIntelligence()
    
    # Fetch matchup intelligence
    matchup = intel.generate_match_intelligence(
        player1_name="Mochizuki",
        player2_name="Cerundolo",
        surface="Hard"
    )
    
    print(f"\n{'='*80}")
    print(f"RECOMMENDED MODEL INPUTS")
    print(f"{'='*80}")
    
    if matchup:
        p1 = matchup['player1']
        p2 = matchup['player2']
        
        print(f"\nFor Live Calculator:")
        print(f"   Player 1: {p1['player_name']}")
        print(f"   - Serve Win %:  {p1['recent_form']['serve_win_pct']}")
        print(f"   - Return Win %: {p1['recent_form']['return_win_pct']}")
        
        print(f"\n   Player 2: {p2['player_name']}")
        print(f"   - Serve Win %:  {p2['recent_form']['serve_win_pct']}")
        print(f"   - Return Win %: {p2['recent_form']['return_win_pct']}")
        
        print(f"\nSurface: {matchup['surface']}")
        print(f"H2H: {matchup['h2h']['player1_wins']}-{matchup['h2h']['player2_wins']}")
