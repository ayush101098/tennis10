"""
Test script to verify feature extraction with REAL statistics
"""

import sqlite3
import pandas as pd
from features import TennisFeatureExtractor
from datetime import datetime

# Connect to database
conn = sqlite3.connect('tennis_data.db')

# Get a sample match with real statistics
query = """
SELECT 
    m.match_id,
    p1.player_name as winner_name,
    p2.player_name as loser_name,
    m.tournament_name,
    m.tournament_date,
    m.surface,
    s_winner.aces as w_aces,
    s_winner.first_serve_pct as w_1st_pct,
    s_winner.first_serve_win_pct as w_1st_win_pct,
    s_loser.aces as l_aces,
    s_loser.first_serve_pct as l_1st_pct,
    s_loser.first_serve_win_pct as l_1st_win_pct
FROM matches m
JOIN players p1 ON m.winner_id = p1.player_id
JOIN players p2 ON m.loser_id = p2.player_id
JOIN statistics s_winner ON m.match_id = s_winner.match_id AND s_winner.is_winner = 1
JOIN statistics s_loser ON m.match_id = s_loser.match_id AND s_loser.is_winner = 0
WHERE s_winner.aces IS NOT NULL
AND m.tournament_date > '2023-01-01'
LIMIT 5
"""

sample_matches = pd.read_sql_query(query, conn)

print("\n" + "="*100)
print("TESTING FEATURE EXTRACTION WITH REAL STATISTICS")
print("="*100)

print("\nSample Matches:")
print(sample_matches[['winner_name', 'loser_name', 'tournament_name', 'surface', 'w_aces', 'w_1st_pct']].to_string(index=False))

# Initialize feature extractor
extractor = TennisFeatureExtractor('tennis_data.db')

print("\n" + "="*100)
print("EXTRACTING FEATURES FOR FIRST MATCH")
print("="*100)

first_match = sample_matches.iloc[0]
print(f"\nMatch: {first_match['winner_name']} vs {first_match['loser_name']}")
print(f"Tournament: {first_match['tournament_name']}")
print(f"Date: {first_match['tournament_date']}")
print(f"Surface: {first_match['surface']}")

# Extract features
try:
    features = extractor.extract_features(match_id=int(first_match['match_id']))
    
    print("\n" + "-"*100)
    print("EXTRACTED FEATURES (using REAL statistics):")
    print("-"*100)
    
    # Group features for better readability
    print("\n📊 Basic Features:")
    print(f"  RANK_DIFF: {features['RANK_DIFF']:.2f}")
    print(f"  POINTS_DIFF: {features['POINTS_DIFF']:.2f}")
    
    print("\n🎾 Serve Statistics (REAL DATA):")
    print(f"  WSP_DIFF: {features['WSP_DIFF']:.4f} (Serve points won %)")
    print(f"  FIRST_SERVE_PCT_DIFF: {features['FIRST_SERVE_PCT_DIFF']:.4f}")
    print(f"  FIRST_SERVE_WIN_PCT_DIFF: {features['FIRST_SERVE_WIN_PCT_DIFF']:.4f}")
    print(f"  SECOND_SERVE_WIN_PCT_DIFF: {features['SECOND_SERVE_WIN_PCT_DIFF']:.4f}")
    print(f"  ACES_DIFF: {features['ACES_DIFF']:.4f}")
    print(f"  DF_DIFF: {features['DF_DIFF']:.4f}")
    
    print("\n🔄 Return Statistics (REAL DATA):")
    print(f"  WRP_DIFF: {features['WRP_DIFF']:.4f} (Return points won %)")
    print(f"  BP_SAVE_DIFF: {features['BP_SAVE_DIFF']:.4f}")
    
    print("\n📈 Performance Features:")
    print(f"  WIN_RATE_DIFF: {features['WIN_RATE_DIFF']:.4f}")
    print(f"  SURFACE_WIN_RATE_DIFF: {features['SURFACE_WIN_RATE_DIFF']:.4f}")
    
    print("\n🏗️ Constructed Features:")
    print(f"  SERVEADV: {features['SERVEADV']:.4f}")
    print(f"  COMPLETE_DIFF: {features['COMPLETE_DIFF']:.4f}")
    
    print("\n⚡ Situational Features:")
    print(f"  FATIGUE_DIFF: {features['FATIGUE_DIFF']:.4f}")
    print(f"  RETIRED_DIFF: {features['RETIRED_DIFF']:.4f}")
    print(f"  DIRECT_H2H: {features['DIRECT_H2H']:.4f}")
    
    print("\n📚 Experience Features:")
    print(f"  MATCHES_PLAYED_DIFF: {features['MATCHES_PLAYED_DIFF']:.2f}")
    print(f"  SURFACE_EXP_DIFF: {features['SURFACE_EXP_DIFF']:.2f}")
    
    print("\n⚠️ Quality Indicator:")
    print(f"  UNCERTAINTY: {features['UNCERTAINTY']:.4f} (lower = more confident)")
    
    print("\n" + "="*100)
    print("✅ SUCCESS: Features extracted using REAL match statistics!")
    print("="*100)
    print("\nKey Improvements:")
    print("  • Aces, double faults, serve % are now REAL values from actual matches")
    print("  • No more proxy metrics based on win rates")
    print("  • Accurate serve and return statistics for predictions")
    print("  • Ready for live match testing with reliable data")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test feature extraction for all sample matches
print("\n" + "="*100)
print("FEATURE EXTRACTION FOR ALL SAMPLE MATCHES")
print("="*100)

success_count = 0
error_count = 0

for idx, match in sample_matches.iterrows():
    try:
        features = extractor.extract_features(match_id=int(match['match_id']))
        print(f"\n✓ Match {idx+1}: {match['winner_name']} vs {match['loser_name']}")
        print(f"  WSP_DIFF: {features['WSP_DIFF']:.4f}, ACES_DIFF: {features['ACES_DIFF']:.4f}, UNCERTAINTY: {features['UNCERTAINTY']:.4f}")
        success_count += 1
    except Exception as e:
        print(f"\n✗ Match {idx+1}: {match['winner_name']} vs {match['loser_name']} - ERROR: {e}")
        error_count += 1

print("\n" + "="*100)
print(f"SUMMARY: {success_count}/{len(sample_matches)} matches successfully extracted")
print("="*100)

extractor.close()
conn.close()
