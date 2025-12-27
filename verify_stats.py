import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('tennis_data.db')

# Query sample matches with real statistics
query = """
SELECT 
    p1.player_name as winner_name,
    p2.player_name as loser_name,
    m.tournament_name,
    m.surface,
    s_winner.aces as w_aces,
    s_winner.double_faults as w_double_faults,
    s_winner.first_serve_pct as w_first_serve_pct,
    s_winner.first_serve_win_pct as w_first_serve_win_pct,
    s_winner.break_point_save_pct as w_break_point_save_pct,
    s_loser.aces as l_aces,
    s_loser.double_faults as l_double_faults,
    s_loser.first_serve_pct as l_first_serve_pct,
    s_loser.first_serve_win_pct as l_first_serve_win_pct
FROM matches m
JOIN players p1 ON m.winner_id = p1.player_id
JOIN players p2 ON m.loser_id = p2.player_id
JOIN statistics s_winner ON m.match_id = s_winner.match_id AND s_winner.is_winner = 1
JOIN statistics s_loser ON m.match_id = s_loser.match_id AND s_loser.is_winner = 0
WHERE s_winner.aces IS NOT NULL
LIMIT 10
"""

df = pd.read_sql_query(query, conn)

print("\n" + "="*100)
print("VERIFICATION: REAL MATCH STATISTICS CAPTURED")
print("="*100)
print("\nSample Matches with Real Statistics:\n")
print(df.to_string(index=False))

# Summary statistics
print("\n" + "="*100)
print("STATISTICS SUMMARY")
print("="*100)

stats_summary = f"""
Winner Aces:        Mean={df['w_aces'].mean():.1f}, Min={df['w_aces'].min()}, Max={df['w_aces'].max()}
Winner Double Faults: Mean={df['w_double_faults'].mean():.1f}, Min={df['w_double_faults'].min()}, Max={df['w_double_faults'].max()}
Winner 1st Serve %:  Mean={df['w_first_serve_pct'].mean():.1f}%, Min={df['w_first_serve_pct'].min():.1f}%, Max={df['w_first_serve_pct'].max():.1f}%
Winner 1st Serve Won %: Mean={df['w_first_serve_win_pct'].mean():.1f}%, Min={df['w_first_serve_win_pct'].min():.1f}%, Max={df['w_first_serve_win_pct'].max():.1f}%
Winner BP Save %:    Mean={df['w_break_point_save_pct'].mean():.1f}%, Min={df['w_break_point_save_pct'].min():.1f}%, Max={df['w_break_point_save_pct'].max():.1f}%
"""
print(stats_summary)

# Check data coverage
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM matches")
total_matches = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(DISTINCT match_id) FROM statistics WHERE aces IS NOT NULL")
matches_with_stats = cursor.fetchone()[0]

coverage = (matches_with_stats / total_matches) * 100

print(f"\nData Coverage: {matches_with_stats}/{total_matches} matches ({coverage:.1f}%) have detailed statistics")
print("="*100)

conn.close()
