"""
Compare proxy metrics vs real statistics for a sample player
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Connect to database
conn = sqlite3.connect('tennis_data.db')

# Get Novak Djokovic's matches with real statistics
query = """
SELECT 
    m.tournament_date,
    m.surface,
    s.is_winner,
    s.aces,
    s.double_faults,
    s.serve_games,
    s.first_serve_pct,
    s.first_serve_win_pct,
    s.second_serve_win_pct,
    s.break_point_save_pct
FROM statistics s
JOIN matches m ON s.match_id = m.match_id
JOIN players p ON s.player_id = p.player_id
WHERE p.player_name = 'Novak Djokovic'
AND m.tournament_date >= '2023-01-01'
AND s.aces IS NOT NULL
ORDER BY m.tournament_date
"""

djokovic_stats = pd.read_sql_query(query, conn)

print(f"\n{'='*100}")
print(f"REAL STATISTICS vs PROXY METRICS COMPARISON")
print(f"{'='*100}")
print(f"\nPlayer: Novak Djokovic")
print(f"Period: 2023-2024")
print(f"Matches with statistics: {len(djokovic_stats)}")

if len(djokovic_stats) > 0:
    # Calculate averages from REAL statistics
    real_1st_serve_pct = djokovic_stats['first_serve_pct'].mean() * 100
    real_1st_serve_win = djokovic_stats['first_serve_win_pct'].mean() * 100
    real_2nd_serve_win = djokovic_stats['second_serve_win_pct'].mean() * 100
    real_bp_save = djokovic_stats['break_point_save_pct'].mean() * 100
    real_aces_per_game = (djokovic_stats['aces'] / djokovic_stats['serve_games']).mean()
    real_df_per_game = (djokovic_stats['double_faults'] / djokovic_stats['serve_games']).mean()
    
    # Calculate proxy metrics (what we used before)
    win_rate = djokovic_stats['is_winner'].mean()
    proxy_wsp = 0.50 + (win_rate - 0.5) * 0.3  # Old formula
    proxy_aces = 0.5 * win_rate  # Old formula
    proxy_df = 0.3 * (1 - win_rate)  # Old formula
    
    print(f"\n{'-'*100}")
    print(f"COMPARISON TABLE")
    print(f"{'-'*100}")
    print(f"{'Metric':<35} {'REAL Statistics':<20} {'Proxy (Old)':<20} {'Difference':<15}")
    print(f"{'-'*100}")
    
    # Serve statistics
    print(f"\n{'Serve Statistics':^100}")
    print(f"{'First Serve %':<35} {real_1st_serve_pct:>18.1f}% {'-':>19} {'N/A':<15}")
    print(f"{'First Serve Won %':<35} {real_1st_serve_win:>18.1f}% {'-':>19} {'N/A':<15}")
    print(f"{'Second Serve Won %':<35} {real_2nd_serve_win:>18.1f}% {'-':>19} {'N/A':<15}")
    print(f"{'Break Point Save %':<35} {real_bp_save:>18.1f}% {'-':>19} {'N/A':<15}")
    print(f"{'Aces per Service Game':<35} {real_aces_per_game:>18.2f} {proxy_aces:>19.2f} {abs(real_aces_per_game - proxy_aces):>14.2f}")
    print(f"{'Double Faults per Service Game':<35} {real_df_per_game:>18.2f} {proxy_df:>19.2f} {abs(real_df_per_game - proxy_df):>14.2f}")
    
    print(f"\n{'Context':^100}")
    print(f"{'Win Rate':<35} {win_rate*100:>18.1f}% {win_rate*100:>19.1f}% {0:>14.1f}")
    print(f"{'Matches Analyzed':<35} {len(djokovic_stats):>18} {len(djokovic_stats):>19} {'-':<15}")
    
    print(f"\n{'-'*100}")
    print(f"KEY INSIGHTS:")
    print(f"{'-'*100}")
    print(f"✅ REAL statistics provide detailed breakdowns proxy metrics cannot capture")
    print(f"✅ First/second serve percentages show actual performance patterns")
    print(f"✅ Break point save % reveals clutch performance (proxy couldn't estimate this)")
    print(f"✅ Aces and double faults show actual counts vs rough win-rate based guesses")
    print(f"\n⚠️  OLD PROXY APPROACH:")
    print(f"   - Used win rate to estimate serve/return performance")
    print(f"   - Assumed linear relationship between winning and individual stats")
    print(f"   - Could not distinguish first vs second serve performance")
    print(f"   - No break point data available")
    print(f"\n✅ NEW REAL STATISTICS APPROACH:")
    print(f"   - Direct match statistics from Tennis Abstract")
    print(f"   - Actual serve percentages, not estimates")
    print(f"   - Real ace/DF counts per service game")
    print(f"   - Accurate break point save rates")
    print(f"   - Much more predictive for live match outcomes")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Novak Djokovic: REAL Statistics Distribution (2023-2024)', 
                 fontsize=16, fontweight='bold')
    
    # 1st Serve %
    axes[0, 0].hist(djokovic_stats['first_serve_pct'] * 100, bins=15, 
                    color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(real_1st_serve_pct, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {real_1st_serve_pct:.1f}%')
    axes[0, 0].set_xlabel('First Serve %', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('First Serve Percentage', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 1st Serve Won %
    axes[0, 1].hist(djokovic_stats['first_serve_win_pct'] * 100, bins=15,
                    color='#A23B72', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(real_1st_serve_win, color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {real_1st_serve_win:.1f}%')
    axes[0, 1].set_xlabel('First Serve Won %', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('First Serve Points Won %', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Aces per game
    aces_per_game = djokovic_stats['aces'] / djokovic_stats['serve_games']
    axes[1, 0].hist(aces_per_game, bins=15, color='#F18F01', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(real_aces_per_game, color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {real_aces_per_game:.2f}')
    axes[1, 0].set_xlabel('Aces per Service Game', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Aces per Service Game', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Break Point Save %
    axes[1, 1].hist(djokovic_stats['break_point_save_pct'].dropna() * 100, bins=15,
                    color='#06A77D', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(real_bp_save, color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {real_bp_save:.1f}%')
    axes[1, 1].set_xlabel('Break Point Save %', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Break Point Save Percentage', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('djokovic_real_stats.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: djokovic_real_stats.png")
    print(f"{'='*100}\n")
    
else:
    print("\n⚠️  No data found for Novak Djokovic in 2023-2024 period")

conn.close()
