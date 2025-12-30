#!/usr/bin/env python3
"""
Fetch comprehensive ATP tennis data from Jeff Sackmann's GitHub repository
https://github.com/JeffSackmann/tennis_atp
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import ssl
import urllib.request
import warnings
warnings.filterwarnings('ignore')

# Disable SSL verification for GitHub
ssl._create_default_https_context = ssl._create_unverified_context

print("="*60)
print("FETCHING JEFF SACKMANN ATP DATA")
print("="*60)

# GitHub raw URLs for ATP match data
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"

# Fetch data for years 2010-2024 (comprehensive recent data)
years = list(range(2010, 2025))

all_matches = []
for year in years:
    url = f"{BASE_URL}atp_matches_{year}.csv"
    print(f"Fetching {year}...", end=" ", flush=True)
    try:
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
        df = pd.read_csv(StringIO(data))
        all_matches.append(df)
        print(f"OK ({len(df)} matches)")
    except Exception as e:
        print(f"ERROR: {e}")

# Combine all data
matches_df = pd.concat(all_matches, ignore_index=True)
print(f"\nTotal matches fetched: {len(matches_df):,}")
print(f"Columns: {list(matches_df.columns)}")

# Create fresh database
print("\n" + "="*60)
print("CREATING DATABASE")
print("="*60)

conn = sqlite3.connect('tennis_data.db')
cursor = conn.cursor()

# Drop existing tables
cursor.execute("DROP TABLE IF EXISTS odds")
cursor.execute("DROP TABLE IF EXISTS statistics")
cursor.execute("DROP TABLE IF EXISTS matches")
cursor.execute("DROP TABLE IF EXISTS players")

# Create players table
cursor.execute("""
    CREATE TABLE players (
        player_id INTEGER PRIMARY KEY,
        player_name TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Create matches table with all needed columns
cursor.execute("""
    CREATE TABLE matches (
        match_id INTEGER PRIMARY KEY AUTOINCREMENT,
        tourney_id TEXT,
        tournament_name TEXT,
        tournament_date DATE,
        location TEXT,
        surface TEXT,
        draw_size INTEGER,
        tourney_level TEXT,
        round TEXT,
        best_of INTEGER,
        winner_id INTEGER,
        loser_id INTEGER,
        winner_rank INTEGER,
        loser_rank INTEGER,
        winner_rank_points REAL,
        loser_rank_points REAL,
        winner_seed INTEGER,
        loser_seed INTEGER,
        winner_entry TEXT,
        loser_entry TEXT,
        winner_hand TEXT,
        loser_hand TEXT,
        winner_ht INTEGER,
        loser_ht INTEGER,
        winner_age REAL,
        loser_age REAL,
        score TEXT,
        w_sets INTEGER,
        l_sets INTEGER,
        minutes INTEGER,
        w_ace INTEGER,
        w_df INTEGER,
        w_svpt INTEGER,
        w_1stIn INTEGER,
        w_1stWon INTEGER,
        w_2ndWon INTEGER,
        w_SvGms INTEGER,
        w_bpSaved INTEGER,
        w_bpFaced INTEGER,
        l_ace INTEGER,
        l_df INTEGER,
        l_svpt INTEGER,
        l_1stIn INTEGER,
        l_1stWon INTEGER,
        l_2ndWon INTEGER,
        l_SvGms INTEGER,
        l_bpSaved INTEGER,
        l_bpFaced INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (winner_id) REFERENCES players(player_id),
        FOREIGN KEY (loser_id) REFERENCES players(player_id)
    )
""")

# Create statistics table
cursor.execute("""
    CREATE TABLE statistics (
        stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        player_id INTEGER,
        is_winner BOOLEAN,
        aces INTEGER,
        double_faults INTEGER,
        first_serve_in INTEGER,
        first_serve_total INTEGER,
        first_serve_pct REAL,
        first_serve_points_won INTEGER,
        first_serve_points_total INTEGER,
        second_serve_points_won INTEGER,
        second_serve_points_total INTEGER,
        break_points_saved INTEGER,
        break_points_faced INTEGER,
        service_games INTEGER,
        FOREIGN KEY (match_id) REFERENCES matches(match_id),
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    )
""")

# Create odds table
cursor.execute("""
    CREATE TABLE odds (
        odds_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER,
        bookmaker TEXT,
        winner_odds REAL,
        loser_odds REAL,
        FOREIGN KEY (match_id) REFERENCES matches(match_id)
    )
""")

# Create indexes
cursor.execute("CREATE INDEX idx_matches_date ON matches(tournament_date)")
cursor.execute("CREATE INDEX idx_matches_surface ON matches(surface)")
cursor.execute("CREATE INDEX idx_matches_winner ON matches(winner_id)")
cursor.execute("CREATE INDEX idx_matches_loser ON matches(loser_id)")
cursor.execute("CREATE INDEX idx_stats_match ON statistics(match_id)")

conn.commit()
print("Database schema created")

# Insert players
print("\nInserting players...")
players = set(matches_df['winner_name'].unique()) | set(matches_df['loser_name'].unique())
player_map = {}
for i, name in enumerate(players, 1):
    if pd.notna(name):
        cursor.execute("INSERT INTO players (player_id, player_name) VALUES (?, ?)", (i, name))
        player_map[name] = i
conn.commit()
print(f"Inserted {len(player_map):,} players")

# Insert matches
print("\nInserting matches...")
inserted = 0
for idx, row in matches_df.iterrows():
    try:
        # Parse date
        tourney_date = None
        if pd.notna(row.get('tourney_date')):
            date_str = str(int(row['tourney_date']))
            tourney_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        winner_id = player_map.get(row.get('winner_name'))
        loser_id = player_map.get(row.get('loser_name'))
        
        if not winner_id or not loser_id:
            continue
        
        # Parse score to get sets
        score = row.get('score', '')
        w_sets, l_sets = 0, 0
        if pd.notna(score) and score:
            sets = str(score).split()
            for s in sets:
                if '-' in s:
                    parts = s.replace('(', ' ').replace(')', '').split('-')
                    try:
                        g1, g2 = int(parts[0].split()[0]), int(parts[1].split()[0])
                        if g1 > g2:
                            w_sets += 1
                        else:
                            l_sets += 1
                    except:
                        pass
        
        cursor.execute("""
            INSERT INTO matches (
                tourney_id, tournament_name, tournament_date, surface, draw_size,
                tourney_level, round, best_of, winner_id, loser_id,
                winner_rank, loser_rank, winner_rank_points, loser_rank_points,
                winner_seed, loser_seed, winner_entry, loser_entry,
                winner_hand, loser_hand, winner_ht, loser_ht,
                winner_age, loser_age, score, w_sets, l_sets, minutes,
                w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced,
                l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get('tourney_id'),
            row.get('tourney_name'),
            tourney_date,
            row.get('surface'),
            row.get('draw_size') if pd.notna(row.get('draw_size')) else None,
            row.get('tourney_level'),
            row.get('round'),
            row.get('best_of') if pd.notna(row.get('best_of')) else 3,
            winner_id,
            loser_id,
            row.get('winner_rank') if pd.notna(row.get('winner_rank')) else None,
            row.get('loser_rank') if pd.notna(row.get('loser_rank')) else None,
            row.get('winner_rank_points') if pd.notna(row.get('winner_rank_points')) else None,
            row.get('loser_rank_points') if pd.notna(row.get('loser_rank_points')) else None,
            row.get('winner_seed') if pd.notna(row.get('winner_seed')) else None,
            row.get('loser_seed') if pd.notna(row.get('loser_seed')) else None,
            row.get('winner_entry'),
            row.get('loser_entry'),
            row.get('winner_hand'),
            row.get('loser_hand'),
            row.get('winner_ht') if pd.notna(row.get('winner_ht')) else None,
            row.get('loser_ht') if pd.notna(row.get('loser_ht')) else None,
            row.get('winner_age') if pd.notna(row.get('winner_age')) else None,
            row.get('loser_age') if pd.notna(row.get('loser_age')) else None,
            score if pd.notna(score) else None,
            w_sets,
            l_sets,
            row.get('minutes') if pd.notna(row.get('minutes')) else None,
            row.get('w_ace') if pd.notna(row.get('w_ace')) else None,
            row.get('w_df') if pd.notna(row.get('w_df')) else None,
            row.get('w_svpt') if pd.notna(row.get('w_svpt')) else None,
            row.get('w_1stIn') if pd.notna(row.get('w_1stIn')) else None,
            row.get('w_1stWon') if pd.notna(row.get('w_1stWon')) else None,
            row.get('w_2ndWon') if pd.notna(row.get('w_2ndWon')) else None,
            row.get('w_SvGms') if pd.notna(row.get('w_SvGms')) else None,
            row.get('w_bpSaved') if pd.notna(row.get('w_bpSaved')) else None,
            row.get('w_bpFaced') if pd.notna(row.get('w_bpFaced')) else None,
            row.get('l_ace') if pd.notna(row.get('l_ace')) else None,
            row.get('l_df') if pd.notna(row.get('l_df')) else None,
            row.get('l_svpt') if pd.notna(row.get('l_svpt')) else None,
            row.get('l_1stIn') if pd.notna(row.get('l_1stIn')) else None,
            row.get('l_1stWon') if pd.notna(row.get('l_1stWon')) else None,
            row.get('l_2ndWon') if pd.notna(row.get('l_2ndWon')) else None,
            row.get('l_SvGms') if pd.notna(row.get('l_SvGms')) else None,
            row.get('l_bpSaved') if pd.notna(row.get('l_bpSaved')) else None,
            row.get('l_bpFaced') if pd.notna(row.get('l_bpFaced')) else None,
        ))
        
        match_id = cursor.lastrowid
        
        # Insert winner statistics
        if pd.notna(row.get('w_svpt')) and row.get('w_svpt', 0) > 0:
            first_serve_pct = row.get('w_1stIn', 0) / row.get('w_svpt', 1) if row.get('w_svpt', 0) > 0 else None
            cursor.execute("""
                INSERT INTO statistics (
                    match_id, player_id, is_winner, aces, double_faults,
                    first_serve_in, first_serve_total, first_serve_pct,
                    first_serve_points_won, first_serve_points_total,
                    second_serve_points_won, second_serve_points_total,
                    break_points_saved, break_points_faced, service_games
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id, winner_id, True,
                row.get('w_ace'), row.get('w_df'),
                row.get('w_1stIn'), row.get('w_svpt'), first_serve_pct,
                row.get('w_1stWon'), row.get('w_1stIn'),
                row.get('w_2ndWon'), (row.get('w_svpt', 0) or 0) - (row.get('w_1stIn', 0) or 0),
                row.get('w_bpSaved'), row.get('w_bpFaced'), row.get('w_SvGms')
            ))
        
        # Insert loser statistics
        if pd.notna(row.get('l_svpt')) and row.get('l_svpt', 0) > 0:
            first_serve_pct = row.get('l_1stIn', 0) / row.get('l_svpt', 1) if row.get('l_svpt', 0) > 0 else None
            cursor.execute("""
                INSERT INTO statistics (
                    match_id, player_id, is_winner, aces, double_faults,
                    first_serve_in, first_serve_total, first_serve_pct,
                    first_serve_points_won, first_serve_points_total,
                    second_serve_points_won, second_serve_points_total,
                    break_points_saved, break_points_faced, service_games
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id, loser_id, False,
                row.get('l_ace'), row.get('l_df'),
                row.get('l_1stIn'), row.get('l_svpt'), first_serve_pct,
                row.get('l_1stWon'), row.get('l_1stIn'),
                row.get('l_2ndWon'), (row.get('l_svpt', 0) or 0) - (row.get('l_1stIn', 0) or 0),
                row.get('l_bpSaved'), row.get('l_bpFaced'), row.get('l_SvGms')
            ))
        
        inserted += 1
        if inserted % 5000 == 0:
            conn.commit()
            print(f"  Inserted {inserted:,} matches...", end="\r")
            
    except Exception as e:
        continue

conn.commit()
print(f"\nInserted {inserted:,} matches total")

# Verify
cursor.execute("SELECT COUNT(*) FROM matches")
match_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM statistics")
stat_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM players")
player_count = cursor.fetchone()[0]
cursor.execute("SELECT MIN(tournament_date), MAX(tournament_date) FROM matches")
date_range = cursor.fetchone()

print(f"\n" + "="*60)
print("DATABASE SUMMARY")
print("="*60)
print(f"Players: {player_count:,}")
print(f"Matches: {match_count:,}")
print(f"Statistics: {stat_count:,}")
print(f"Date range: {date_range[0]} to {date_range[1]}")

# Surface breakdown
cursor.execute("SELECT surface, COUNT(*) FROM matches GROUP BY surface ORDER BY COUNT(*) DESC")
print("\nMatches by surface:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]:,}")

# Tournament level breakdown
cursor.execute("SELECT tourney_level, COUNT(*) FROM matches GROUP BY tourney_level ORDER BY COUNT(*) DESC")
print("\nMatches by tournament level:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]:,}")

conn.close()
print("\n" + "="*60)
print("DATA FETCH COMPLETE!")
print("="*60)
