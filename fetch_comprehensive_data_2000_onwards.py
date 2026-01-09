#!/usr/bin/env python3
"""
🎾 COMPREHENSIVE ATP & WTA DATA INTEGRATION (2000-2026)
========================================================
Fetches historical tennis data from Jeff Sackmann's repositories:
- ATP: https://github.com/JeffSackmann/tennis_atp
- WTA: https://github.com/JeffSackmann/tennis_wta

Features:
- 26+ years of match data (2000-2026)
- Both men's and women's tennis
- Advanced statistics extraction
- Special parameters: momentum, surface mastery, clutch performance
- Player career progression tracking
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import ssl
import urllib.request
import warnings
import time
warnings.filterwarnings('ignore')

# Disable SSL verification for GitHub
ssl._create_default_https_context = ssl._create_unverified_context

class TennisDataIntegrator:
    """Fetch and integrate comprehensive tennis data"""
    
    def __init__(self, db_path='tennis_betting.db'):
        self.db_path = db_path
        self.atp_base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
        self.wta_base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/"
        
    def fetch_year_data(self, year, tour='atp'):
        """Fetch data for a specific year and tour"""
        base_url = self.atp_base_url if tour == 'atp' else self.wta_base_url
        url = f"{base_url}{tour}_matches_{year}.csv"
        
        try:
            response = urllib.request.urlopen(url, timeout=15)
            data = response.read().decode('utf-8')
            df = pd.read_csv(StringIO(data))
            df['tour'] = tour.upper()
            df['year'] = year
            return df
        except Exception as e:
            print(f"  ⚠️ {tour.upper()} {year}: {str(e)[:50]}")
            return None
            
    def fetch_all_data(self, start_year=2000, end_year=2026):
        """Fetch all ATP and WTA data from start_year to end_year"""
        print("="*70)
        print("🎾 FETCHING COMPREHENSIVE TENNIS DATA")
        print("="*70)
        
        all_matches = []
        years = list(range(start_year, end_year + 1))
        
        for tour in ['atp', 'wta']:
            print(f"\n📥 Fetching {tour.upper()} data...")
            for year in years:
                print(f"  {year}...", end=" ", flush=True)
                df = self.fetch_year_data(year, tour)
                if df is not None:
                    all_matches.append(df)
                    print(f"✓ ({len(df)} matches)")
                time.sleep(0.1)  # Be nice to GitHub servers
        
        if not all_matches:
            raise Exception("No data fetched!")
            
        combined_df = pd.concat(all_matches, ignore_index=True)
        print(f"\n✅ Total matches fetched: {len(combined_df):,}")
        return combined_df
    
    def calculate_special_parameters(self, matches_df):
        """
        Calculate advanced tennis parameters:
        1. Momentum Score (recent form weight)
        2. Surface Mastery (surface-specific win rate)
        3. Clutch Performance (performance in critical matches)
        4. Break Point Conversion (offensive pressure)
        5. Break Point Defense (defensive resilience)
        6. Consistency Rating (variance in performance)
        7. Big Match Experience (Grand Slam/Masters performance)
        """
        print("\n" + "="*70)
        print("🧠 CALCULATING SPECIAL PARAMETERS")
        print("="*70)
        
        # Convert date to datetime
        matches_df['tourney_date'] = pd.to_datetime(matches_df['tourney_date'], format='%Y%m%d', errors='coerce')
        
        # Sort by date
        matches_df = matches_df.sort_values('tourney_date')
        
        special_stats = []
        
        # Process each player
        all_players = pd.concat([
            matches_df[['winner_id', 'winner_name']].rename(columns={'winner_id': 'player_id', 'winner_name': 'player_name'}),
            matches_df[['loser_id', 'loser_name']].rename(columns={'loser_id': 'player_id', 'loser_name': 'player_name'})
        ]).drop_duplicates('player_id')
        
        print(f"\n📊 Processing {len(all_players):,} unique players...")
        
        for idx, player_row in all_players.iterrows():
            player_id = player_row['player_id']
            player_name = player_row['player_name']
            
            if pd.isna(player_id) or pd.isna(player_name):
                continue
                
            # Get player's matches
            player_wins = matches_df[matches_df['winner_id'] == player_id].copy()
            player_losses = matches_df[matches_df['loser_id'] == player_id].copy()
            
            player_wins['won'] = 1
            player_losses['won'] = 0
            
            # Combine wins and losses
            player_matches = pd.concat([
                player_wins[['tourney_date', 'surface', 'tourney_level', 'won', 'w_svpt', 'w_1stIn', 'w_1stWon', 
                            'w_2ndWon', 'w_bpSaved', 'w_bpFaced']].rename(columns={
                    'w_svpt': 'svpt', 'w_1stIn': '1stIn', 'w_1stWon': '1stWon',
                    'w_2ndWon': '2ndWon', 'w_bpSaved': 'bpSaved', 'w_bpFaced': 'bpFaced'
                }),
                player_losses[['tourney_date', 'surface', 'tourney_level', 'won', 'l_svpt', 'l_1stIn', 'l_1stWon',
                              'l_2ndWon', 'l_bpSaved', 'l_bpFaced']].rename(columns={
                    'l_svpt': 'svpt', 'l_1stIn': '1stIn', 'l_1stWon': '1stWon',
                    'l_2ndWon': '2ndWon', 'l_bpSaved': 'bpSaved', 'l_bpFaced': 'bpFaced'
                })
            ]).sort_values('tourney_date')
            
            if len(player_matches) < 10:  # Need minimum matches
                continue
            
            # 1. MOMENTUM SCORE (exponentially weighted recent form)
            recent_20 = player_matches.tail(20)
            weights = np.exp(np.linspace(0, 1, len(recent_20)))  # Exponential weight to recent
            momentum_score = np.average(recent_20['won'], weights=weights) if len(recent_20) > 0 else 0.5
            
            # 2. SURFACE MASTERY (best surface win rate)
            surface_stats = {}
            for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
                surf_matches = player_matches[player_matches['surface'] == surface]
                if len(surf_matches) >= 5:
                    surface_stats[surface] = surf_matches['won'].mean()
            
            best_surface = max(surface_stats.items(), key=lambda x: x[1])[0] if surface_stats else 'Hard'
            surface_mastery = max(surface_stats.values()) if surface_stats else 0.5
            
            # 3. CLUTCH PERFORMANCE (performance in important tournaments)
            big_matches = player_matches[player_matches['tourney_level'].isin(['G', 'M', 'F'])]  # Grand Slams, Masters, Finals
            clutch_performance = big_matches['won'].mean() if len(big_matches) >= 3 else 0.5
            
            # 4. BREAK POINT CONVERSION (opponent's break points faced that we saved)
            valid_bp = player_matches.dropna(subset=['bpFaced'])
            bp_defense = (valid_bp['bpSaved'] / valid_bp['bpFaced']).mean() if len(valid_bp) > 0 else 0.5
            
            # 5. SERVE QUALITY (first serve effectiveness)
            valid_serve = player_matches.dropna(subset=['svpt', '1stIn', '1stWon'])
            first_serve_win_pct = (valid_serve['1stWon'] / valid_serve['1stIn']).mean() if len(valid_serve) > 0 else 0.65
            
            # 6. CONSISTENCY RATING (inverse of win rate variance)
            # Rolling 10-match win rate variance
            if len(player_matches) >= 10:
                rolling_wr = player_matches['won'].rolling(10).mean()
                consistency = 1 - min(rolling_wr.std(), 0.5) if len(rolling_wr) > 0 else 0.5
            else:
                consistency = 0.5
            
            # 7. CAREER WIN RATE
            career_win_rate = player_matches['won'].mean()
            
            # 8. PEAK RATING (best 20-match rolling win rate)
            if len(player_matches) >= 20:
                rolling_20 = player_matches['won'].rolling(20).mean()
                peak_rating = rolling_20.max()
            else:
                peak_rating = career_win_rate
            
            special_stats.append({
                'player_id': int(player_id),
                'player_name': player_name,
                'total_matches': len(player_matches),
                'career_win_rate': round(career_win_rate, 4),
                'momentum_score': round(momentum_score, 4),
                'best_surface': best_surface,
                'surface_mastery': round(surface_mastery, 4),
                'clutch_performance': round(clutch_performance, 4),
                'bp_defense_rate': round(bp_defense, 4),
                'first_serve_win_pct': round(first_serve_win_pct, 4),
                'consistency_rating': round(consistency, 4),
                'peak_rating': round(peak_rating, 4)
            })
            
            if len(special_stats) % 500 == 0:
                print(f"  Processed {len(special_stats):,} players...")
        
        special_df = pd.DataFrame(special_stats)
        print(f"\n✅ Calculated special parameters for {len(special_df):,} players")
        return special_df
    
    def setup_database(self, conn):
        """Create database schema with advanced fields"""
        print("\n" + "="*70)
        print("🗄️  SETTING UP DATABASE SCHEMA")
        print("="*70)
        
        cursor = conn.cursor()
        
        # Create players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY,
                player_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create matches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tourney_id TEXT,
                tourney_name TEXT,
                tourney_date DATE,
                surface TEXT,
                draw_size INTEGER,
                tourney_level TEXT,
                tour TEXT,
                best_of INTEGER,
                round TEXT,
                winner_id INTEGER,
                loser_id INTEGER,
                winner_rank INTEGER,
                loser_rank INTEGER,
                winner_rank_points REAL,
                loser_rank_points REAL,
                winner_seed INTEGER,
                loser_seed INTEGER,
                winner_age REAL,
                loser_age REAL,
                score TEXT,
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
                minutes INTEGER,
                FOREIGN KEY (winner_id) REFERENCES players(player_id),
                FOREIGN KEY (loser_id) REFERENCES players(loser_id)
            )
        """)
        
        # Create special_parameters table (NEW!)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS special_parameters (
                player_id INTEGER PRIMARY KEY,
                player_name TEXT,
                total_matches INTEGER,
                career_win_rate REAL,
                momentum_score REAL,
                best_surface TEXT,
                surface_mastery REAL,
                clutch_performance REAL,
                bp_defense_rate REAL,
                first_serve_win_pct REAL,
                consistency_rating REAL,
                peak_rating REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        """)
        
        # Create statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                player_id INTEGER,
                is_winner BOOLEAN,
                aces INTEGER,
                double_faults INTEGER,
                first_serve_pct REAL,
                first_serve_won_pct REAL,
                second_serve_won_pct REAL,
                break_points_saved INTEGER,
                break_points_faced INTEGER,
                service_points_won REAL,
                return_points_won REAL,
                FOREIGN KEY (match_id) REFERENCES matches(match_id),
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        """)
        
        conn.commit()
        print("✅ Database schema created")
    
    def insert_players(self, conn, matches_df):
        """Insert unique players"""
        print("\n📝 Inserting players...")
        
        # Get unique winner IDs and names
        winners = matches_df[['winner_id', 'winner_name']].rename(
            columns={'winner_id': 'player_id', 'winner_name': 'player_name'}
        )
        losers = matches_df[['loser_id', 'loser_name']].rename(
            columns={'loser_id': 'player_id', 'loser_name': 'player_name'}
        )
        
        all_players = pd.concat([winners, losers]).drop_duplicates('player_id')
        all_players = all_players.dropna(subset=['player_id', 'player_name'])
        
        cursor = conn.cursor()
        for _, row in all_players.iterrows():
            cursor.execute("""
                INSERT OR IGNORE INTO players (player_id, player_name)
                VALUES (?, ?)
            """, (int(row['player_id']), row['player_name']))
        
        conn.commit()
        print(f"✅ {len(all_players):,} players inserted")
        return len(all_players)
    
    def insert_matches(self, conn, matches_df):
        """Insert match data"""
        print("\n📝 Inserting matches...")
        
        # Convert date format
        matches_df['tourney_date'] = matches_df['tourney_date'].astype(str)
        
        cursor = conn.cursor()
        inserted = 0
        
        for _, row in matches_df.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO matches (
                        tourney_id, tourney_name, tourney_date, surface, draw_size,
                        tourney_level, tour, best_of, round, winner_id, loser_id,
                        winner_rank, loser_rank, winner_rank_points, loser_rank_points,
                        winner_seed, loser_seed, winner_age, loser_age, score,
                        w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon,
                        w_SvGms, w_bpSaved, w_bpFaced,
                        l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon,
                        l_SvGms, l_bpSaved, l_bpFaced, minutes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                             ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.get('tourney_id'), row.get('tourney_name'), row.get('tourney_date'),
                    row.get('surface'), row.get('draw_size'), row.get('tourney_level'),
                    row.get('tour'), row.get('best_of'), row.get('round'),
                    int(row['winner_id']) if pd.notna(row.get('winner_id')) else None,
                    int(row['loser_id']) if pd.notna(row.get('loser_id')) else None,
                    row.get('winner_rank'), row.get('loser_rank'),
                    row.get('winner_rank_points'), row.get('loser_rank_points'),
                    row.get('winner_seed'), row.get('loser_seed'),
                    row.get('winner_age'), row.get('loser_age'), row.get('score'),
                    row.get('w_ace'), row.get('w_df'), row.get('w_svpt'),
                    row.get('w_1stIn'), row.get('w_1stWon'), row.get('w_2ndWon'),
                    row.get('w_SvGms'), row.get('w_bpSaved'), row.get('w_bpFaced'),
                    row.get('l_ace'), row.get('l_df'), row.get('l_svpt'),
                    row.get('l_1stIn'), row.get('l_1stWon'), row.get('l_2ndWon'),
                    row.get('l_SvGms'), row.get('l_bpSaved'), row.get('l_bpFaced'),
                    row.get('minutes')
                ))
                inserted += 1
                
                if inserted % 5000 == 0:
                    conn.commit()
                    print(f"  {inserted:,} matches inserted...")
                    
            except Exception as e:
                print(f"  Error inserting match: {e}")
                continue
        
        conn.commit()
        print(f"✅ {inserted:,} matches inserted")
        return inserted
    
    def insert_special_parameters(self, conn, special_df):
        """Insert special parameters"""
        print("\n📝 Inserting special parameters...")
        
        cursor = conn.cursor()
        for _, row in special_df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO special_parameters (
                    player_id, player_name, total_matches, career_win_rate,
                    momentum_score, best_surface, surface_mastery, clutch_performance,
                    bp_defense_rate, first_serve_win_pct, consistency_rating, peak_rating
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['player_id'], row['player_name'], row['total_matches'],
                row['career_win_rate'], row['momentum_score'], row['best_surface'],
                row['surface_mastery'], row['clutch_performance'], row['bp_defense_rate'],
                row['first_serve_win_pct'], row['consistency_rating'], row['peak_rating']
            ))
        
        conn.commit()
        print(f"✅ {len(special_df):,} player special parameters inserted")
    
    def run(self, start_year=2000, end_year=2026):
        """Main execution pipeline"""
        print("\n" + "="*70)
        print(f"🚀 STARTING DATA INTEGRATION ({start_year}-{end_year})")
        print("="*70)
        
        # 1. Fetch data
        matches_df = self.fetch_all_data(start_year, end_year)
        
        # 2. Calculate special parameters
        special_df = self.calculate_special_parameters(matches_df)
        
        # 3. Setup database
        conn = sqlite3.connect(self.db_path)
        self.setup_database(conn)
        
        # 4. Insert data
        self.insert_players(conn, matches_df)
        self.insert_matches(conn, matches_df)
        self.insert_special_parameters(conn, special_df)
        
        # 5. Create indexes for performance
        print("\n🔧 Creating database indexes...")
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(tourney_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches(winner_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_loser ON matches(loser_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_surface ON matches(surface)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_special_params_player ON special_parameters(player_id)")
        conn.commit()
        print("✅ Indexes created")
        
        # 6. Display summary
        print("\n" + "="*70)
        print("📊 DATABASE SUMMARY")
        print("="*70)
        
        stats = cursor.execute("""
            SELECT 
                (SELECT COUNT(*) FROM players) as players,
                (SELECT COUNT(*) FROM matches) as matches,
                (SELECT COUNT(*) FROM special_parameters) as special_params,
                (SELECT MIN(tourney_date) FROM matches) as earliest_match,
                (SELECT MAX(tourney_date) FROM matches) as latest_match
        """).fetchone()
        
        print(f"👥 Total Players: {stats[0]:,}")
        print(f"🎾 Total Matches: {stats[1]:,}")
        print(f"🧠 Players with Special Parameters: {stats[2]:,}")
        print(f"📅 Date Range: {stats[3]} to {stats[4]}")
        
        # Show top players by special parameters
        print("\n🏆 TOP 10 PLAYERS BY PEAK RATING:")
        top_players = cursor.execute("""
            SELECT player_name, peak_rating, momentum_score, clutch_performance, best_surface
            FROM special_parameters
            ORDER BY peak_rating DESC
            LIMIT 10
        """).fetchall()
        
        for i, p in enumerate(top_players, 1):
            print(f"  {i}. {p[0]}: Peak={p[1]:.3f}, Momentum={p[2]:.3f}, Clutch={p[3]:.3f}, Best={p[4]}")
        
        conn.close()
        
        print("\n" + "="*70)
        print("✅ DATA INTEGRATION COMPLETE!")
        print("="*70)
        print(f"\nDatabase saved to: {self.db_path}")
        print("\nNext steps:")
        print("  1. Run training script to build ML models")
        print("  2. Use special_parameters table for advanced predictions")
        print("  3. Launch dashboard to visualize insights")


if __name__ == '__main__':
    integrator = TennisDataIntegrator(db_path='tennis_betting.db')
    integrator.run(start_year=2000, end_year=2026)
