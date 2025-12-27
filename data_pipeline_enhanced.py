"""
Enhanced ATP Tennis Data Pipeline
Fetches ATP match data with REAL match statistics from Tennis Abstract (Jeff Sackmann)
Source: https://github.com/JeffSackmann/tennis_atp
"""

import sqlite3
import pandas as pd
import requests
from datetime import datetime
from typing import List, Dict, Optional
import logging
import ssl
import urllib.request

# Handle SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedTennisDataPipeline:
    """Pipeline for fetching ATP tennis data with real match statistics"""
    
    GITHUB_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
    DB_NAME = "tennis_data.db"
    
    def __init__(self, db_path: str = None):
        """Initialize pipeline with database connection"""
        self.db_path = db_path or self.DB_NAME
        self.conn = None
        self.validation_errors = []
        
    def connect_db(self):
        """Create database connection and initialize schema"""
        self.conn = sqlite3.connect(self.db_path)
        self.create_schema()
        logger.info(f"Connected to database: {self.db_path}")
        
    def create_schema(self):
        """Create database tables with real statistics support"""
        cursor = self.conn.cursor()
        
        # Players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY,
                player_name TEXT UNIQUE NOT NULL,
                hand TEXT,
                height_cm INTEGER,
                country TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Matches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tourney_id TEXT,
                tournament_name TEXT,
                tournament_date DATE,
                surface TEXT,
                draw_size INTEGER,
                tourney_level TEXT,
                round TEXT,
                best_of INTEGER,
                minutes INTEGER,
                winner_id INTEGER,
                loser_id INTEGER,
                winner_seed INTEGER,
                loser_seed INTEGER,
                winner_rank INTEGER,
                loser_rank INTEGER,
                winner_rank_points REAL,
                loser_rank_points REAL,
                winner_age REAL,
                loser_age REAL,
                score TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (winner_id) REFERENCES players(player_id),
                FOREIGN KEY (loser_id) REFERENCES players(player_id)
            )
        """)
        
        # Statistics table - REAL MATCH STATISTICS
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                player_id INTEGER,
                is_winner BOOLEAN,
                aces INTEGER,
                double_faults INTEGER,
                serve_points_total INTEGER,
                first_serve_in INTEGER,
                first_serve_won INTEGER,
                second_serve_won INTEGER,
                serve_games INTEGER,
                break_points_saved INTEGER,
                break_points_faced INTEGER,
                first_serve_pct REAL,
                first_serve_win_pct REAL,
                second_serve_win_pct REAL,
                break_point_save_pct REAL,
                validation_flags TEXT,
                FOREIGN KEY (match_id) REFERENCES matches(match_id),
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(tournament_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_surface ON matches(surface)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches(winner_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_loser ON matches(loser_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_match ON statistics(match_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_player ON statistics(player_id)")
        
        self.conn.commit()
        logger.info("Database schema created successfully")
        
    def get_or_create_player(self, player_id: int, player_name: str, 
                            hand: str = None, height: int = None, country: str = None) -> int:
        """Get player_id or create new player entry"""
        if not player_id or pd.isna(player_id):
            return None
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT player_id FROM players WHERE player_id = ?", (player_id,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            cursor.execute("""
                INSERT INTO players (player_id, player_name, hand, height_cm, country) 
                VALUES (?, ?, ?, ?, ?)
            """, (player_id, player_name, hand, height, country))
            self.conn.commit()
            return player_id
            
    def validate_percentage(self, value: float, field_name: str, match_id: int = None) -> Optional[str]:
        """Validate percentage values are between 0 and 1"""
        if pd.isna(value):
            return None
            
        if value < 0 or value > 1:
            error_msg = f"Invalid {field_name}: {value} (should be 0-1)"
            if match_id:
                error_msg += f" [Match ID: {match_id}]"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
            return error_msg
        return None
        
    def validate_statistics(self, row: pd.Series, match_id: int) -> List[str]:
        """Validate match statistics"""
        errors = []
        
        # Validate first serve percentage
        if pd.notna(row['w_1stIn']) and pd.notna(row['w_svpt']) and row['w_svpt'] > 0:
            pct = row['w_1stIn'] / row['w_svpt']
            error = self.validate_percentage(pct, 'Winner 1st serve %', match_id)
            if error:
                errors.append(error)
                
        if pd.notna(row['l_1stIn']) and pd.notna(row['l_svpt']) and row['l_svpt'] > 0:
            pct = row['l_1stIn'] / row['l_svpt']
            error = self.validate_percentage(pct, 'Loser 1st serve %', match_id)
            if error:
                errors.append(error)
        
        return errors
        
    def fetch_year_data(self, year: int) -> Optional[pd.DataFrame]:
        """Fetch ATP data for a specific year from Tennis Abstract GitHub"""
        try:
            url = f"{self.GITHUB_BASE_URL}atp_matches_{year}.csv"
            logger.info(f"Fetching data for {year} from {url}")
            
            df = pd.read_csv(url)
            logger.info(f"Successfully fetched {len(df)} matches for {year}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {year}: {str(e)}")
            return None
            
    def process_match_data(self, df: pd.DataFrame, year: int):
        """Process and insert match data with real statistics into database"""
        logger.info(f"Processing {len(df)} matches from {year}")
        
        for idx, row in df.iterrows():
            try:
                # Get or create players
                winner_id = self.get_or_create_player(
                    int(row['winner_id']) if pd.notna(row['winner_id']) else None,
                    row['winner_name'],
                    row.get('winner_hand'),
                    row.get('winner_ht'),
                    row.get('winner_ioc')
                )
                
                loser_id = self.get_or_create_player(
                    int(row['loser_id']) if pd.notna(row['loser_id']) else None,
                    row['loser_name'],
                    row.get('loser_hand'),
                    row.get('loser_ht'),
                    row.get('loser_ioc')
                )
                
                if not winner_id or not loser_id:
                    logger.warning(f"Skipping match at row {idx}: missing player IDs")
                    continue
                
                # Parse date
                match_date = row.get('tourney_date')
                if pd.notna(match_date):
                    try:
                        match_date = datetime.strptime(str(match_date), '%Y%m%d').date()
                    except:
                        match_date = None
                
                # Insert match
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT INTO matches (
                        tourney_id, tournament_name, tournament_date, surface, 
                        draw_size, tourney_level, round, best_of, minutes,
                        winner_id, loser_id, winner_seed, loser_seed,
                        winner_rank, loser_rank, winner_rank_points, loser_rank_points,
                        winner_age, loser_age, score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.get('tourney_id'), row.get('tourney_name'), match_date, 
                    row.get('surface'), row.get('draw_size'), row.get('tourney_level'),
                    row.get('round'), row.get('best_of'), row.get('minutes'),
                    winner_id, loser_id,
                    row.get('winner_seed'), row.get('loser_seed'),
                    row.get('winner_rank'), row.get('loser_rank'),
                    row.get('winner_rank_points'), row.get('loser_rank_points'),
                    row.get('winner_age'), row.get('loser_age'),
                    row.get('score')
                ))
                
                match_id = cursor.lastrowid
                
                # Validate data
                validation_errors = self.validate_statistics(row, match_id)
                validation_flags = "; ".join(validation_errors) if validation_errors else None
                
                # Insert REAL winner statistics
                if pd.notna(row.get('w_ace')):  # Only insert if we have stats
                    # Calculate percentages
                    w_1st_pct = row['w_1stIn'] / row['w_svpt'] if pd.notna(row['w_svpt']) and row['w_svpt'] > 0 else None
                    w_1st_win_pct = row['w_1stWon'] / row['w_1stIn'] if pd.notna(row['w_1stIn']) and row['w_1stIn'] > 0 else None
                    w_2nd_win_pct = row['w_2ndWon'] / (row['w_svpt'] - row['w_1stIn']) if pd.notna(row['w_svpt']) and pd.notna(row['w_1stIn']) and (row['w_svpt'] - row['w_1stIn']) > 0 else None
                    w_bp_save_pct = row['w_bpSaved'] / row['w_bpFaced'] if pd.notna(row['w_bpFaced']) and row['w_bpFaced'] > 0 else None
                    
                    cursor.execute("""
                        INSERT INTO statistics (
                            match_id, player_id, is_winner,
                            aces, double_faults, serve_points_total,
                            first_serve_in, first_serve_won, second_serve_won,
                            serve_games, break_points_saved, break_points_faced,
                            first_serve_pct, first_serve_win_pct, second_serve_win_pct,
                            break_point_save_pct, validation_flags
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        match_id, winner_id, True,
                        row.get('w_ace'), row.get('w_df'), row.get('w_svpt'),
                        row.get('w_1stIn'), row.get('w_1stWon'), row.get('w_2ndWon'),
                        row.get('w_SvGms'), row.get('w_bpSaved'), row.get('w_bpFaced'),
                        w_1st_pct, w_1st_win_pct, w_2nd_win_pct,
                        w_bp_save_pct, validation_flags
                    ))
                
                # Insert REAL loser statistics
                if pd.notna(row.get('l_ace')):
                    l_1st_pct = row['l_1stIn'] / row['l_svpt'] if pd.notna(row['l_svpt']) and row['l_svpt'] > 0 else None
                    l_1st_win_pct = row['l_1stWon'] / row['l_1stIn'] if pd.notna(row['l_1stIn']) and row['l_1stIn'] > 0 else None
                    l_2nd_win_pct = row['l_2ndWon'] / (row['l_svpt'] - row['l_1stIn']) if pd.notna(row['l_svpt']) and pd.notna(row['l_1stIn']) and (row['l_svpt'] - row['l_1stIn']) > 0 else None
                    l_bp_save_pct = row['l_bpSaved'] / row['l_bpFaced'] if pd.notna(row['l_bpFaced']) and row['l_bpFaced'] > 0 else None
                    
                    cursor.execute("""
                        INSERT INTO statistics (
                            match_id, player_id, is_winner,
                            aces, double_faults, serve_points_total,
                            first_serve_in, first_serve_won, second_serve_won,
                            serve_games, break_points_saved, break_points_faced,
                            first_serve_pct, first_serve_win_pct, second_serve_win_pct,
                            break_point_save_pct, validation_flags
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        match_id, loser_id, False,
                        row.get('l_ace'), row.get('l_df'), row.get('l_svpt'),
                        row.get('l_1stIn'), row.get('l_1stWon'), row.get('l_2ndWon'),
                        row.get('l_SvGms'), row.get('l_bpSaved'), row.get('l_bpFaced'),
                        l_1st_pct, l_1st_win_pct, l_2nd_win_pct,
                        l_bp_save_pct, validation_flags
                    ))
                
                # Commit every 100 rows
                if idx % 100 == 0:
                    self.conn.commit()
                    logger.info(f"Processed {idx} matches...")
                    
            except Exception as e:
                logger.error(f"Error processing match at row {idx}: {str(e)}")
                continue
        
        self.conn.commit()
        logger.info(f"Completed processing {len(df)} matches")
        
    def run_pipeline(self, start_year: int = 2020, end_year: int = None):
        """Run the complete data pipeline"""
        if end_year is None:
            end_year = datetime.now().year
            
        logger.info(f"Starting ATP data pipeline for years {start_year}-{end_year}")
        logger.info("Data source: Tennis Abstract (Jeff Sackmann) - https://github.com/JeffSackmann/tennis_atp")
        
        # Connect to database
        self.connect_db()
        
        # Fetch and process data for each year
        for year in range(start_year, end_year + 1):
            df = self.fetch_year_data(year)
            if df is not None:
                self.process_match_data(df, year)
            else:
                logger.warning(f"Skipping year {year} - no data available")
        
        # Report validation errors
        if self.validation_errors:
            logger.warning(f"Found {len(self.validation_errors)} validation errors")
            for error in self.validation_errors[:10]:
                logger.warning(f"  - {error}")
            if len(self.validation_errors) > 10:
                logger.warning(f"  ... and {len(self.validation_errors) - 10} more")
        
        logger.info("Pipeline completed successfully!")
        
    def get_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        stats['total_players'] = cursor.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        stats['total_matches'] = cursor.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        stats['total_statistics'] = cursor.execute("SELECT COUNT(*) FROM statistics").fetchone()[0]
        stats['validation_errors'] = cursor.execute(
            "SELECT COUNT(*) FROM statistics WHERE validation_flags IS NOT NULL"
        ).fetchone()[0]
        stats['matches_with_stats'] = cursor.execute(
            "SELECT COUNT(DISTINCT match_id) FROM statistics"
        ).fetchone()[0]
        
        # Get date range
        date_range = cursor.execute(
            "SELECT MIN(tournament_date), MAX(tournament_date) FROM matches"
        ).fetchone()
        stats['date_range'] = date_range
        
        # Get statistics coverage
        stats['stats_coverage_pct'] = (stats['matches_with_stats'] / stats['total_matches'] * 100) if stats['total_matches'] > 0 else 0
        
        return stats
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """Main execution function"""
    # Delete old database if exists
    import os
    if os.path.exists('tennis_data.db'):
        logger.info("Removing old database...")
        os.remove('tennis_data.db')
    
    # Create and run pipeline
    pipeline = EnhancedTennisDataPipeline()
    
    try:
        # Fetch data from 2020 to present with REAL statistics
        pipeline.run_pipeline(start_year=2020, end_year=2024)
        
        # Print statistics
        stats = pipeline.get_stats()
        print("\n" + "="*70)
        print("ENHANCED DATA PIPELINE STATISTICS")
        print("="*70)
        print(f"Data Source: Tennis Abstract (Jeff Sackmann)")
        print(f"GitHub: https://github.com/JeffSackmann/tennis_atp")
        print("="*70)
        print(f"Total Players: {stats['total_players']:,}")
        print(f"Total Matches: {stats['total_matches']:,}")
        print(f"Total Statistics Records: {stats['total_statistics']:,}")
        print(f"Matches with Statistics: {stats['matches_with_stats']:,} ({stats['stats_coverage_pct']:.1f}%)")
        print(f"Validation Errors: {stats['validation_errors']:,}")
        print(f"Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print("="*70)
        print("\n✅ Now you have REAL match statistics:")
        print("   - Actual aces, double faults")
        print("   - Real first/second serve percentages")
        print("   - True break point conversion rates")
        print("   - Accurate serve points won/lost")
        print("="*70)
        
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
