"""
ATP Tennis Data Pipeline
Fetches ATP match data from tennis-data.co.uk (2020-present)
Stores in SQLite database with validation
"""

import sqlite3
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TennisDataPipeline:
    """Pipeline for fetching and storing ATP tennis data"""
    
    BASE_URL = "http://www.tennis-data.co.uk/{year}/{file}"
    DB_NAME = "tennis_data.db"
    
    # tennis-data.co.uk file patterns
    ATP_FILES = {
        2020: "2020/2020.xlsx",
        2021: "2021/2021.xlsx",
        2022: "2022/2022.xlsx",
        2023: "2023/2023.xlsx",
        2024: "2024/2024.xlsx"
    }
    
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
        """Create database tables for matches, players, statistics, and odds"""
        cursor = self.conn.cursor()
        
        # Players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Matches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_name TEXT,
                tournament_date DATE,
                location TEXT,
                series TEXT,
                surface TEXT,
                round TEXT,
                winner_id INTEGER,
                loser_id INTEGER,
                winner_rank INTEGER,
                loser_rank INTEGER,
                winner_rank_points REAL,
                loser_rank_points REAL,
                best_of INTEGER,
                w_sets INTEGER,
                l_sets INTEGER,
                set1_w INTEGER,
                set1_l INTEGER,
                set2_w INTEGER,
                set2_l INTEGER,
                set3_w INTEGER,
                set3_l INTEGER,
                set4_w INTEGER,
                set4_l INTEGER,
                set5_w INTEGER,
                set5_l INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (winner_id) REFERENCES players(player_id),
                FOREIGN KEY (loser_id) REFERENCES players(player_id)
            )
        """)
        
        # Statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
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
                break_points_converted INTEGER,
                break_points_total INTEGER,
                return_points_won INTEGER,
                return_points_total INTEGER,
                total_points_won INTEGER,
                validation_flags TEXT,
                FOREIGN KEY (match_id) REFERENCES matches(match_id),
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        """)
        
        # Odds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds (
                odds_id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                bookmaker TEXT,
                winner_odds REAL,
                loser_odds REAL,
                FOREIGN KEY (match_id) REFERENCES matches(match_id)
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(tournament_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_surface ON matches(surface)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches(winner_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_loser ON matches(loser_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_match ON statistics(match_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_odds_match ON odds(match_id)")
        
        self.conn.commit()
        logger.info("Database schema created successfully")
        
    def get_or_create_player(self, player_name: str) -> int:
        """Get player_id or create new player entry"""
        if not player_name or pd.isna(player_name):
            return None
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT player_id FROM players WHERE player_name = ?", (player_name,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        else:
            cursor.execute("INSERT INTO players (player_name) VALUES (?)", (player_name,))
            self.conn.commit()
            return cursor.lastrowid
            
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
        
    def validate_data(self, row: pd.Series, match_id: int) -> List[str]:
        """Validate data for a single match"""
        errors = []
        
        # Validate first serve percentage
        if 'W1stIn' in row and 'W1stWon' in row and not pd.isna(row['W1stIn']):
            if row['W1stIn'] > 0:
                pct = row.get('W1stWon', 0) / row['W1stIn']
                error = self.validate_percentage(pct, 'Winner 1st serve points won %', match_id)
                if error:
                    errors.append(error)
                    
        if 'L1stIn' in row and 'L1stWon' in row and not pd.isna(row['L1stIn']):
            if row['L1stIn'] > 0:
                pct = row.get('L1stWon', 0) / row['L1stIn']
                error = self.validate_percentage(pct, 'Loser 1st serve points won %', match_id)
                if error:
                    errors.append(error)
        
        return errors
        
    def fetch_atp_data(self, year: int) -> Optional[pd.DataFrame]:
        """Fetch ATP data for a specific year from tennis-data.co.uk"""
        try:
            # tennis-data.co.uk provides Excel files
            url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
            logger.info(f"Fetching data for {year} from {url}")
            
            df = pd.read_excel(url)
            logger.info(f"Successfully fetched {len(df)} matches for {year}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {year}: {str(e)}")
            return None
            
    def process_match_data(self, df: pd.DataFrame, year: int):
        """Process and insert match data into database"""
        logger.info(f"Processing {len(df)} matches from {year}")
        
        for idx, row in df.iterrows():
            try:
                # Get or create players
                winner_id = self.get_or_create_player(row.get('Winner'))
                loser_id = self.get_or_create_player(row.get('Loser'))
                
                if not winner_id or not loser_id:
                    logger.warning(f"Skipping match at row {idx}: missing player names")
                    continue
                
                # Parse date
                match_date = row.get('Date')
                if pd.notna(match_date):
                    if isinstance(match_date, str):
                        try:
                            match_date = datetime.strptime(match_date, '%Y-%m-%d').date()
                        except:
                            match_date = None
                    elif hasattr(match_date, 'date'):
                        match_date = match_date.date()
                
                # Insert match
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT INTO matches (
                        tournament_name, tournament_date, location, series, surface, round,
                        winner_id, loser_id, winner_rank, loser_rank, 
                        winner_rank_points, loser_rank_points, best_of,
                        w_sets, l_sets,
                        set1_w, set1_l, set2_w, set2_l, set3_w, set3_l,
                        set4_w, set4_l, set5_w, set5_l
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.get('Tournament'), match_date, row.get('Location'), 
                    row.get('Series'), row.get('Surface'), row.get('Round'),
                    winner_id, loser_id, 
                    row.get('WRank'), row.get('LRank'),
                    row.get('WPts'), row.get('LPts'),
                    row.get('Best of'),
                    row.get('Wsets'), row.get('Lsets'),
                    row.get('W1'), row.get('L1'),
                    row.get('W2'), row.get('L2'),
                    row.get('W3'), row.get('L3'),
                    row.get('W4'), row.get('L4'),
                    row.get('W5'), row.get('L5')
                ))
                
                match_id = cursor.lastrowid
                
                # Validate data
                validation_errors = self.validate_data(row, match_id)
                validation_flags = "; ".join(validation_errors) if validation_errors else None
                
                # Note: tennis-data.co.uk Excel files don't include detailed statistics
                # Statistics like aces, double faults, serve percentages are not available
                # in the free Excel download. For now, statistics table will remain empty.
                # To get detailed stats, you would need to scrape individual match pages
                # or use a different data source like the ATP website or paid APIs.
                
                # Insert betting odds
                # Pinnacle odds
                if pd.notna(row.get('PSW')) and pd.notna(row.get('PSL')):
                    cursor.execute("""
                        INSERT INTO odds (match_id, bookmaker, winner_odds, loser_odds)
                        VALUES (?, ?, ?, ?)
                    """, (match_id, 'Pinnacle', row.get('PSW'), row.get('PSL')))
                
                # Bet365 odds
                if pd.notna(row.get('B365W')) and pd.notna(row.get('B365L')):
                    cursor.execute("""
                        INSERT INTO odds (match_id, bookmaker, winner_odds, loser_odds)
                        VALUES (?, ?, ?, ?)
                    """, (match_id, 'Bet365', row.get('B365W'), row.get('B365L')))
                
                # Max odds (typically best available)
                if pd.notna(row.get('MaxW')) and pd.notna(row.get('MaxL')):
                    cursor.execute("""
                        INSERT INTO odds (match_id, bookmaker, winner_odds, loser_odds)
                        VALUES (?, ?, ?, ?)
                    """, (match_id, 'Max', row.get('MaxW'), row.get('MaxL')))
                
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
        
        # Connect to database
        self.connect_db()
        
        # Fetch and process data for each year
        for year in range(start_year, end_year + 1):
            df = self.fetch_atp_data(year)
            if df is not None:
                self.process_match_data(df, year)
            else:
                logger.warning(f"Skipping year {year} - no data available")
        
        # Report validation errors
        if self.validation_errors:
            logger.warning(f"Found {len(self.validation_errors)} validation errors")
            for error in self.validation_errors[:10]:  # Show first 10
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
        stats['total_odds'] = cursor.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
        stats['validation_errors'] = cursor.execute(
            "SELECT COUNT(*) FROM statistics WHERE validation_flags IS NOT NULL"
        ).fetchone()[0]
        
        # Get date range
        date_range = cursor.execute(
            "SELECT MIN(tournament_date), MAX(tournament_date) FROM matches"
        ).fetchone()
        stats['date_range'] = date_range
        
        return stats
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """Main execution function"""
    # Create and run pipeline
    pipeline = TennisDataPipeline()
    
    try:
        # Fetch data from 2020 to present
        pipeline.run_pipeline(start_year=2020, end_year=2024)
        
        # Print statistics
        stats = pipeline.get_stats()
        print("\n" + "="*50)
        print("DATA PIPELINE STATISTICS")
        print("="*50)
        print(f"Total Players: {stats['total_players']}")
        print(f"Total Matches: {stats['total_matches']}")
        print(f"Total Statistics: {stats['total_statistics']}")
        print(f"Total Odds Records: {stats['total_odds']}")
        print(f"Validation Errors: {stats['validation_errors']}")
        print(f"Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print("="*50)
        
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
