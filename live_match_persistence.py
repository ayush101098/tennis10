"""
Live Match State Persistence and Bet Tracking
==============================================
Handles saving/loading live match tracking data and selected bets to database
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

DB_PATH = 'tennis_betting.db'

def init_live_tracking_tables():
    """Create tables for live match tracking and bet selection"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for active/paused live matches
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS live_matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            player1_name TEXT NOT NULL,
            player2_name TEXT NOT NULL,
            surface TEXT,
            p1_serve_win REAL,
            p2_serve_win REAL,
            p1_return_win REAL,
            p2_return_win REAL,
            p1_sets INTEGER DEFAULT 0,
            p2_sets INTEGER DEFAULT 0,
            p1_games INTEGER DEFAULT 0,
            p2_games INTEGER DEFAULT 0,
            p1_points INTEGER DEFAULT 0,
            p2_points INTEGER DEFAULT 0,
            total_points INTEGER DEFAULT 0,
            p1_breaks INTEGER DEFAULT 0,
            p2_breaks INTEGER DEFAULT 0,
            probability_history TEXT,
            score_history TEXT,
            point_winner_history TEXT,
            p1_games_won_history TEXT,
            p2_games_won_history TEXT,
            advanced_params TEXT,
            match_snapshots TEXT,
            prematch_odds TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table for selected/placed bets
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS selected_bets (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            player1_name TEXT NOT NULL,
            player2_name TEXT NOT NULL,
            bet_type TEXT NOT NULL,
            selection TEXT NOT NULL,
            odds REAL NOT NULL,
            probability REAL NOT NULL,
            edge REAL NOT NULL,
            expected_value REAL NOT NULL,
            recommended_stake REAL NOT NULL,
            actual_stake REAL,
            model_confidence REAL,
            current_score TEXT,
            bet_status TEXT DEFAULT 'pending',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            settled_at TIMESTAMP,
            result TEXT,
            profit_loss REAL,
            FOREIGN KEY (match_id) REFERENCES live_matches(match_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Live tracking tables initialized")

def save_live_match(match_data: Dict) -> int:
    """Save current live match state to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Convert lists/dicts to JSON
    prob_hist = json.dumps(match_data.get('probability_history', []))
    score_hist = json.dumps(match_data.get('score_history', []))
    point_hist = json.dumps(match_data.get('point_winner_history', []))
    p1_games_hist = json.dumps(match_data.get('p1_games_won_history', []))
    p2_games_hist = json.dumps(match_data.get('p2_games_won_history', []))
    adv_params = json.dumps(match_data.get('advanced_params', {}))
    match_snaps = json.dumps(match_data.get('match_snapshots', []))
    prematch = json.dumps(match_data.get('prematch_odds', {'p1': 1.85, 'p2': 2.10}))
    
    # Check if match already exists (update if so)
    existing = cursor.execute('''
        SELECT match_id FROM live_matches 
        WHERE player1_name = ? AND player2_name = ? AND status = 'active'
    ''', (match_data['player1_name'], match_data['player2_name'])).fetchone()
    
    if existing:
        # Update existing match
        cursor.execute('''
            UPDATE live_matches SET
                p1_sets = ?, p2_sets = ?, p1_games = ?, p2_games = ?,
                p1_points = ?, p2_points = ?, total_points = ?,
                p1_breaks = ?, p2_breaks = ?,
                probability_history = ?, score_history = ?,
                point_winner_history = ?, p1_games_won_history = ?,
                p2_games_won_history = ?, advanced_params = ?,
                match_snapshots = ?, prematch_odds = ?,
                last_updated = CURRENT_TIMESTAMP
            WHERE match_id = ?
        ''', (
            match_data['p1_sets'], match_data['p2_sets'],
            match_data['p1_games'], match_data['p2_games'],
            match_data['p1_points'], match_data['p2_points'],
            match_data['total_points'], match_data['p1_breaks'], match_data['p2_breaks'],
            prob_hist, score_hist, point_hist, p1_games_hist, p2_games_hist,
            adv_params, match_snaps, prematch, existing[0]
        ))
        match_id = existing[0]
    else:
        # Insert new match
        cursor.execute('''
            INSERT INTO live_matches (
                player1_name, player2_name, surface,
                p1_serve_win, p2_serve_win, p1_return_win, p2_return_win,
                p1_sets, p2_sets, p1_games, p2_games,
                p1_points, p2_points, total_points,
                p1_breaks, p2_breaks,
                probability_history, score_history, point_winner_history,
                p1_games_won_history, p2_games_won_history, advanced_params,
                match_snapshots, prematch_odds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            match_data['player1_name'], match_data['player2_name'],
            match_data.get('surface', 'Hard'),
            match_data.get('p1_serve_win', 0.65), match_data.get('p2_serve_win', 0.65),
            match_data.get('p1_return_win', 0.35), match_data.get('p2_return_win', 0.35),
            match_data['p1_sets'], match_data['p2_sets'],
            match_data['p1_games'], match_data['p2_games'],
            match_data['p1_points'], match_data['p2_points'],
            match_data['total_points'], match_data['p1_breaks'], match_data['p2_breaks'],
            prob_hist, score_hist, point_hist, p1_games_hist, p2_games_hist, adv_params,
            match_snaps, prematch
        ))
        match_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    return match_id

def load_live_match(player1_name: str, player2_name: str) -> Optional[Dict]:
    """Load active live match from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    match = cursor.execute('''
        SELECT * FROM live_matches 
        WHERE player1_name = ? AND player2_name = ? AND status = 'active'
        ORDER BY last_updated DESC LIMIT 1
    ''', (player1_name, player2_name)).fetchone()
    
    conn.close()
    
    if not match:
        return None
    
    # Column names
    cols = ['match_id', 'player1_name', 'player2_name', 'surface',
            'p1_serve_win', 'p2_serve_win', 'p1_return_win', 'p2_return_win',
            'p1_sets', 'p2_sets', 'p1_games', 'p2_games',
            'p1_points', 'p2_points', 'total_points',
            'p1_breaks', 'p2_breaks',
            'probability_history', 'score_history', 'point_winner_history',
            'p1_games_won_history', 'p2_games_won_history', 'advanced_params',
            'match_snapshots', 'prematch_odds',
            'status', 'created_at', 'last_updated']
    
    data = dict(zip(cols, match))
    
    # Parse JSON fields (handle None/NULL values)
    def safe_json_load(value, default):
        """Safely parse JSON, handling None and invalid values"""
        if value is None or value == '' or value == 'null':
            return default
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    
    data['probability_history'] = safe_json_load(data.get('probability_history'), [])
    data['score_history'] = safe_json_load(data.get('score_history'), [])
    data['point_winner_history'] = safe_json_load(data.get('point_winner_history'), [])
    data['p1_games_won_history'] = safe_json_load(data.get('p1_games_won_history'), [])
    data['p2_games_won_history'] = safe_json_load(data.get('p2_games_won_history'), [])
    data['advanced_params'] = safe_json_load(data.get('advanced_params'), {})
    data['match_snapshots'] = safe_json_load(data.get('match_snapshots'), [])
    data['prematch_odds'] = safe_json_load(data.get('prematch_odds'), {'p1': 1.85, 'p2': 2.10})
    
    return data

def finish_live_match(player1_name: str, player2_name: str):
    """Mark live match as finished"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE live_matches 
        SET status = 'finished', last_updated = CURRENT_TIMESTAMP
        WHERE player1_name = ? AND player2_name = ? AND status = 'active'
    ''', (player1_name, player2_name))
    
    conn.commit()
    conn.close()

def save_selected_bet(bet_data: Dict) -> int:
    """Save user's selected bet to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO selected_bets (
            match_id, player1_name, player2_name,
            bet_type, selection, odds, probability, edge,
            expected_value, recommended_stake, actual_stake,
            model_confidence, current_score, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        bet_data.get('match_id'),
        bet_data['player1_name'],
        bet_data['player2_name'],
        bet_data['bet_type'],
        bet_data['selection'],
        bet_data['odds'],
        bet_data['probability'],
        bet_data['edge'],
        bet_data['expected_value'],
        bet_data['recommended_stake'],
        bet_data.get('actual_stake'),
        bet_data.get('model_confidence', 0.0),
        bet_data.get('current_score', '0-0'),
        bet_data.get('notes', '')
    ))
    
    bet_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return bet_id

def get_pending_bets(limit: int = 10) -> List[Dict]:
    """Get recent pending bets"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    bets = cursor.execute('''
        SELECT * FROM selected_bets 
        WHERE bet_status = 'pending'
        ORDER BY created_at DESC
        LIMIT ?
    ''', (limit,)).fetchall()
    
    conn.close()
    
    if not bets:
        return []
    
    cols = ['bet_id', 'match_id', 'player1_name', 'player2_name',
            'bet_type', 'selection', 'odds', 'probability', 'edge',
            'expected_value', 'recommended_stake', 'actual_stake',
            'model_confidence', 'current_score', 'bet_status', 'notes',
            'created_at', 'settled_at', 'result', 'profit_loss']
    
    return [dict(zip(cols, bet)) for bet in bets]

def get_all_selected_bets(status: Optional[str] = None) -> List[Dict]:
    """Get all selected bets, optionally filtered by status"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if status:
        bets = cursor.execute('''
            SELECT * FROM selected_bets WHERE bet_status = ?
            ORDER BY created_at DESC
        ''', (status,)).fetchall()
    else:
        bets = cursor.execute('''
            SELECT * FROM selected_bets ORDER BY created_at DESC
        ''').fetchall()
    
    conn.close()
    
    if not bets:
        return []
    
    cols = ['bet_id', 'match_id', 'player1_name', 'player2_name',
            'bet_type', 'selection', 'odds', 'probability', 'edge',
            'expected_value', 'recommended_stake', 'actual_stake',
            'model_confidence', 'current_score', 'bet_status', 'notes',
            'created_at', 'settled_at', 'result', 'profit_loss']
    
    return [dict(zip(cols, bet)) for bet in bets]

# Initialize tables on import
try:
    init_live_tracking_tables()
except Exception as e:
    print(f"⚠️ Error initializing live tracking tables: {e}")
