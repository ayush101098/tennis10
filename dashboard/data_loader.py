"""
Data Loader - Database queries and caching for dashboard
=========================================================
Provides clean interface for dashboard to access data
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import streamlit as st

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tennis_betting.db')


def get_database_connection():
    """Get database connection"""
    if not os.path.exists(DB_PATH):
        # Create empty database with schema
        conn = sqlite3.connect(DB_PATH)
        _create_schema(conn)
        conn.close()
    
    return sqlite3.connect(DB_PATH)


def _create_schema(conn):
    """Create database schema if doesn't exist"""
    cursor = conn.cursor()
    
    # Matches table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS upcoming_matches (
            match_id TEXT PRIMARY KEY,
            player1_id TEXT,
            player2_id TEXT,
            player1_name TEXT,
            player2_name TEXT,
            tournament TEXT,
            surface TEXT,
            scheduled_time TIMESTAMP,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Odds table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS live_odds (
            odds_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            bookmaker TEXT,
            player1_odds REAL,
            player2_odds REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (match_id) REFERENCES upcoming_matches(match_id)
        )
    """)
    
    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            markov_p1_win REAL,
            lr_p1_win REAL,
            nn_p1_win REAL,
            ensemble_p1_win REAL,
            model_agreement REAL,
            confidence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (match_id) REFERENCES upcoming_matches(match_id)
        )
    """)
    
    # Bets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            player_bet_on TEXT,
            odds REAL,
            stake REAL,
            edge REAL,
            expected_value REAL,
            confidence TEXT,
            status TEXT DEFAULT 'active',
            result TEXT,
            profit REAL,
            placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            settled_at TIMESTAMP,
            FOREIGN KEY (match_id) REFERENCES upcoming_matches(match_id)
        )
    """)
    
    # Bankroll history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bankroll_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bankroll REAL,
            daily_pnl REAL,
            roi REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert initial bankroll
    cursor.execute("""
        INSERT INTO bankroll_history (bankroll, daily_pnl, roi)
        VALUES (1000.0, 0.0, 0.0)
    """)
    
    conn.commit()


# ============================================================================
# BANKROLL & PORTFOLIO QUERIES
# ============================================================================

@st.cache_data(ttl=60)
def get_bankroll_status() -> Dict:
    """Get current bankroll and change"""
    conn = get_database_connection()
    
    df = pd.read_sql("""
        SELECT bankroll, daily_pnl, roi, timestamp
        FROM bankroll_history
        ORDER BY timestamp DESC
        LIMIT 30
    """, conn)
    
    conn.close()
    
    if len(df) == 0:
        return {'current': 1000, 'change_pct': 0}
    
    current = df.iloc[0]['bankroll']
    previous = df.iloc[1]['bankroll'] if len(df) > 1 else current
    change_pct = ((current - previous) / previous * 100) if previous > 0 else 0
    
    return {
        'current': current,
        'previous': previous,
        'change': current - previous,
        'change_pct': change_pct
    }


@st.cache_data(ttl=60)
def get_roi_metrics(days: int = 30) -> Dict:
    """Get ROI metrics for specified period"""
    conn = get_database_connection()
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    df = pd.read_sql(f"""
        SELECT *
        FROM bets
        WHERE placed_at >= '{cutoff_date.isoformat()}'
        AND status = 'settled'
    """, conn)
    
    conn.close()
    
    if len(df) == 0:
        return {'roi': 0, 'change': 0, 'win_rate': 0, 'win_rate_change': 0}
    
    total_staked = df['stake'].sum()
    total_profit = df['profit'].sum()
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    
    wins = (df['result'] == 'won').sum()
    total_bets = len(df)
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
    
    return {
        'roi': roi,
        'change': 0,  # TODO: Calculate vs previous period
        'win_rate': win_rate,
        'win_rate_change': 0  # TODO: Calculate vs previous period
    }


# ============================================================================
# MATCHES & PREDICTIONS QUERIES
# ============================================================================

@st.cache_data(ttl=60)
def get_upcoming_matches(hours: int = 24) -> pd.DataFrame:
    """Get upcoming matches within specified hours"""
    conn = get_database_connection()
    
    cutoff_time = datetime.now() + timedelta(hours=hours)
    
    df = pd.read_sql(f"""
        SELECT *
        FROM upcoming_matches
        WHERE scheduled_time <= '{cutoff_time.isoformat()}'
        AND scheduled_time >= '{datetime.now().isoformat()}'
        ORDER BY scheduled_time ASC
    """, conn)
    
    conn.close()
    
    if len(df) > 0:
        df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
    
    return df


@st.cache_data(ttl=60)
def get_upcoming_matches_count(hours: int = 24) -> Dict:
    """Get count of upcoming matches"""
    matches = get_upcoming_matches(hours=hours)
    
    # Get yesterday's count for comparison
    yesterday_matches = get_upcoming_matches(hours=hours - 24)
    
    return {
        'count': len(matches),
        'change': len(matches) - len(yesterday_matches)
    }


@st.cache_data(ttl=60)
def get_predictions(match_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """Get predictions for matches"""
    conn = get_database_connection()
    
    if match_ids:
        placeholders = ','.join('?' * len(match_ids))
        query = f"""
            SELECT p.*, m.player1_name, m.player2_name, m.tournament, m.surface, m.scheduled_time
            FROM predictions p
            JOIN upcoming_matches m ON p.match_id = m.match_id
            WHERE p.match_id IN ({placeholders})
            ORDER BY p.created_at DESC
        """
        df = pd.read_sql(query, conn, params=match_ids)
    else:
        df = pd.read_sql("""
            SELECT p.*, m.player1_name, m.player2_name, m.tournament, m.surface, m.scheduled_time
            FROM predictions p
            JOIN upcoming_matches m ON p.match_id = m.match_id
            ORDER BY m.scheduled_time ASC
        """, conn)
    
    conn.close()
    
    if len(df) > 0:
        df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
    
    return df


@st.cache_data(ttl=60)
def get_recommended_bets_count() -> Dict:
    """Get count of recommended bets"""
    conn = get_database_connection()
    
    df = pd.read_sql("""
        SELECT COUNT(*) as count, SUM(edge * stake) as total_edge
        FROM bets
        WHERE status = 'recommended'
        AND placed_at >= datetime('now', '-1 hour')
    """, conn)
    
    conn.close()
    
    return {
        'count': int(df.iloc[0]['count']) if len(df) > 0 else 0,
        'total_edge': float(df.iloc[0]['total_edge']) if len(df) > 0 and df.iloc[0]['total_edge'] else 0,
        'change': 0,  # TODO: Calculate vs previous hour
        'edge_change': 0
    }


@st.cache_data(ttl=60)
def get_model_agreement() -> Dict:
    """Get average model agreement"""
    conn = get_database_connection()
    
    df = pd.read_sql("""
        SELECT AVG(model_agreement) as avg_agreement
        FROM predictions
        WHERE created_at >= datetime('now', '-24 hours')
    """, conn)
    
    conn.close()
    
    avg = df.iloc[0]['avg_agreement'] if len(df) > 0 and df.iloc[0]['avg_agreement'] else 0
    
    return {
        'avg': avg * 100,
        'change': 0  # TODO: Calculate vs previous period
    }


# ============================================================================
# BETTING HISTORY QUERIES
# ============================================================================

@st.cache_data(ttl=60)
def get_active_bets() -> pd.DataFrame:
    """Get all active bets"""
    conn = get_database_connection()
    
    df = pd.read_sql("""
        SELECT b.*, m.player1_name, m.player2_name, m.scheduled_time,
               (m.player1_name || ' vs ' || m.player2_name) as match
        FROM bets b
        JOIN upcoming_matches m ON b.match_id = m.match_id
        WHERE b.status = 'active'
        ORDER BY m.scheduled_time ASC
    """, conn)
    
    conn.close()
    
    if len(df) > 0:
        df['scheduled_time'] = pd.to_datetime(df['scheduled_time'])
        df['potential_profit'] = df['stake'] * (df['odds'] - 1)
        df['selection'] = df['player_bet_on']
        df['start_time'] = df['scheduled_time']
    
    return df


@st.cache_data(ttl=60)
def get_active_bets_count() -> int:
    """Get count of active bets"""
    return len(get_active_bets())


@st.cache_data(ttl=60)
def get_settled_bets(days: int = 30) -> pd.DataFrame:
    """Get settled bets history"""
    conn = get_database_connection()
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    df = pd.read_sql(f"""
        SELECT b.*, m.player1_name, m.player2_name,
               (m.player1_name || ' vs ' || m.player2_name) as match,
               date(b.placed_at) as date
        FROM bets b
        JOIN upcoming_matches m ON b.match_id = m.match_id
        WHERE b.status = 'settled'
        AND b.placed_at >= '{cutoff_date.isoformat()}'
        ORDER BY b.settled_at DESC
    """, conn)
    
    conn.close()
    
    if len(df) > 0:
        df['date'] = pd.to_datetime(df['date'])
        df['selection'] = df['player_bet_on']
    
    return df


# ============================================================================
# PERFORMANCE & ANALYTICS QUERIES
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_pnl_history(days: int = 30) -> pd.DataFrame:
    """Get PnL history for charting"""
    conn = get_database_connection()
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    df = pd.read_sql(f"""
        SELECT timestamp, bankroll, daily_pnl, roi
        FROM bankroll_history
        WHERE timestamp >= '{cutoff_date.isoformat()}'
        ORDER BY timestamp ASC
    """, conn)
    
    conn.close()
    
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


@st.cache_data(ttl=300)
def get_performance_metrics(period: str = 'Last 30 Days') -> Dict:
    """Get comprehensive performance metrics"""
    
    days_map = {
        'Last 7 Days': 7,
        'Last 30 Days': 30,
        'Last 3 Months': 90,
        'All Time': 36500
    }
    
    days = days_map.get(period, 30)
    
    bets = get_settled_bets(days=days)
    
    if len(bets) == 0:
        return {
            'roi': 0,
            'roi_change': 0,
            'win_rate': 0,
            'win_rate_change': 0,
            'log_loss': 0,
            'log_loss_change': 0,
            'sharpe': 0,
            'sharpe_change': 0
        }
    
    total_staked = bets['stake'].sum()
    total_profit = bets['profit'].sum()
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
    
    wins = (bets['result'] == 'won').sum()
    win_rate = (wins / len(bets) * 100) if len(bets) > 0 else 0
    
    # Sharpe ratio (simplified)
    if len(bets) > 1:
        returns = bets['profit'] / bets['stake']
        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
    else:
        sharpe = 0
    
    return {
        'roi': roi,
        'roi_change': 0,  # TODO: Compare to previous period
        'win_rate': win_rate,
        'win_rate_change': 0,
        'log_loss': 0.61,  # TODO: Calculate actual log loss
        'log_loss_change': 0,
        'sharpe': sharpe,
        'sharpe_change': 0
    }


@st.cache_data(ttl=300)
def get_calibration_data() -> pd.DataFrame:
    """Get data for calibration curve"""
    conn = get_database_connection()
    
    df = pd.read_sql("""
        SELECT b.result, p.ensemble_p1_win, b.player_bet_on, m.player1_name
        FROM bets b
        JOIN predictions p ON b.match_id = p.match_id
        JOIN upcoming_matches m ON b.match_id = m.match_id
        WHERE b.status = 'settled'
    """, conn)
    
    conn.close()
    
    if len(df) == 0:
        return pd.DataFrame({
            'predicted_prob': [],
            'actual_freq': [],
            'count': []
        })
    
    # Calculate calibration buckets
    df['predicted_prob'] = df.apply(
        lambda x: x['ensemble_p1_win'] if x['player_bet_on'] == x['player1_name'] else 1 - x['ensemble_p1_win'],
        axis=1
    )
    df['won'] = (df['result'] == 'won').astype(int)
    
    # Bin predictions
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    df['bin'] = pd.cut(df['predicted_prob'], bins=bins)
    
    calibration = df.groupby('bin').agg({
        'won': ['mean', 'count'],
        'predicted_prob': 'mean'
    }).reset_index()
    
    calibration.columns = ['bin', 'actual_freq', 'count', 'predicted_prob']
    
    return calibration


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def record_bet(bet_data: Dict, actual_stake: float):
    """Record a new bet in the database"""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO bets (
            match_id, player_bet_on, odds, stake, edge,
            expected_value, confidence, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
    """, (
        bet_data['match_id'],
        bet_data['player_bet_on'],
        bet_data['odds'],
        actual_stake,
        bet_data['edge'],
        bet_data['expected_value'],
        bet_data['confidence']
    ))
    
    conn.commit()
    conn.close()
    
    # Clear cache
    get_active_bets.clear()
    get_active_bets_count.clear()


def update_bet_result(bet_id: int, result: str, profit: float):
    """Update bet with result"""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE bets
        SET status = 'settled', result = ?, profit = ?, settled_at = CURRENT_TIMESTAMP
        WHERE bet_id = ?
    """, (result, profit, bet_id))
    
    conn.commit()
    conn.close()
    
    # Clear cache
    get_active_bets.clear()
    get_settled_bets.clear()


def update_bankroll(new_bankroll: float, daily_pnl: float, roi: float):
    """Update bankroll history"""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO bankroll_history (bankroll, daily_pnl, roi)
        VALUES (?, ?, ?)
    """, (new_bankroll, daily_pnl, roi))
    
    conn.commit()
    conn.close()
    
    # Clear cache
    get_bankroll_status.clear()
    get_pnl_history.clear()
