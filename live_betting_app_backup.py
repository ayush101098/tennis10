"""
🎾 Tennis Betting Intelligence Hub
===================================
A beautiful, easy-to-navigate betting analysis platform

Run: streamlit run live_betting_app.py
"""

import streamlit as st
import numpy as np
from scipy.special import comb
import sqlite3
import os
from pathlib import Path
from datetime import datetime
import json
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import ML models
try:
    from hierarchical_model import HierarchicalTennisModel
    from ml_models.logistic_regression import SymmetricLogisticRegression
    from ml_models.neural_network import NeuralNetworkTrainer
    import torch
    MODELS_AVAILABLE = True
except Exception as e:
    MODELS_AVAILABLE = False
    MODEL_ERROR = str(e)

# Try to import TennisRatio integration
try:
    from api.tennisratio_integration import get_tennisratio_insights, TennisRatioAPI
    TENNISRATIO_AVAILABLE = True
except Exception as e:
    TENNISRATIO_AVAILABLE = False
    TENNISRATIO_ERROR = str(e)

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🎾 Tennis Betting Hub",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== BOOKMAKER TERMINAL CSS ====================
st.markdown("""
<style>
    /* Clean terminal background */
    .main { background-color: #0a0e27; color: #e0e0e0; }
    
    /* Market header */
    .market-header {
        background: #151932;
        border: 1px solid #2d3561;
        padding: 8px 15px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-family: 'Monaco', 'Courier New', monospace;
    }
    
    /* Score display - compact */
    .score-panel {
        background: #1a1f3a;
        border: 1px solid #2d3561;
        padding: 10px;
        border-radius: 4px;
        font-family: 'Monaco', monospace;
        margin-bottom: 10px;
    }
    .score-panel table { width: 100%; border-collapse: collapse; }
    .score-panel td { padding: 4px 8px; border-bottom: 1px solid #2d3561; }
    .score-panel .serve-dot { color: #00ff88; font-weight: bold; }
    
    /* Odds table */
    .odds-table {
        background: #151932;
        border: 1px solid #2d3561;
        border-radius: 4px;
        font-family: 'Monaco', monospace;
        margin: 10px 0;
    }
    .odds-row {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
        padding: 8px 12px;
        border-bottom: 1px solid #2d3561;
        align-items: center;
    }
    .odds-row:hover { background: #1a1f3a; }
    .odds-header { background: #1a1f3a; font-weight: bold; color: #00ff88; }
    .odds-value { 
        background: #2d3561; 
        padding: 6px 10px; 
        border-radius: 3px; 
        text-align: center;
        font-weight: bold;
        cursor: pointer;
    }
    .odds-value:hover { background: #3d4571; }
    .odds-value.back { background: #1e4d7b; color: #7ec8e3; }
    .odds-value.lay { background: #7b1e2e; color: #e37e7e; }
    .odds-value.value { background: #1e7b4d; color: #7ee3a3; animation: glow 2s infinite; }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(126, 227, 163, 0.5); }
        50% { box-shadow: 0 0 15px rgba(126, 227, 163, 0.8); }
    }
    
    /* Probability panel */
    .prob-panel {
        background: #151932;
        border: 1px solid #2d3561;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
        font-family: 'Monaco', monospace;
        font-size: 0.85rem;
    }
    .prob-row {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        border-bottom: 1px solid #2d3561;
    }
    .prob-label { color: #888; }
    .prob-value { color: #00ff88; font-weight: bold; }
    .edge-positive { color: #00ff88; }
    .edge-negative { color: #ff4466; }
    
    /* P&L tracker */
    .pl-panel {
        background: #1a1f3a;
        border: 1px solid #2d3561;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 8px 0;
        font-family: 'Monaco', monospace;
        font-size: 0.9rem;
    }
    .pl-positive { color: #00ff88; font-weight: bold; }
    .pl-negative { color: #ff4466; font-weight: bold; }
    
    /* Position table */
    .position-table {
        background: #151932;
        border: 1px solid #2d3561;
        border-radius: 4px;
        margin: 10px 0;
        font-family: 'Monaco', monospace;
        font-size: 0.85rem;
    }
    .position-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr 1fr;
        padding: 6px 10px;
        border-bottom: 1px solid #2d3561;
    }
    .position-header { background: #1a1f3a; font-weight: bold; color: #888; }
    
    /* Compact buttons */
    .stButton button {
        background: #2d3561;
        color: #e0e0e0;
        border: 1px solid #3d4571;
        border-radius: 3px;
        padding: 4px 12px;
        font-family: 'Monaco', monospace;
    }
    .stButton button:hover { background: #3d4571; border-color: #4d5581; }
    
    /* Hide unnecessary Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    }
    .momentum-marker {
        width: 20px;
        height: 20px;
        background: white;
        border-radius: 50%;
        position: absolute;
        top: -5px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        transition: left 0.3s ease;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* AI Insights Card */
    .ai-insight {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.4);
    }
    .ai-insight h3 {
        margin: 0 0 15px 0;
        font-size: 1.5rem;
    }
    .ai-insight .model-consensus {
        background: rgba(255,255,255,0.2);
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
    }
    
    /* Model prediction cards */
    .model-card {
        background: linear-gradient(135deg, #2d3436 0%, #000000 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        border: 2px solid rgba(255,255,255,0.1);
    }
    .model-card h4 {
        margin: 0 0 10px 0;
        color: #38ef7d;
    }
    .model-card .prob-bar {
        height: 30px;
        background: linear-gradient(90deg, #ef473a 0%, #ffc107 50%, #38ef7d 100%);
        border-radius: 5px;
        position: relative;
        margin: 10px 0;
    }
    .model-card .prob-marker {
        position: absolute;
        width: 4px;
        height: 30px;
        background: white;
        box-shadow: 0 0 10px rgba(255,255,255,0.8);
    }
</style>
""", unsafe_allow_html=True)


# ==================== MODEL LOADING ====================
@st.cache_resource
def load_ml_models():
    """Load all three trained models"""
    models = {
        'hierarchical': None,
        'logistic': None,
        'neural': None,
        'status': {}
    }
    
    if not MODELS_AVAILABLE:
        models['status']['error'] = 'Models not available: ' + MODEL_ERROR
        return models
    
    try:
        # Load Hierarchical Markov Model
        if os.path.exists('tennis_data.db'):
            models['hierarchical'] = HierarchicalTennisModel('tennis_data.db')
            models['status']['hierarchical'] = 'Loaded ✓'
        else:
            models['status']['hierarchical'] = 'No database'
    except Exception as e:
        models['status']['hierarchical'] = f'Error: {e}'
    
    try:
        # Load Logistic Regression
        lr_path = Path('ml_models/logistic_regression_trained.pkl')
        if lr_path.exists():
            with open(lr_path, 'rb') as f:
                models['logistic'] = pickle.load(f)
            models['status']['logistic'] = 'Loaded ✓'
        else:
            models['status']['logistic'] = 'Not trained'
    except Exception as e:
        models['status']['logistic'] = f'Error: {e}'
    
    try:
        # Load Neural Network Ensemble
        nn_path = Path('ml_models/neural_network_ensemble.pkl')
        if nn_path.exists():
            with open(nn_path, 'rb') as f:
                models['neural'] = pickle.load(f)
            models['status']['neural'] = 'Loaded ✓'
        else:
            models['status']['neural'] = 'Not trained'
    except Exception as e:
        models['status']['neural'] = f'Error: {e}'
    
    return models


# ==================== DATABASE FUNCTIONS ====================
DB_PATH = Path(__file__).parent / "tennis_data.db"

@st.cache_resource
def get_db_connection():
    if os.path.exists(DB_PATH):
        return sqlite3.connect(str(DB_PATH), check_same_thread=False)
    return None

def search_player(name: str, limit: int = 5):
    conn = get_db_connection()
    if not conn or len(name) < 2:
        return []
    try:
        query = """
        SELECT DISTINCT p.player_id, p.player_name
        FROM players p
        WHERE LOWER(p.player_name) LIKE LOWER(?)
        LIMIT ?
        """
        cursor = conn.execute(query, (f"%{name}%", limit))
        return [{'id': r[0], 'name': r[1], 'country': ''} for r in cursor.fetchall()]
    except:
        return []

def get_player_serve_stats(player_id: int, surface: str = None):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        surface_filter = f"AND m.surface = '{surface}'" if surface and surface != "All" else ""
        query = f"""
        SELECT 
            AVG(s.first_serve_pct) as first_serve_pct,
            AVG(CAST(s.first_serve_points_won AS FLOAT) / NULLIF(s.first_serve_points_total, 0)) as first_serve_win_pct,
            AVG(CAST(s.second_serve_points_won AS FLOAT) / NULLIF(s.second_serve_points_total, 0)) as second_serve_win_pct,
            AVG(CAST(s.aces AS FLOAT) / NULLIF(s.service_games, 0)) as aces_per_game,
            AVG(CAST(s.double_faults AS FLOAT) / NULLIF(s.service_games, 0)) as df_per_game,
            COUNT(*) as match_count
        FROM statistics s
        JOIN matches m ON s.match_id = m.match_id
        WHERE s.player_id = ?
            AND m.tournament_date >= date('now', '-365 days')
            AND s.first_serve_pct IS NOT NULL
            {surface_filter}
        """
        cursor = conn.execute(query, (player_id,))
        row = cursor.fetchone()
        if row and row[5] >= 3:
            first_pct = row[0] or 0.62
            first_win = row[1] or 0.70
            second_win = row[2] or 0.50
            serve_point_pct = first_pct * first_win + (1 - first_pct) * 0.95 * second_win
            return {
                'first_serve_pct': row[0], 'first_serve_win_pct': row[1],
                'second_serve_win_pct': row[2], 'aces_per_game': row[3],
                'df_per_game': row[4], 'match_count': row[5],
                'serve_point_pct': serve_point_pct
            }
        return None
    except Exception as e:
        print(f"Error in get_player_serve_stats: {e}")
        return None

def get_h2h_record(player1_id: int, player2_id: int):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        query = """
        SELECT 
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as p1_wins,
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as p2_wins
        FROM matches
        WHERE (winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)
        """
        cursor = conn.execute(query, (player1_id, player2_id, player1_id, player2_id, player2_id, player1_id))
        row = cursor.fetchone()
        if row and (row[0] or row[1]):
            return {'p1_wins': row[0] or 0, 'p2_wins': row[1] or 0}
        return None
    except:
        return None


# ==================== ML PREDICTION FUNCTIONS ====================
def get_ml_predictions(p1_serve_pct, p2_serve_pct, surface, best_of=3):
    """
    Get predictions from all three models + TennisRatio data
    NOW RUNS MARKOV FOR ALL MATCHES - with database stats OR manual serve percentages
    
    Returns:
        dict with model predictions and ensemble consensus
    """
    predictions = {
        'hierarchical': None,
        'logistic': None,
        'neural': None,
        'tennisratio': None,
        'ensemble': None,
        'insights': [],
        'features': {}  # Store extracted features for analysis
    }
    
    models = load_ml_models()
    
    # ========================================
    # HIERARCHICAL MARKOV MODEL - RUNS FOR ALL MATCHES
    # ========================================
    if models['hierarchical']:
        try:
            # Try to use database-backed prediction first
            if st.session_state.p1_id and st.session_state.p2_id:
                result = models['hierarchical'].predict_match(
                    st.session_state.p1_id,
                    st.session_state.p2_id,
                    surface,
                    num_sets=best_of
                )
                predictions['hierarchical'] = result['p_match']
                predictions['insights'].append({
                    'model': '⚡ Markov Chain (Database)',
                    'prob': result['p_match'],
                    'insight': f"Historical point-level analysis: {result['p_match']:.1%} win probability from {st.session_state.p1_stats['match_count'] if st.session_state.p1_stats else 0}+ matches"
                })
            else:
                # MANUAL MARKOV - Use current serve percentages for LIVE matches
                # Calculate game probability from serve percentages
                p1_serve = p1_serve_pct / 100.0
                p2_serve = p2_serve_pct / 100.0
                
                # Estimate game win probabilities
                p1_hold = p_game_from_points(0, 0, p1_serve)
                p2_hold = p_game_from_points(0, 0, p2_serve)
                
                # Estimate set win probability (player 1 serving first)
                p1_set = p_set_from_games(0, 0, 1, p1_serve, p2_serve)
                
                # Estimate match win probability
                p1_match = p_match_from_sets(0, 0, p1_set, best_of)
                
                predictions['hierarchical'] = p1_match
                predictions['features']['p1_hold_prob'] = p1_hold
                predictions['features']['p2_hold_prob'] = p2_hold
                predictions['features']['p1_set_prob'] = p1_set
                
                predictions['insights'].append({
                    'model': '⚡ Markov Chain (Live)',
                    'prob': p1_match,
                    'insight': f"Real-time point-level simulation from serve stats ({p1_serve_pct}% vs {p2_serve_pct}%)"
                })
        except Exception as e:
            predictions['insights'].append({
                'model': '⚡ Markov Chain Model',
                'prob': None,
                'insight': f"Unable to compute: {str(e)[:50]}"
            })
    
    # ========================================
    # TENNISRATIO INTEGRATION - LIVE WEB DATA
    # ========================================
    if TENNISRATIO_AVAILABLE and st.session_state.p1 and st.session_state.p2:
        try:
            h2h_data, tr_pred = get_tennisratio_insights(
                st.session_state.p1,
                st.session_state.p2
            )
            if tr_pred and tr_pred.get('probability_p1'):
                predictions['tennisratio'] = tr_pred['probability_p1']
                
                # Store TennisRatio features for comprehensive analysis
                if h2h_data:
                    predictions['features']['tr_h2h'] = h2h_data
                    predictions['features']['tr_dominance'] = tr_pred.get('dominance_factor', 0)
                    predictions['features']['tr_efficiency'] = tr_pred.get('efficiency_factor', 0)
                
                factors_str = " | ".join(tr_pred['factors'][:2]) if tr_pred.get('factors') else ""
                predictions['insights'].append({
                    'model': '🌐 TennisRatio Advanced Stats',
                    'prob': tr_pred['probability_p1'],
                    'insight': f"Real-time data from tennisratio.com ({tr_pred['confidence']} confidence): {factors_str}"
                })
            else:
                predictions['insights'].append({
                    'model': '🌐 TennisRatio Advanced Stats',
                    'prob': None,
                    'insight': 'Fetching live data from tennisratio.com...'
                })
        except Exception as e:
            predictions['insights'].append({
                'model': '🌐 TennisRatio Advanced Stats',
                'prob': None,
                'insight': f'Data fetch error: {str(e)[:40]}'
            })
    
    # ========================================
    # LOGISTIC REGRESSION & NEURAL NETWORK
    # These require full match features - show status for now
    # ========================================
    if models['logistic']:
        predictions['insights'].append({
            'model': '📈 Logistic Regression',
            'prob': None,
            'insight': 'Requires full match statistics (available after first set completion)'
        })
    
    if models['neural']:
        predictions['insights'].append({
            'model': '🧠 Neural Network Ensemble',
            'prob': None,
            'insight': 'Requires full match statistics (available after first set completion)'
        })
    
    # ========================================
    # TRUE PROBABILITY CALCULATION
    # Weighted ensemble based on model confidence and data quality
    # ========================================
    model_weights = []
    model_values = []
    
    # Hierarchical Markov - highest weight when using database stats
    if predictions['hierarchical'] is not None:
        if st.session_state.p1_id and st.session_state.p2_id:
            # Database-backed: high confidence (weight = 0.40)
            model_weights.append(0.40)
        else:
            # Live serve stats: moderate confidence (weight = 0.25)
            model_weights.append(0.25)
        model_values.append(predictions['hierarchical'])
    
    # TennisRatio - weight based on confidence level
    if predictions['tennisratio'] is not None:
        confidence = h2h_data.get('confidence', 'low') if 'h2h_data' in locals() else 'low'
        if confidence == 'high':
            model_weights.append(0.35)
        elif confidence == 'medium':
            model_weights.append(0.25)
        else:
            model_weights.append(0.15)
        model_values.append(predictions['tennisratio'])
    
    # Logistic Regression (when available)
    if predictions['logistic'] is not None:
        model_weights.append(0.30)
        model_values.append(predictions['logistic'])
    
    # Neural Network (when available)
    if predictions['neural'] is not None:
        model_weights.append(0.30)
        model_values.append(predictions['neural'])
    
    # Calculate weighted ensemble
    if model_values:
        # Normalize weights to sum to 1.0
        total_weight = sum(model_weights)
        normalized_weights = [w / total_weight for w in model_weights]
        
        # Calculate weighted average
        predictions['ensemble'] = sum(v * w for v, w in zip(model_values, normalized_weights))
        predictions['confidence'] = sum(normalized_weights[:2]) if len(normalized_weights) >= 2 else 0.5  # Higher when multiple models agree
        
        # Store for transparency
        predictions['features']['model_weights'] = {
            'hierarchical': normalized_weights[0] if len(normalized_weights) > 0 else 0,
            'tennisratio': normalized_weights[1] if len(normalized_weights) > 1 else 0,
            'logistic': normalized_weights[2] if len(normalized_weights) > 2 else 0,
            'neural': normalized_weights[3] if len(normalized_weights) > 3 else 0
        }
    
    return predictions


def get_ai_insights_html(predictions, p1_name, p2_name, current_score):
    """Generate beautiful AI insights display"""
    
    if not predictions or 'insights' not in predictions:
        return ""
    
    ensemble = predictions.get('ensemble')
    insights = predictions.get('insights', [])
    
    # Determine winner
    if ensemble:
        favorite = p1_name if ensemble > 0.5 else p2_name
        confidence = max(ensemble, 1 - ensemble)
        confidence_text = "Very High" if confidence > 0.75 else ("High" if confidence > 0.65 else ("Moderate" if confidence > 0.55 else "Low"))
        
        # Generate recommendation
        edge = abs(ensemble - 0.5)
        if edge > 0.15:
            recommendation = f"🎯 Strong edge detected on {favorite}"
        elif edge > 0.08:
            recommendation = f"✅ Moderate edge on {favorite}"
        else:
            recommendation = "⚠️ Tight matchup - minimal edge"
    else:
        favorite = "Analysis pending"
        confidence_text = "N/A"
        recommendation = "📊 Gathering data..."
    
    html = f"""
    <div class="ai-insight">
        <h3>🤖 AI Model Intelligence Hub</h3>
        """
    
    if ensemble:
        p1_pct = ensemble * 100
        p2_pct = (1 - ensemble) * 100
        marker_pos = ensemble * 100
        
        html += f"""
        <div class="model-consensus">
            <div style="font-size: 1.3rem; margin-bottom: 5px; font-weight: bold;">
                {recommendation}
            </div>
            <div style="font-size: 1.1rem; margin-bottom: 15px; opacity: 0.9;">
                Consensus: <strong>{favorite}</strong> ({confidence_text} Confidence)
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 1.05rem;">
                <span><strong>{p1_name}:</strong> {p1_pct:.1f}%</span>
                <span><strong>{p2_name}:</strong> {p2_pct:.1f}%</span>
            </div>
            <div style="height: 35px; background: linear-gradient(90deg, #ef473a 0%, #ffc107 50%, #38ef7d 100%); border-radius: 8px; position: relative; box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);">
                <div style="position: absolute; width: 5px; height: 35px; background: white; left: {marker_pos}%; box-shadow: 0 0 15px rgba(255,255,255,0.9); border-radius: 2px;"></div>
            </div>
        </div>
        """
    else:
        html += f"""
        <div class="model-consensus">
            <div style="font-size: 1.2rem; margin-bottom: 10px;">
                {recommendation}
            </div>
            <div style="opacity: 0.8;">Models are analyzing match data...</div>
        </div>
        """
    
    # Individual model predictions
    html += "<div style='margin-top: 20px;'>"
    for insight in insights:
        if insight['prob']:
            prob_pct = insight['prob'] * 100
            html += f"""
            <div class="model-card">
                <h4>{insight['model']}</h4>
                <div style="margin: 8px 0; opacity: 0.9;">{insight['insight']}</div>
                <div style="margin-top: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 5px;">
                        <span>{p1_name}: {prob_pct:.1f}%</span>
                        <span>{p2_name}: {100-prob_pct:.1f}%</span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-marker" style="left: {prob_pct}%;"></div>
                    </div>
                </div>
            </div>
            """
        else:
            html += f"""
            <div class="model-card" style="opacity: 0.6; border: 2px solid rgba(255,255,255,0.05);">
                <h4>{insight['model']}</h4>
                <div style="font-size: 0.9rem; opacity: 0.8;">{insight['insight']}</div>
            </div>
            """
    html += "</div>"
    
    html += "</div>"
    return html


def get_db_stats():
    conn = get_db_connection()
    if not conn:
        return None
    try:
        stats = {}
        stats['matches'] = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        stats['players'] = conn.execute("SELECT COUNT(DISTINCT player_id) FROM players").fetchone()[0]
        return stats
    except:
        return None


def get_recent_form(player_id: int, limit: int = 5):
    """Get last N matches for a player with results."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        query = """
        SELECT 
            m.tournament_name,
            m.surface,
            CASE WHEN m.winner_id = ? THEN 'W' ELSE 'L' END as result,
            CASE WHEN m.winner_id = ? THEN p2.name ELSE p1.name END as opponent,
            m.score,
            m.tournament_date
        FROM matches m
        JOIN players p1 ON m.winner_id = p1.player_id
        JOIN players p2 ON m.loser_id = p2.player_id
        WHERE m.winner_id = ? OR m.loser_id = ?
        ORDER BY m.tournament_date DESC
        LIMIT ?
        """
        cursor = conn.execute(query, (player_id, player_id, player_id, player_id, limit))
        rows = cursor.fetchall()
        if rows:
            return [{
                'tournament': r[0] or 'Unknown',
                'surface': r[1] or 'Hard',
                'result': r[2],
                'opponent': r[3],
                'score': r[4] or '',
                'date': r[5]
            } for r in rows]
        return None
    except:
        return None


def get_surface_win_rates(player_id: int):
    """Get win rates by surface."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        query = """
        SELECT 
            m.surface,
            SUM(CASE WHEN m.winner_id = ? THEN 1 ELSE 0 END) as wins,
            COUNT(*) as total
        FROM matches m
        WHERE (m.winner_id = ? OR m.loser_id = ?)
            AND m.tournament_date >= date('now', '-730 days')
        GROUP BY m.surface
        HAVING total >= 3
        """
        cursor = conn.execute(query, (player_id, player_id, player_id))
        rows = cursor.fetchall()
        if rows:
            return {r[0]: {'wins': r[1], 'total': r[2], 'pct': r[1]/r[2] if r[2] > 0 else 0} for r in rows if r[0]}
        return None
    except:
        return None


def get_hold_break_stats(player_id: int, surface: str = None):
    """Get average hold and break percentages."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        surface_filter = f"AND m.surface = '{surface}'" if surface and surface != "All" else ""
        query = f"""
        SELECT 
            AVG(s.service_games_won_pct) as hold_pct,
            AVG(s.return_games_won_pct) as break_pct,
            AVG(s.break_points_saved_pct) as bp_saved_pct,
            AVG(s.break_points_converted_pct) as bp_converted_pct,
            COUNT(*) as matches
        FROM statistics s
        JOIN matches m ON s.match_id = m.match_id
        WHERE s.player_id = ?
            AND m.tournament_date >= date('now', '-365 days')
            {surface_filter}
        """
        cursor = conn.execute(query, (player_id,))
        row = cursor.fetchone()
        if row and row[4] >= 3:
            return {
                'hold_pct': row[0],
                'break_pct': row[1],
                'bp_saved_pct': row[2],
                'bp_converted_pct': row[3],
                'matches': row[4]
            }
        return None
    except:
        return None


def get_h2h_details(player1_id: int, player2_id: int):
    """Get detailed H2H with recent matches."""
    conn = get_db_connection()
    if not conn:
        return None
    try:
        # Overall record
        query = """
        SELECT 
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as p1_wins,
            SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as p2_wins
        FROM matches
        WHERE (winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)
        """
        cursor = conn.execute(query, (player1_id, player2_id, player1_id, player2_id, player2_id, player1_id))
        row = cursor.fetchone()
        
        if not row or (not row[0] and not row[1]):
            return None
        
        result = {'p1_wins': row[0] or 0, 'p2_wins': row[1] or 0, 'matches': []}
        
        # Recent H2H matches
        query2 = """
        SELECT 
            m.tournament_name,
            m.surface,
            p1.name as winner,
            m.score,
            m.tournament_date
        FROM matches m
        JOIN players p1 ON m.winner_id = p1.player_id
        WHERE (m.winner_id = ? AND m.loser_id = ?) OR (m.winner_id = ? AND m.loser_id = ?)
        ORDER BY m.tournament_date DESC
        LIMIT 5
        """
        cursor2 = conn.execute(query2, (player1_id, player2_id, player2_id, player1_id))
        for r in cursor2.fetchall():
            result['matches'].append({
                'tournament': r[0] or 'Unknown',
                'surface': r[1] or 'Hard',
                'winner': r[2],
                'score': r[3] or '',
                'date': r[4]
            })
        
        return result
    except:
        return None


# ==================== PROBABILITY FUNCTIONS ====================

def p_game_from_points(server_pts: int, returner_pts: int, p_point: float) -> float:
    if server_pts >= 4 and server_pts >= returner_pts + 2:
        return 1.0
    if returner_pts >= 4 and returner_pts >= server_pts + 2:
        return 0.0
    
    if server_pts >= 3 and returner_pts >= 3:
        p_d = (p_point ** 2) / (p_point ** 2 + (1 - p_point) ** 2)
        if server_pts == returner_pts:
            return p_d
        elif server_pts > returner_pts:
            return p_point + (1 - p_point) * p_d
        else:
            return p_point * p_d
    
    cache = {}
    def prob_from(s, r):
        if (s, r) in cache:
            return cache[(s, r)]
        if s == 4 and r <= 2:
            return 1.0
        if r == 4 and s <= 2:
            return 0.0
        if s >= 3 and r >= 3:
            p_d = (p_point ** 2) / (p_point ** 2 + (1 - p_point) ** 2)
            return p_d if s == r else (p_point + (1-p_point)*p_d if s > r else p_point*p_d)
        result = p_point * prob_from(min(s+1, 4), r) + (1-p_point) * prob_from(s, min(r+1, 4))
        cache[(s, r)] = result
        return result
    return prob_from(server_pts, returner_pts)


def p_win_game(p_point: float) -> float:
    p, q = p_point, 1 - p_point
    p_before_deuce = p**4 * (1 + 4*q + 10*q**2)
    p_deuce = comb(6, 3) * p**3 * q**3 * (p**2 / (p**2 + q**2))
    return p_before_deuce + p_deuce


def p_set_from_games(g1: int, g2: int, server: int, p1_serve: float, p2_serve: float) -> float:
    p_hold_p1 = p_win_game(p1_serve)
    p_hold_p2 = p_win_game(p2_serve)
    
    cache = {}
    def prob_from(g1, g2, srv):
        if (g1, g2, srv) in cache:
            return cache[(g1, g2, srv)]
        if g1 >= 6 and g1 >= g2 + 2: return 1.0
        if g2 >= 6 and g2 >= g1 + 2: return 0.0
        if g1 == 7: return 1.0
        if g2 == 7: return 0.0
        if g1 == 6 and g2 == 6:
            p_avg = (p1_serve + (1 - p2_serve)) / 2
            return (p_avg ** 2) / (p_avg ** 2 + (1 - p_avg) ** 2)
        p_p1_wins = p_hold_p1 if srv == 1 else (1 - p_hold_p2)
        result = p_p1_wins * prob_from(g1+1, g2, 3-srv) + (1-p_p1_wins) * prob_from(g1, g2+1, 3-srv)
        cache[(g1, g2, srv)] = result
        return result
    return prob_from(g1, g2, server)


def p_match_from_sets(s1: int, s2: int, p_set: float, best_of: int = 3) -> float:
    sets_to_win = (best_of + 1) // 2
    cache = {}
    def prob_from(s1, s2):
        if (s1, s2) in cache: return cache[(s1, s2)]
        if s1 >= sets_to_win: return 1.0
        if s2 >= sets_to_win: return 0.0
        result = p_set * prob_from(s1+1, s2) + (1-p_set) * prob_from(s1, s2+1)
        cache[(s1, s2)] = result
        return result
    return prob_from(s1, s2)


def adjust_serve_for_surface(base_serve: float, surface: str, serve_speed: str) -> float:
    surface_adj = {'Hard': 0, 'Clay': -0.03, 'Grass': +0.03, 'Indoor': +0.02, 'All': 0}
    speed_adj = {'Average': 0, 'Big Server (+4%)': +0.04, 'Weak Server (-3%)': -0.03}
    adjusted = base_serve + surface_adj.get(surface, 0) + speed_adj.get(serve_speed, 0)
    return np.clip(adjusted, 0.45, 0.80)


# ==================== SESSION STATE ====================
defaults = {
    'p1': 'Player 1', 'p2': 'Player 2',
    'p1_serve': 62, 'p2_serve': 62,
    'sets': [0, 0], 'games': [0, 0], 'points': [0, 0],
    'server': 1, 'best_of': 3,
    'surface': 'Hard',
    'p1_speed': 'Average', 'p2_speed': 'Average',
    'p1_id': None, 'p2_id': None,
    'p1_stats': None, 'p2_stats': None, 'h2h': None,
    'bankroll': 100.0,
    'game_hold_odds': 1.20, 'game_break_odds': 4.50,
    'set_p1_odds': 1.80, 'set_p2_odds': 2.00,
    'match_p1_odds': 1.50, 'match_p2_odds': 2.50,
    'bets': [],
    'total_staked': 0.0,
    'total_profit': 0.0,
    'point_history': [],
    'games_history': [],
    'current_page': 'Live Match',
}
for k, v in defaults.items():
    if k not in st.session_state:
        # Make a copy of lists to avoid reference issues
        if isinstance(v, list):
            st.session_state[k] = v.copy()
        else:
            st.session_state[k] = v


# ==================== HELPER FUNCTIONS ====================

def get_point_display(pts):
    if pts == 0: return '0'
    elif pts == 1: return '15'
    elif pts == 2: return '30'
    elif pts == 3: return '40'
    else: return 'AD'

def get_score_string():
    p1_pt = get_point_display(st.session_state.points[0])
    p2_pt = get_point_display(st.session_state.points[1])
    
    if st.session_state.points[0] >= 3 and st.session_state.points[1] >= 3:
        if st.session_state.points[0] == st.session_state.points[1]:
            point_str = "DEUCE"
        elif st.session_state.points[0] > st.session_state.points[1]:
            point_str = f"AD-{st.session_state.p1[:3].upper()}"
        else:
            point_str = f"AD-{st.session_state.p2[:3].upper()}"
    else:
        point_str = f"{p1_pt}-{p2_pt}"
    
    return f"{st.session_state.sets[0]}-{st.session_state.sets[1]}  •  {st.session_state.games[0]}-{st.session_state.games[1]}  •  {point_str}"

def record_point(winner: int):
    st.session_state.point_history.append(winner)
    st.session_state.points[winner - 1] += 1
    
    p1, p2 = st.session_state.points
    game_won = None
    
    if p1 >= 4 and p1 >= p2 + 2:
        game_won = 1
    elif p2 >= 4 and p2 >= p1 + 2:
        game_won = 2
    
    if game_won:
        st.session_state.points = [0, 0]
        st.session_state.games[game_won - 1] += 1
        st.session_state.games_history.append({
            'winner': game_won,
            'server': st.session_state.server,
            'was_break': game_won != st.session_state.server
        })
        st.session_state.server = 3 - st.session_state.server
        
        g1, g2 = st.session_state.games
        if (g1 >= 6 and g1 >= g2 + 2) or g1 == 7:
            st.session_state.sets[0] += 1
            st.session_state.games = [0, 0]
        elif (g2 >= 6 and g2 >= g1 + 2) or g2 == 7:
            st.session_state.sets[1] += 1
            st.session_state.games = [0, 0]

def record_bet(bet_type: str, selection: str, odds: float, stake: float, model_prob: float):
    st.session_state.bets.append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'type': bet_type,
        'selection': selection,
        'odds': odds,
        'stake': stake,
        'model_prob': model_prob,
        'ev': (model_prob * odds - 1) * stake,
        'result': None,
        'profit': None
    })
    st.session_state.total_staked += stake

def settle_bet(idx: int, won: bool):
    bet = st.session_state.bets[idx]
    if won:
        bet['result'] = 'WON'
        bet['profit'] = bet['stake'] * (bet['odds'] - 1)
    else:
        bet['result'] = 'LOST'
        bet['profit'] = -bet['stake']
    st.session_state.total_profit += bet['profit']

def get_momentum():
    if len(st.session_state.point_history) < 3:
        return 0.5, "Even", "⚖️"
    recent = st.session_state.point_history[-5:]
    p1_won = sum(1 for p in recent if p == 1)
    ratio = p1_won / len(recent)
    if ratio >= 0.7:
        return ratio, st.session_state.p1, "🔥"
    elif ratio <= 0.3:
        return ratio, st.session_state.p2, "🔥"
    else:
        return ratio, "Even", "⚖️"

def is_break_point():
    server = st.session_state.server
    s_pts = st.session_state.points[server - 1]
    r_pts = st.session_state.points[2 - server]
    if r_pts >= 3 and r_pts > s_pts:
        return True
    return False

def get_break_point_count():
    server = st.session_state.server
    s_pts = st.session_state.points[server - 1]
    r_pts = st.session_state.points[2 - server]
    if r_pts < 3:
        return 0
    if s_pts < 3:
        return r_pts - 2
    return 1


# ==================== SIDEBAR - PLAYER SETUP ====================
with st.sidebar:
    st.markdown("## 🎾 Match Setup")
    
    # Player 1
    st.markdown("### Player 1")
    p1_input = st.text_input("Name", st.session_state.p1, key="p1_name_input")
    if p1_input and len(p1_input) >= 2 and p1_input != st.session_state.p1:
        players = search_player(p1_input)
        if players:
            for p in players[:3]:
                if st.button(f"📚 {p['name']}", key=f"sp1_{p['id']}", use_container_width=True):
                    st.session_state.p1 = p['name']
                    st.session_state.p1_id = p['id']
                    stats = get_player_serve_stats(p['id'], st.session_state.surface)
                    if stats:
                        st.session_state.p1_stats = stats
                        st.session_state.p1_serve = int(stats['serve_point_pct'] * 100)
                    st.rerun()
    if p1_input:
        st.session_state.p1 = p1_input
    
    # Player 2
    st.markdown("### Player 2")
    p2_input = st.text_input("Name", st.session_state.p2, key="p2_name_input")
    if p2_input and len(p2_input) >= 2 and p2_input != st.session_state.p2:
        players = search_player(p2_input)
        if players:
            for p in players[:3]:
                if st.button(f"📚 {p['name']}", key=f"sp2_{p['id']}", use_container_width=True):
                    st.session_state.p2 = p['name']
                    st.session_state.p2_id = p['id']
                    stats = get_player_serve_stats(p['id'], st.session_state.surface)
                    if stats:
                        st.session_state.p2_stats = stats
                        st.session_state.p2_serve = int(stats['serve_point_pct'] * 100)
                    st.rerun()
    if p2_input:
        st.session_state.p2 = p2_input
    
    st.markdown("---")
    
    # Match settings
    st.markdown("### ⚙️ Settings")
    st.session_state.surface = st.selectbox("Surface", ['Hard', 'Clay', 'Grass', 'Indoor'])
    st.session_state.best_of = st.selectbox("Format", [3, 5], index=0 if st.session_state.best_of == 3 else 1)
    
    server_options = [st.session_state.p1, st.session_state.p2]
    current_idx = st.session_state.server - 1
    selected = st.selectbox("Server", server_options, index=current_idx)
    st.session_state.server = 1 if selected == st.session_state.p1 else 2
    
    st.markdown("---")
    
    # Serve stats adjustments
    st.markdown("### 🎾 Serve Power")
    st.session_state.p1_serve = st.slider(f"{st.session_state.p1[:10]} serve %", 50, 75, st.session_state.p1_serve)
    st.session_state.p2_serve = st.slider(f"{st.session_state.p2[:10]} serve %", 50, 75, st.session_state.p2_serve)
    
    st.markdown("---")
    
    # Database stats
    db_stats = get_db_stats()
    if db_stats:
        st.markdown("### 📚 Database")
        st.metric("Matches", f"{db_stats['matches']:,}")
        st.metric("Players", f"{db_stats['players']:,}")
    
    st.markdown("---")
    
    # Session P&L
    profit_color = "green" if st.session_state.total_profit >= 0 else "red"
    st.markdown(f"""
    ### 💰 Session P&L
    <h2 style='color: {profit_color};'>${st.session_state.total_profit:+.2f}</h2>
    <p style='opacity: 0.7;'>Staked: ${st.session_state.total_staked:.2f}</p>
    """, unsafe_allow_html=True)
    
    # Reset button
    if st.button("🔄 Reset Match", use_container_width=True):
        for k in ['sets', 'games', 'points', 'point_history', 'games_history', 'bets']:
            if k in defaults:
                st.session_state[k] = defaults[k].copy() if isinstance(defaults[k], list) else defaults[k]
        st.session_state.total_staked = 0.0
        st.session_state.total_profit = 0.0
        st.rerun()


# ==================== MAIN PAGE ====================
    # Hero header
    st.markdown(f"""
    <div class="hero-header">
        <h1>🎾 {st.session_state.p1} vs {st.session_state.p2}</h1>
        <div class="hero-subtitle">{st.session_state.surface} Court • Best of {st.session_state.best_of}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # === PLAYER INSIGHTS FROM DATABASE ===
    if st.session_state.p1_id and st.session_state.p2_id:
        h2h = get_h2h_details(st.session_state.p1_id, st.session_state.p2_id)
        if h2h:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2d3436 0%, #000000 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: white; text-align: center; margin: 0;">⚔️ Head to Head</h3>
                <div style="display: flex; justify-content: center; align-items: center; gap: 30px; margin-top: 15px;">
                    <div style="text-align: center;">
                        <div style="color: #38ef7d; font-size: 2.5rem; font-weight: bold;">{h2h['p1_wins']}</div>
                        <div style="color: white; opacity: 0.8;">{st.session_state.p1[:15]}</div>
                    </div>
                    <div style="color: white; font-size: 1.5rem;">-</div>
                    <div style="text-align: center;">
                        <div style="color: #ff6b6b; font-size: 2.5rem; font-weight: bold;">{h2h['p2_wins']}</div>
                        <div style="color: white; opacity: 0.8;">{st.session_state.p2[:15]}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ==================== BOOKMAKER TERMINAL VIEW ====================
    
    # Market header
    st.markdown(f"""
    <div class="market-header">
        LIVE TENNIS MARKET | {st.session_state.surface.upper()} | BEST OF {st.session_state.best_of}
    </div>
    """, unsafe_allow_html=True)
    
    # Score panel - compact table format
    server_dot_1 = "●" if st.session_state.server == 1 else ""
    server_dot_2 = "●" if st.session_state.server == 2 else ""
    
    point_map = {0: "0", 1: "15", 2: "30", 3: "40", 4: "AD"}
    p1_pts = point_map.get(st.session_state.points[0], "")
    p2_pts = point_map.get(st.session_state.points[1], "")
    
    st.markdown(f"""
    <div class="score-panel">
        <table>
            <tr style="font-weight: bold; color: #00ff88;">
                <td width="25%">PLAYER</td>
                <td width="15%">SET 1</td>
                <td width="15%">SET 2</td>
                <td width="15%">SET 3</td>
                <td width="15%">GAME</td>
                <td width="15%">PTS</td>
            </tr>
            <tr>
                <td><span class="serve-dot">{server_dot_1}</span> {st.session_state.p1[:20]}</td>
                <td>{st.session_state.sets[0] if len(st.session_state.sets) > 0 else 0}</td>
                <td>{st.session_state.set_history[0][0] if len(st.session_state.set_history) > 0 else '-'}</td>
                <td>{st.session_state.set_history[1][0] if len(st.session_state.set_history) > 1 else '-'}</td>
                <td>{st.session_state.games[0]}</td>
                <td>{p1_pts}</td>
            </tr>
            <tr>
                <td><span class="serve-dot">{server_dot_2}</span> {st.session_state.p2[:20]}</td>
                <td>{st.session_state.sets[1] if len(st.session_state.sets) > 1 else 0}</td>
                <td>{st.session_state.set_history[0][1] if len(st.session_state.set_history) > 0 else '-'}</td>
                <td>{st.session_state.set_history[1][1] if len(st.session_state.set_history) > 1 else '-'}</td>
                <td>{st.session_state.games[1]}</td>
                <td>{p2_pts}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Get model probabilities
    ml_predictions = get_ml_predictions(
        st.session_state.p1_serve,
        st.session_state.p2_serve,
        st.session_state.surface,
        st.session_state.best_of
    )
    
    # Calculate game probabilities
    p1_serve = st.session_state.p1_serve / 100
    p2_serve = st.session_state.p2_serve / 100
    
    if st.session_state.server == 1:
        server_pts, returner_pts = st.session_state.points
        p_serve = p1_serve
        server_name = st.session_state.p1
        returner_name = st.session_state.p2
    else:
        server_pts, returner_pts = st.session_state.points[1], st.session_state.points[0]
        p_serve = p2_serve
        server_name = st.session_state.p2
        returner_name = st.session_state.p1
    
    p_hold_base = p_game_from_points(server_pts, returner_pts, p_serve)
    
    # Adjust with ensemble if available
    if ml_predictions and ml_predictions.get('ensemble'):
        ensemble_p1 = ml_predictions['ensemble']
        confidence = ml_predictions.get('confidence', 0.5)
        adjustment_factor = (ensemble_p1 - 0.5) * 0.1 * confidence
        
        if st.session_state.server == 1:
            p_hold = min(0.95, max(0.05, p_hold_base + adjustment_factor))
        else:
            p_hold = min(0.95, max(0.05, p_hold_base - adjustment_factor))
    else:
        p_hold = p_hold_base
    
    p_break = 1 - p_hold
    
    # Two-column layout: Odds | Probabilities
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # ODDS TABLE - BOOKMAKER STYLE
        st.markdown("### 📊 LIVE MARKETS")
        
        # Game market
        st.markdown(f"""
        <div class="odds-table">
            <div class="odds-row odds-header">
                <div>MARKET</div>
                <div>BACK</div>
                <div>TRUE %</div>
                <div>EDGE</div>
                <div>STAKE</div>
            </div>
            <div class="odds-row">
                <div>{server_name[:15]} HOLD</div>
                <div><input type="number" id="hold_odds" value="{st.session_state.game_hold_odds:.2f}" step="0.05" style="width: 70px; background: #2d3561; color: #7ec8e3; border: none; padding: 4px; border-radius: 3px; text-align: center;"></div>
                <div class="prob-value">{p_hold:.1%}</div>
                <div class="{'edge-positive' if p_hold > 1/st.session_state.game_hold_odds else 'edge-negative'}">{(p_hold - 1/st.session_state.game_hold_odds):+.1%}</div>
                <div>—</div>
            </div>
            <div class="odds-row">
                <div>{returner_name[:15]} BREAK</div>
                <div><input type="number" id="break_odds" value="{st.session_state.game_break_odds:.2f}" step="0.05" style="width: 70px; background: #2d3561; color: #e37e7e; border: none; padding: 4px; border-radius: 3px; text-align: center;"></div>
                <div class="prob-value">{p_break:.1%}</div>
                <div class="{'edge-positive' if p_break > 1/st.session_state.game_break_odds else 'edge-negative'}">{(p_break - 1/st.session_state.game_break_odds):+.1%}</div>
                <div>—</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Input actual odds
        odds_c1, odds_c2 = st.columns(2)
        with odds_c1:
            st.session_state.game_hold_odds = st.number_input(
                f"HOLD ODDS", 1.01, 20.0, float(st.session_state.game_hold_odds), 0.05, 
                key="hold_input", label_visibility="collapsed"
            )
        with odds_c2:
            st.session_state.game_break_odds = st.number_input(
                f"BREAK ODDS", 1.01, 20.0, float(st.session_state.game_break_odds), 0.05,
                key="break_input", label_visibility="collapsed"
            )
        
        # Point buttons
        st.markdown("### ⚡ SCORE POINT")
        btn_cols = st.columns([2, 2, 1, 1])
        
        with btn_cols[0]:
            if st.button(f"✓ {st.session_state.p1[:10]}", use_container_width=True):
                record_point(1)
                st.rerun()
        
        with btn_cols[1]:
            if st.button(f"✓ {st.session_state.p2[:10]}", use_container_width=True):
                record_point(2)
                st.rerun()
        
        with btn_cols[2]:
            if st.button("↩ UNDO", use_container_width=True):
                if st.session_state.point_history:
                    last = st.session_state.point_history.pop()
                    if st.session_state.points[last - 1] > 0:
                        st.session_state.points[last - 1] -= 1
                st.rerun()
        
        with btn_cols[3]:
            if st.button("⇄ SWAP", use_container_width=True):
                st.session_state.server = 3 - st.session_state.server
                st.rerun()
    
    with col_right:
        # PROBABILITY PANEL
        st.markdown("### 🎯 TRUE PROBABILITIES")
        
        fair_hold = 1/st.session_state.game_hold_odds
        fair_break = 1/st.session_state.game_break_odds
        total_implied = fair_hold + fair_break
        
        edge_hold = p_hold - fair_hold
        edge_break = p_break - fair_break
        
        st.markdown(f"""
        <div class="prob-panel">
            <div class="prob-row">
                <span class="prob-label">Model Ensemble:</span>
                <span class="prob-value">{ml_predictions.get('ensemble', 0.5):.1%}</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Hold Prob:</span>
                <span class="prob-value">{p_hold:.1%}</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Break Prob:</span>
                <span class="prob-value">{p_break:.1%}</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Book %:</span>
                <span class="prob-value">{total_implied:.1%}</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Hold Edge:</span>
                <span class="{'edge-positive' if edge_hold > 0 else 'edge-negative'}">{edge_hold:+.1%}</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Break Edge:</span>
                <span class="{'edge-positive' if edge_break > 0 else 'edge-negative'}">{edge_break:+.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # P&L tracker
        total_profit = sum(b['profit'] for b in st.session_state.bets if b['profit'] is not None)
        pl_class = "pl-positive" if total_profit >= 0 else "pl-negative"
        
        st.markdown(f"""
        <div class="pl-panel">
            <div style="display: flex; justify-content: space-between;">
                <span>SESSION P&L:</span>
                <span class="{pl_class}">${total_profit:+.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                <span>BETS:</span>
                <span>{len(st.session_state.bets)}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                <span>BANKROLL:</span>
                <span>${st.session_state.bankroll:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==================== FOOTER ====================
st.markdown("---")
st.caption(f"TENNIS TRADING TERMINAL | Database: {'✓' if os.path.exists(DB_PATH) else '✗'} | Models: 4 Active")

            if st.session_state.p1_id:
                # Recent form
                form = get_recent_form(st.session_state.p1_id, 5)
                if form:
                    form_str = ' '.join(['🟢' if m['result'] == 'W' else '🔴' for m in form])
                    wins = sum(1 for m in form if m['result'] == 'W')
                    st.markdown(f"**Recent Form:** {form_str} ({wins}/5)")
                    for m in form[:3]:
                        st.caption(f"{m['result']} vs {m['opponent'][:15]} ({m['surface']})")
                
                # Surface win rates
                surface_stats = get_surface_win_rates(st.session_state.p1_id)
                if surface_stats:
                    st.markdown("**Surface Win Rates:**")
                    for surf, data in surface_stats.items():
                        emoji = '🏟️' if surf == 'Hard' else ('🧱' if surf == 'Clay' else '🌿')
                        st.markdown(f"{emoji} {surf}: **{data['pct']:.0%}** ({data['wins']}/{data['total']})")
                
                # Hold/Break stats
                hb_stats = get_hold_break_stats(st.session_state.p1_id, st.session_state.surface)
                if hb_stats:
                    st.markdown(f"**On {st.session_state.surface}:**")
                    if hb_stats['hold_pct']:
                        st.markdown(f"Hold: **{hb_stats['hold_pct']:.0%}** | Break: **{hb_stats['break_pct']:.0%}**")
                    if hb_stats['bp_saved_pct']:
                        st.markdown(f"BP Saved: {hb_stats['bp_saved_pct']:.0%} | BP Conv: {hb_stats['bp_converted_pct']:.0%}")
            else:
                st.caption("Player not in database - using manual stats")
        
        with pi_col2:
            st.markdown(f"#### {st.session_state.p2}")
            if st.session_state.p2_id:
                # Recent form
                form = get_recent_form(st.session_state.p2_id, 5)
                if form:
                    form_str = ' '.join(['🟢' if m['result'] == 'W' else '🔴' for m in form])
                    wins = sum(1 for m in form if m['result'] == 'W')
                    st.markdown(f"**Recent Form:** {form_str} ({wins}/5)")
                    for m in form[:3]:
                        st.caption(f"{m['result']} vs {m['opponent'][:15]} ({m['surface']})")
                
                # Surface win rates
                surface_stats = get_surface_win_rates(st.session_state.p2_id)
                if surface_stats:
                    st.markdown("**Surface Win Rates:**")
                    for surf, data in surface_stats.items():
                        emoji = '🏟️' if surf == 'Hard' else ('🧱' if surf == 'Clay' else '🌿')
                        st.markdown(f"{emoji} {surf}: **{data['pct']:.0%}** ({data['wins']}/{data['total']})")
                
                # Hold/Break stats
                hb_stats = get_hold_break_stats(st.session_state.p2_id, st.session_state.surface)
                if hb_stats:
                    st.markdown(f"**On {st.session_state.surface}:**")
                    if hb_stats['hold_pct']:
                        st.markdown(f"Hold: **{hb_stats['hold_pct']:.0%}** | Break: **{hb_stats['break_pct']:.0%}**")
                    if hb_stats['bp_saved_pct']:
                        st.markdown(f"BP Saved: {hb_stats['bp_saved_pct']:.0%} | BP Conv: {hb_stats['bp_converted_pct']:.0%}")
            else:
                st.caption("Player not in database - using manual stats")
        
        # H2H Details
        if st.session_state.p1_id and st.session_state.p2_id:
            h2h = get_h2h_details(st.session_state.p1_id, st.session_state.p2_id)
            if h2h and h2h['matches']:
                st.markdown("---")
                st.markdown("#### ⚔️ Recent H2H Matches")
                for m in h2h['matches']:
                    st.markdown(f"**{m['winner']}** won | {m['tournament']} ({m['surface']}) | {m['score']}")
    
    # Live score
    server_name = st.session_state.p1 if st.session_state.server == 1 else st.session_state.p2
    st.markdown(f"""
    <div class="live-score">
        {get_score_string()}
        <div class="server-indicator">🎾 {server_name} serving</div>
    </div>
    """, unsafe_allow_html=True)
    
    # === AI INSIGHTS (BEFORE STATS) ===
    ml_predictions = get_ml_predictions(
        st.session_state.p1_serve,
        st.session_state.p2_serve,
        st.session_state.surface,
        st.session_state.best_of
    )
    
    ai_insights_html = get_ai_insights_html(
        ml_predictions,
        st.session_state.p1,
        st.session_state.p2,
        get_score_string()
    )
    
    if ai_insights_html:
        st.markdown(ai_insights_html, unsafe_allow_html=True)
    
    # === FEATURE ENGINEERING & DATA INTEGRATION ===
    if ml_predictions and 'features' in ml_predictions and ml_predictions['features']:
        with st.expander("📊 Advanced Analytics & Feature Engineering", expanded=False):
            features = ml_predictions['features']
            
            # Model weights visualization
            if 'model_weights' in features:
                st.markdown("#### 🎯 Model Contribution to True Probability")
                weights = features['model_weights']
                
                wc1, wc2, wc3, wc4 = st.columns(4)
                with wc1:
                    st.metric("Markov Chain", f"{weights.get('hierarchical', 0):.0%}")
                with wc2:
                    st.metric("TennisRatio", f"{weights.get('tennisratio', 0):.0%}")
                with wc3:
                    st.metric("Logistic Reg", f"{weights.get('logistic', 0):.0%}")
                with wc4:
                    st.metric("Neural Net", f"{weights.get('neural', 0):.0%}")
            
            # Markov chain probabilities
            if 'p1_hold_prob' in features:
                st.markdown("#### ⚡ Point-to-Game Transition Probabilities")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric(f"{st.session_state.p1} Hold", f"{features['p1_hold_prob']:.1%}")
                with mc2:
                    st.metric(f"{st.session_state.p2} Hold", f"{features['p2_hold_prob']:.1%}")
                with mc3:
                    st.metric(f"{st.session_state.p1} Set Win", f"{features.get('p1_set_prob', 0):.1%}")
            
            # TennisRatio features
            if 'tr_h2h' in features:
                h2h = features['tr_h2h']
                st.markdown("#### 🌐 TennisRatio Live Data")
                tr1, tr2, tr3, tr4 = st.columns(4)
                
                with tr1:
                    dom = features.get('tr_dominance', 0)
                    st.metric("Dominance", f"{dom:.1%}" if dom > 0 else "N/A")
                with tr2:
                    eff = features.get('tr_efficiency', 0)
                    st.metric("Efficiency", f"{eff:.1%}" if eff > 0 else "N/A")
                with tr3:
                    h2h_wins = h2h.get('p1_wins', 0) if isinstance(h2h, dict) else 0
                    h2h_total = h2h.get('total_matches', 0) if isinstance(h2h, dict) else 0
                    st.metric("H2H Wins", f"{h2h_wins}/{h2h_total}" if h2h_total > 0 else "No data")
                with tr4:
                    st.metric("Source", "Live Web")
            
            st.markdown("---")
            st.caption("💡 True Probability (P) is calculated using weighted ensemble of all available models based on data quality and confidence")
    
    # Break point alert
    if is_break_point():
        bp_count = get_break_point_count()
        returner = st.session_state.p2 if st.session_state.server == 1 else st.session_state.p1
        st.markdown(f"""
        <div class="break-alert">
            🔴 BREAK POINT{'S' if bp_count > 1 else ''} ×{bp_count} — {returner}!
        </div>
        """, unsafe_allow_html=True)
    
    # Quick point buttons
    st.markdown("### ⚡ Score a Point")
    btn_cols = st.columns([2, 2, 1, 1])
    
    with btn_cols[0]:
        if st.button(f"✅ {st.session_state.p1}", type="primary", use_container_width=True):
            record_point(1)
            st.rerun()
    
    with btn_cols[1]:
        if st.button(f"✅ {st.session_state.p2}", type="primary", use_container_width=True):
            record_point(2)
            st.rerun()
    
    with btn_cols[2]:
        if st.button("↩️ Undo", use_container_width=True):
            if st.session_state.point_history:
                last = st.session_state.point_history.pop()
                if st.session_state.points[last - 1] > 0:
                    st.session_state.points[last - 1] -= 1
            st.rerun()
    
    with btn_cols[3]:
        if st.button("🔄 Swap", use_container_width=True):
            st.session_state.server = 3 - st.session_state.server
            st.rerun()
    
    st.markdown("---")
    
    # === ODDS INPUT & VALUE ANALYSIS ===
    st.markdown("### 💰 Live Game Odds")
    
    server_name = st.session_state.p1 if st.session_state.server == 1 else st.session_state.p2
    returner_name = st.session_state.p2 if st.session_state.server == 1 else st.session_state.p1
    
    oc1, oc2 = st.columns(2)
    with oc1:
        st.session_state.game_hold_odds = st.number_input(
            f"🟢 {server_name} HOLD", 1.01, 20.0,
            float(st.session_state.game_hold_odds), 0.05, key="hold_odds"
        )
    with oc2:
        st.session_state.game_break_odds = st.number_input(
            f"🔴 {returner_name} BREAK", 1.01, 20.0,
            float(st.session_state.game_break_odds), 0.05, key="break_odds"
        )
    
    # Calculate probabilities using TRUE P from ensemble when available
    p1_serve = st.session_state.p1_serve / 100
    p2_serve = st.session_state.p2_serve / 100
    
    if st.session_state.server == 1:
        server_pts, returner_pts = st.session_state.points
        p_serve = p1_serve
    else:
        server_pts, returner_pts = st.session_state.points[1], st.session_state.points[0]
        p_serve = p2_serve
    
    # Base probability from Markov point-level model
    p_hold_base = p_game_from_points(server_pts, returner_pts, p_serve)
    p_break_base = 1 - p_hold_base
    
    # Adjust with ensemble confidence if available
    if ml_predictions and ml_predictions.get('ensemble'):
        ensemble_p1 = ml_predictions['ensemble']
        confidence = ml_predictions.get('confidence', 0.5)
        
        # Adjust game probability based on match-level ensemble
        # Higher ensemble for P1 suggests stronger overall performance
        adjustment_factor = (ensemble_p1 - 0.5) * 0.1 * confidence  # ±5% max adjustment
        
        if st.session_state.server == 1:
            p_hold = min(0.95, max(0.05, p_hold_base + adjustment_factor))
        else:
            p_hold = min(0.95, max(0.05, p_hold_base - adjustment_factor))
        
        p_break = 1 - p_hold
        
        st.caption(f"✨ Using True P (ensemble-adjusted): {abs(adjustment_factor):.1%} confidence boost")
    else:
        # Fallback to base Markov calculation
        p_hold = p_hold_base
        p_break = p_break_base
        st.caption("📊 Using base Markov probability (collecting more data...)")
    
    # Fair odds & edge
    hold_implied = 1 / st.session_state.game_hold_odds
    break_implied = 1 / st.session_state.game_break_odds
    total_implied = hold_implied + break_implied
    hold_fair = hold_implied / total_implied
    break_fair = break_implied / total_implied
    
    edge_hold = p_hold - hold_fair
    edge_break = p_break - break_fair
    
    kelly_hold = max(0, (p_hold * st.session_state.game_hold_odds - 1) / (st.session_state.game_hold_odds - 1)) if edge_hold > 0 else 0
    kelly_break = max(0, (p_break * st.session_state.game_break_odds - 1) / (st.session_state.game_break_odds - 1)) if edge_break > 0 else 0
    
    # Value cards
    st.markdown("### 🎯 Value Analysis")
    val_cols = st.columns(2)
    
    with val_cols[0]:
        fair_hold = 1/p_hold if p_hold > 0.01 else 99
        if edge_hold > 0.03:
            stake = kelly_hold * 0.25 * st.session_state.bankroll
            st.markdown(f"""
            <div class="value-card">
                <div class="label">🟢 {server_name} HOLD</div>
                <div class="value">+{edge_hold:.1%} EDGE</div>
                <div class="subtext">Model: {p_hold:.1%} | Fair: {fair_hold:.2f} | Book: {st.session_state.game_hold_odds}</div>
                <div class="subtext" style="margin-top: 10px;">💵 Stake: ${stake:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📝 Record Hold Bet", key="rec_hold", use_container_width=True):
                record_bet("Game", f"{server_name} Hold", st.session_state.game_hold_odds, stake, p_hold)
                st.success(f"✅ Bet recorded: ${stake:.2f} @ {st.session_state.game_hold_odds}")
        elif edge_hold > 0:
            st.markdown(f"""
            <div class="neutral-card">
                <div class="label">🟢 {server_name} HOLD</div>
                <div class="value">+{edge_hold:.1%}</div>
                <div class="subtext">Marginal edge - small value</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="no-value-card">
                <div class="label">🟢 {server_name} HOLD</div>
                <div class="value">{edge_hold:+.1%}</div>
                <div class="subtext">No value</div>
            </div>
            """, unsafe_allow_html=True)
    
    with val_cols[1]:
        fair_break = 1/p_break if p_break > 0.01 else 99
        if edge_break > 0.03:
            stake = kelly_break * 0.25 * st.session_state.bankroll
            st.markdown(f"""
            <div class="value-card">
                <div class="label">🔴 {returner_name} BREAK</div>
                <div class="value">+{edge_break:.1%} EDGE</div>
                <div class="subtext">Model: {p_break:.1%} | Fair: {fair_break:.2f} | Book: {st.session_state.game_break_odds}</div>
                <div class="subtext" style="margin-top: 10px;">💵 Stake: ${stake:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📝 Record Break Bet", key="rec_break", use_container_width=True):
                record_bet("Game", f"{returner_name} Break", st.session_state.game_break_odds, stake, p_break)
                st.success(f"✅ Bet recorded: ${stake:.2f} @ {st.session_state.game_break_odds}")
        elif edge_break > 0:
            st.markdown(f"""
            <div class="neutral-card">
                <div class="label">🔴 {returner_name} BREAK</div>
                <div class="value">+{edge_break:.1%}</div>
                <div class="subtext">Marginal edge - small value</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="no-value-card">
                <div class="label">🔴 {returner_name} BREAK</div>
                <div class="value">{edge_break:+.1%}</div>
                <div class="subtext">No value</div>
            </div>
            """, unsafe_allow_html=True)


# ==================== FOOTER ====================
st.markdown("---")
db_status = "✅ Connected" if os.path.exists(DB_PATH) else "❌ Not found"
st.caption(f"🎾 Tennis Betting Hub Pro | Database: {db_status} | 4 ML Models Integrated | Built with Streamlit")
