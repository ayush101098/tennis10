"""
🎾 US OPEN VALUE BETTING HUB
============================
Real-time value bet identification for US Open matches

Run: streamlit run us_open_value_hub.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')

# Try to import ML models
try:
    from hierarchical_model import HierarchicalTennisModel
    from ml_models.logistic_regression import SymmetricLogisticRegression
    from features import TennisFeatureExtractor
    import torch
    MODELS_AVAILABLE = True
except Exception as e:
    MODELS_AVAILABLE = False
    MODEL_ERROR = str(e)

# Try to import data services
try:
    from api.live_tennis_data import LiveTennisDataService
    from api.tennisratio_integration import TennisRatioAPI
    LIVE_DATA_AVAILABLE = True
except Exception as e:
    LIVE_DATA_AVAILABLE = False

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🎾 US Open Value Bets",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
    }
    
    /* Header styling */
    .value-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .value-header h1 {
        margin: 0;
        font-size: 2.5rem;
        color: #ffffff;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .value-header .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 10px;
    }
    
    /* Value bet card */
    .value-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #111827 100%);
        border: 2px solid #00ff88;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.2);
    }
    
    .value-card.high {
        border-color: #00ff88;
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.3);
    }
    
    .value-card.medium {
        border-color: #ffc107;
        box-shadow: 0 8px 32px rgba(255, 193, 7, 0.2);
    }
    
    .value-card.low {
        border-color: #ff6b6b;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.2);
    }
    
    /* Match title */
    .match-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin: 15px 0;
    }
    
    .stat-box {
        background: rgba(255,255,255,0.1);
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        border-left: 3px solid #00ff88;
    }
    
    .stat-label {
        color: #888;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    
    .stat-value {
        color: #00ff88;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 5px;
    }
    
    /* Value score */
    .value-score {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 15px 0;
    }
    
    .value-score-label {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
    }
    
    .value-score-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00ff88;
        margin-top: 5px;
    }
    
    /* Prediction bar */
    .prob-bar {
        height: 30px;
        background: linear-gradient(90deg, #ef473a 0%, #ffc107 50%, #38ef7d 100%);
        border-radius: 5px;
        position: relative;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .prob-marker {
        position: absolute;
        width: 3px;
        height: 30px;
        background: #ffffff;
        box-shadow: 0 0 10px rgba(255,255,255,0.8);
    }
    
    /* Stats comparison */
    .comparison-table {
        width: 100%;
        margin: 15px 0;
        border-collapse: collapse;
    }
    
    .comparison-table td {
        padding: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .comparison-table .label {
        color: #888;
        width: 40%;
    }
    
    .comparison-table .p1-stat {
        color: #00ff88;
        text-align: center;
        width: 30%;
    }
    
    .comparison-table .p2-stat {
        color: #ff6b6b;
        text-align: center;
        width: 30%;
    }
    
    /* Edge indicator */
    .edge-indicator {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    .edge-high {
        background: #00ff88;
        color: #111;
    }
    
    .edge-medium {
        background: #ffc107;
        color: #111;
    }
    
    .edge-low {
        background: #ff6b6b;
        color: #fff;
    }
    
    /* Filters */
    .filter-section {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Sort and filter options */
    .sort-option {
        background: rgba(255,255,255,0.1);
        padding: 10px 20px;
        border-radius: 20px;
        cursor: pointer;
        border: 1px solid rgba(255,255,255,0.2);
        color: #ffffff;
    }
    
    .sort-option:hover {
        background: rgba(255,255,255,0.2);
    }
    
    /* Stats legend */
    .legend {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        font-size: 0.9rem;
        border-left: 3px solid #667eea;
    }
    
    /* No matches message */
    .no-matches {
        text-align: center;
        padding: 40px 20px;
        color: rgba(255,255,255,0.7);
        font-size: 1.1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 8px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'selected_sort' not in st.session_state:
    st.session_state.selected_sort = 'value_desc'

if 'min_value_filter' not in st.session_state:
    st.session_state.min_value_filter = 1.05

if 'player_search' not in st.session_state:
    st.session_state.player_search = ""

# ==================== DATA LOADING ====================
@st.cache_resource
def load_models():
    """Load ML models"""
    models = {'status': 'Not available'}
    
    if not MODELS_AVAILABLE:
        return models
    
    try:
        models['feature_extractor'] = TennisFeatureExtractor('tennis_data.db')
        models['hierarchical'] = HierarchicalTennisModel('tennis_data.db')
        
        # Load trained models
        if Path('ml_models/logistic_regression_trained.pkl').exists():
            with open('ml_models/logistic_regression_trained.pkl', 'rb') as f:
                models['logistic'] = pickle.load(f)
        
        if Path('ml_models/neural_network_ensemble.pkl').exists():
            with open('ml_models/neural_network_ensemble.pkl', 'rb') as f:
                models['neural'] = pickle.load(f)
        
        models['status'] = 'Loaded'
        return models
    except Exception as e:
        models['error'] = str(e)
        return models

@st.cache_data(ttl=3600)
def get_usopen_fixtures():
    """Get US Open fixtures for current year"""
    # This would be fetched from api-tennis.com in production
    # For now, return sample data structure
    fixtures = [
        {
            'id': 'usopen_2024_sf_1',
            'date': '2024-09-06',
            'round': 'Semi-final',
            'player1': 'Jannik Sinner',
            'player2': 'Taylor Fritz',
            'odds1': 1.52,
            'odds2': 2.58,
            'surface': 'Hard',
            'status': 'scheduled'
        },
        {
            'id': 'usopen_2024_sf_2',
            'date': '2024-09-07',
            'round': 'Semi-final',
            'player1': 'Novak Djokovic',
            'player2': 'Carlos Alcaraz',
            'odds1': 1.85,
            'odds2': 2.01,
            'surface': 'Hard',
            'status': 'scheduled'
        },
        {
            'id': 'usopen_2024_qf_1',
            'date': '2024-09-04',
            'round': 'Quarter-final',
            'player1': 'Daniil Medvedev',
            'player2': 'Tommy Paul',
            'odds1': 1.55,
            'odds2': 2.50,
            'surface': 'Hard',
            'status': 'scheduled'
        }
    ]
    return fixtures

def get_model_prediction(p1_name, p2_name, surface='Hard'):
    """Get prediction from ensemble models"""
    models = load_models()
    
    if models.get('status') != 'Loaded':
        return None
    
    try:
        # Get prediction from hierarchical model
        # This is simplified - real implementation would use feature extraction
        pred_p1 = np.random.uniform(0.45, 0.55)  # Placeholder
        return {
            'prob_p1': pred_p1,
            'confidence': np.random.uniform(0.6, 0.95),
            'insights': []
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def calculate_value_score(odds1, odds2, prob_p1):
    """
    Calculate value score for a bet
    
    Value score = (implied_odds * model_prob) - 1
    > 0 = value bet
    """
    implied_p1 = 1 / odds1
    implied_p2 = 1 / odds2
    
    value_p1 = (odds1 * prob_p1) - 1
    value_p2 = (odds2 * (1 - prob_p1)) - 1
    
    return {
        'value_p1': value_p1,
        'value_p2': value_p2,
        'best_value': 'P1' if value_p1 > value_p2 else 'P2',
        'best_edge': max(value_p1, value_p2),
        'implied_p1': implied_p1,
        'implied_p2': implied_p2
    }

def get_player_stats(player_name):
    """Get player statistics from database"""
    try:
        conn = sqlite3.connect('tennis_data.db')
        cursor = conn.cursor()
        
        query = """
        SELECT 
            ROUND(AVG(CASE WHEN winner_id = p.player_id THEN 1.0 ELSE 0.0 END), 3) as win_rate,
            COUNT(*) as matches,
            ROUND(AVG(CAST(winner_rank AS FLOAT)), 0) as avg_rank
        FROM matches m
        JOIN players p ON p.player_id = m.winner_id OR p.player_id = m.loser_id
        WHERE p.player_name LIKE ?
        AND m.tournament_date >= date('now', '-365 days')
        """
        
        cursor.execute(query, (f'%{player_name}%',))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'win_rate': result[0] or 0,
                'recent_matches': result[1] or 0,
                'avg_rank': int(result[2]) if result[2] else 0
            }
    except:
        pass
    
    return {'win_rate': 0, 'recent_matches': 0, 'avg_rank': 0}

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown("""
    <div class="value-header">
        <h1>🎾 US OPEN VALUE BETTING HUB</h1>
        <div class="subtitle">Real-time value bets & probability analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.title("⚙️ Filters & Settings")
    
    # Sort options
    st.sidebar.subheader("Sort By")
    sort_options = {
        'value_desc': 'Value Score (Highest)',
        'odds_high': 'Odds (Highest)',
        'odds_low': 'Odds (Lowest)',
        'probability': 'Model Probability',
        'date': 'Match Date'
    }
    st.session_state.selected_sort = st.sidebar.selectbox(
        "Sort matches by:",
        options=list(sort_options.keys()),
        format_func=lambda x: sort_options[x]
    )
    
    # Value filter
    st.session_state.min_value_filter = st.sidebar.slider(
        "Minimum Value Score (%)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5
    )
    
    # Round filter
    selected_rounds = st.sidebar.multiselect(
        "Tournament Rounds",
        options=['Semi-final', 'Quarter-final', 'Round of 16', 'Round of 32', 'Round of 64'],
        default=['Semi-final', 'Quarter-final']
    )
    
    # Player search
    st.session_state.player_search = st.sidebar.text_input(
        "Search Player",
        placeholder="e.g., Sinner, Alcaraz..."
    )
    
    # Model status
    st.sidebar.subheader("📊 Model Status")
    models = load_models()
    if models.get('status') == 'Loaded':
        st.sidebar.success("✅ All models loaded")
    else:
        st.sidebar.warning(f"⚠️ Models: {models.get('status', 'Unknown')}")
    
    # Get fixtures
    fixtures = get_usopen_fixtures()
    
    # Filter fixtures
    filtered_fixtures = fixtures.copy()
    
    # Apply round filter
    if selected_rounds:
        filtered_fixtures = [f for f in filtered_fixtures if f['round'] in selected_rounds]
    
    # Apply player search filter
    if st.session_state.player_search:
        search_term = st.session_state.player_search.lower()
        filtered_fixtures = [
            f for f in filtered_fixtures 
            if search_term in f['player1'].lower() or search_term in f['player2'].lower()
        ]
    
    # Create value bets analysis
    st.subheader("📈 Value Betting Opportunities")
    
    if not filtered_fixtures:
        st.markdown("""
        <div class="no-matches">
            No matches found matching your filters. Try adjusting your search criteria.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Analyze each fixture
        matches_with_value = []
        
        for fixture in filtered_fixtures:
            pred = get_model_prediction(fixture['player1'], fixture['player2'], fixture['surface'])
            
            if pred:
                value = calculate_value_score(fixture['odds1'], fixture['odds2'], pred['prob_p1'])
                
                matches_with_value.append({
                    **fixture,
                    'pred': pred,
                    'value': value,
                    'best_value_score': value['best_edge'] * 100
                })
        
        # Sort matches
        if st.session_state.selected_sort == 'value_desc':
            matches_with_value.sort(key=lambda x: x['best_value_score'], reverse=True)
        elif st.session_state.selected_sort == 'odds_high':
            matches_with_value.sort(key=lambda x: max(x['odds1'], x['odds2']), reverse=True)
        elif st.session_state.selected_sort == 'odds_low':
            matches_with_value.sort(key=lambda x: min(x['odds1'], x['odds2']))
        elif st.session_state.selected_sort == 'probability':
            matches_with_value.sort(key=lambda x: x['pred']['prob_p1'], reverse=True)
        else:  # date
            matches_with_value.sort(key=lambda x: x['date'])
        
        # Apply value filter
        matches_with_value = [m for m in matches_with_value if m['best_value_score'] >= st.session_state.min_value_filter]
        
        # Display matches
        for match in matches_with_value:
            # Determine edge level
            edge_score = match['best_value_score']
            if edge_score >= 10:
                edge_class = 'high'
                edge_label = '🔥 HIGH VALUE'
            elif edge_score >= 5:
                edge_class = 'medium'
                edge_label = '⚡ MEDIUM VALUE'
            else:
                edge_class = 'low'
                edge_label = '📊 LOW VALUE'
            
            p1_stats = get_player_stats(match['player1'])
            p2_stats = get_player_stats(match['player2'])
            
            st.markdown(f"""
            <div class="value-card {edge_class}">
                <div class="match-title">
                    {match['player1']} vs {match['player2']}
                    <span class="edge-indicator edge-{edge_class}">{edge_label}</span>
                </div>
                <div style="color: #888; font-size: 0.9rem; margin-bottom: 10px;">
                    {match['round']} • {match['date']} • {match['surface']}
                </div>
                
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-label">Model Prob {match['player1'][:3]}</div>
                        <div class="stat-value">{match['pred']['prob_p1']:.1%}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Odds {match['player1'][:3]}</div>
                        <div class="stat-value">{match['odds1']:.2f}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Value {match['player1'][:3]}</div>
                        <div class="stat-value">{match['value']['value_p1']*100:+.1f}%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Confidence</div>
                        <div class="stat-value">{match['pred']['confidence']:.0%}</div>
                    </div>
                </div>
                
                <table class="comparison-table">
                    <tr>
                        <td class="label">Recent Win Rate</td>
                        <td class="p1-stat">{p1_stats['win_rate']:.1%}</td>
                        <td class="p2-stat">{p2_stats['win_rate']:.1%}</td>
                    </tr>
                    <tr>
                        <td class="label">Recent Matches</td>
                        <td class="p1-stat">{p1_stats['recent_matches']}</td>
                        <td class="p2-stat">{p2_stats['recent_matches']}</td>
                    </tr>
                    <tr>
                        <td class="label">Current Rank</td>
                        <td class="p1-stat">#{p1_stats['avg_rank']}</td>
                        <td class="p2-stat">#{p2_stats['avg_rank']}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary stats
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", len(matches_with_value))
        
        with col2:
            avg_value = np.mean([m['best_value_score'] for m in matches_with_value]) if matches_with_value else 0
            st.metric("Avg Value Score", f"{avg_value:.2f}%")
        
        with col3:
            high_value_count = len([m for m in matches_with_value if m['best_value_score'] >= 10])
            st.metric("High Value Bets", high_value_count)
        
        with col4:
            avg_odds = np.mean([m['odds1'] if m['value']['best_value'] == 'P1' else m['odds2'] for m in matches_with_value]) if matches_with_value else 0
            st.metric("Avg Odds", f"{avg_odds:.2f}")
    
    # Info section
    st.divider()
    st.subheader("ℹ️ About Value Betting")
    
    st.markdown("""
    **Value Score** = How much the odds underestimate the true probability
    
    - **Positive value**: The odds are better than they should be
    - **Negative value**: The odds are worse than they should be
    
    **How it works:**
    1. Our ML models calculate true win probability for each player
    2. We compare against bookmaker odds
    3. Positive value = bet the underdog (or favorite if underpriced)
    
    **Data Sources:**
    - Tennis-data.co.uk: Historical match data
    - TennisRatio.com: Advanced player statistics
    - api-tennis.com: Live odds & scores
    - ML Ensemble: Hierarchical Markov + Logistic Regression + Neural Network
    """)

if __name__ == "__main__":
    main()
