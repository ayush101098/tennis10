"""
Tennis Match Prediction API Service

Simple Flask API for match predictions and bet recommendations.
"""

from flask import Flask, request, jsonify
import sqlite3
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Optional

from features import MatchFeatures
from hierarchical_model import HierarchicalTennisModel


app = Flask(__name__)

# Global model storage
MODELS = {}
feature_gen = None
markov_model = None


def load_models():
    """Load all trained models on startup."""
    global MODELS, feature_gen, markov_model
    
    print("Loading models...")
    
    # Initialize feature generator
    feature_gen = MatchFeatures('tennis_data.db')
    
    # Load Markov model
    markov_model = HierarchicalTennisModel('tennis_data.db')
    print("✅ Markov model loaded")
    
    # Load Logistic Regression
    try:
        with open('ml_models/logistic_model.pkl', 'rb') as f:
            MODELS['logistic'] = pickle.load(f)
        print("✅ Logistic Regression loaded")
    except FileNotFoundError:
        print("⚠️  Logistic model not found")
    
    # Load Neural Network Ensemble
    try:
        with open('ml_models/nn_ensemble.pkl', 'rb') as f:
            MODELS['neural_net'] = pickle.load(f)
        print("✅ Neural Network Ensemble loaded")
    except FileNotFoundError:
        print("⚠️  Neural Network not found")
    
    print("Models loaded successfully!")


def get_player_id(player_name: str) -> Optional[int]:
    """Look up player ID from name."""
    conn = sqlite3.connect('tennis_data.db')
    cursor = conn.cursor()
    
    query = "SELECT player_id FROM players WHERE player_id LIKE ?"
    cursor.execute(query, (f"%{player_name}%",))
    result = cursor.fetchone()
    
    conn.close()
    
    return result[0] if result else None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(MODELS),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict_match', methods=['POST'])
def predict_match():
    """
    Predict match outcome and recommend bet.
    
    Body:
    {
        "player1_id": "roger-federer",
        "player2_id": "rafael-nadal",
        "surface": "clay",
        "date": "2024-06-05"
    }
    """
    data = request.json
    
    # Validate input
    required_fields = ['player1_id', 'player2_id', 'surface']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Get player IDs
    player1_id = data['player1_id']
    player2_id = data['player2_id']
    surface = data['surface']
    match_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    # Ensure player1_id < player2_id
    if player1_id > player2_id:
        player1_id, player2_id = player2_id, player1_id
        swap_players = True
    else:
        swap_players = False
    
    # Generate features
    features = feature_gen.generate_features(
        player1_id,
        player2_id,
        surface,
        match_date=match_date
    )
    
    predictions = {}
    
    # Markov prediction
    if markov_model:
        result = markov_model.predict_match(
            player1_id, player2_id, surface, best_of=3, match_date=match_date
        )
        predictions['markov'] = result['p_player1_win']
    
    # Logistic prediction
    if 'logistic' in MODELS:
        from ml_models.logistic_regression import prepare_features
        import pandas as pd
        
        df = pd.DataFrame([features])
        df['winner'] = 1
        
        lr_model = MODELS['logistic']['model']
        predictions['logistic'] = lr_model.predict_proba(df)[0]
    
    # Neural Network prediction
    if 'neural_net' in MODELS:
        from ml_models.neural_network import predict_ensemble
        import pandas as pd
        
        df = pd.DataFrame([features])
        nn_models = MODELS['neural_net']['models']
        nn_features = MODELS['neural_net']['features']
        
        predictions['neural_net'] = predict_ensemble(nn_models, df, nn_features)[0]
    
    # Ensemble prediction (average)
    if predictions:
        p_player1 = np.mean(list(predictions.values()))
        p_player2 = 1 - p_player1
    else:
        return jsonify({'error': 'No models available'}), 500
    
    # Swap back if needed
    if swap_players:
        p_player1, p_player2 = p_player2, p_player1
    
    # Calculate uncertainty (std of model predictions)
    uncertainty = np.std(list(predictions.values())) if len(predictions) > 1 else 0.5
    
    # Determine confidence
    if uncertainty < 0.1:
        confidence = "high"
    elif uncertainty < 0.2:
        confidence = "medium"
    else:
        confidence = "low"
    
    # Build response
    response = {
        'player1_id': data['player1_id'],
        'player2_id': data['player2_id'],
        'surface': surface,
        'date': match_date,
        'predictions': {
            'p_player1_win': round(p_player1, 3),
            'p_player2_win': round(p_player2, 3)
        },
        'model_breakdown': {k: round(v, 3) for k, v in predictions.items()},
        'model_confidence': confidence,
        'uncertainty_score': round(uncertainty, 3),
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(response)


@app.route('/models', methods=['GET'])
def list_models():
    """List available models."""
    return jsonify({
        'markov': markov_model is not None,
        'logistic': 'logistic' in MODELS,
        'neural_net': 'neural_net' in MODELS,
        'total_models': len(MODELS) + (1 if markov_model else 0)
    })


if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
