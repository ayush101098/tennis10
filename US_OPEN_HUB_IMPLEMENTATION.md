# US Open Value Betting Hub - Implementation Guide

## Overview

The `us_open_value_hub.py` Streamlit app provides a real-time value betting dashboard for US Open matches. It identifies opportunities where bookmaker odds are better than the model's predicted probability.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run us_open_value_hub.py
```

The app will open at `http://localhost:8501` and display:
- US Open match fixtures
- Model probability predictions
- Bookmaker odds comparison
- Value scores for each match
- Player statistics and recent form

---

## Key Features

### 1. Value Bet Identification
```
Value Score = (Odds × Model Probability) - 1
```

- **Positive value** = The odds are better than the model says
- **Negative value** = The odds are worse than they should be

Example:
- Model says: Player A has 55% win probability
- Odds offer: 2.00 (50% implied)
- Value Score: (2.00 × 0.55) - 1 = 0.10 = **+10% VALUE**

### 2. Filtering Options
- **Sort by Value Score** (highest first)
- **Sort by Odds** (lowest/highest)
- **Filter by Round** (SF, QF, R16, etc.)
- **Player Search** (find specific matchups)
- **Minimum Value Filter** (show only edges > X%)

### 3. Match Cards Display
Each match shows:
- Player names and round
- Model probability prediction
- Bookmaker odds (P1 & P2)
- Value score for each side
- Recent form (win rate, matches played, ranking)
- Confidence level
- Edge classification (🔥 HIGH / ⚡ MEDIUM / 📊 LOW)

### 4. Summary Statistics
- Total value bets found
- Average value score
- Count of high-value opportunities
- Average odds offered

---

## Integration Steps

### Step 1: Connect Live Data Sources (Week 1)

#### A. api-tennis.com Integration
```python
# Update: api/live_tennis_data.py
from api.live_tennis_data import LiveTennisDataService

service = LiveTennisDataService(api_key=os.environ.get("API_TENNIS_KEY"))

# Get US Open fixtures
fixtures = service.get_fixtures(tournament_key="us_open_2024")

# Get live odds for a match
odds = service.get_odds(event_key=match_id)
```

#### B. TennisRatio Integration
```python
# Already integrated in: api/tennisratio_integration.py
from api.tennisratio_integration import TennisRatioAPI

api = TennisRatioAPI()
h2h_data = api.fetch_h2h_data("Jannik Sinner", "Taylor Fritz")
```

### Step 2: Enhance Feature Extraction (Week 1-2)

Add to `features.py`:

```python
class LiveFeatureExtractor:
    """Extract features specifically for live matches"""
    
    def get_momentum_features(self, event_id: str):
        """Get real-time momentum from point-level data"""
        # Fetch live match data
        match_data = self.live_service.get_event_details(event_id)
        
        # Calculate momentum: points won in last 10 points
        recent_points = match_data['point_history'][-10:]
        p1_recent_wins = sum(1 for p in recent_points if p['winner_id'] == match_data['player1_id'])
        
        return {
            'momentum_p1': p1_recent_wins / 10,
            'rally_length_avg': np.mean([p['rally_length'] for p in recent_points]),
            'break_points_this_set': self._count_break_points(recent_points)
        }
    
    def get_pressure_features(self, player1_name: str, player2_name: str):
        """Get pressure point performance from TennisRatio"""
        h2h_data = self.tennisratio_api.fetch_h2h_data(player1_name, player2_name)
        
        return {
            'pp_serve_p1': h2h_data['stats']['pressure_points_serve']['player1'],
            'pp_return_p1': h2h_data['stats']['pressure_points_return']['player1'],
            'pp_deuce_p1': h2h_data['stats']['pressure_points_deuce']['player1']
        }
    
    def get_odds_movement_features(self, event_id: str):
        """Track odds movement to detect sharp money"""
        odds_history = self.live_service.get_odds_history(event_id)
        
        open_odds_p1 = odds_history[0]['odds_p1']
        current_odds_p1 = odds_history[-1]['odds_p1']
        
        return {
            'odds_movement_p1': (current_odds_p1 - open_odds_p1) / open_odds_p1,
            'odds_volatility': np.std([o['odds_p1'] for o in odds_history]),
            'odds_consensus': self._calculate_consensus(odds_history)
        }
```

### Step 3: Create Enhanced Model (Week 2)

```python
# New file: models/live_match_model.py

class LiveMatchModel(nn.Module):
    """Specialized model for live match probability updates"""
    
    def __init__(self, input_dim):
        super().__init__()
        # Input: historical stats + live features
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

# Training uses:
# - Historical stats (serve %, H2H, rankings)
# - Live features (momentum, pressure points, odds movement)
# - Target: actual match outcomes
```

### Step 4: Update US Open Hub (Week 2)

Modify `us_open_value_hub.py`:

```python
def get_model_prediction(p1_name, p2_name, surface='Hard', event_id=None):
    """Enhanced prediction with live data"""
    models = load_models()
    
    # Get historical prediction
    features = models['feature_extractor'].extract_features(
        p1_name, p2_name, surface
    )
    
    # If live match, add live features
    if event_id:
        live_service = LiveTennisDataService()
        live_features = {
            **models['live_extractor'].get_momentum_features(event_id),
            **models['live_extractor'].get_pressure_features(p1_name, p2_name),
            **models['live_extractor'].get_odds_movement_features(event_id)
        }
        features.update(live_features)
    
    # Get predictions from all models
    hierarchical_pred = models['hierarchical'].predict_match(p1_name, p2_name, surface)
    logistic_pred = models['logistic'].predict(features)
    neural_pred = models['neural'].predict(features)
    
    # Ensemble with context-aware weighting
    weights = {
        'hierarchical': 0.3,
        'logistic': 0.35,
        'neural': 0.35
    }
    
    if event_id:  # Live match - boost live model if available
        weights['live'] = 0.2
        weights = {k: v * 0.8 for k, v in weights.items() if k != 'live'}
    
    ensemble_pred = (
        weights['hierarchical'] * hierarchical_pred +
        weights['logistic'] * logistic_pred +
        weights['neural'] * neural_pred
    )
    
    return {
        'prob_p1': ensemble_pred,
        'confidence': calculate_confidence(models),
        'insights': []
    }
```

### Step 5: Real-time Updates (Week 3)

```python
# Add to us_open_value_hub.py

@st.cache_resource
def start_live_update_worker():
    """Background worker for real-time updates"""
    import threading
    
    def update_loop():
        while True:
            # Every 30 seconds during matches:
            live_service = LiveTennisDataService()
            
            for fixture in st.session_state.fixtures:
                if fixture['status'] == 'live':
                    # Update odds
                    fixture['odds1'] = live_service.get_odds(fixture['id'])['odds_p1']
                    
                    # Update prediction
                    pred = get_model_prediction(
                        fixture['player1'],
                        fixture['player2'],
                        fixture['surface'],
                        event_id=fixture['id']
                    )
                    fixture['prediction'] = pred
                    
                    # Recalculate value
                    fixture['value'] = calculate_value_score(
                        fixture['odds1'],
                        fixture['odds2'],
                        pred['prob_p1']
                    )
            
            time.sleep(30)
    
    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()
```

---

## Data Sources Configuration

### Environment Variables (.env)
```env
# API Keys
API_TENNIS_KEY=your_api_tennis_key
BETFAIR_API_KEY=your_betfair_key

# Database
TENNIS_DB_PATH=./tennis_data.db

# Model paths
LOGISTIC_MODEL_PATH=./ml_models/logistic_regression_trained.pkl
NEURAL_MODEL_PATH=./ml_models/neural_network_ensemble.pkl

# Live data refresh intervals
LIVE_ODDS_REFRESH_SECS=30
LIVE_SCORE_REFRESH_SECS=20
```

### API Endpoints

**api-tennis.com (Live odds & scores)**
```
/events/live              - Current live matches
/fixtures?date=YYYY-MM-DD - Scheduled matches
/odds/{event_id}          - Pre-match & in-play odds
/players/{player_key}     - Player profile
/h2h/{p1_key}/{p2_key}    - Head-to-head record
```

**TennisRatio.com (Advanced stats)**
```
/h2h-compare/{p1}-vs-{p2}.html - H2H comparison page
```

---

## Feature Engineering Pipeline

### Historical Features (Pre-Match)
```python
historical_features = {
    # Serve performance
    'p1_first_serve_pct': 0.625,
    'p1_first_serve_win_pct': 0.715,
    'p1_ace_rate': 0.85,
    
    # Return performance  
    'p1_bp_save_pct': 0.620,
    'p1_bp_win_pct': 0.280,
    
    # Surface adjustment
    'p1_hard_correlation': surface_correlation_factor,
    
    # H2H
    'p1_h2h_wins': 5,
    'p1_h2h_win_pct': 0.625,
    
    # Ranking
    'p1_current_rank': 2,
    'p1_rank_change_6m': -1,
}
```

### Live Features (During Match)
```python
live_features = {
    # Momentum
    'momentum_p1': 0.7,  # 70% of last 10 points won
    'rally_length_trend': 15,  # avg rally length
    
    # Pressure points
    'pp_serve_p1': 0.65,  # pressure point win % on serve
    'pp_return_p1': 0.55,
    
    # Current game state
    'current_set_p1_games': 3,
    'current_game_p1_points': 15,
    'break_points_created': 2,
    
    # Odds
    'odds_movement_p1': 0.05,  # +5% movement from open
    'implied_prob_p1': 0.52,
    'odds_consensus': 0.98,  # agreement across books
}
```

---

## Model Performance Targets

After implementing all Tier 1 features:
- **Pre-match accuracy**: 55-58% (vs 52% baseline)
- **Live match accuracy**: 58-62% (vs 54% baseline)
- **Value bet ROI**: +8-12% over 100+ bets

---

## Testing & Validation

### Backtest Historical US Open Matches
```python
# test_us_open_predictions.py

def backtest_us_open():
    """Test predictions against historical US Open results"""
    
    # Load all US Open matches from database
    conn = sqlite3.connect('tennis_data.db')
    usopen_matches = pd.read_sql_query("""
        SELECT * FROM matches 
        WHERE tournament_name LIKE '%US Open%'
        ORDER BY tournament_date DESC
    """, conn)
    
    # For each match, test prediction accuracy
    correct = 0
    value_positive = 0
    value_profit = 0
    
    for _, match in usopen_matches.iterrows():
        # Get prediction from model
        pred = get_model_prediction(
            match['player1_name'],
            match['player2_name'],
            match['surface']
        )
        
        # Check if correct
        actual_winner = match['winner_id']
        if pred['prob_p1'] > 0.5 and actual_winner == match['player1_id']:
            correct += 1
        elif pred['prob_p1'] <= 0.5 and actual_winner == match['player2_id']:
            correct += 1
        
        # Check if had value
        value = calculate_value_score(
            match['odds_p1'],
            match['odds_p2'],
            pred['prob_p1']
        )
        
        if value['best_edge'] > 0:
            value_positive += 1
            # Calculate profit if bet on best value side
            if value['best_value'] == 'P1':
                if actual_winner == match['player1_id']:
                    value_profit += match['odds_p1'] - 1
                else:
                    value_profit -= 1
    
    accuracy = correct / len(usopen_matches)
    value_bet_roi = value_profit / value_positive if value_positive > 0 else 0
    
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Value bet ROI: {value_bet_roi:.1%}")
    print(f"Value opportunities: {value_positive}")
```

---

## Deployment

### Docker Deployment
```dockerfile
# Add to existing Dockerfile
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "us_open_value_hub.py"]
```

### Cloud Deployment (GCP/Render)
```yaml
# render.yaml addition
services:
  - type: web
    name: tennis-value-hub
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run us_open_value_hub.py
    envVars:
      - key: API_TENNIS_KEY
        scope: run
      - key: PORT
        value: "8501"
```

---

## Monitoring & Alerts

### Track Performance Metrics
```python
# Log predictions and outcomes for continuous monitoring
import logging

class PredictionLogger:
    def __init__(self):
        self.logger = logging.getLogger('predictions')
        
    def log_prediction(self, match_id, pred, odds, value_score):
        """Log all predictions for analysis"""
        self.logger.info(json.dumps({
            'match_id': match_id,
            'predicted_prob': pred['prob_p1'],
            'odds_p1': odds[0],
            'value_score': value_score,
            'timestamp': datetime.now().isoformat()
        }))
    
    def log_outcome(self, match_id, winner_id, odds_p1):
        """Log actual outcomes"""
        self.logger.info(json.dumps({
            'match_id': match_id,
            'actual_winner': winner_id,
            'odds_p1': odds_p1,
            'timestamp': datetime.now().isoformat()
        }))
```

---

## Next Steps

1. **Week 1**: Integrate api-tennis.com for live fixtures & odds
2. **Week 2**: Add pressure point features from TennisRatio
3. **Week 2**: Deploy enhanced US Open hub to production
4. **Week 3**: Implement live momentum tracking
5. **Week 4**: Add recent form & tournament history features
6. **Week 5+**: Continuous optimization based on actual results

---

## Support & Troubleshooting

**Models not loading?**
- Check `ml_models/` directory exists
- Verify pickle files are not corrupted: `python -c "import pickle; pickle.load(open('ml_models/logistic_regression_trained.pkl', 'rb'))"`

**No fixtures appearing?**
- Verify API_TENNIS_KEY is set
- Check api-tennis.com is accessible
- Fallback to cached sample data while troubleshooting

**Odds not updating?**
- Check internet connection
- Verify API key rate limits not exceeded
- Review error logs in sidebar

