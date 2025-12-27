# Tennis Prediction System - Deployment Guide

## Quick Start (Local)

### 1. Install Dependencies
```bash
cd /Users/ayushmishra/tennis10
source .venv/bin/activate
pip install flask
```

### 2. Train Models (Optional)
```bash
# Run notebooks to train models (if not already trained)
jupyter notebook logistic_regression_model.ipynb
jupyter notebook neural_network_model.ipynb
```

### 3. Start API Server
```bash
python api/prediction_service.py
```

Server runs on: `http://localhost:5000`

## API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### List Models
```bash
curl http://localhost:5000/models
```

### Predict Match
```bash
curl -X POST http://localhost:5000/predict_match \
  -H "Content-Type: application/json" \
  -d '{
    "player1_id": "novak-djokovic",
    "player2_id": "carlos-alcaraz",
    "surface": "hard",
    "date": "2024-01-15"
  }'
```

**Response:**
```json
{
  "player1_id": "novak-djokovic",
  "player2_id": "carlos-alcaraz",
  "surface": "hard",
  "date": "2024-01-15",
  "predictions": {
    "p_player1_win": 0.456,
    "p_player2_win": 0.544
  },
  "model_breakdown": {
    "markov": 0.445,
    "logistic": 0.460,
    "neural_net": 0.462
  },
  "model_confidence": "high",
  "uncertainty_score": 0.009,
  "timestamp": "2024-12-28T10:30:00"
}
```

## Testing Workflow

### 1. Test Naive Baselines
```bash
jupyter notebook validation/08_naive_backtest.ipynb
```
Confirms infrastructure works. Expected:
- Rank model: ~66% accuracy
- Odds model: ~70% accuracy
- Random: ~50% accuracy

### 2. Test Markov Model
```python
from hierarchical_model import HierarchicalTennisModel

model = HierarchicalTennisModel('tennis_data.db')
result = model.predict_match(
    'novak-djokovic',
    'rafael-nadal',
    'clay',
    best_of=5
)

print(f"P(Djokovic wins): {result['p_player1_win']:.2%}")
```

### 3. Calculate True Probabilities
The Markov model calculates true match probabilities by:

1. **Point Level**: Uses serve/return statistics
   ```python
   p_point = p_server_win_on_serve
   ```

2. **Game Level**: Binomial expansion
   ```python
   p_game = sum(binomial(n, k) * p^k * (1-p)^(n-k))
   # for k >= 4 points to win
   ```

3. **Set Level**: Accounts for tiebreaks
   ```python
   p_set = p(win 6+ games) + p(tiebreak at 6-6)
   ```

4. **Match Level**: Best-of-3 or Best-of-5
   ```python
   p_match = p(win 2/3 sets) or p(win 3/5 sets)
   ```

## File Structure
```
tennis10/
├── api/
│   └── prediction_service.py    # Flask API
├── ml_models/
│   ├── logistic_model.pkl       # Trained logistic model
│   └── nn_ensemble.pkl          # Trained neural net
├── validation/
│   └── 08_naive_backtest.ipynb  # Baseline tests
├── features.py                   # Feature generation
├── hierarchical_model.py         # Markov chain model
└── tennis_data.db               # Match database
```

## Next Steps

1. **Run Naive Baseline**: Validate infrastructure
2. **Test Markov Model**: Verify probability calculations
3. **Train ML Models**: Run logistic + neural net notebooks
4. **Compare Models**: Run final_model_evaluation.ipynb
5. **Deploy API**: Start prediction service

## Troubleshooting

### Models Not Loading
```bash
# Check if models exist
ls -lh ml_models/*.pkl

# If missing, train them first
jupyter notebook logistic_regression_model.ipynb
```

### Database Not Found
```bash
# Check database exists
ls -lh tennis_data.db

# If missing, run data pipeline
python data_pipeline/tennis_abstract_scraper.py
```

### API Port Already in Use
```bash
# Change port in prediction_service.py
app.run(port=5001)  # Use different port
```

## Production Deployment (Future)

1. **Add Authentication**: API keys for security
2. **Add Rate Limiting**: Prevent abuse
3. **Add Caching**: Cache predictions for same match
4. **Add Logging**: Track all predictions and outcomes
5. **Add Monitoring**: Grafana/Prometheus for metrics
6. **Live Odds Integration**: Fetch real bookmaker odds
7. **Database Logging**: Store predictions and calculate ROI

## Contact

For issues or questions, check the code comments or review validation notebooks.
