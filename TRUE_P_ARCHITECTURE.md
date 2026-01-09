# 🧮 True Probability (True P) Architecture

## Overview
The "True P" is an intelligent weighted ensemble that combines predictions from multiple data sources to calculate the most accurate probability estimate.

---

## 🔄 Data Flow

```
┌─────────────────┐
│  User Input     │
│  - Player names │
│  - Serve %      │
│  - Surface      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│         get_ml_predictions()                        │
│  Orchestrates all models and data sources          │
└──┬──────┬─────────┬─────────┬─────────────────┬────┘
   │      │         │         │                 │
   ▼      ▼         ▼         ▼                 ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌─────────┐ ┌───────────┐
│Markov│ │Tennis│ │Log   │ │Neural   │ │Feature    │
│Chain │ │Ratio │ │Reg   │ │Network  │ │Extraction │
└──┬───┘ └──┬───┘ └──┬───┘ └────┬────┘ └─────┬─────┘
   │        │        │          │            │
   ▼        ▼        ▼          ▼            ▼
┌────────────────────────────────────────────────────┐
│          Weighted Ensemble Calculator              │
│  - Assigns weights based on confidence             │
│  - Normalizes to 100%                              │
│  - Computes weighted average                       │
└───────────────────────┬────────────────────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │   TRUE P    │
                 │  (Ensemble) │
                 └─────────────┘
```

---

## 🎯 Model Weights

### Hierarchical Markov Chain
**Weight**: 25% - 40% (depends on data source)

#### Mode 1: Database-Backed (40%)
- **Condition**: Player IDs available in database
- **Data**: Historical serve/return statistics from past matches
- **Confidence**: HIGH ✅
- **Rationale**: Large sample size of actual match data

#### Mode 2: Live Serve % (25%)
- **Condition**: No database IDs, using manual serve %
- **Data**: Current/estimated serve percentages
- **Confidence**: MODERATE ⚠️
- **Rationale**: Point-level model still valid but less historical context

**Formula**:
```python
if player_in_database:
    markov_weight = 0.40
else:
    markov_weight = 0.25
```

---

### TennisRatio Web Data
**Weight**: 15% - 35% (depends on confidence)

#### High Confidence (35%)
- **Condition**: Rich H2H data, recent matches, detailed stats
- **Features**: Dominance, efficiency, pressure points, form
- **Rationale**: Real-time professional data

#### Medium Confidence (25%)
- **Condition**: Some H2H data or limited stats available
- **Rationale**: Partial but useful information

#### Low Confidence (15%)
- **Condition**: Minimal data fetched or players not well-known
- **Rationale**: Better than nothing but unreliable

**Formula**:
```python
if tennisratio_confidence == 'high':
    tr_weight = 0.35
elif tennisratio_confidence == 'medium':
    tr_weight = 0.25
else:
    tr_weight = 0.15
```

---

### Logistic Regression
**Weight**: 30% (when available)

- **Condition**: Full match features available (after set 1+)
- **Model**: Symmetric logistic regression (no bias)
- **Features**: Serve %, return %, break points, aces, etc.
- **Confidence**: HIGH ✅
- **Rationale**: Trained on large historical dataset

**Status**: Currently shows "Requires full match statistics"

---

### Neural Network Ensemble
**Weight**: 30% (when available)

- **Condition**: Full match features available (after set 1+)
- **Architecture**: 100 hidden neurons, tanh activation
- **Training**: Ensemble of networks
- **Confidence**: HIGH ✅
- **Rationale**: Captures non-linear patterns

**Status**: Currently shows "Requires full match statistics"

---

## 🧮 True P Calculation Algorithm

### Step 1: Collect Model Predictions
```python
predictions = {
    'hierarchical': 0.63,  # Markov
    'tennisratio': 0.58,   # Web data
    'logistic': None,      # Not available yet
    'neural': None         # Not available yet
}
```

### Step 2: Assign Weights
```python
weights = {
    'hierarchical': 0.40,  # Database mode
    'tennisratio': 0.25    # Medium confidence
}
```

### Step 3: Normalize Weights
```python
total_weight = 0.40 + 0.25 = 0.65
normalized_weights = {
    'hierarchical': 0.40 / 0.65 = 0.615 (61.5%)
    'tennisratio': 0.25 / 0.65 = 0.385 (38.5%)
}
```

### Step 4: Calculate Weighted Average
```python
true_p = (0.63 × 0.615) + (0.58 × 0.385)
       = 0.387 + 0.223
       = 0.610 (61.0%)
```

### Step 5: Calculate Confidence Score
```python
# Higher when multiple models with high weights agree
confidence = sum(top_2_normalized_weights)
           = 0.615 + 0.385
           = 1.0 (if only 2 models available)
```

---

## 📊 Example Scenarios

### Scenario A: Database Players + TennisRatio
```
Player 1 ID: 1091 (Munar)
Player 2 ID: 1125 (Baez)
Surface: Hard

Models Available:
✅ Markov (Database): 0.58 → Weight: 40%
✅ TennisRatio (High): 0.62 → Weight: 35%
❌ Logistic: N/A
❌ Neural: N/A

Normalized Weights:
- Markov: 40/75 = 53.3%
- TennisRatio: 35/75 = 46.7%

True P = (0.58 × 0.533) + (0.62 × 0.467)
       = 0.309 + 0.290
       = 0.599 ≈ 60%
```

### Scenario B: Custom Players (No Database)
```
Player 1: "Roger" (65% serve)
Player 2: "Rafael" (62% serve)

Models Available:
✅ Markov (Live): 0.56 → Weight: 25%
✅ TennisRatio (Low): 0.51 → Weight: 15%
❌ Logistic: N/A
❌ Neural: N/A

Normalized Weights:
- Markov: 25/40 = 62.5%
- TennisRatio: 15/40 = 37.5%

True P = (0.56 × 0.625) + (0.51 × 0.375)
       = 0.350 + 0.191
       = 0.541 ≈ 54%
```

### Scenario C: All 4 Models Available (Future)
```
Models Available:
✅ Markov (Database): 0.58 → Weight: 40%
✅ TennisRatio (High): 0.62 → Weight: 35%
✅ Logistic: 0.60 → Weight: 30%
✅ Neural: 0.61 → Weight: 30%

Total Weight: 135%

Normalized Weights:
- Markov: 40/135 = 29.6%
- TennisRatio: 35/135 = 25.9%
- Logistic: 30/135 = 22.2%
- Neural: 30/135 = 22.2%

True P = (0.58 × 0.296) + (0.62 × 0.259) + (0.60 × 0.222) + (0.61 × 0.222)
       = 0.172 + 0.161 + 0.133 + 0.135
       = 0.601 ≈ 60%
```

---

## 🎯 Integration with Bet Identification

### Game-Level Adjustment
Once True P is calculated for the match, it adjusts game-level probabilities:

```python
# Base game probability from Markov point model
p_hold_base = 0.68

# True P suggests Player 1 is stronger overall
ensemble_p1 = 0.60  # (60% to win match)
confidence = 0.85

# Adjustment factor: (0.60 - 0.50) × 0.1 × 0.85 = 0.0085
adjustment = (ensemble_p1 - 0.5) × 0.1 × confidence

# If Player 1 serving:
p_hold_adjusted = 0.68 + 0.0085 = 0.6885 ≈ 69%

# This gives more accurate edge calculations for betting
```

---

## 📈 Benefits of True P

1. **Robustness**: No single point of failure - uses all available data
2. **Adaptability**: Weights adjust based on data quality
3. **Transparency**: Users see exact model contributions
4. **Accuracy**: Weighted ensemble proven to outperform single models
5. **Confidence Tracking**: Know when predictions are reliable

---

## 🔮 Future Enhancements

1. **Dynamic Weight Learning**: Adjust weights based on historical accuracy
2. **Bayesian Updating**: Update probabilities as match progresses
3. **Context Awareness**: Adjust for tournament importance, fatigue, etc.
4. **Calibration**: Track and improve probability calibration over time
5. **ML Meta-Model**: Train a model to optimally combine base models

---

## 📝 Summary

True P is not just a simple average - it's an intelligent weighted ensemble that:
- ✅ Uses the best available data sources
- ✅ Adjusts for data quality and confidence
- ✅ Normalizes weights for consistency
- ✅ Provides transparency into calculations
- ✅ Integrates into bet identification

**Result**: More accurate probabilities = Better betting decisions = Higher ROI 🎯
