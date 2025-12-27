# Feature Engineering Module - Documentation

## Overview
The `features.py` module extracts sophisticated features from tennis match data for predictive modeling.

## Features Implemented

### 1. Basic Features
- **RANK_DIFF**: ATP ranking difference (player1 - player2)
- **POINTS_DIFF**: ATP points difference

### 2. Performance Features (with Time Decay & Surface Weighting)
- **WSP_DIFF**: Winning on serve percentage difference
- **WRP_DIFF**: Winning on return percentage difference
- **ACES_DIFF**: Aces per game difference
- **DF_DIFF**: Double faults per game difference
- **BP_CONV_DIFF**: Break points conversion percentage difference
- **WIN_RATE_DIFF**: Overall win rate difference (time-weighted)
- **SURFACE_WIN_RATE_DIFF**: Surface-specific win rate difference

### 3. Constructed Features
- **SERVEADV**: Serve advantage metric
  - Formula: `(WSP_p1 - WRP_p2) - (WSP_p2 - WRP_p1)`
- **COMPLETE_DIFF**: Complete performance metric
  - Formula: `(WSP_p1 * WRP_p1) - (WSP_p2 * WRP_p2)`
- **FATIGUE_DIFF**: Fatigue differential (games played in last 3 days)
- **RETIRED_DIFF**: Retirement status (1 if first match after 90+ days)
- **DIRECT_H2H**: Head-to-head win percentage difference (centered at 0)

### 4. Experience Features
- **MATCHES_PLAYED_DIFF**: Total match experience difference
- **SURFACE_EXP_DIFF**: Surface-specific experience difference

### 5. Uncertainty Score
- Composite metric based on:
  - Match count for both players (40% weight)
  - Surface-specific match count (30% weight)
  - Head-to-head match count (30% weight)
- Range: 0 (very confident) to 1 (uncertain)
- Threshold: 0.7 (exclude matches above this)

## Technical Implementation

### Time Decay
- **Formula**: `weight = 0.5^(years_diff / half_life)`
- **Half-life**: 0.8 years (matches from 0.8 years ago have 50% weight)
- **Rationale**: Recent form is more predictive than distant past

### Surface Weighting
Surface correlation matrix:
```
          Hard   Clay   Grass
Hard      1.00   0.28   0.24
Clay      0.28   1.00   0.15
Grass     0.24   0.15   1.00
```

- Same surface: full weight (1.0)
- Hard-Clay: moderate correlation (0.28)
- Hard-Grass: moderate correlation (0.24)
- Clay-Grass: low correlation (0.15)

### Combined Weighting
Each historical match gets a combined weight:
```python
combined_weight = time_weight * surface_weight
```

## Usage

### Single Match Extraction
```python
from features import TennisFeatureExtractor

extractor = TennisFeatureExtractor('tennis_data.db')

# Extract features for match ID 5000
features = extractor.extract_features(
    match_id=5000,
    lookback_months=36  # 3 years of history
)

print(f"RANK_DIFF: {features['RANK_DIFF']}")
print(f"Uncertainty: {features['UNCERTAINTY']}")

extractor.close()
```

### Batch Extraction
```python
# Extract features for multiple matches
features_df = extractor.extract_features_batch(
    match_ids=[5000, 5001, 5002],  # Or None for all matches
    lookback_months=36,
    uncertainty_threshold=0.7  # Exclude high-uncertainty matches
)

# Save to CSV
features_df.to_csv('features.csv', index=False)
```

### Predicting New Matches
```python
# For a match that hasn't happened yet
features = extractor.extract_features(
    player1_id=42,
    player2_id=87,
    match_date=datetime(2024, 12, 28),
    surface='Hard',
    lookback_months=36
)
```

## Important Notes

### Data Limitations
⚠️ **Tennis-data.co.uk Constraint**: The free Excel files don't include detailed statistics (aces, serve %, etc.). 

**Current Solution**: We use **proxy metrics** based on win rates:
- WSP/WRP estimated from weighted win rates
- Aces/DFs approximated from performance level
- Break point conversion derived from win patterns

**For Production**: Consider:
- Scraping ATP website for real statistics
- Using paid APIs (Tennis API, Sportradar)
- OnCourt or other professional data providers

### Feature Interpretation

**Positive values favor Player 1** (the first player in the comparison):
- `RANK_DIFF = -50` → Player 2 ranked 50 spots higher (disadvantage for Player 1)
- `WIN_RATE_DIFF = 0.15` → Player 1 has 15% better win rate (advantage)
- `DIRECT_H2H = 0.3` → Player 1 wins 80% of h2h matches (0.5 + 0.3)

**Uncertainty thresholds**:
- < 0.3: High confidence (both players have extensive history)
- 0.3-0.5: Medium confidence
- 0.5-0.7: Low confidence (include with caution)
- \> 0.7: Very low confidence (exclude from training)

## Testing

Run the test notebook:
```bash
jupyter notebook feature_engineering_test.ipynb
```

Or run the standalone test:
```bash
python features.py
```

## Performance

- **Single match**: ~50-100ms (depending on match history)
- **Batch (500 matches)**: ~1-3 minutes
- **Full dataset (11,794 matches)**: ~20-30 minutes

## Next Steps

1. **Extract features for full dataset**:
   ```python
   features_df = extractor.extract_features_batch(
       match_ids=None,  # All matches
       uncertainty_threshold=0.6
   )
   ```

2. **Train ML models** using these features
3. **Validate** on hold-out test set
4. **Fine-tune** uncertainty threshold for optimal dataset size vs quality trade-off

## Class Reference

### `TennisFeatureExtractor`

#### Methods
- `__init__(db_path)`: Initialize with database path
- `extract_features(match_id, lookback_months)`: Extract single match
- `extract_features_batch(match_ids, lookback_months, uncertainty_threshold)`: Batch extraction
- `apply_time_discount(date_current, date_past, half_life_years)`: Calculate time decay
- `get_surface_weight(surface1, surface2)`: Get surface correlation
- `calculate_uncertainty(p1_features, p2_features)`: Calculate uncertainty score
- `close()`: Close database connection

#### Attributes
- `SURFACE_CORRELATIONS`: Surface correlation matrix
- `conn`: SQLite database connection

## Example Output

```python
{
    'RANK_DIFF': -53.0,
    'POINTS_DIFF': -402.0,
    'WSP_DIFF': -0.0484,
    'WRP_DIFF': -0.0323,
    'WIN_RATE_DIFF': -0.1613,
    'SURFACE_WIN_RATE_DIFF': -0.1512,
    'SERVEADV': -0.0807,
    'COMPLETE_DIFF': -0.0242,
    'FATIGUE_DIFF': 0.0,
    'RETIRED_DIFF': 0,
    'DIRECT_H2H': 0.0,
    'MATCHES_PLAYED_DIFF': -15,
    'SURFACE_EXP_DIFF': -8,
    'UNCERTAINTY': 0.6069,
    'match_id': 5000,
    'surface': 'Hard',
    'match_date': datetime(2021, 8, 15)
}
```

## License
MIT License - Free to use and modify
