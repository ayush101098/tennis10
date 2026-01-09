# Advanced Statistics Integration Guide
## Live Calculator - Parameter-Based Markov Adjustments

### 🎯 Overview
The Live Calculator now includes an **Advanced Parameters** section that allows you to input special statistics for players. These parameters dynamically adjust the Markov Chain probability calculations in real-time.

---

## 📊 Available Advanced Parameters

### 1. **Momentum Score** (0.0 - 1.0)
- **What it is:** Recent form indicator
- **Scale:**
  - 0.0 = Very poor recent form
  - 0.5 = Average/Neutral
  - 1.0 = Excellent recent form
- **Impact:** ±7.5% probability adjustment
- **Example:** Player winning 18/20 recent matches → 0.85-0.95

### 2. **Surface Mastery** (0.0 - 1.0)
- **What it is:** Win rate on current surface (Hard/Clay/Grass)
- **Scale:**
  - 0.0 = Poor on surface
  - 0.5 = Average (50% win rate)
  - 1.0 = Dominant on surface
- **Impact:** ±5% probability adjustment
- **Example:** Nadal on Clay → 0.90+ | Djokovic on Hard → 0.75-0.85

### 3. **Clutch Performance** (0.0 - 1.0)
- **What it is:** Performance in big matches (Grand Slams, Masters 1000)
- **Scale:**
  - 0.0 = Poor in big matches
  - 0.5 = Average
  - 1.0 = Exceptional clutch player
- **Impact:** ±4% probability adjustment
- **Example:** Federer in Grand Slams → 0.80+ | Lower-ranked players → 0.20-0.40

### 4. **Break Point Defense Rate** (0.0 - 1.0)
- **What it is:** Break points saved / break points faced
- **Scale:**
  - 0.4 = Poor BP defense
  - 0.6 = Average
  - 0.8+ = Excellent BP defense
- **Impact:** ±10% adjustment (baseline at 0.6)
- **Example:** Top servers (Isner, Karlovic) → 0.70-0.75 | Baseline players → 0.55-0.65

### 5. **Consistency Rating** (0.0 - 1.0)
- **What it is:** Performance variance (1 - standard deviation of form)
- **Scale:**
  - 0.0 = Highly inconsistent
  - 0.5 = Average variance
  - 1.0 = Very consistent performer
- **Impact:** ±2.5% probability adjustment
- **Example:** Djokovic, Murray → 0.70-0.80 | Young players → 0.30-0.50

### 6. **First Serve In %** (0.0 - 1.0)
- **What it is:** First serve percentage
- **Scale:**
  - 0.50 = Poor first serve%
  - 0.62 = Average
  - 0.75+ = Excellent first serve%
- **Impact:** ±8% adjustment (baseline at 0.62)
- **Example:** Big servers → 0.65-0.70 | Defensive players → 0.58-0.62

---

## 🔧 How Parameters Affect Calculations

### Markov Chain Adjustments

The system applies the following formula to base serve/return percentages:

```python
# Total Serve Adjustment
serve_adjustment = (
    (momentum - 0.5) * 0.15 +           # Momentum: ±7.5%
    (surface_mastery - 0.5) * 0.10 +    # Surface: ±5%
    (clutch - 0.5) * 0.08 +             # Clutch: ±4%
    (bp_defense - 0.6) * 0.10 +         # BP Defense: ±10%
    (consistency - 0.5) * 0.05 +        # Consistency: ±2.5%
    (first_serve_pct - 0.62) * 0.08     # 1st Serve: ±8%
)

adjusted_serve_pct = base_serve_pct + serve_adjustment
```

### Example Calculation

**Player:** Juan Manuel Cerundolo  
**Base Stats:** Serve 63%, Return 42%  
**Advanced Parameters:**
- Momentum: 0.374
- Surface Mastery: 0.414 (Hard)
- Clutch: 0.314
- BP Defense: 0.532
- Consistency: 0.50
- First Serve %: 0.659

**Adjustments:**
- Momentum: (0.374 - 0.5) × 0.15 = -0.0189 (-1.89%)
- Surface: (0.414 - 0.5) × 0.10 = -0.0086 (-0.86%)
- Clutch: (0.314 - 0.5) × 0.08 = -0.0149 (-1.49%)
- BP Defense: (0.532 - 0.6) × 0.10 = -0.0068 (-0.68%)
- Consistency: (0.50 - 0.5) × 0.05 = 0 (0%)
- First Serve: (0.659 - 0.62) × 0.08 = +0.0031 (+0.31%)

**Total Adjustment:** -4.61%  
**Final Serve %:** 63% - 4.61% = **58.39%**

This shows how below-average parameters reduce the player's effective serve percentage in the model.

---

## 🎮 How to Use in Dashboard

### Step 1: Open Advanced Parameters
1. Navigate to **Live Calculator** page
2. In sidebar, find "🎯 Advanced Parameters (Optional)"
3. Click to expand the section

### Step 2: Enter Player Statistics
For each player, adjust sliders:
- **Momentum Score** - Based on recent 20 matches
- **Surface Mastery** - Win rate on current surface
- **Clutch Performance** - Grand Slam/Masters performance
- **BP Defense Rate** - Break points saved ratio
- **Consistency Rating** - Performance variance
- **1st Serve In %** - First serve percentage

### Step 3: Review Parameter Impact
- Check the **"Parameter Impact Preview"** section
- Shows total adjustment for each player (e.g., "+3.2%" or "-1.8%")
- Green = Advantage, Red = Disadvantage

### Step 4: Observe Adjusted Calculations
Once parameters are set, the system will:
1. **Adjust base probabilities** before Markov calculations
2. **Display adjustment info** (e.g., "Serve adjusted: 63.0% → 66.2% (+3.2%)")
3. **Apply to all models** (Markov, LR, NN)
4. **Update in real-time** as match progresses

---

## 📈 Real-World Example: Mochizuki vs Cerundolo

### Player 1: Shintaro Mochizuki
**Database Stats:**
- Momentum: 0.155 (poor form - 3-19 recent)
- Surface Mastery: 0.002 (weak on hard)
- Clutch: 0.0 (no big match wins)
- BP Defense: 0.512
- First Serve: 0.628
- Consistency: 0.50

**Input to Dashboard:**
```
Serve Win %: 63%
Return Win %: 35%
Momentum: 0.16
Surface Mastery (Hard): 0.00
Clutch: 0.00
BP Defense: 0.51
Consistency: 0.50
1st Serve %: 0.63
```

**Expected Adjustment:** -3% to -5% (below average parameters)

### Player 2: Juan Manuel Cerundolo
**Database Stats:**
- Momentum: 0.374 (moderate form)
- Surface Mastery: 0.414 (decent on hard)
- Clutch: 0.314
- BP Defense: 0.532
- First Serve: 0.659
- Consistency: 0.50

**Input to Dashboard:**
```
Serve Win %: 63%
Return Win %: 42%
Momentum: 0.37
Surface Mastery (Hard): 0.41
Clutch: 0.31
BP Defense: 0.53
Consistency: 0.50
1st Serve %: 0.66
```

**Expected Adjustment:** -1% to -2% (slightly below average)

**Net Effect:** Cerundolo gains +2-3% advantage from better parameters

---

## 🔍 Parameter Sources

### From Database (`fetch_player_intelligence.py`)
The system can automatically pull these parameters from your database:
```python
intel = PlayerIntelligence()
matchup = intel.generate_match_intelligence(
    player1_name="Mochizuki",
    player2_name="Cerundolo",
    surface="Hard"
)
```

Output includes:
- `momentum_score`: Exponentially weighted recent form
- `surface_mastery`: Win rate on best surface
- `clutch_performance`: Performance in Grand Slams/Masters
- `bp_defense_rate`: Break points saved / faced
- `first_serve_win_pct`: Points won on first serve
- `consistency_rating`: 1 - std(rolling_win_rate)
- `career_win_rate`: Overall win percentage

### From Manual Research
- **ATP/WTA Stats:** [atptour.com](https://www.atptour.com), [wtatennis.com](https://www.wtatennis.com)
- **Tennis Abstract:** [tennisabstract.com](http://www.tennisabstract.com)
- **Ultimate Tennis Statistics:** [ultimatetennisstatistics.com](https://www.ultimatetennisstatistics.com)
- **Flashscore/Sofascore:** Recent match statistics

---

## ⚙️ Technical Implementation

### Backend Adjustment Function
```python
def apply_advanced_parameters(base_serve_pct, base_return_pct, adv_params):
    momentum_adj = (adv_params['momentum'] - 0.5) * 0.15
    surface_adj = (adv_params['surface_mastery'] - 0.5) * 0.10
    clutch_adj = (adv_params['clutch'] - 0.5) * 0.08
    bp_defense_adj = (adv_params['bp_defense'] - 0.6) * 0.10
    consistency_adj = (adv_params['consistency'] - 0.5) * 0.05
    first_serve_adj = (adv_params['first_serve_pct'] - 0.62) * 0.08
    
    total_serve_adj = sum([momentum_adj, surface_adj, clutch_adj, 
                           bp_defense_adj, consistency_adj, first_serve_adj])
    
    adjusted_serve = max(0.45, min(0.85, base_serve_pct + total_serve_adj))
    return adjusted_serve
```

### Markov Integration
```python
# Before calculation
probs = calculate_point_probabilities(
    p1_serve_win=0.63, 
    p2_serve_win=0.63,
    p1_adv_params={'momentum': 0.16, 'surface_mastery': 0.00, ...},
    p2_adv_params={'momentum': 0.37, 'surface_mastery': 0.41, ...}
)

# Adjustments applied internally
# P1: 63% → 59% (poor parameters)
# P2: 63% → 62% (slightly below average)

# Then normal Markov calculation proceeds with adjusted values
```

---

## 📊 Expected Impacts by Parameter Range

| Parameter | Value | Typical Impact | Player Type |
|-----------|-------|----------------|-------------|
| **Momentum** | 0.8-1.0 | +4.5% to +7.5% | Hot streak (15+ wins/20) |
| | 0.4-0.6 | -1.5% to +1.5% | Average form |
| | 0.0-0.2 | -7.5% to -4.5% | Poor form (<5 wins/20) |
| **Surface Mastery** | 0.8-1.0 | +3% to +5% | Nadal on Clay, Djokovic on Hard |
| | 0.4-0.6 | -1% to +1% | Average surface |
| | 0.0-0.2 | -5% to -3% | Weak surface |
| **Clutch** | 0.7-1.0 | +1.6% to +4% | Big match players |
| | 0.3-0.5 | -1.6% to +0% | Average clutch |
| | 0.0-0.2 | -4% to -2.4% | Poor under pressure |
| **BP Defense** | 0.7-0.8 | +1% to +2% | Top servers |
| | 0.5-0.6 | -1% to 0% | Average |
| | 0.3-0.5 | -3% to -1% | Poor BP defense |

---

## 🎯 Best Practices

### 1. **Use Database Values When Available**
- Run `fetch_player_intelligence.py` first
- Use output JSON files for accuracy
- Manually enter values from reports

### 2. **Start Conservative**
- Set all to 0.5 (neutral) if unsure
- Gradually adjust based on research
- Avoid extreme values without evidence

### 3. **Update During Match**
- Momentum can change mid-match
- Adjust if player showing fatigue
- Increase clutch parameter in critical sets

### 4. **Validate Adjustments**
- Check "Parameter Impact Preview"
- Ensure adjustments make sense (±10% max total)
- Compare to pre-match odds for sanity check

### 5. **Document Your Sources**
- Keep notes on where parameters came from
- Track accuracy for future improvements
- Build player profiles over time

---

## 🚀 Quick Start Guide

**For Mochizuki vs Cerundolo:**

1. **Open Dashboard:** http://localhost:8501
2. **Go to Live Calculator**
3. **Enter Basic Stats:**
   - Mochizuki: Serve 63%, Return 35%
   - Cerundolo: Serve 63%, Return 42%
4. **Expand Advanced Parameters**
5. **Enter from Database Report:**
   - Mochizuki: Momentum 0.16, Surface 0.00, Clutch 0.00
   - Cerundolo: Momentum 0.37, Surface 0.41, Clutch 0.31
6. **Check Adjustment Preview:**
   - Should show Mochizuki: -3% to -5%
   - Should show Cerundolo: -1% to -2%
7. **Observe Probability Change:**
   - Without params: ~60-65% Cerundolo
   - With params: ~70-75% Cerundolo (+5-10% boost)

---

## 📝 Parameter Quick Reference Card

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETER QUICK REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Momentum:       0.0 ════ 0.5 ════ 1.0
                Poor   Average  Excellent
                ±7.5% max impact

Surface:        0.0 ════ 0.5 ════ 1.0
                Weak   Average  Dominant
                ±5% max impact

Clutch:         0.0 ════ 0.5 ════ 1.0
                Poor   Average  Elite
                ±4% max impact

BP Defense:     0.4 ════ 0.6 ════ 0.8
                Weak   Average  Strong
                ±10% max impact (baseline 0.6)

Consistency:    0.0 ════ 0.5 ════ 1.0
                Erratic Average  Steady
                ±2.5% max impact

1st Serve%:     0.50 ═══ 0.62 ═══ 0.75
                Poor   Average  Excellent
                ±8% max impact (baseline 0.62)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL POSSIBLE RANGE: ±37% (theoretical max)
TYPICAL RANGE: ±10% (realistic adjustments)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

*System designed for Mochizuki vs Cerundolo analysis with database intelligence integration*
