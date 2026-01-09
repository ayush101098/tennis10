# 🧪 Testing Checklist - Tennis Betting App

## Quick Test Guide

### ✅ Test 1: App Launch
- [ ] Open http://localhost:8501
- [ ] Verify single-page interface loads
- [ ] Check sidebar shows player setup (not multi-page navigation)
- [ ] No console errors

### ✅ Test 2: Markov Analysis WITHOUT Database Players
**Purpose**: Verify Markov runs for ALL matches

1. Enter custom player names (e.g., "Roger" and "Rafael")
2. Don't select from database
3. Adjust serve percentages (e.g., 65% vs 62%)
4. Check AI Insights section shows:
   - ⚡ Markov Chain (Live) model
   - Probability displayed (not "Select players from database")
   - "Real-time point-level simulation from serve stats"

**Expected**: Markov model works WITHOUT database IDs ✅

### ✅ Test 3: Markov Analysis WITH Database Players
**Purpose**: Verify enhanced database mode

1. Search for "Munar" in Player 1
2. Select "Jaume Munar" from dropdown
3. Search for "Baez" in Player 2
4. Select "Sebastian Baez" from dropdown
5. Check AI Insights section shows:
   - ⚡ Markov Chain (Database) model
   - "Historical point-level analysis"
   - Match count displayed

**Expected**: Markov uses database stats with higher weight ✅

### ✅ Test 4: True Probability (True P) Calculation
**Purpose**: Verify weighted ensemble

1. With both players loaded, check AI Insights
2. Click "Advanced Analytics & Feature Engineering" expander
3. Verify you see:
   - Model Contribution section
   - Markov Chain weight (should be 40% or 25%)
   - TennisRatio weight (if data fetched)
   - Point-to-Game Transition Probabilities
4. Check consensus probability is shown at top

**Expected**: True P = weighted average of available models ✅

### ✅ Test 5: Feature Engineering Display
**Purpose**: Verify comprehensive data extraction

1. Open "Advanced Analytics & Feature Engineering" expander
2. Verify sections shown:
   - 🎯 Model Contribution to True Probability (weights)
   - ⚡ Point-to-Game Transition Probabilities (hold/break %)
   - 🌐 TennisRatio Live Data (if fetched)
3. Check caption: "True Probability (P) is calculated using weighted ensemble..."

**Expected**: All extracted features visible ✅

### ✅ Test 6: Bet Identification with True P
**Purpose**: Verify ensemble integration in betting

1. Enter game odds (e.g., Hold: 1.50, Break: 3.00)
2. Check caption below odds input shows:
   - "Using True P (ensemble-adjusted)" when models available
   - OR "Using base Markov probability (collecting more data...)"
3. Verify edge calculations adjust based on True P

**Expected**: Bet edges use ensemble-adjusted probabilities ✅

### ✅ Test 7: TennisRatio Integration
**Purpose**: Verify live web data

1. Use real ATP player names (e.g., "Djokovic" and "Alcaraz")
2. Check AI Insights for:
   - 🌐 TennisRatio Advanced Stats model
   - Confidence level (high/medium/low)
   - Factors displayed
3. Open Analytics expander, check TennisRatio section shows:
   - Dominance %
   - Efficiency %
   - H2H record

**Expected**: Live web data integrated when available ✅

### ✅ Test 8: All 4 Models Status
**Purpose**: Verify all models tracked

Check AI Insights shows:
1. ⚡ Markov Chain (Database OR Live) - Should work ✅
2. 🌐 TennisRatio Advanced Stats - Should attempt fetch
3. 📈 Logistic Regression - Shows status message
4. 🧠 Neural Network Ensemble - Shows status message

**Expected**: All 4 models listed with appropriate status ✅

### ✅ Test 9: Live Scoring Integration
**Purpose**: Verify probability updates with score

1. Score several points using point buttons
2. Watch AI Insights update
3. Check True P adjusts as match progresses
4. Verify confidence boost shown in bet section

**Expected**: Dynamic probability updates ✅

### ✅ Test 10: No Database Connection
**Purpose**: Verify app works without database

1. Temporarily rename tennis_data.db
2. Reload app
3. Enter custom players
4. Verify:
   - Markov still works (Live mode)
   - TennisRatio still fetches
   - No crashes
   - Footer shows "Database: ❌ Not found"

**Expected**: Graceful degradation, app still functional ✅

---

## 🐛 Known Issues to Monitor

1. **TennisRatio Parsing**: Currently returns 50/50 often, needs better HTML extraction
2. **Neural Network Loading**: May show pickle error with TennisNN attribute
3. **BeautifulSoup**: Ensure installed: `pip install beautifulsoup4 lxml`

---

## 🎯 Success Criteria

- ✅ Markov works for ALL matches (not just database)
- ✅ True P uses weighted ensemble
- ✅ All 4 models tracked and displayed
- ✅ Feature engineering data visible
- ✅ Bet identification uses True P
- ✅ Single-page interface clean and functional
- ✅ No crashes or errors

---

## 📊 Test Results Template

```
Test Date: ___________
Tester: ___________

✅ Test 1 (Launch): PASS / FAIL
✅ Test 2 (Markov without DB): PASS / FAIL
✅ Test 3 (Markov with DB): PASS / FAIL
✅ Test 4 (True P): PASS / FAIL
✅ Test 5 (Features): PASS / FAIL
✅ Test 6 (Bet ID): PASS / FAIL
✅ Test 7 (TennisRatio): PASS / FAIL
✅ Test 8 (4 Models): PASS / FAIL
✅ Test 9 (Live Scoring): PASS / FAIL
✅ Test 10 (No DB): PASS / FAIL

Overall Status: PASS / FAIL
Notes: ___________
```
