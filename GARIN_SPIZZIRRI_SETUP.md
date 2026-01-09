# Garin vs Spizzirri - Live Calculator Setup Guide

## Match Details
- **Player 1:** Cristian Garin
- **Player 2:** Eliot Spizzirri
- **Surface:** Hard Court
- **Intelligence Report:** match_intelligence_Cristian_Garin_vs_Eliot_Spizzirri.json

---

## Step-by-Step Setup

### 1. Open Live Calculator V2
- Go to: http://localhost:8501
- Navigate to: **🎯 Live Calculator V2** page

### 2. Enter Player Names
```
Player 1 Name: Cristian Garin
Player 2 Name: Eliot Spizzirri
Surface: Hard
```

### 3. Enter Basic Stats

#### Cristian Garin (Player 1):
```
Serve Win %: 0.655 (65.5%)
Return Win %: 0.397 (39.7%)
```

#### Eliot Spizzirri (Player 2):
```
Serve Win %: 0.703 (70.3%)
Return Win %: 0.452 (45.2%)
```

---

### 4. Expand "Advanced Player Parameters"

#### Cristian Garin Parameters:
```
Momentum:        0.478 (from intelligence report)
Surface Mastery: 0.40  (hard court mastery)
Clutch:          0.50  (default - no data)
BP Defense:      0.69  (68.9% break point save rate)
Consistency:     0.50  (default - use match observation)
First Serve %:   0.631 (63.1% from stats)
```

#### Eliot Spizzirri Parameters:
```
Momentum:        0.25  (poor recent form: 25% win rate)
Surface Mastery: 0.00  (0% hard court win rate, limited data)
Clutch:          0.50  (default - no data)
BP Defense:      0.80  (80% break point save rate)
Consistency:     0.50  (default - use match observation)
First Serve %:   0.641 (64.1% from stats)
```

---

### 5. Enter Pre-Match Bookmaker Odds

Click "📰 Pre-Match Bookmaker Odds" expander:

```
Garin Pre-Match Odds:     1.85 (or current market odds)
Spizzirri Pre-Match Odds: 2.10 (or current market odds)
```

**Note:** Adjust these based on actual bookmaker odds before match starts.

**Implied Probabilities:**
- Garin: 54.1% (1/1.85)
- Spizzirri: 47.6% (1/2.10)

---

### 6. Start Tracking Points

Go to **Tab 1: 🎾 Live Tracker**

As match progresses:
- Click "🟢 Cristian Garin WINS POINT" when Garin wins
- Click "🔵 Eliot Spizzirri WINS POINT" when Spizzirri wins
- Score updates automatically
- Data saves to database after EVERY point

---

### 7. Monitor Value Bets

Go to **Tab 2: 📊 Probability & Bets**

#### Enter Current Live Odds:

**Match Winner:**
```
Garin Match Odds:     (enter current odds)
Spizzirri Match Odds: (enter current odds)
```

**Current Set Winner:**
```
Garin Set Odds:     (enter current odds)
Spizzirri Set Odds: (enter current odds)
```

**Next Game Winner:**
```
Garin Game Odds:     (enter current odds)
Spizzirri Game Odds: (enter current odds)
```

#### Save Value Bets:
- System calculates edge for all markets
- Shows value bets with:
  - ✅ Recommended stake
  - 📈 Expected Value
  - 🎯 Edge percentage
- Click "💾 SAVE MATCH BET", "💾 SAVE SET BET", or "💾 SAVE GAME BET"
- Bets saved to database with complete context

---

### 8. View Your Bets

Go to **Tab 3: 💰 My Selected Bets**

See all saved bets:
- Time placed
- Bet type
- Selection
- Odds
- Edge & EV
- Recommended stake
- Score when placed

---

### 9. Track Match Progression

Go to **Tab 4: 📈 Analytics**

View:
- Probability evolution chart
- Momentum shifts
- Key turning points
- Win probability over time

---

### 10. View Complete Match History

Go to **Tab 5: 📸 Match Snapshots**

Features:
- Complete point-by-point history
- Filter by: All / Sets Only / Games Only / Every Point
- Shows last N snapshots (adjustable)
- Score, probabilities, and odds at each point
- Pre-match vs current odds comparison

---

## Expected Model Predictions

### Pre-Match Assessment:

**Garin Advantages:**
- ✅ Experience (251 matches vs 4)
- ✅ Better hard court record (40% vs 0%)
- ✅ Higher momentum (0.478 vs 0.25)
- ✅ More consistent performance history

**Spizzirri Advantages:**
- ✅ Higher serve win % (70.3% vs 65.5%)
- ✅ Higher return win % (45.2% vs 39.7%)
- ✅ Better BP save rate (80% vs 68.9%)
- ✅ Higher first serve % (64.1% vs 63.1%)

**Likely Scenario:**
- Model should favor **Garin** due to experience and momentum
- But Spizzirri's superior serve/return stats may create value
- Watch for value on Spizzirri if odds drift too high
- In-play odds will fluctuate based on break points

---

## Betting Strategy Recommendations

### Pre-Match:
1. Wait for model probability calculation
2. Compare with bookmaker odds
3. If Spizzirri odds > 2.50 and model shows >40% chance, consider value
4. If Garin odds > 2.00, likely strong value (experience edge)

### In-Play:
1. Track every point to update probabilities
2. Enter live odds frequently (every game)
3. Look for:
   - **Set winner bets** when one player leads in games
   - **Game winner bets** when server has high hold %
   - **Match winner bets** after break points
4. Save all value bets with edge > 2.5%

### Markets to Focus:
1. **Match Winner** - Main market, highest stakes
2. **Set Winner** - Good value when games close
3. **Next Game** - Server advantage plays

---

## Data Persistence

### Auto-Save Features:
- ✅ Every point saves to database
- ✅ Match snapshots created automatically
- ✅ Odds history tracked
- ✅ Bet selections preserved

### If You Refresh Page:
1. Click "📂 Load Match"
2. Enter: "Cristian Garin" and "Eliot Spizzirri"
3. All data restored:
   - Complete score
   - Probability history
   - Advanced parameters
   - Match snapshots
   - Pre-match odds

---

## Tips for Success

### Parameter Adjustments:
- Adjust **Momentum** if you see momentum shifts during match
- Adjust **Consistency** based on unforced error rate
- Adjust **Clutch** if players perform differently on big points

### Odds Management:
- Update odds every 2-3 games minimum
- Save odds snapshots at set changes
- Compare live odds movement vs pre-match

### Bet Tracking:
- Don't just save value bets - track ALL recommendations
- Review Tab 3 to see bet portfolio
- Use snapshots to analyze timing of bets

---

## Expected Workflow

```
1. Setup (2 min)
   ↓
2. Enter pre-match odds
   ↓
3. First serve → Start tracking
   ↓
4. After each point → Click winner
   ↓
5. Every 2-3 games → Update live odds
   ↓
6. When value appears → Save bet
   ↓
7. Check Tab 3 → Review saved bets
   ↓
8. Check Tab 5 → See match progression
   ↓
9. Match ends → Review analytics
```

---

## Troubleshooting

**Q: Data lost on refresh?**
A: Click "📂 Load Match" and enter player names

**Q: Odds not being saved in snapshots?**
A: Enter odds in Tab 2, then track next point - odds saved with snapshot

**Q: Value bet not showing?**
A: Edge must be > 2.5% to trigger alert

**Q: Can't see all parameters?**
A: Expand "Advanced Player Parameters" section

**Q: Snapshots showing wrong data?**
A: Check filter settings in Tab 5

---

## Ready to Start!

Dashboard URL: **http://localhost:8501**

Navigate to: **🎯 Live Calculator V2**

Good luck with the match tracking!
