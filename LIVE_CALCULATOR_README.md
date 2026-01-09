# 🎯 Live Match Calculator - User Guide

## Overview

The Live Match Calculator is a **real-time Markov chain probability calculator** designed for in-play tennis betting. It allows you to manually enter live match scores and get instant win probability updates based on mathematical modeling.

## 🚀 Quick Start

### 1. Launch Dashboard
```bash
./launch_dashboard.sh
```

### 2. Navigate to Live Calculator
- Open dashboard at `http://localhost:8501`
- Click on **"🎯 Live Calculator"** in the sidebar (6th page)

### 3. Set Up Match
**In the sidebar:**
- Enter player names
- Set player statistics (serve win %, return win %)
- Input current bookmaker odds

**In the main area:**
- Enter current score (sets, games, points)
- Select who is serving
- Click through the match as it progresses

## 📊 Features

### Real-Time Probability Calculation
- **Markov Chain Model**: Mathematical probability based on point-by-point transitions
- **Live Updates**: Probabilities adjust instantly as you update the score
- **Multiple Stages**: Shows probabilities for match, set, and game levels

### Value Bet Detection
- **Edge Calculation**: Compares your model probability vs bookmaker implied odds
- **EV Analysis**: Expected value calculation for each bet
- **Automatic Alerts**: Green "✅ VALUE BET" when edge > 2.5%
- **Kelly Stakes**: Recommended bet size based on edge and bankroll

### Match Tracking
- **Save States**: Capture probability snapshots throughout the match
- **History Review**: See how probabilities evolved
- **Notes**: Add context about momentum, injuries, weather, etc.
- **Export Data**: Download match history for analysis

## 🎮 Step-by-Step Usage

### Before Match Starts

1. **Enter Player Names**
   ```
   Player 1: Novak Djokovic
   Player 2: Carlos Alcaraz
   ```

2. **Set Player Statistics** (sidebar)
   - Use historical averages or recent form
   - Typical serve win %: 60-75%
   - Typical return win %: 30-45%
   
   Example:
   ```
   Djokovic Serve Win: 68%
   Djokovic Return Win: 35%
   Alcaraz Serve Win: 65%
   Alcaraz Return Win: 32%
   ```

3. **Input Bookmaker Odds** (sidebar)
   ```
   Djokovic Odds: 1.85
   Alcaraz Odds: 2.10
   ```

### During the Match

4. **Update Score After Each Game**
   
   **Example progression:**
   
   **Start of match:**
   ```
   Sets: 0-0
   Games: 0-0
   Points: 0-0
   Server: Djokovic
   → Win Prob: 52% (Djokovic)
   ```
   
   **After first game (Djokovic holds):**
   ```
   Sets: 0-0
   Games: 1-0
   Points: 0-0
   Server: Alcaraz
   → Win Prob: 53.5% (Djokovic)
   ```
   
   **Djokovic breaks to lead 3-1:**
   ```
   Sets: 0-0
   Games: 3-1
   Points: 0-0
   Server: Djokovic
   → Win Prob: 58.2% (Djokovic) ↑
   → VALUE BET DETECTED ✅ (Edge: +3.8%)
   ```
   
   **Djokovic wins first set:**
   ```
   Sets: 1-0
   Games: 0-0
   Points: 0-0
   Server: Alcaraz
   → Win Prob: 64.7% (Djokovic) ↑↑
   → VALUE BET DETECTED ✅ (Edge: +5.2%)
   → Recommended Stake: $208
   ```

5. **Save Important Moments**
   - Click "💾 Save Match State" when:
     - Value bet opportunities appear
     - Major momentum shifts occur
     - Critical break points happen
   
6. **Add Notes**
   ```
   "Alcaraz looks tired after 10-minute game at 3-3"
   "Djokovic serving percentage dropped to 55% in 2nd set"
   "Wind picked up, affecting serve placement"
   ```

## 📈 Interpreting the Output

### Probability Metrics

| Metric | Meaning | Action |
|--------|---------|--------|
| **Win Probability** | Model's estimate of match win chance | Compare to bookmaker implied odds |
| **Edge** | Difference from bookmaker probability | Look for >2.5% for value |
| **Expected Value (EV)** | Average profit per $1 bet | Positive EV = profitable long-term |
| **Recommended Stake** | Kelly criterion calculation | Bet this % of bankroll |

### Example Interpretation

```
Djokovic Win Probability: 64.7%
Bookmaker Odds: 1.85 (implied 54.1%)
Edge: +10.6%
EV: +15.8%
✅ VALUE BET DETECTED
Recommended Stake: $424
```

**What this means:**
- Your model thinks Djokovic has 64.7% chance to win
- Bookmaker odds imply only 54.1% chance
- You have a 10.6% edge over the market
- If you bet $1, you expect to make 15.8 cents on average
- Kelly suggests betting $424 (on $10,000 bankroll)

## 💡 Best Practices

### Getting Accurate Statistics

**Option 1: Use Historical Averages**
- Check player's serve/return stats from past tournaments
- Sources: ATP website, Tennis Abstract, Flashscore
- Adjust for surface (clay, grass, hard)

**Option 2: Use Live Match Stats**
- Many streaming services show live stats
- Update stats every set based on current match
- More accurate but requires real-time tracking

**Option 3: Start with Defaults and Adjust**
- Begin with typical values (65% serve, 35% return)
- Adjust based on what you're seeing in the match
- Update if player is clearly under-performing

### When to Bet

✅ **Good Scenarios:**
- Edge > 2.5%
- EV > 5%
- You've observed momentum shift not reflected in odds
- Player statistics are accurate (not defaults)

❌ **Avoid:**
- Edge < 2.5% (too small to overcome variance)
- Using default stats without verification
- Betting on every game (wait for clear opportunities)
- Chasing losses with increased stakes

### Managing Bankroll

1. **Never bet more than recommended stake**
   - Calculator uses fractional Kelly (safe)
   - Full Kelly can be volatile

2. **Update odds frequently**
   - Bookmaker odds change during match
   - Re-calculate before placing bet

3. **Track your bets**
   - Save match states when you bet
   - Review later to improve model

## 🔧 Advanced Features

### Markov Chain Breakdown

The calculator shows probabilities at multiple levels:

1. **Point Probabilities** - Based on serve/return stats
2. **Game Probabilities** - Compound from point probabilities
3. **Set Probabilities** - Factor in tiebreak scenarios
4. **Match Probabilities** - Best of 3 or 5 sets calculation

### Probability Visualization

The chart shows:
- **Match Win %** - Overall chance to win
- **Set Win %** - Chance to win current set
- **Game Win % (on serve)** - Chance to hold serve

Compare both players to identify:
- Serving dominance
- Return quality
- Overall match control

### Match History Export

1. Save states throughout the match
2. Click table to view historical probabilities
3. Export to CSV for:
   - Model validation
   - Strategy refinement
   - Bet journal records

## 🎯 Example Live Session

### Australian Open 2026 - Djokovic vs Alcaraz

**Pre-match Setup (11:00 AM):**
```
Player Stats:
- Djokovic: 68% serve, 35% return
- Alcaraz: 65% serve, 32% return

Bookmaker Odds:
- Djokovic: 1.85
- Alcaraz: 2.10

Initial Probabilities:
- Djokovic: 52.4%
- Alcaraz: 47.6%
Edge: -1.7% (no value)
```

**First Set Progress:**

| Score | Prob | Edge | Action |
|-------|------|------|--------|
| 0-0, 0-0 | 52.4% | -1.7% | Wait |
| 1-0, 0-0 (Djokovic) | 53.8% | -0.3% | Wait |
| 2-1, 0-0 (Djokovic) | 55.2% | +1.1% | Wait |
| 3-1, 0-0 (Djokovic) | 58.3% | +4.2% | ✅ **BET $168** |
| 4-2, 0-0 (Djokovic) | 61.7% | +7.6% | ✅ **BET $304** |
| **6-3 SET 1** | 64.9% | +10.8% | ✅ **BET $432** |

**Match Notes:**
```
11:23 AM - Djokovic breaks at 2-1, odds now 1.90 but prob jumped to 58%
11:45 AM - Holds serve 4-2, Alcaraz looking frustrated
12:08 PM - Wins set 6-3, massive value at current odds
```

**Result:**
- Djokovic wins 6-3, 7-5
- 3 value bets placed
- Total profit: $847 (on $1000 total stakes)
- ROI: 84.7%

## ❓ Troubleshooting

### "Probabilities seem wrong"
- ✅ Check player statistics are realistic
- ✅ Verify score entered correctly
- ✅ Ensure correct server selected
- ✅ Update stats if player is injured/tired

### "No value bets showing"
- ✅ Odds may be efficient (accurate)
- ✅ Try updating player stats based on current form
- ✅ Wait for momentum shifts
- ✅ Consider lower edge threshold (>2% instead of >2.5%)

### "Recommended stakes too high"
- ✅ Calculator assumes $10,000 bankroll by default
- ✅ Scale down proportionally to your bankroll
- ✅ Use fractional Kelly (25% of full Kelly)
- ✅ Never bet more than you're comfortable losing

### "Match history not saving"
- ✅ Click "💾 Save Match State" button
- ✅ Check browser console for errors
- ✅ Refresh page to see if state persists
- ✅ Export to CSV before closing browser

## 📚 Additional Resources

### Recommended Reading:
- **Fortune's Formula** by William Poundstone (Kelly Criterion)
- **Beat the Market** by Ed Thorp (Mathematical betting)
- **Tennis Abstract** blog (Player statistics)

### Useful Websites:
- **ATP Stats**: https://www.atptour.com/en/stats
- **Tennis Abstract**: http://www.tennisabstract.com/
- **Flashscore**: https://www.flashscore.com/tennis/
- **The Odds API**: https://the-odds-api.com/

### Further Learning:
- Markov chain theory in tennis
- Kelly criterion mathematics
- Value betting strategies
- Bankroll management

## 🤝 Support

For issues or questions:
1. Check this guide first
2. Review main README.md
3. Check TESTING_GUIDE.md for troubleshooting
4. Open GitHub issue with:
   - Match details
   - Input values
   - Expected vs actual output
   - Screenshots if applicable

## 🔐 Responsible Betting

⚠️ **Important Reminders:**
- Only bet what you can afford to lose
- This is a mathematical tool, not a guarantee
- Past performance doesn't guarantee future results
- Take breaks and set limits
- Seek help if betting becomes problematic

**Gambling Support Resources:**
- National Council on Problem Gambling: 1-800-522-4700
- Gamblers Anonymous: https://www.gamblersanonymous.org/

---

**Built with:** Streamlit, Markov Chain Theory, Kelly Criterion  
**Last Updated:** December 2024  
**Version:** 1.0.0
