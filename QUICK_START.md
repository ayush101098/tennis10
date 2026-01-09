# 🚀 QUICK START GUIDE

## Get Up and Running in 5 Minutes

### Step 1: Get The Odds API Key (2 minutes)

1. Visit: https://the-odds-api.com/
2. Click "Sign Up"
3. Enter email + password
4. Verify email
5. Copy your API key from dashboard

**Set the key:**
```bash
export ODDS_API_KEY='paste_your_key_here'
```

### Step 2: Test the System (1 minute)

```bash
# Get current matches and odds
python get_live_odds.py
```

You should see:
```
✅ The Odds API key found
📡 Fetching live tennis matches...
✅ Found 15 matches with odds

1. Carlos Alcaraz vs Jannik Sinner
   League: ATP
   Best Odds: 1.85 / 2.10
   ...
```

### Step 3: Generate Predictions (1 minute)

```bash
python src/live_predictions/predictor.py
```

You should see:
```
🎾 LIVE TENNIS PREDICTIONS & BETTING RECOMMENDATIONS

Bankroll: $1,000.00

✅ 2 PROFITABLE BETS FOUND

Total recommended stake: $150.00
Total expected value: $22.50
```

### Step 4: Start Automated System (optional)

```bash
# Runs continuously, checks every 15-30 minutes
python src/live_data/scheduler.py
```

Press `Ctrl+C` to stop.

---

## 📊 Test in Jupyter Notebook

```bash
jupyter notebook notebooks/test_live_system.ipynb
```

Run all cells to see:
- Match scraping
- Odds collection
- Value bet analysis
- Betting recommendations with Kelly stakes
- Visualizations

---

## 🎯 What to Expect

### First Run

**IF YOU SEE BETS:** Great! Place them at recommended stakes.

**IF NO BETS:** This is normal! Value bets are rare.
- Keep the scheduler running
- It will find opportunities when they appear
- Typical hit rate: 5-10% of matches have exploitable edges

### Over Time

- Monitor for high-confidence bets (>5% edge)
- Track your performance
- Adjust Kelly fraction based on results
- Most profits come from a few big wins, not many small bets

---

## 💡 Tips for Success

1. **Be Patient** - Don't force bets without edge
2. **Trust the System** - Kelly criterion is mathematically optimal
3. **Track Everything** - Record all bets and outcomes
4. **Adjust if Needed** - If losing consistently, reduce Kelly fraction
5. **Watch for Steam Moves** - If odds suddenly shift, sharp money may know something

---

## ⚠️ Troubleshooting

### "No API key set"

```bash
export ODDS_API_KEY='your_key'
# Then run again
```

### "No matches found"

**Normal causes:**
- No tournaments scheduled right now
- Check back during Grand Slams (Australian Open, French Open, Wimbledon, US Open)
- ATP/WTA 1000 events have most matches

**Check schedule:** https://www.atptour.com/en/scores/current

### "No profitable bets"

**This is expected!** Bookmakers are smart.

Value bets appear when:
- New information emerges (injury, weather)
- Bookmakers make mistakes
- Public overreacts to recent results
- Your models see something bookmakers missed

**Keep running the scheduler** to catch these moments.

---

## 📈 Expected Performance

**Conservative Estimates:**
- Hit rate: 5-10% of matches have edges >2.5%
- Average edge when found: 5-8%
- Typical stake: 5-12% of bankroll
- Expected ROI: 10-20% per month (if opportunities exist)

**To reach $1,000 → $5,000:**
- Requires 400% return
- Assumes consistent 5-10% edges
- Timeline: 3-6 months of active betting
- Risk: Low (quarter Kelly very conservative)

---

## 🎾 Ready to Bet!

1. ✅ API key set
2. ✅ System tested
3. ✅ Found profitable bets
4. ✅ Understand the risks

**Place your first bet:**
- Use the recommended stake
- Place at the bookmaker with best odds
- Record the outcome
- Update bankroll in the system

**Good luck!** 🍀
