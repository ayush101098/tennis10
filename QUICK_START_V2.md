# Quick Start Guide - Live Calculator V2

## 🎯 Access the New Calculator

**Dashboard**: http://localhost:8501
**Page**: Navigate to **"🎯 Live Calculator V2"** in sidebar

## ✨ Three Main Improvements

### 1. NO MORE DATA LOSS! 
```
Track points → Refresh page → Click "📂 Load Match" → Everything restored!
```

### 2. SAVE BETS TO DATABASE
```
See value bet → Click "💾 SAVE THIS BET" → Bet saved → View in "My Selected Bets" tab
```

### 3. COMPACT LAYOUT
```
All inputs in ONE row
Everything visible in 4 TABS
No endless scrolling!
```

## 🚀 Quick Workflow

### Starting a New Match:
1. Top row: Enter "Player 1 Name" | "Player 2 Name" | Select "Surface"
2. (Optional) Click "⚙️ Player Stats & Advanced Parameters" → Set serve/return %
3. Tab 1: "🎾 Live Tracker" → Click winning player after each point
4. Data AUTO-SAVES after every point!

### Loading Saved Match:
1. Enter same player names as before
2. Click "📂 Load Match" button (top right)
3. ✅ Everything restored!
   - Current score
   - Probability history
   - Momentum tracking
   - All statistics

### Saving Value Bets:
1. Tab 2: "📊 Probability & Bets"
2. Enter bookmaker odds for each player
3. Model shows: ✅ VALUE BET alerts with green background
4. Click "💾 SAVE THIS BET" button
5. Bet saved with:
   - Current score
   - Probability
   - Edge
   - Expected Value
   - Recommended stake
   - Timestamp

### Viewing Your Bets:
1. Tab 3: "💰 My Selected Bets"
2. Table shows ALL saved bets:
   - Time placed
   - Selection
   - Odds
   - Edge & EV
   - Stake amount
   - Score when placed
3. Bottom shows: Total Bets & Total Stake

### Analytics:
1. Tab 4: "📈 Analytics"
2. View:
   - Probability evolution chart
   - Momentum shifts
   - Match trends

## 📊 Layout Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Top Bar                                                     │
│ [Player 1 Input] [Player 2 Input] [Surface] [📂 Load Match]│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Collapsible Section (click to expand)                      │
│ ⚙️ Player Stats & Advanced Parameters                      │
│ [Serve %] [Return %] [Momentum] [Surface] [Clutch] [BP Def]│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ TABS                                                        │
├───────┬──────────┬─────────────┬──────────────┐
│🎾Live │📊Prob&Bets│💰My Bets   │📈Analytics   │
├───────┴──────────┴─────────────┴──────────────┤
│                                                            │
│ TAB 1: Live Tracker                                        │
│ [🟢 PLAYER 1 WINS POINT] [🔵 PLAYER 2 WINS POINT] [🔄 RST]│
│                                                            │
│ Current Score:                                             │
│ Player 1: 1 | 3 | 40        Player 2: 1 | 2 | 30         │
│                                                            │
└─────────────────────────────────────────────────────────────┘

TAB 2: Probability & Bets
┌─────────────────────────────────────────────────────────────┐
│ Win Probability                                             │
│ ┌────────────────────┐  ┌────────────────────┐            │
│ │ Player 1           │  │ Player 2           │            │
│ │ 67.3%              │  │ 32.7%              │            │
│ │ Momentum: 🟢 +4.2% │  │ Momentum: 🔴 -4.2% │            │
│ └────────────────────┘  └────────────────────┘            │
│                                                            │
│ Recommended Value Bets:                                    │
│ ┌────────────────────────────────────────────────────────┐│
│ │ ✅ VALUE BET: Player 1                                ││
│ │ 💰 Recommended Stake: $120                            ││
│ │ 📈 Expected Value: +8.5%                              ││
│ │ 🎯 Edge: +5.2%                                        ││
│ │ [💾 SAVE THIS BET] ← CLICK HERE!                      ││
│ └────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

TAB 3: My Selected Bets
┌─────────────────────────────────────────────────────────────┐
│ Table of Saved Bets:                                        │
│ ┌──────┬──────┬──────────┬─────┬─────┬─────┬───────┬─────┐│
│ │Time  │Type  │Selection │Odds │Edge │EV   │Stake$ │Score││
│ ├──────┼──────┼──────────┼─────┼─────┼─────┼───────┼─────┤│
│ │10:15 │Match │Djokovic  │1.85 │+5.2%│+8.5%│$120   │1-1  ││
│ │10:32 │Match │Alcaraz   │2.10 │+3.8%│+6.2%│$80    │1-2  ││
│ └──────┴──────┴──────────┴─────┴─────┴─────┴───────┴─────┘│
│ Total Bets: 2 | Total Stake: $200                          │
└─────────────────────────────────────────────────────────────┘

TAB 4: Analytics
┌─────────────────────────────────────────────────────────────┐
│ Probability Evolution Chart                                 │
│     %                                                       │
│ 100 │                                                       │
│  80 │     ▲                                                │
│  60 │    ╱ ╲      ╱─────                                   │
│  40 │   ╱   ╲    ╱                                         │
│  20 │  ╱     ╲  ╱                                          │
│   0 └────────────────────────────────────────────         │
│     0   10   20   30   40   50  (Points)                   │
│                                                            │
│ 🔥 Key Momentum Shifts:                                    │
│ Point 23: Djokovic won → 📈 4.5% shift                     │
│ Point 38: Alcaraz won → 📉 3.8% shift                      │
└─────────────────────────────────────────────────────────────┘
```

## 💾 Data Persistence Example

### Scenario: You're tracking a live match but need to take a break

**What you do:**
1. Track 50 points (clicking winner after each)
2. Close browser OR refresh page
3. Come back later
4. Enter same player names
5. Click "📂 Load Match"

**What gets restored:**
- ✅ Score: 1-1 sets, 3-2 games, 30-15 points
- ✅ Probability history: All 50 points
- ✅ Momentum: Current +4.2%
- ✅ Breaks: 2-1
- ✅ Analytics: Full probability chart
- ✅ Match state: Ready to continue from point 51

**What you DON'T need to do:**
- ❌ Re-enter scores manually
- ❌ Rebuild probability history
- ❌ Recalculate anything
- ❌ Remember where you left off

## 🎯 Bet Tracking Example

### Scenario: Model shows value bet, you want to save it

**Old way (before):**
1. See value bet recommendation
2. Write down on paper or separate app
3. Easy to forget details
4. No tracking system

**New way (now):**
1. See green "✅ VALUE BET" alert
2. Click "💾 SAVE THIS BET" button
3. Done! Saved to database with:
   - Player names
   - Odds (1.85)
   - Probability (67.3%)
   - Edge (+5.2%)
   - EV (+8.5%)
   - Recommended stake ($120)
   - Current score (1-1, 3-2)
   - Exact timestamp

**View later:**
- Go to "My Selected Bets" tab
- See complete table
- Know exactly what you bet and why
- Track total exposure

## 🔥 Pro Tips

1. **Always load first**: When returning to a match, click Load before doing anything
2. **Save interesting bets**: Even if you don't place them, save for analysis
3. **Use analytics tab**: Check momentum shifts to time your bets
4. **Advanced params matter**: Small adjustments (0.1-0.2) can shift probabilities 2-3%
5. **Trust the auto-save**: Don't worry about losing data, every point is saved

## ⚠️ Important Notes

- **Match persistence**: Uses player names to identify matches
- **Multiple matches**: Can track different matches by changing names
- **Database location**: `tennis_betting.db` in project root
- **Auto-save timing**: Triggers after every point tracked
- **Load button**: Only appears when match exists for those players

## 🆘 Troubleshooting

**"Load Match" shows "No saved match found":**
- Player names don't match exactly (check spelling/capitalization)
- No points tracked yet with these names
- Match was marked as finished

**Bet didn't save:**
- Check "My Selected Bets" tab to confirm
- Make sure you clicked the "💾 SAVE THIS BET" button
- Button only appears next to green value bet alerts

**Page looks different:**
- Make sure you're on "Live Calculator V2" not old version
- Old version: "Live Calculator" (page 6)
- New version: "Live Calculator V2" (page 7)

**Data not loading:**
- Wait for page to fully load
- Click "Load Match" AFTER entering player names
- Check that surface matches (Hard/Clay/Grass)

## 🎓 Comparison

| Feature | Old Calculator | New V2 Calculator |
|---------|---------------|-------------------|
| Data persistence | ❌ Lost on refresh | ✅ Saves to DB |
| Bet tracking | ❌ No tracking | ✅ Full tracking |
| Layout | ❌ 3000px scroll | ✅ 1200px tabs |
| Save bet | ❌ Manual notes | ✅ One click |
| Load match | ❌ Re-enter all | ✅ One click |
| Probability history | ⚠️ Lost on refresh | ✅ Persisted |
| Multiple matches | ❌ Only current | ✅ All saved |

## 📞 Quick Reference

**One-line setup**: Enter names → Optionally adjust stats → Start tracking
**One-click save**: See value bet → Click "💾 SAVE THIS BET"
**One-click load**: Enter names → Click "📂 Load Match"
**Zero data loss**: Everything auto-saves after each point

---

**Ready to use!** Navigate to "🎯 Live Calculator V2" in your dashboard and start tracking! 🚀
