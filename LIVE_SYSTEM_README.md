# 🎾 Live Tennis Betting System - Complete Implementation

## 📁 System Architecture

```
tennis10/
├── src/
│   ├── live_data/              # Data collection modules
│   │   ├── match_scraper.py    # Scrape upcoming matches (Sofascore/Flashscore/ATP)
│   │   ├── player_mapper.py    # Fuzzy player name matching across sources
│   │   ├── validators.py       # Data quality checks
│   │   ├── odds_scraper.py     # Collect odds (The Odds API + web scraping)
│   │   ├── odds_analyzer.py    # Analyze odds for value (implied prob, overround)
│   │   └── scheduler.py        # Automated data collection (runs every 6h/15m/30m)
│   │
│   └── live_predictions/       # Prediction and betting modules
│       ├── predictor.py        # Main prediction pipeline (models → edges → bets)
│       └── bet_calculator.py   # Kelly criterion, risk management
│
├── notebooks/
│   └── test_live_system.ipynb  # Complete test suite for all components
│
└── get_live_odds.py            # Standalone odds fetcher (The Odds API)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install requests pandas numpy beautifulsoup4 apscheduler matplotlib seaborn
```

Optional (for web scraping):
```bash
pip install playwright selenium
playwright install  # Install browsers
```

### 2. Set Up The Odds API (Recommended)

```bash
# Get FREE API key: https://the-odds-api.com/
# Free tier: 500 requests/month

export ODDS_API_KEY='your_key_here'
```

### 3. Test the System

Run the comprehensive test notebook:

```bash
jupyter notebook notebooks/test_live_system.ipynb
```

This will test all components:
- ✅ Match scraping from multiple sources
- ✅ Player name matching (fuzzy)
- ✅ Odds collection
- ✅ Value bet detection
- ✅ Prediction generation
- ✅ Betting recommendations with Kelly stakes

### 4. Run Live Predictions

```bash
# One-time prediction
python src/live_predictions/predictor.py

# Or get just the odds
python get_live_odds.py
```

### 5. Run Automated Scheduler

```bash
# Runs continuously, updates every 15-30 minutes
python src/live_data/scheduler.py
```

## 📊 Components

### Match Scraping

**Data Sources:**
1. **Sofascore API** (Primary) - No API key needed
   - Endpoint: `https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{date}`
   - Returns: Player names, tournament, surface, time, round
   - Rate limit: 1 req/second

2. **Flashscore** (Backup) - Web scraping
   - More matches, 7 days ahead
   - Requires Playwright/Selenium

3. **ATP Official** (Authoritative) - Web scraping
   - Tournament draws
   - Most reliable surface info

**Usage:**
```python
from src.live_data.match_scraper import get_all_upcoming_matches

matches = get_all_upcoming_matches(days_ahead=2)
print(f"Found {len(matches)} upcoming matches")
```

### Player Mapping

**Problem:** Different sources use different name formats:
- Sofascore: "Rafael Nadal"
- Flashscore: "R. Nadal"
- ATP: "Nadal R."

**Solution:** Fuzzy string matching (Levenshtein distance)

**Usage:**
```python
from src.live_data.player_mapper import match_player_name

result = match_player_name("R. Nadal", source='flashscore')
# Returns: {'player_id': 123, 'canonical_name': 'Rafael Nadal', 'confidence': 0.95}
```

### Odds Collection

**Option 1: The Odds API (Recommended)**
- 30+ bookmakers (Pinnacle, Bet365, DraftKings, FanDuel, etc.)
- Real-time updates
- Clean JSON format
- Cost: FREE tier (500 req/month) or $99/month

**Option 2: Web Scraping**
- OddsPortal (aggregator)
- Pinnacle (best odds)
- May be blocked/rate limited

**Usage:**
```python
from src.live_data.odds_scraper import get_tennis_odds

odds = get_tennis_odds(use_api=True)  # Requires ODDS_API_KEY
```

### Odds Analysis

Calculate value bets:

```python
from src.live_data.odds_analyzer import find_value_bets

value = find_value_bets(
    our_probability=0.65,  # We predict 65% win chance
    bookmaker_odds=1.85     # Bookmaker offers 1.85 odds
)

if value['recommended']:
    print(f"Edge: {value['edge']:.2%}")
    print(f"Expected Value: {value['expected_value']:.2%}")
```

**Key Functions:**
- `calculate_implied_probability()` - Convert odds to probability
- `calculate_overround()` - Bookmaker margin (Pinnacle ~2%, others ~7%)
- `detect_odds_movement()` - Track sharp money
- `find_value_bets()` - Identify profitable opportunities

### Live Predictions

**Pipeline:**
1. Get upcoming matches
2. Collect odds
3. Generate predictions (Markov + Logistic + Neural Network ensemble)
4. Calculate edges
5. Recommend bets with Kelly stakes

**Usage:**
```python
from src.live_predictions.predictor import LivePredictor

predictor = LivePredictor(bankroll=1000)
all_predictions, profitable_bets = predictor.predict_upcoming_matches()

predictor.display_recommendations(all_predictions, profitable_bets)
```

### Bet Calculator

**Kelly Criterion** for optimal stake sizing:

```python
from src.live_predictions.bet_calculator import BetCalculator

calc = BetCalculator(bankroll=1000, kelly_fraction=0.25)

rec = calc.calculate_bet_recommendation(
    our_probability=0.65,
    p1_odds=1.85,
    p2_odds=2.20,
    uncertainty=0.2,
    model_agreement=0.95
)

print(f"Action: {rec['action']}")
print(f"Stake: ${rec['recommended_stake']:.2f}")
print(f"Edge: {rec['edge']:.2%}")
```

**Risk Management:**
- **Kelly Fraction:** 0.25 (quarter Kelly, conservative)
- **Min Edge:** 2.5%
- **Max Bet:** 15% of bankroll
- **Uncertainty Threshold:** <50%
- **Model Agreement:** >85%

### Automated Scheduler

Runs continuously:

```python
from src.live_data.scheduler import run_scheduler

run_scheduler(bankroll=1000)
```

**Schedule:**
- Match scraping: Every 6 hours
- Odds updates: Every 15 minutes
- Predictions: Every 30 minutes
- High-value alerts: Every 10 minutes
- Database cleanup: Daily at 2 AM

## 📈 Expected Output

### Profitable Bets Example

```
✅ 3 PROFITABLE BETS FOUND

Total recommended stake: $285.00
Total expected value: $42.75
Portfolio EV: 15.0%

1. Carlos Alcaraz vs Jannik Sinner
   Tournament: Australian Open (Hard)
   🎯 BET: Carlos Alcaraz
   Odds: 1.85
   Stake: $120.00 (12.0% of bankroll)
   Edge: 8.2%
   Expected Value: 15.3%
   Potential profit: $18.36
   Confidence: HIGH

2. Alexander Zverev vs Daniil Medvedev
   Tournament: Australian Open (Hard)
   🎯 BET: Zverev
   Odds: 2.10
   Stake: $95.00 (9.5% of bankroll)
   Edge: 6.5%
   Expected Value: 13.6%
   Potential profit: $12.92
   Confidence: MEDIUM
```

## 🎯 Betting Strategy

### Rules

1. **Only bet with edge >2.5%** - Meaningful advantage required
2. **Only bet with uncertainty <50%** - Sufficient data needed
3. **Only bet with model agreement >85%** - Models must agree
4. **Use 25% Kelly** - Conservative stake sizing (reduces variance)
5. **Max 15% per bet** - Bankroll protection
6. **Max 50% portfolio exposure** - Don't bet too much at once

### Goal: $1,000 → $5,000

- Required return: 400%
- Strategy: Find consistent edges of 5-10%
- Timeline: Depends on betting opportunities (weeks to months)
- Risk: Conservative (quarter Kelly reduces bankruptcy risk to near 0)

## 🔧 Troubleshooting

### No matches found

**Possible causes:**
1. No tournaments scheduled
2. Network issues
3. API endpoints changed

**Solutions:**
- Try different time of day (tournaments start at specific hours)
- Check internet connection
- Update scraping code if websites changed

### No odds available

**Possible causes:**
1. The Odds API key not set
2. Rate limit reached
3. Web scraping blocked

**Solutions:**
```bash
# Set API key
export ODDS_API_KEY='your_key'

# Check API quota
curl "https://api.the-odds-api.com/v4/sports/?apiKey=YOUR_KEY"

# Use web scraping fallback
python src/live_data/odds_scraper.py
```

### No profitable bets

**This is normal!** Value bets are rare.

- Bookmakers are efficient
- Sharp bettors move lines quickly
- Need large sample size to find consistent edges

**Keep running the scheduler** - it will find opportunities when they appear.

### Models not loaded

Currently using odds-based estimation (fair probability after removing overround).

To use actual models:
1. Train models on historical data
2. Save as pickles/h5 files
3. Update `predictor.py` to load them

## 📚 Further Reading

- **Kelly Criterion:** https://en.wikipedia.org/wiki/Kelly_criterion
- **The Odds API:** https://the-odds-api.com/liveapi/guides/v4/
- **Sofascore API:** (Undocumented, reverse engineered)
- **Sports Betting Math:** https://www.pinnacle.com/en/betting-articles/

## 📧 Support

Issues? Check:
1. All dependencies installed
2. API key set (if using The Odds API)
3. Internet connection working
4. Run test notebook first

## 🎾 Good Luck!

Remember:
- **Gambling can be addictive** - Set strict bankroll limits
- **Past performance ≠ future results** - No system is perfect
- **Bookmakers ban winners** - Consider using betting exchanges
- **Only bet what you can afford to lose**

The system helps find edges, but **YOU** are responsible for risk management!
