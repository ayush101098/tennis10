# ATP Tennis Data Pipeline & Analysis 🎾

Complete data pipeline for ATP tennis matches with predictive modeling capabilities.

## 📊 Project Overview

This project provides a comprehensive data pipeline that:
- **Fetches** ATP match data from tennis-data.co.uk (2020-present)
- **Stores** data in a normalized SQLite database
- **Validates** data quality with automated checks
- **Analyzes** match outcomes, player performance, and betting odds
- **Prepares** data for machine learning models

## 🗄️ Database Schema

### Tables
1. **players** - Unique player registry (517 players)
2. **matches** - Match results and tournament info (11,794 matches)
3. **statistics** - Per-match performance stats (expandable)
4. **odds** - Betting odds from 3 bookmakers (35,265 records)

### Coverage
- **Date Range**: January 2020 - November 2024
- **Surfaces**: Hard, Clay, Grass
- **Series**: Grand Slam, Masters 1000, ATP 500, ATP 250
- **Bookmakers**: Pinnacle, Bet365, Max

## 🚀 Quick Start

### 1. Installation
```bash
# Install required packages
pip install pandas openpyxl matplotlib seaborn requests jupyter
```

### 2. Fetch Data
```bash
# Run the data pipeline (takes ~20 seconds)
python data_pipeline.py
```

**Expected Output:**
```
Total Players: 517
Total Matches: 11,794
Total Odds Records: 35,265
Validation Errors: 0
Date Range: 2020-01-06 to 2024-11-17
```

### 3. Explore Data
```bash
# Open Jupyter notebook
jupyter notebook data_exploration.ipynb
```

## 📁 Files

### Core Files
- **`data_pipeline.py`** - Main data pipeline script
  - Fetches ATP data from tennis-data.co.uk
  - Creates SQLite database with 4 tables
  - Validates data quality
  - Includes logging and error handling

- **`data_exploration.ipynb`** - Comprehensive analysis notebook
  - 14 sections of exploratory data analysis
  - Visualizations for player performance, surfaces, odds
  - Data quality checks
  - Summary statistics

- **`tennis_data.db`** - SQLite database (3.7 MB)
  - Generated automatically by pipeline
  - Can be queried with any SQLite tool

## 📈 Data Available

### Match Information
- Tournament name, date, location, series
- Surface type (Hard/Clay/Grass)
- Round (Finals, Semifinals, etc.)
- Match result (sets and games won)
- Best of 3 or 5 sets

### Player Data
- Player names
- ATP rankings (winner & loser)
- Ranking points

### Betting Odds
- Pinnacle Sports odds
- Bet365 odds
- Maximum market odds
- Implied probabilities & bookmaker margins

## ⚠️ Important Notes

### Statistics Limitation
The free tennis-data.co.uk Excel files **do not include** detailed match statistics like:
- Aces, double faults
- Serve percentages (1st/2nd serve)
- Break points won/saved
- Return points won

**Options to get detailed stats:**
1. Scrape ATP website (requires custom scraper)
2. Use paid APIs (Tennis API, Sportradar)
3. Use OnCourt or other paid data providers

The database schema includes a `statistics` table ready for this data when available.

## 🔍 Sample Queries

### SQL Examples
```sql
-- Top players by wins
SELECT p.player_name, COUNT(*) as wins
FROM matches m
JOIN players p ON m.winner_id = p.player_id
GROUP BY p.player_name
ORDER BY wins DESC
LIMIT 10;

-- Matches on clay surface
SELECT COUNT(*) as clay_matches
FROM matches
WHERE surface = 'Clay';

-- Average odds by bookmaker
SELECT bookmaker, 
       AVG(winner_odds) as avg_winner_odds,
       AVG(loser_odds) as avg_loser_odds
FROM odds
GROUP BY bookmaker;
```

### Python Examples
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('tennis_data.db')

# Load all matches
matches = pd.read_sql_query("SELECT * FROM matches", conn)

# Analyze upsets (lower ranked player wins)
matches['rank_diff'] = matches['loser_rank'] - matches['winner_rank']
upsets = matches[matches['rank_diff'] < 0]
print(f"Upset rate: {len(upsets)/len(matches):.1%}")

conn.close()
```

## 📊 Exploratory Analysis Highlights

From `data_exploration.ipynb`:
- **11,794 total matches** analyzed
- **517 unique players** tracked
- **99.9% odds coverage** (35,265 records)
- **Hard court dominance**: 58.6% of all matches
- **Upset rate**: ~30% (lower-ranked players win)
- **Bookmaker margins**: 3-5% average overround

## 🎯 Next Steps: Building Predictive Models

### Feature Engineering Ideas
1. **Player Features**
   - Recent form (last 5/10 matches)
   - Surface-specific win rate
   - Head-to-head record
   - Current ranking momentum

2. **Match Context**
   - Tournament importance (Grand Slam vs ATP 250)
   - Time of season
   - Best of 3 vs Best of 5

3. **Betting Market Features**
   - Implied probabilities
   - Odds movements (if available)
   - Bookmaker disagreements

### Model Approaches
- **Logistic Regression** - Baseline model
- **Random Forest** - Feature importance analysis
- **XGBoost** - High performance
- **Neural Networks** - Complex patterns

### Evaluation Metrics
- Accuracy on match outcomes
- ROI on betting strategy
- Kelly Criterion for bankroll management
- Calibration curves for probability estimates

## 🛠️ Troubleshooting

### "Module not found" errors
```bash
pip install pandas openpyxl matplotlib seaborn requests
```

### Database is empty
```bash
# Re-run the pipeline
python data_pipeline.py
```

### Notebook won't run
```bash
# Ensure Jupyter is installed
pip install jupyter

# Start Jupyter
jupyter notebook
```

## 📝 Code Structure

```
tennis10/
├── data_pipeline.py          # Main pipeline script
├── data_exploration.ipynb    # Analysis notebook
├── tennis_data.db           # SQLite database (auto-generated)
└── README.md                # This file
```

## 🔄 Updating Data

To fetch the latest data:
```bash
# Delete old database
rm tennis_data.db

# Re-run pipeline (fetches fresh data)
python data_pipeline.py
```

## 📚 Resources

- **Data Source**: [tennis-data.co.uk](http://www.tennis-data.co.uk/)
- **ATP Rankings**: [atptour.com](https://www.atptour.com/en/rankings/singles)
- **SQLite**: [sqlite.org](https://www.sqlite.org/)

## 🎓 Learning Resources

- [Predicting Tennis Matches (Scikit-learn)](https://scikit-learn.org/stable/)
- [Sports Betting Models](https://github.com/betfair/predictive-models)
- [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion)

## ✅ Verified & Working

- ✅ Pipeline successfully fetches data from tennis-data.co.uk
- ✅ Database created with 11,794 matches
- ✅ 517 unique players tracked
- ✅ 35,265 betting odds records
- ✅ Notebook runs without errors
- ✅ All visualizations working
- ✅ Data quality validation passing

## 🚀 Ready for Modeling!

The data pipeline is complete and verified. You now have:
- Clean, validated data
- Comprehensive database
- Exploratory analysis
- Foundation for predictive models

**Next**: Build machine learning models to predict match outcomes and identify profitable betting opportunities! 🎯
