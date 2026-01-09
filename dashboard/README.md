# Tennis ML Betting Dashboard 🎾

Professional web interface for monitoring tennis match predictions and managing your betting portfolio.

## 🚀 Quick Start

### Launch the Dashboard

```bash
# From the project root
streamlit run dashboard/streamlit_app.py
```

The dashboard will open at: **http://localhost:8501**

### First-Time Setup

1. **Configure API Key**
   - Go to **Settings** page (⚙️)
   - Enter your The Odds API key
   - Click "Test API Key" to verify
   - Save settings

2. **Set Betting Parameters**
   - Configure your bankroll
   - Set Kelly fraction (default: 0.25 for conservative)
   - Define minimum edge threshold (default: 2.5%)
   - Set confidence levels

3. **Start Data Collection**
   ```bash
   # In a separate terminal
   source setup_env.sh
   python src/live_data/scheduler.py
   ```

## 📱 Dashboard Structure

### Main Dashboard (Home)
- **Real-time metrics**: Bankroll, ROI, active bets
- **Quick navigation** to all pages
- **System status** overview
- **Global filters** in sidebar

### 📊 Live Predictions
- **Next 24/48 hours** match views
- **Recommended bets** with high confidence
- **Match cards** with:
  - Win probabilities
  - Best odds
  - Edge calculations
  - Model agreement
  - One-click bet placement
- **Multiple views**: Cards, Table, Detailed
- **Edge distribution** charts

### 📈 Model Performance
- **Key metrics**: ROI, Win Rate, Log Loss, Sharpe Ratio
- **Cumulative PnL** chart with drawdown analysis
- **Model comparison** radar charts
- **Calibration curves** to assess prediction accuracy
- **ROI by confidence** level analysis
- **Historical performance** trends

### 💰 Betting History
- **Active bets** tracking
- **Settled bets** history with filters
- **Bet confirmation** workflow
- **Performance summaries**
- **Export to CSV** functionality
- **Bankroll status** monitoring

### ⚙️ Settings
- **Betting parameters**: 
  - Bankroll amount
  - Kelly fraction
  - Min edge, max bet size
  - Confidence thresholds
- **API configuration**:
  - The Odds API key
  - API test utility
- **Automation**:
  - Auto-betting toggle
  - Scheduler intervals
- **Notifications**:
  - Email alerts
  - Slack webhooks
- **Data management**:
  - Cache clearing
  - Database export
  - Backup creation

### 🔍 Player Analysis
- **Player search** and profiles
- **Surface-specific** statistics
- **Recent form** tracking (last 10 matches)
- **Serve/Return** statistics
- **Head-to-head** records
- **Betting insights** per player
- **Top players** rankings

## 🎨 Features

### Real-Time Updates
- Auto-refresh every 15 minutes (optional)
- Live bet status tracking
- Dynamic odds updates
- Real-time model predictions

### Smart Filtering
- Confidence level (High/Medium/Low)
- Surface type (Hard/Clay/Grass/Indoor)
- Minimum edge threshold
- Time windows (24h/48h/1week)

### Interactive Visualizations
- Plotly charts for all analytics
- Responsive design (mobile-friendly)
- Color-coded confidence levels
- Edge distribution histograms
- Calibration curves
- Drawdown analysis

### Bet Management
- One-click bet placement
- Confirmation workflow
- Stake adjustment
- Potential profit calculator
- Active/Settled bet tracking
- CSV export for record-keeping

## 🗂️ File Structure

```
dashboard/
├── streamlit_app.py          # Main entry point
├── data_loader.py             # Database queries & caching
├── components/
│   ├── __init__.py
│   ├── match_card.py          # Match display components
│   ├── charts.py              # Visualization components
│   └── tables.py              # Table components
└── pages/
    ├── 1_📊_Live_Predictions.py
    ├── 2_📈_Model_Performance.py
    ├── 3_💰_Betting_History.py
    ├── 4_⚙️_Settings.py
    └── 5_🔍_Player_Analysis.py
```

## 💾 Data Flow

1. **Match Scraper** → Upcoming matches from Sofascore/Flashscore/ATP
2. **Odds Scraper** → Best odds from The Odds API
3. **Predictor** → Ensemble model predictions
4. **Bet Calculator** → Kelly Criterion stake sizing
5. **Dashboard** → Real-time display & user interaction
6. **Database** → SQLite storage for history

## 🔧 Configuration

### Environment Variables
```bash
# Set in setup_env.sh
export ODDS_API_KEY='your_api_key_here'
```

### Settings File
Located at: `settings.json` (auto-created)

```json
{
  "bankroll": 1000.0,
  "kelly_fraction": 0.25,
  "min_edge": 0.025,
  "max_bet_pct": 0.15,
  "min_confidence": "medium",
  "min_model_agreement": 0.85,
  "auto_bet": false,
  "notifications_enabled": false
}
```

## 📊 Database Schema

### Tables
- `upcoming_matches` - Scraped match data
- `live_odds` - Real-time odds from bookmakers
- `predictions` - Model predictions (Markov, LR, NN, Ensemble)
- `bets` - Placed bets (active & settled)
- `bankroll_history` - Daily bankroll tracking

## 🎯 Usage Examples

### View High-Confidence Bets
1. Go to **Live Predictions**
2. Filter: Confidence = "High"
3. Review recommended bets in top section
4. Click "Place Bet" on desired match

### Track Performance
1. Go to **Model Performance**
2. Select time period (7/30/90 days)
3. Review PnL chart and drawdown
4. Check calibration curve for model accuracy

### Manage Active Bets
1. Go to **Betting History**
2. View active bets section
3. Monitor match start times
4. Track potential profits

### Configure System
1. Go to **Settings**
2. Adjust betting parameters
3. Test and save API key
4. Enable/disable automation
5. Set up notifications

## 🚨 Important Notes

### Automatic Betting
⚠️ **USE WITH CAUTION!** When enabled:
- System places bets automatically
- No confirmation required
- Respects all configured thresholds
- Monitor closely for unexpected behavior

### API Limits
- Free tier: 500 requests/month
- ~16 requests/day budget
- System optimized to stay within limits
- Scheduler intervals configured accordingly

### Off-Season Behavior
- Dashboard displays "No matches" during off-season
- Database queries return empty results (normal)
- System activates automatically when matches appear
- Check back during major tournaments

## 🔗 Integration

### With Existing Workflow
```bash
# 1. Start scheduler (data collection)
source setup_env.sh
python src/live_data/scheduler.py &

# 2. Launch dashboard (user interface)
streamlit run dashboard/streamlit_app.py
```

### With Live Predictor
```python
from src.live_predictions.predictor import LivePredictor

predictor = LivePredictor(bankroll=1000)
predictions, bets = predictor.predict_upcoming_matches()

# Dashboard automatically loads these predictions
```

## 📈 Performance Tips

1. **Cache Management**
   - Dashboard caches data for 60 seconds
   - Clear cache in Settings if data seems stale
   - Auto-refresh keeps data current

2. **Database Optimization**
   - Regular cleanup via scheduler (daily 2 AM)
   - Export database weekly for backups
   - Monitor database size (check Settings)

3. **API Efficiency**
   - Odds update every 15 min (not real-time)
   - Match scraping every 6 hours
   - Predictions every 30 minutes
   - Adjust intervals in Settings if needed

## 🐛 Troubleshooting

### "No matches available"
- **Cause**: Off-season or no data yet
- **Solution**: Wait for tennis season, or run manual scraper

### "Could not load metrics"
- **Cause**: Empty database or connection error
- **Solution**: Ensure database exists, check file permissions

### "API key invalid"
- **Cause**: Expired or incorrect key
- **Solution**: Get new key from the-odds-api.com

### Dashboard won't start
- **Cause**: Missing dependencies
- **Solution**: `pip install streamlit plotly`

## 📚 Additional Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [The Odds API Docs](https://the-odds-api.com/liveapi/guides/v4/)
- [Plotly Charts](https://plotly.com/python/)

## 🎓 Best Practices

1. **Start Conservative**
   - Use 0.25 Kelly fraction
   - High confidence only
   - Min 2.5% edge

2. **Monitor Regularly**
   - Check dashboard daily
   - Review settled bets
   - Adjust strategy based on performance

3. **Keep Records**
   - Export betting history weekly
   - Track ROI trends
   - Document strategy changes

4. **Test First**
   - Paper trade for 2 weeks
   - Verify edge calculations
   - Build confidence in system

## 🏆 Success Metrics

Track these KPIs:
- **ROI**: Target 3-5% monthly
- **Win Rate**: Target 65%+ 
- **Sharpe Ratio**: Target 0.5+
- **Max Drawdown**: Keep under 20%
- **Bet Count**: 50+ for statistical significance

---

**Built with:** Python • Streamlit • Plotly • SQLite • Machine Learning

**Version:** 1.0.0

**Last Updated:** January 2026
