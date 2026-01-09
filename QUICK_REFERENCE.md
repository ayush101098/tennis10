# 🎾 Tennis ML Betting System - Quick Reference

## 🚀 Quick Start Commands

### Launch Dashboard
```bash
./launch_dashboard.sh
# OR
streamlit run dashboard/streamlit_app.py
```
**Access:** http://localhost:8501

### Start Data Collection
```bash
source setup_env.sh
python src/live_data/scheduler.py &
```

### Manual Prediction Run
```bash
source setup_env.sh
python src/live_predictions/predictor.py
```

### Test System
```bash
python test_dashboard.py
```

---

## 📂 File Organization

```
tennis10/
├── 🎨 DASHBOARD FILES
│   ├── dashboard/streamlit_app.py      # Main dashboard
│   ├── dashboard/data_loader.py        # Database queries
│   ├── dashboard/pages/                # 5 pages
│   └── dashboard/components/           # Reusable components
│
├── 🤖 CORE WORKFLOW
│   ├── src/live_data/                  # Match & odds scraping
│   └── src/live_predictions/           # Predictions & betting
│
├── 📚 DOCUMENTATION
│   ├── README.md                       # Project overview
│   ├── dashboard/README.md             # Dashboard guide
│   ├── WORKFLOW_VERIFICATION.md        # System status
│   ├── INTEGRATION_CHECKLIST.md        # Verification steps
│   └── DASHBOARD_SUMMARY.md            # Creation summary
│
└── 🔧 UTILITIES
    ├── launch_dashboard.sh             # Quick launcher
    ├── setup_env.sh                    # API key setup
    └── test_dashboard.py               # System testing
```

---

## 🎯 Dashboard Pages

| Page | Purpose | Key Features |
|------|---------|-------------|
| **📊 Live Predictions** | Find betting opportunities | Match cards, odds, recommendations |
| **📈 Model Performance** | Track ROI and accuracy | PnL charts, calibration, metrics |
| **💰 Betting History** | Manage bets | Active/settled bets, performance |
| **⚙️ Settings** | Configure system | Bankroll, API key, automation |
| **🔍 Player Analysis** | Research players | Stats, H2H, betting insights |

---

## ⚙️ Key Settings

| Setting | Default | Recommended | Purpose |
|---------|---------|-------------|---------|
| Bankroll | $1,000 | Your amount | Total betting capital |
| Kelly Fraction | 0.25 | 0.25 | Conservative bet sizing |
| Min Edge | 2.5% | 2.5-5% | Bet quality filter |
| Max Bet % | 15% | 10-15% | Risk management |
| Min Confidence | Medium | Medium/High | Quality threshold |
| API Key | - | Your key | The Odds API access |

---

## 📊 Database Tables

| Table | Purpose | Updated By |
|-------|---------|-----------|
| `upcoming_matches` | Future matches | Match scraper (6h) |
| `live_odds` | Best odds | Odds scraper (15min) |
| `predictions` | Model outputs | Predictor (30min) |
| `bets` | Bet tracking | Dashboard + User |
| `bankroll_history` | Performance | Daily calculation |
| `player_mappings` | Name resolution | Player mapper |

---

## 🔄 Scheduler Jobs

| Job | Frequency | Purpose |
|-----|-----------|---------|
| Match Scraper | 6 hours | Fetch upcoming matches |
| Odds Updater | 15 minutes | Get latest odds |
| Prediction Generator | 30 minutes | Calculate predictions |
| Alert Checker | 10 minutes | Find high-value bets |
| Cleanup | Daily 2 AM | Database maintenance |

---

## 🎨 Component Library

### Match Cards (`match_card.py`)
- `render_match_card()` - Standard card view
- `render_detailed_match_view()` - Full details with tabs
- `render_compact_match_row()` - Table row format

### Charts (`charts.py`)
- `create_pnl_chart()` - Cumulative profit/loss
- `create_drawdown_chart()` - Risk analysis
- `create_calibration_plot()` - Model accuracy
- `create_edge_distribution_chart()` - Opportunity distribution
- `create_roi_by_confidence_chart()` - Performance breakdown
- Plus 3 more chart types

### Tables (`tables.py`)
- `render_predictions_table()` - Match predictions
- `render_bets_table()` - Betting history
- `render_performance_summary()` - Statistics
- `render_model_performance_table()` - Model comparison
- Plus 2 more table types

---

## 🔑 API Configuration

**Provider:** The Odds API (the-odds-api.com)  
**Your Key:** a0292044f825f2b560225751fd782851  
**Tier:** Free (500 requests/month)  
**Usage:** ~16 requests/day with default schedule

**Set API Key:**
```bash
# Method 1: Environment variable
export ODDS_API_KEY='your_key'

# Method 2: Edit setup_env.sh
echo "export ODDS_API_KEY='your_key'" > setup_env.sh

# Method 3: Dashboard Settings page
# Go to Settings → API Configuration → Enter key → Save
```

---

## 📈 Success Metrics to Track

| Metric | Target | Where to Find |
|--------|--------|---------------|
| ROI | >3% monthly | Model Performance page |
| Win Rate | >60% | Model Performance page |
| Sharpe Ratio | >0.5 | Model Performance page |
| Max Drawdown | <20% | Model Performance page |
| Bets/Month | 20-50 | Betting History page |
| Avg Edge | >4% | Live Predictions page |

---

## 🐛 Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| "No matches available" | Normal during off-season (wait for tournaments) |
| "API key invalid" | Update key in Settings → API Configuration |
| Dashboard won't start | Run: `pip install streamlit plotly` |
| "Could not load metrics" | Database empty → normal if no bets yet |
| Charts not rendering | Check Plotly: `pip install plotly` |
| Scheduler not running | Start: `python src/live_data/scheduler.py &` |

---

## 📞 Help Resources

| Resource | Location |
|----------|----------|
| **Dashboard Guide** | `dashboard/README.md` |
| **System Verification** | `WORKFLOW_VERIFICATION.md` |
| **Integration Checklist** | `INTEGRATION_CHECKLIST.md` |
| **Test System** | `python test_dashboard.py` |
| **API Docs** | https://the-odds-api.com/liveapi/guides/v4/ |
| **Streamlit Docs** | https://docs.streamlit.io |

---

## ⚡ Power User Tips

1. **Keyboard Shortcuts**
   - `Ctrl+C` in terminal: Stop dashboard/scheduler
   - Browser `Ctrl+R`: Refresh dashboard manually
   - `Esc`: Close modals in dashboard

2. **Data Management**
   - Export bets weekly: Betting History → Export CSV
   - Backup database: Settings → Export Database
   - Clear cache: Settings → Clear Cache

3. **Optimization**
   - Reduce scheduler frequency to save API calls
   - Use confidence filters to focus on best bets
   - Export data regularly for external analysis

4. **Monitoring**
   - Check logs: `tail -f logs/system.log`
   - Monitor API usage in Settings page
   - Review model agreement before betting

---

## 🎯 Recommended Workflow

### Daily Routine
1. ✅ Open dashboard: `./launch_dashboard.sh`
2. ✅ Check Live Predictions for opportunities
3. ✅ Review Model Performance metrics
4. ✅ Confirm any recommended bets
5. ✅ Update bet results in Betting History

### Weekly Review
1. 📊 Export betting history to CSV
2. 📈 Review ROI and win rate trends
3. ⚙️ Adjust settings if needed
4. 🔄 Check API usage remaining
5. 💾 Create database backup

### Monthly Analysis
1. 📉 Analyze drawdown periods
2. 🎯 Review strategy effectiveness
3. 📊 Compare model performances
4. 💰 Calculate actual vs expected returns
5. 🔧 Fine-tune parameters

---

## 🏁 Launch Checklist

Before first use:

- [ ] Virtual environment activated
- [ ] All dependencies installed (`test_dashboard.py`)
- [ ] API key configured (`setup_env.sh`)
- [ ] Settings configured (Settings page)
- [ ] Database initialized (auto-created)
- [ ] Documentation reviewed (`dashboard/README.md`)

Ready to go:

- [ ] Dashboard launched and accessible
- [ ] All pages loading correctly
- [ ] Can navigate between pages
- [ ] Filters work in sidebar
- [ ] Charts render properly

When tennis season starts:

- [ ] Scheduler running in background
- [ ] Matches appearing in Live Predictions
- [ ] Odds updating every 15 minutes
- [ ] Predictions generating automatically
- [ ] Alerts configured (optional)

---

## 🎉 You're All Set!

**System Status:** ✅ Production Ready  
**Files Created:** 14 new files  
**Components:** 17 reusable components  
**Features:** 30+ dashboard features  
**Documentation:** Complete  
**Testing:** All passed  

**Launch command:**
```bash
./launch_dashboard.sh
```

**Visit:** http://localhost:8501

---

**Last Updated:** January 9, 2026  
**Version:** 1.0.0  
**Status:** 🚀 Ready to Win! 🎾
