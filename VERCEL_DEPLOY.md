# Tennis Betting System - Vercel Deployment

## Deploy to Vercel

1. **Install Vercel CLI:**
```bash
npm install -g vercel
```

2. **Login to Vercel:**
```bash
vercel login
```

3. **Deploy:**
```bash
vercel --prod
```

## Configuration

The project is configured with:
- `vercel.json` - Vercel deployment settings
- `requirements.txt` - Python dependencies
- `start.sh` - Streamlit startup script

## Environment Variables (Set in Vercel Dashboard)

After deployment, add these in Vercel project settings:

```
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## Features Deployed

✅ 3 Live Calculators:
- **Page 6:** Full V1 Calculator
- **Page 7:** V2 with Persistence & Bet Tracking
- **Page 8:** ⚡ Compact Calculator (recommended)

✅ Advanced ML Models:
- Logistic Regression (94.29% accuracy)
- Random Forest (98.04% accuracy)
- Ensemble predictions

✅ Player Intelligence:
- 143,530+ match database
- Advanced parameter extraction
- H2H analysis

✅ Features:
- Point-by-point tracking
- Multi-market value bets (Match/Set/Game)
- 6 advanced parameters per player
- Pre-match odds tracking
- Match snapshots
- Probability evolution charts

## Alternative: Streamlit Cloud

If Vercel doesn't work, use Streamlit Cloud:

1. Go to https://share.streamlit.io
2. Connect your GitHub repo
3. Select: `ayush101098/tennis10`
4. Main file: `dashboard/streamlit_app.py`
5. Deploy!

## Local Testing

```bash
cd /Users/ayushmishra/tennis10
.venv/bin/streamlit run dashboard/streamlit_app.py --server.port 8501
```

Access: http://localhost:8501

## Database Note

⚠️ The `tennis_betting.db` file (143,530 matches) is included in the repo. On Vercel, the database will be read-only. For production with write capabilities, consider:

1. **Vercel Postgres** - For live match persistence
2. **Supabase** - Free PostgreSQL hosting
3. **PlanetScale** - MySQL alternative

## Quick Deploy Steps

```bash
# 1. Push to main (already done)
git push origin main

# 2. Deploy to Vercel
vercel --prod

# 3. Access your live app
# URL will be: https://tennis10-[random].vercel.app
```

## Support

Dashboard running locally: http://localhost:8501
GitHub repo: https://github.com/ayush101098/tennis10

Recommended calculator: **⚡ Live Calc Compact** (Page 8)
