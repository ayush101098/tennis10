#!/bin/bash
# Set The Odds API key for tennis betting system
export ODDS_API_KEY='a0292044f825f2b560225751fd782851'

echo "✅ The Odds API key set: ${ODDS_API_KEY:0:10}..."
echo "🎾 Ready to fetch live tennis odds!"
echo ""
echo "Run:"
echo "  source setup_env.sh && python get_live_odds.py"
echo "  source setup_env.sh && python src/live_predictions/predictor.py"
