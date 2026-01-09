#!/bin/bash
# Launch Tennis ML Betting Dashboard
# Quick start script for the Streamlit dashboard

echo "🎾 Tennis ML Betting Dashboard"
echo "=============================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚡ Activating virtual environment..."
    source .venv/bin/activate
fi

# Load API key
if [ -f "setup_env.sh" ]; then
    echo "🔑 Loading API key..."
    source setup_env.sh
else
    echo "⚠️  Warning: setup_env.sh not found. API key may not be set."
fi

# Check database exists
if [ ! -f "tennis_betting.db" ]; then
    echo "📦 Creating new database..."
    python -c "from dashboard.data_loader import get_database_connection; get_database_connection()"
    echo "✅ Database created"
fi

echo ""
echo "🚀 Launching dashboard..."
echo "📍 Access at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Launch Streamlit
streamlit run dashboard/streamlit_app.py
