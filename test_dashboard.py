"""
Quick Dashboard Test Script
===========================
Verifies dashboard can launch without errors
"""

import sys
import os

print("🔍 Testing Dashboard Setup...\n")

# Test 1: Check file structure
print("1️⃣ Checking file structure...")
required_files = [
    'dashboard/streamlit_app.py',
    'dashboard/data_loader.py',
    'dashboard/components/__init__.py',
    'dashboard/components/match_card.py',
    'dashboard/components/charts.py',
    'dashboard/components/tables.py',
    'dashboard/pages/1_📊_Live_Predictions.py',
    'dashboard/pages/2_📈_Model_Performance.py',
    'dashboard/pages/3_💰_Betting_History.py',
    'dashboard/pages/4_⚙️_Settings.py',
    'dashboard/pages/5_🔍_Player_Analysis.py',
]

missing_files = []
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"  ✅ {file_path}")
    else:
        print(f"  ❌ {file_path} - MISSING")
        missing_files.append(file_path)

if missing_files:
    print(f"\n❌ {len(missing_files)} files missing!")
    sys.exit(1)
else:
    print(f"\n✅ All {len(required_files)} files found!\n")

# Test 2: Check Python dependencies
print("2️⃣ Checking Python dependencies...")
dependencies = [
    'streamlit',
    'plotly',
    'pandas',
    'numpy',
    'requests'
]

missing_deps = []
for dep in dependencies:
    try:
        __import__(dep)
        print(f"  ✅ {dep}")
    except ImportError:
        print(f"  ❌ {dep} - NOT INSTALLED")
        missing_deps.append(dep)

if missing_deps:
    print(f"\n❌ {len(missing_deps)} dependencies missing!")
    print(f"\nInstall with: pip install {' '.join(missing_deps)}")
    sys.exit(1)
else:
    print(f"\n✅ All dependencies installed!\n")

# Test 3: Check database can be created
print("3️⃣ Testing database connection...")
try:
    import sqlite3
    from dashboard.data_loader import get_database_connection
    
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    table_count = cursor.fetchone()[0]
    conn.close()
    
    print(f"  ✅ Database connected ({table_count} tables)")
except Exception as e:
    print(f"  ❌ Database error: {str(e)}")
    sys.exit(1)

print()

# Test 4: Import dashboard modules
print("4️⃣ Testing dashboard imports...")
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from dashboard.data_loader import (
        get_bankroll_status,
        get_upcoming_matches,
        get_active_bets
    )
    print("  ✅ data_loader imports")
    
    from dashboard.components import (
        render_match_card,
        create_pnl_chart,
        render_predictions_table
    )
    print("  ✅ components imports")
    
    print("\n✅ All imports successful!\n")
    
except Exception as e:
    print(f"  ❌ Import error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check live data modules
print("5️⃣ Testing live data modules...")
try:
    from src.live_data.match_scraper import get_all_upcoming_matches
    print("  ✅ match_scraper")
    
    from src.live_data.odds_scraper import get_tennis_odds
    print("  ✅ odds_scraper")
    
    from src.live_predictions.predictor import LivePredictor
    print("  ✅ predictor")
    
    from src.live_predictions.bet_calculator import BetCalculator
    print("  ✅ bet_calculator")
    
    print("\n✅ All modules accessible!\n")
    
except Exception as e:
    print(f"  ❌ Module error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("="*60)
print("🎉 Dashboard Setup Test: ALL PASSED!")
print("="*60)
print()
print("✨ Next steps:")
print("   1. Set your API key: export ODDS_API_KEY='your_key'")
print("   2. Launch dashboard: streamlit run dashboard/streamlit_app.py")
print("   3. Visit: http://localhost:8501")
print()
print("📚 Documentation: dashboard/README.md")
print("✅ Integration checklist: INTEGRATION_CHECKLIST.md")
print()
