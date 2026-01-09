"""
Page 4: Settings
================
Configure system parameters and preferences
"""

import streamlit as st
import os
import json

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ System Settings")

# Load current settings
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.json')

def load_settings():
    """Load settings from file"""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    else:
        # Default settings
        return {
            'bankroll': 1000.0,
            'kelly_fraction': 0.25,
            'min_edge': 0.025,
            'max_bet_pct': 0.15,
            'min_confidence': 'medium',
            'min_model_agreement': 0.85,
            'api_key': os.getenv('ODDS_API_KEY', ''),
            'auto_bet': False,
            'notifications_enabled': False,
            'email': '',
            'slack_webhook': ''
        }

def save_settings(settings):
    """Save settings to file"""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)
    st.success("✅ Settings saved successfully!")

# Load current settings
settings = load_settings()

# ============================================================================
# BETTING PARAMETERS
# ============================================================================

st.subheader("💰 Betting Parameters")

with st.form("betting_params"):
    col1, col2 = st.columns(2)
    
    with col1:
        bankroll = st.number_input(
            "Starting Bankroll ($)",
            min_value=100.0,
            max_value=100000.0,
            value=settings['bankroll'],
            step=100.0,
            help="Your total betting bankroll"
        )
        
        kelly_fraction = st.slider(
            "Kelly Fraction",
            min_value=0.1,
            max_value=1.0,
            value=settings['kelly_fraction'],
            step=0.05,
            help="Fraction of Kelly Criterion to use (0.25 = quarter Kelly, conservative)"
        )
        
        min_edge = st.slider(
            "Minimum Edge (%)",
            min_value=0.0,
            max_value=10.0,
            value=settings['min_edge'] * 100,
            step=0.5,
            help="Minimum edge required to place a bet"
        ) / 100
    
    with col2:
        max_bet_pct = st.slider(
            "Max Bet Size (% of Bankroll)",
            min_value=1.0,
            max_value=25.0,
            value=settings['max_bet_pct'] * 100,
            step=1.0,
            help="Maximum stake as percentage of bankroll"
        ) / 100
        
        min_confidence = st.selectbox(
            "Minimum Confidence Level",
            ['low', 'medium', 'high'],
            index=['low', 'medium', 'high'].index(settings['min_confidence']),
            help="Only bet on matches with this confidence or higher"
        )
        
        min_model_agreement = st.slider(
            "Minimum Model Agreement",
            min_value=0.70,
            max_value=1.0,
            value=settings['min_model_agreement'],
            step=0.05,
            format="%.0f%%",
            help="Only bet when models agree above this threshold"
        )
    
    if st.form_submit_button("💾 Save Betting Parameters", use_container_width=True, type="primary"):
        settings['bankroll'] = bankroll
        settings['kelly_fraction'] = kelly_fraction
        settings['min_edge'] = min_edge
        settings['max_bet_pct'] = max_bet_pct
        settings['min_confidence'] = min_confidence
        settings['min_model_agreement'] = min_model_agreement
        save_settings(settings)

st.divider()

# ============================================================================
# API CONFIGURATION
# ============================================================================

st.subheader("🔑 API Configuration")

with st.form("api_config"):
    api_key = st.text_input(
        "The Odds API Key",
        value=settings['api_key'],
        type="password",
        help="Your API key from the-odds-api.com"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Free Tier Limits:**
        - 500 requests/month
        - Updates every 15 minutes
        
        Get your key at: [the-odds-api.com](https://the-odds-api.com)
        """)
    
    with col2:
        if st.form_submit_button("🔍 Test API Key", use_container_width=True):
            if api_key:
                import requests
                try:
                    response = requests.get(
                        "https://api.the-odds-api.com/v4/sports/",
                        params={'apiKey': api_key}
                    )
                    if response.status_code == 200:
                        st.success("✅ API key is valid!")
                        remaining = response.headers.get('x-requests-remaining', 'Unknown')
                        st.info(f"Requests remaining: {remaining}")
                    else:
                        st.error(f"❌ Invalid API key (Status: {response.status_code})")
                except Exception as e:
                    st.error(f"❌ Error testing API key: {str(e)}")
            else:
                st.warning("Please enter an API key")
    
    if st.form_submit_button("💾 Save API Key", use_container_width=True, type="primary"):
        settings['api_key'] = api_key
        save_settings(settings)
        
        # Also update environment variable
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'setup_env.sh'), 'w') as f:
            f.write(f"export ODDS_API_KEY='{api_key}'\n")
        
        st.success("✅ API key saved to settings and setup_env.sh")

st.divider()

# ============================================================================
# AUTOMATION
# ============================================================================

st.subheader("🤖 Automation Settings")

with st.form("automation"):
    auto_bet = st.checkbox(
        "Enable Automatic Betting",
        value=settings['auto_bet'],
        help="⚠️ WARNING: This will automatically place bets without confirmation!"
    )
    
    if auto_bet:
        st.warning("""
        ⚠️ **Automatic betting is ENABLED**
        
        The system will:
        - Automatically place bets when opportunities are found
        - Use the betting parameters configured above
        - Only bet on matches meeting confidence and edge thresholds
        
        **USE WITH CAUTION!** Monitor the system closely.
        """)
    
    st.divider()
    
    st.write("**Scheduler Configuration:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        match_scrape_interval = st.selectbox(
            "Match Scraping Interval",
            [1, 3, 6, 12, 24],
            index=2,
            format_func=lambda x: f"Every {x} hours"
        )
        
        odds_update_interval = st.selectbox(
            "Odds Update Interval",
            [5, 10, 15, 30, 60],
            index=2,
            format_func=lambda x: f"Every {x} minutes"
        )
    
    with col2:
        prediction_interval = st.selectbox(
            "Prediction Generation Interval",
            [10, 15, 30, 60],
            index=2,
            format_func=lambda x: f"Every {x} minutes"
        )
        
        alert_check_interval = st.selectbox(
            "Alert Check Interval",
            [5, 10, 15, 30],
            index=1,
            format_func=lambda x: f"Every {x} minutes"
        )
    
    if st.form_submit_button("💾 Save Automation Settings", use_container_width=True, type="primary"):
        settings['auto_bet'] = auto_bet
        settings['match_scrape_interval'] = match_scrape_interval
        settings['odds_update_interval'] = odds_update_interval
        settings['prediction_interval'] = prediction_interval
        settings['alert_check_interval'] = alert_check_interval
        save_settings(settings)

st.divider()

# ============================================================================
# NOTIFICATIONS
# ============================================================================

st.subheader("🔔 Notifications")

with st.form("notifications"):
    notifications_enabled = st.checkbox(
        "Enable Notifications",
        value=settings['notifications_enabled']
    )
    
    if notifications_enabled:
        notification_method = st.radio(
            "Notification Method",
            ['Email', 'Slack', 'Both'],
            horizontal=True
        )
        
        if notification_method in ['Email', 'Both']:
            email = st.text_input(
                "Email Address",
                value=settings.get('email', '')
            )
        
        if notification_method in ['Slack', 'Both']:
            slack_webhook = st.text_input(
                "Slack Webhook URL",
                value=settings.get('slack_webhook', ''),
                type="password"
            )
        
        st.info("""
        **You'll receive notifications for:**
        - High-confidence betting opportunities (>5% edge)
        - Settled bets (wins and losses)
        - System errors or data issues
        - Daily performance summaries
        """)
    
    if st.form_submit_button("💾 Save Notification Settings", use_container_width=True, type="primary"):
        settings['notifications_enabled'] = notifications_enabled
        if notifications_enabled:
            if notification_method in ['Email', 'Both']:
                settings['email'] = email
            if notification_method in ['Slack', 'Both']:
                settings['slack_webhook'] = slack_webhook
        save_settings(settings)

st.divider()

# ============================================================================
# DATA MANAGEMENT
# ============================================================================

st.subheader("💾 Data Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared!")

with col2:
    if st.button("📥 Export Database", use_container_width=True):
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tennis_betting.db')
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                st.download_button(
                    label="Download Database",
                    data=f,
                    file_name=f"tennis_betting_backup_{datetime.now().strftime('%Y%m%d')}.db",
                    mime="application/octet-stream"
                )
        else:
            st.warning("Database not found")

with col3:
    if st.button("⚠️ Reset All Data", use_container_width=True):
        st.error("This feature is disabled for safety. Manually delete tennis_betting.db to reset.")

# ============================================================================
# CURRENT CONFIGURATION SUMMARY
# ============================================================================

st.divider()

st.subheader("📋 Current Configuration")

config_display = {
    "Bankroll": f"${settings['bankroll']:.2f}",
    "Kelly Fraction": f"{settings['kelly_fraction']:.2f}",
    "Min Edge": f"{settings['min_edge']:.2%}",
    "Max Bet %": f"{settings['max_bet_pct']:.1%}",
    "Min Confidence": settings['min_confidence'].capitalize(),
    "Min Model Agreement": f"{settings['min_model_agreement']:.0%}",
    "API Key": "✅ Set" if settings['api_key'] else "❌ Not set",
    "Auto Betting": "🟢 Enabled" if settings['auto_bet'] else "🔴 Disabled",
    "Notifications": "🟢 Enabled" if settings['notifications_enabled'] else "🔴 Disabled"
}

import datetime

for key, value in config_display.items():
    st.text(f"{key}: {value}")

st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
