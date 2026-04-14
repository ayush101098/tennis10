"""
🔴 LIVE MATCHES — Real-Time Tennis Dashboard
=============================================
Auto-refreshing view of all live tennis matches worldwide.
Uses ESPN public API — 100% FREE, no API key needed.
Click any match to load it into the Calculator V2 for full analysis.
"""

import streamlit as st
import time
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.free_live_data import FreeLiveTennisService, get_free_service

st.set_page_config(page_title="Live Tennis Matches", page_icon="🔴", layout="wide")

# ──────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.live-badge {
    background: #dc3545; color: white; padding: 2px 8px;
    border-radius: 10px; font-size: 12px; font-weight: bold;
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.6} }
.match-card {
    background: white; border: 1px solid #dee2e6;
    border-radius: 10px; padding: 16px; margin: 8px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.06);
}
.match-card:hover { border-color: #007bff; box-shadow: 0 4px 8px rgba(0,123,255,0.15); }
.score-live { font-size: 28px; font-weight: bold; text-align: center; }
.player-name { font-size: 16px; font-weight: 600; }
.tournament-tag {
    background: #e9ecef; padding: 2px 8px; border-radius: 4px;
    font-size: 12px; color: #495057;
}
.finished-badge {
    background: #6c757d; color: white; padding: 2px 8px;
    border-radius: 10px; font-size: 12px;
}
.upcoming-badge {
    background: #17a2b8; color: white; padding: 2px 8px;
    border-radius: 10px; font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = 30
if "filter_tour" not in st.session_state:
    st.session_state.filter_tour = "All"
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None

# ──────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────
st.title("🔴 Live Tennis Matches")
st.caption("Real-time scores & rankings from ESPN — 100% FREE, no API key needed")

# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✅ ESPN Free API")
    st.success("No API key required! Data loads automatically.")

    st.markdown("---")
    st.markdown("## 🔄 Refresh Settings")
    st.session_state.auto_refresh = st.checkbox(
        "Auto-refresh", value=st.session_state.auto_refresh
    )
    st.session_state.refresh_interval = st.selectbox(
        "Interval (seconds)", [15, 30, 60, 120], index=1
    )

    st.markdown("---")
    st.markdown("## 🎾 Filters")
    st.session_state.filter_tour = st.selectbox(
        "Tour", ["All", "ATP", "WTA"]
    )

    if st.session_state.last_refresh:
        st.caption(f"Last refresh: {st.session_state.last_refresh}")


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def status_badge(match: dict) -> str:
    """Return HTML badge for match status."""
    if match["is_live"]:
        return '<span class="live-badge">● LIVE</span>'
    elif match["is_finished"]:
        return '<span class="finished-badge">✓ Finished</span>'
    else:
        return f'<span class="upcoming-badge">⏳ {match.get("event_time", "Upcoming")}</span>'


def render_match_card(match: dict, idx: int):
    """Render a single match as a card."""
    p1 = match["p1_name"]
    p2 = match["p2_name"]
    tournament = match["tournament"]
    score = match["score_display"]
    badge = status_badge(match)
    tour = match.get("tour", "")
    rnd = match.get("round", "")
    surface = match.get("surface", "")
    seed1 = f"[{match['p1_seed']}] " if match.get("p1_seed") else ""
    seed2 = f" [{match['p2_seed']}]" if match.get("p2_seed") else ""

    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1.5])

        with col1:
            st.markdown(f"""
            {badge} <span class="tournament-tag">{tour} — {tournament}</span>
            <br/>
            <span class="player-name">{seed1}{p1}</span>
            <br/>
            <span class="player-name">{p2}{seed2}</span>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="score-live">{score}</div>',
                        unsafe_allow_html=True)
            if rnd:
                st.caption(rnd)

        with col3:
            st.caption(f"🎾 Surface: {surface}")
            if match.get("p1_winner"):
                st.caption(f"🏆 Winner: {p1}")
            elif match.get("p2_winner"):
                st.caption(f"🏆 Winner: {p2}")
            if match.get("event_time"):
                st.caption(f"🕐 {match['event_time']}")

        with col4:
            if match["is_live"] or match["is_finished"]:
                if st.button("🎯 Load into V2", key=f"load_{idx}",
                             use_container_width=True):
                    st.session_state["api_loaded_match"] = match
                    st.success(f"✅ Loaded! Go to Live Calculator V2")
            else:
                st.caption("Not started")

        st.markdown("---")


def render_rankings_sidebar(svc: FreeLiveTennisService):
    """Show rankings in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🏆 Rankings")
        rank_tour = st.selectbox("Tour", ["ATP", "WTA"], key="rank_tour_sb")
        rankings = svc.get_rankings(rank_tour)
        if rankings:
            for r in rankings[:20]:
                trend = ""
                if r["trend"] and r["trend"] not in ("-", ""):
                    trend = f" ({r['trend']})"
                st.caption(f"**{r['rank']}.** {r['player']} — {r['points']:.0f} pts{trend}")
        else:
            st.caption("Rankings unavailable")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

svc = get_free_service()

# Main tabs
tab_live, tab_schedule, tab_search = st.tabs(
    ["🔴 Live Matches", "📅 Today's Schedule", "🔍 Search Player"]
)

with tab_live:
    st.markdown("### 🔴 Currently Live")

    with st.spinner("Fetching live scores..."):
        live_matches = svc.get_live_matches()

    st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")

    # Filter by tour
    if st.session_state.filter_tour != "All":
        live_matches = [m for m in live_matches
                        if m["tour"] == st.session_state.filter_tour.upper()]

    if live_matches:
        st.success(f"**{len(live_matches)} live matches** found")
        for i, m in enumerate(live_matches):
            render_match_card(m, i)
    else:
        st.info(
            "No live matches right now. "
            "Check **Today's Schedule** tab for upcoming & finished matches."
        )

with tab_schedule:
    st.markdown("## 📅 Today's Schedule")

    with st.spinner("Loading schedule..."):
        today_matches = svc.get_todays_matches()

    # Filter
    if st.session_state.filter_tour != "All":
        today_matches = [m for m in today_matches
                         if m["tour"] == st.session_state.filter_tour.upper()]

    if not today_matches:
        st.info("No matches found for today.")
    else:
        # Separate by status
        live_now = [m for m in today_matches if m["is_live"]]
        finished = [m for m in today_matches if m["is_finished"]]
        upcoming = [m for m in today_matches
                    if not m["is_live"] and not m["is_finished"]]

        if live_now:
            st.markdown(f"### 🔴 Live Now ({len(live_now)})")
            for i, m in enumerate(live_now):
                render_match_card(m, 3000 + i)

        if upcoming:
            with st.expander(f"⏳ Upcoming ({len(upcoming)})", expanded=not live_now):
                for i, m in enumerate(upcoming):
                    render_match_card(m, 4000 + i)

        if finished:
            with st.expander(f"✅ Finished ({len(finished)})", expanded=not live_now and not upcoming):
                for i, m in enumerate(finished[:50]):  # Limit to 50
                    render_match_card(m, 5000 + i)

        st.caption(f"Total: {len(today_matches)} matches ({len(live_now)} live, "
                   f"{len(upcoming)} upcoming, {len(finished)} finished)")

with tab_search:
    st.markdown("### 🔍 Search Player")
    st.caption("Search ATP & WTA rankings for any player")

    search_name = st.text_input("Player name", placeholder="e.g. Sinner, Sabalenka, Djokovic")

    if search_name:
        results = svc.search_player(search_name)
        if results:
            st.success(f"Found {len(results)} match(es)")
            for p in results:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**{p['name']}**")
                with col2:
                    st.metric("Rank", f"#{p['rank']}")
                with col3:
                    st.caption(f"{p['tour']} | {p['points']:.0f} pts | {p['country']}")
        else:
            st.warning(f"No player found matching '{search_name}'")

    st.markdown("---")
    st.markdown("### 🏆 Full Rankings")
    rank_tab_tour = st.selectbox("Select Tour", ["ATP", "WTA"], key="rank_tab_tour")
    rankings = svc.get_rankings(rank_tab_tour)

    if rankings:
        # Show top N with expandable rest
        top_n = st.slider("Show top N", 10, 150, 50, 10)
        for r in rankings[:top_n]:
            trend = ""
            if r["trend"] and r["trend"] not in ("-", ""):
                trend = f" ↕{r['trend']}"
            st.caption(
                f"**#{r['rank']}** {r['player']} ({r['country']}) — "
                f"{r['points']:.0f} pts{trend}"
            )
    else:
        st.info("Rankings unavailable")

# Rankings in sidebar
try:
    render_rankings_sidebar(svc)
except Exception:
    pass

# Auto-refresh
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()
