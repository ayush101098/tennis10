#!/usr/bin/env python3
"""
Test multiple endpoints and URL patterns to discover ALL tennis tours.
"""
import requests
import time

# ═══════════════════════════════════════════════════════════════
# 1. FLASHSCORE — Try different URL patterns
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("FLASHSCORE URL VARIATIONS")
print("=" * 70)

fs_headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "X-Fsign": "SW9D1eZo",
    "Accept": "*/*",
    "Referer": "https://www.flashscore.com/tennis/",
}

# Try different URL patterns
fs_urls = {
    "f_2_0_1_en_1": "https://flashscore.com/x/feed/f_2_0_1_en_1",     # original
    "f_2_-1_1_en_1": "https://flashscore.com/x/feed/f_2_-1_1_en_1",   # all?
    "f_2_1_1_en_1": "https://flashscore.com/x/feed/f_2_1_1_en_1",     # page 2?
    "f_2_2_1_en_1": "https://flashscore.com/x/feed/f_2_2_1_en_1",     # page 3?
    "f_2_0_2_en_1": "https://flashscore.com/x/feed/f_2_0_2_en_1",     # type 2?
    "f_2_0_3_en_1": "https://flashscore.com/x/feed/f_2_0_3_en_1",     # type 3?
    "f_2_0_1_en_2": "https://flashscore.com/x/feed/f_2_0_1_en_2",     # variant 2?
    "f_2_0_1_en_3": "https://flashscore.com/x/feed/f_2_0_1_en_3",     # variant 3?
    "dc_2_en_1": "https://flashscore.com/x/feed/dc_2_en_1",           # dc format?
    "tr_2_0_en_1": "https://flashscore.com/x/feed/tr_2_0_en_1",       # tr format?
    "tr_2_1_en_1": "https://flashscore.com/x/feed/tr_2_1_en_1",       # tr format?
    "s_2_en_1": "https://flashscore.com/x/feed/s_2_en_1",             # s format?
}

for name, url in fs_urls.items():
    try:
        r = requests.get(url, headers=fs_headers, timeout=10)
        tours_found = set()
        if r.status_code == 200 and len(r.text) > 100:
            # Extract tour names
            for part in r.text.split("ZA÷"):
                if "¬" in part:
                    tour = part.split("¬")[0].strip()
                    if tour and len(tour) > 2:
                        tours_found.add(tour)
        print(f"  {name}: {r.status_code} | {len(r.text):,} chars | Tours: {tours_found if tours_found else 'none'}")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")
    time.sleep(0.3)

# ═══════════════════════════════════════════════════════════════
# 2. Try Flashscore CATEGORY-SPECIFIC endpoints
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FLASHSCORE CATEGORY ENDPOINTS")
print("=" * 70)

# Flashscore has category IDs for different tours
# Try common patterns
cat_urls = {
    "tennis-all": "https://www.flashscore.com/x/feed/f_2_0_1_en_1",
    "tennis-results": "https://www.flashscore.com/x/feed/f_2_0_2_en_1",
    "tennis-tomorrow": "https://www.flashscore.com/x/feed/f_2_0_3_en_1",
}

for name, url in cat_urls.items():
    try:
        r = requests.get(url, headers=fs_headers, timeout=10)
        if r.status_code == 200 and len(r.text) > 100:
            tours = set()
            for part in r.text.split("ZA÷"):
                if "¬" in part:
                    tour = part.split("¬")[0].strip()
                    if tour and len(tour) > 2:
                        tours.add(tour)
            print(f"  {name}: {r.status_code} | {len(r.text):,} chars | {len(tours)} tours: {tours}")
        else:
            print(f"  {name}: {r.status_code} | {len(r.text)} chars")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")
    time.sleep(0.3)

# ═══════════════════════════════════════════════════════════════
# 3. Try the Flashscore TENNIS page directly for tournament links
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FLASHSCORE TENNIS LANDING PAGE")
print("=" * 70)

try:
    r = requests.get("https://www.flashscore.com/tennis/", 
                     headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}, 
                     timeout=10)
    print(f"  Status: {r.status_code} | {len(r.text):,} chars")
    # Look for tournament/category references
    import re
    # Find all /tennis/XXXX patterns
    links = re.findall(r'/tennis/([a-zA-Z0-9_-]+)/?', r.text)
    unique_links = sorted(set(links))
    print(f"  Unique /tennis/XXX paths: {len(unique_links)}")
    for l in unique_links[:30]:
        print(f"    → {l}")
except Exception as e:
    print(f"  ERROR: {e}")

# ═══════════════════════════════════════════════════════════════
# 4. ESPN — Check scoreboard with dates for WTA 
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ESPN SCOREBOARD CHECK (ATP & WTA)")
print("=" * 70)

for tour in ["atp", "wta"]:
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard"
        r = requests.get(url, timeout=10)
        data = r.json()
        events = data.get("events", [])
        leagues = data.get("leagues", [])
        league_names = [l.get("name", "?") for l in leagues]
        print(f"  {tour.upper()}: {len(events)} events | Leagues: {league_names}")
        
        # Show first 3 events
        for ev in events[:3]:
            name = ev.get("name", "?")
            status = ev.get("status", {}).get("type", {}).get("description", "?")
            print(f"    → {name} [{status}]")
    except Exception as e:
        print(f"  {tour.upper()}: ERROR - {e}")

# ═══════════════════════════════════════════════════════════════
# 5. Try sofascore with different headers / mobile endpoint
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SOFASCORE MOBILE API")
print("=" * 70)

sofa_headers = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)",
    "Accept": "application/json",
}

sofa_urls = {
    "live": "https://api.sofascore.com/api/v1/sport/tennis/events/live",
    "scheduled": "https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/2025-04-15",
    "categories": "https://api.sofascore.com/api/v1/sport/tennis/categories",
}

for name, url in sofa_urls.items():
    try:
        r = requests.get(url, headers=sofa_headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "events" in data:
                events = data["events"]
                # Count by tournament
                tours = {}
                for ev in events:
                    t = ev.get("tournament", {}).get("category", {}).get("name", "Unknown")
                    tours[t] = tours.get(t, 0) + 1
                print(f"  {name}: {r.status_code} | {len(events)} events | Tours: {dict(sorted(tours.items(), key=lambda x: -x[1])[:10])}")
            elif "categories" in data:
                cats = data["categories"]
                print(f"  {name}: {r.status_code} | {len(cats)} categories")
                for c in cats[:15]:
                    print(f"    → {c.get('name', '?')} (id={c.get('id', '?')})")
            else:
                print(f"  {name}: {r.status_code} | Keys: {list(data.keys())[:10]}")
        else:
            print(f"  {name}: {r.status_code}")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")
    time.sleep(0.3)

# ═══════════════════════════════════════════════════════════════
# 6. Tennis-Live-Data (free tier)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("OPENLIVETENNIS / OTHER FREE")
print("=" * 70)

other_urls = {
    "tennisapi.com": "https://tennisapi.com/api/v1/tennis/events/live",
    "tennis-live-data": "https://tennis-live-data.p.rapidapi.com/matches-results/today",
    "livescore-api": "https://livescore-api.com/api-client/scores/live.json?competition_id=2",
}

for name, url in other_urls.items():
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        print(f"  {name}: {r.status_code} | {len(r.text):,} chars | Preview: {r.text[:150]}")
    except Exception as e:
        print(f"  {name}: ERROR - {e}")
    time.sleep(0.3)

print("\n✅ All tests complete!")
