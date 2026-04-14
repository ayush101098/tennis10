"""Test alternative free APIs for ITF/Challenger tennis data."""
import requests
import json

# ── 1. Try Tennis Live / Open data endpoints ──
print("=" * 70)
print("1. TESTING FLASHSCORE PUBLIC API")
print("=" * 70)

# Flashscore uses sport IDs - tennis = 2
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "X-Fsign": "SW9D1eZo",
}

# Try Flashscore's public API (used by their mobile app)
urls_to_test = [
    ("Flashscore live", "https://flashscore.com/x/feed/f_2_0_1_en_1"),
    ("Flashscore tennis", "https://www.flashscore.com/x/feed/tr_2_0_1_en_1"),
]
for name, url in urls_to_test:
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"  {name}: Status {r.status_code} | Length: {len(r.text)} chars")
        if r.status_code == 200:
            print(f"  Preview: {r.text[:200]}")
    except Exception as e:
        print(f"  {name}: Error - {e}")

# ── 2. Try Sofascore with different approaches ──
print("\n" + "=" * 70)
print("2. TESTING SOFASCORE (different endpoints)")
print("=" * 70)

sofa_headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "*/*",
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
}

sofa_urls = [
    ("Tennis live", "https://api.sofascore.com/api/v1/sport/tennis/events/live"),
    ("Tennis today", "https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/2026-04-09"),
    ("Rankings", "https://api.sofascore.com/api/v1/rankings/type/1"),
]
for name, url in sofa_urls:
    try:
        r = requests.get(url, headers=sofa_headers, timeout=10)
        print(f"  {name}: Status {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if "events" in data:
                print(f"    Events: {len(data['events'])}")
                if data['events']:
                    ev = data['events'][0]
                    t = ev.get('tournament', {})
                    cat = t.get('category', {}).get('name', '?')
                    uname = t.get('uniqueTournament', {}).get('name', '?')
                    print(f"    First: {ev.get('homeTeam',{}).get('name','?')} vs {ev.get('awayTeam',{}).get('name','?')} | {cat} | {uname}")
    except Exception as e:
        print(f"  {name}: Error - {e}")

# ── 3. Try the-odds-api.com (free tier) ──
print("\n" + "=" * 70)
print("3. TESTING THE-ODDS-API (free tier)")
print("=" * 70)
try:
    r = requests.get("https://api.the-odds-api.com/v4/sports", 
                      params={"apiKey": "demo"}, timeout=10)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        sports = r.json()
        tennis_sports = [s for s in sports if 'tennis' in s.get('key', '').lower()]
        print(f"  Tennis sports: {len(tennis_sports)}")
        for s in tennis_sports:
            print(f"    {s['key']}: {s['title']} | Active: {s.get('active', '?')}")
except Exception as e:
    print(f"  Error: {e}")

# ── 4. Try tennisexplorer ──
print("\n" + "=" * 70) 
print("4. TESTING TENNISEXPLORER")
print("=" * 70)
try:
    r = requests.get("https://www.tennisexplorer.com/matches/?type=all",
                      headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    print(f"  Status: {r.status_code} | Length: {len(r.text)}")
    # Check for ITF mentions
    itf_count = r.text.lower().count("itf")
    challenger_count = r.text.lower().count("challenger")
    atp_count = r.text.lower().count("atp")
    wta_count = r.text.lower().count("wta")
    print(f"  Mentions: ATP={atp_count}, WTA={wta_count}, ITF={itf_count}, Challenger={challenger_count}")
except Exception as e:
    print(f"  Error: {e}")

# ── 5. Try tennis-data RSS / JSON feeds ──
print("\n" + "=" * 70)
print("5. TESTING OPEN TENNIS DATA SOURCES")
print("=" * 70)

open_urls = [
    ("ATP official calendar", "https://www.atptour.com/en/-/ajax/scores/GetInitialScores"),
    ("Pointbet tennis", "https://api.pointsbet.com/api/v2/sports/tennis/events"),
]
for name, url in open_urls:
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=10)
        print(f"  {name}: Status {r.status_code} | Length: {len(r.text)}")
        if r.status_code == 200 and r.text.startswith('{'):
            data = r.json()
            print(f"    Keys: {list(data.keys())[:10]}")
    except Exception as e:
        print(f"  {name}: Error - {e}")

print("\n✅ Done!")
