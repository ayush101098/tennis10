"""Discover all ESPN tennis tour slugs and endpoints."""
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"

# Known slugs to test
SLUGS = [
    "atp", "wta",
    # Possible ITF / Challenger / other slugs
    "itf", "itf-men", "itf-women",
    "challenger", "atp-challenger",
    "atp-doubles", "wta-doubles",
    "davis-cup", "fed-cup", "billie-jean-king-cup",
    "united-cup",
    "atp-singles", "wta-singles",
    "grand-slam", "grand-slams",
    "atp-qualifying", "wta-qualifying",
    "men", "women",
    "itf-men-singles", "itf-women-singles",
]

print("=" * 70)
print("TESTING ESPN TENNIS SLUGS — scoreboard endpoint")
print("=" * 70)

working = []
for slug in SLUGS:
    url = f"{BASE}/{slug}/scoreboard"
    try:
        r = requests.get(url, timeout=8)
        events = []
        if r.status_code == 200:
            data = r.json()
            events = data.get("events", [])
            leagues = data.get("leagues", [])
            league_names = [lg.get("name", "?") for lg in leagues]
            match_count = 0
            for ev in events:
                for g in ev.get("groupings", []):
                    match_count += len(g.get("competitions", []))
            status = f"✅ {r.status_code} | {len(events)} events, {match_count} matches | Leagues: {league_names}"
            working.append((slug, league_names, len(events), match_count))
        else:
            status = f"❌ {r.status_code}"
    except Exception as e:
        status = f"💥 Error: {e}"
    print(f"  {slug:30s} → {status}")

print("\n" + "=" * 70)
print("TESTING RANKINGS")
print("=" * 70)
for slug in SLUGS[:10]:
    url = f"{BASE}/{slug}/rankings"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            data = r.json()
            rankings = data.get("rankings", [])
            if rankings:
                ranks = rankings[0].get("ranks", [])
                print(f"  {slug:30s} → ✅ {len(ranks)} ranked players")
            else:
                print(f"  {slug:30s} → ✅ 200 but no rankings data")
        else:
            print(f"  {slug:30s} → ❌ {r.status_code}")
    except Exception as e:
        print(f"  {slug:30s} → 💥 {e}")

print("\n" + "=" * 70)
print("WORKING SLUGS SUMMARY")
print("=" * 70)
for slug, leagues, events, matches in working:
    print(f"  {slug:30s} | {events} events, {matches} matches | {leagues}")

# Also check what leagues the main API lists
print("\n" + "=" * 70)
print("EXPLORING LEAGUE IDS")
print("=" * 70)
for slug in ["atp", "wta"]:
    r = requests.get(f"{BASE}/{slug}/scoreboard", timeout=8)
    if r.status_code == 200:
        data = r.json()
        for lg in data.get("leagues", []):
            print(f"  {slug}: League '{lg.get('name')}' ID={lg.get('id')} slug={lg.get('slug')}")

print("\n✅ Done!")
