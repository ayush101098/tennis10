"""
Generate a compact rankings JSON from Sackmann's GitHub data.
Outputs: trading-terminal/public/rankings.json
"""
import json, urllib.request, csv, io

def fetch_csv(url):
    raw = urllib.request.urlopen(url).read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))

def build_rankings(tour):
    players_url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{tour}_players.csv"
    rankings_url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master/{tour}_rankings_current.csv"

    print(f"Fetching {tour} players...")
    players = fetch_csv(players_url)
    id_to_name = {}
    for p in players:
        pid = p.get("player_id", "").strip()
        fn = p.get("name_first", "").strip()
        ln = p.get("name_last", "").strip()
        if pid and ln:
            id_to_name[pid] = f"{fn} {ln}".strip()

    print(f"Fetching {tour} rankings...")
    rankings = fetch_csv(rankings_url)

    # Find latest date
    dates = set(r["ranking_date"] for r in rankings if "ranking_date" in r)
    latest = max(dates)
    print(f"  Latest date: {latest}, total dates: {len(dates)}")

    # Get entries for latest date, top 500
    result = {}
    for r in rankings:
        if r.get("ranking_date") != latest:
            continue
        rank = int(r.get("rank", 0))
        if rank > 500:
            continue
        pid = r.get("player", "").strip()
        pts = int(r.get("points", 0))
        name = id_to_name.get(pid, "")
        if not name:
            continue
        result[name] = {"rank": rank, "points": pts}

    print(f"  {len(result)} players with rank <= 500")
    return result, latest

atp, atp_date = build_rankings("atp")
wta, wta_date = build_rankings("wta")

output = {
    "atp_date": atp_date,
    "wta_date": wta_date,
    "atp": atp,
    "wta": wta,
}

out_path = "trading-terminal/public/rankings.json"
with open(out_path, "w") as f:
    json.dump(output, f, separators=(",", ":"))

size_kb = len(json.dumps(output, separators=(",", ":"))) / 1024
print(f"\nWrote {out_path} ({size_kb:.0f} KB)")
print(f"ATP: {len(atp)} players (as of {atp_date})")
print(f"WTA: {len(wta)} players (as of {wta_date})")
