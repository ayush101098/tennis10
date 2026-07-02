"""
Generate a compact rankings JSON for the trading terminal.

Source: SofaScore rankings API via the local sofa_proxy (type 5 = ATP,
type 6 = WTA) — always current, unlike the retired Sackmann
*_rankings_current.csv files this script used before.

Run with sofa_proxy.py up:   python generate_rankings.py
Outputs: trading-terminal/public/rankings.json
"""
import json
import sys
import urllib.request
from datetime import date

PROXY = "http://127.0.0.1:3001"
OUT = "trading-terminal/public/rankings.json"

RANKING_TYPES = {"atp": 5, "wta": 6}


def fetch_rankings(tour: str) -> dict:
    url = f"{PROXY}/rankings/type/{RANKING_TYPES[tour]}"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    entries = data.get("rankings", [])
    result = {}
    for e in entries:
        name = (e.get("team") or {}).get("name") or e.get("rowName", "")
        rank = e.get("ranking") or e.get("position")
        points = e.get("points", 0)
        if name and rank:
            result[name] = {"rank": int(rank), "points": int(points or 0)}
    print(f"{tour.upper()}: {len(result)} players (#1 = {entries[0]['team']['name'] if entries else '?'})")
    return result


def main():
    today = date.today().strftime("%Y%m%d")
    try:
        atp = fetch_rankings("atp")
        wta = fetch_rankings("wta")
    except Exception as e:
        print(f"ERROR: could not fetch rankings — is sofa_proxy.py running on 3001? ({e})")
        sys.exit(1)

    if len(atp) < 100 or len(wta) < 100:
        print("ERROR: suspiciously few players — refusing to overwrite rankings.json")
        sys.exit(1)

    out = {"atp_date": today, "wta_date": today, "atp": atp, "wta": wta}
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {OUT}: {len(atp)} ATP + {len(wta)} WTA players, dated {today}")


if __name__ == "__main__":
    main()
