import json, urllib.request

for tour in ["atp", "wta"]:
    url = f"https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard"
    data = json.loads(urllib.request.urlopen(url).read())
    events = data.get("events", [])
    print(f"\n{tour.upper()}: {len(events)} events")
    for ev in events[:2]:
        print(f"  {ev.get('name','?')}")
        allComps = []
        for g in ev.get("groupings", []):
            for c in g.get("competitions", []):
                allComps.append(c)
        if not allComps:
            for c in ev.get("competitions", []):
                allComps.append(c)
        live = [c for c in allComps if c.get("status",{}).get("type",{}).get("state") == "in"]
        pre = [c for c in allComps if c.get("status",{}).get("type",{}).get("state") == "pre"]
        post = [c for c in allComps if c.get("status",{}).get("type",{}).get("state") == "post"]
        print(f"    live={len(live)}, pre={len(pre)}, post={len(post)}")
        if live:
            comp = live[0]
            print("    FULL LIVE COMP DUMP:")
            print(json.dumps(comp, indent=2)[:3000])
