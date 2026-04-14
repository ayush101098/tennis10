import json, urllib.request

url = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"
data = json.loads(urllib.request.urlopen(url).read())
events = data.get("events", [])
print(f"Total events: {len(events)}")

for ev in events[:3]:
    name = ev.get("name", "?")
    print(f"\n=== EVENT: {name} ===")
    allComps = []
    for g in ev.get("groupings", []):
        for c in g.get("competitions", []):
            allComps.append(c)
    if not allComps:
        for c in ev.get("competitions", []):
            allComps.append(c)

    for comp in allComps[:5]:
        state = comp.get("status", {}).get("type", {}).get("state", "")
        detail = comp.get("status", {}).get("type", {}).get("detail", "")
        compets = comp.get("competitors", [])
        if len(compets) < 2:
            continue
        if compets[0].get("type") == "team":
            continue

        p1 = compets[0].get("athlete", {}).get("displayName", "?") if isinstance(compets[0].get("athlete"), dict) else "?"
        p2 = compets[1].get("athlete", {}).get("displayName", "?") if isinstance(compets[1].get("athlete"), dict) else "?"
        print(f"  {p1} vs {p2} [{state}] detail={detail}")

        for i, c in enumerate(compets):
            ls = c.get("linescores", [])
            score_val = c.get("score", "")
            print(f"    P{i+1} score={score_val}, linescores={json.dumps(ls)}")

        situation = comp.get("situation", {})
        if situation:
            print(f"    SITUATION: {json.dumps(situation, indent=2)[:800]}")

        status = comp.get("status", {})
        print(f"    STATUS detail={status.get('type',{}).get('detail')}, clock={status.get('clock','')}")

        extra_keys = set(comp.keys()) - {"id", "uid", "date", "startDate", "timeValid", "recent", "competitors", "status", "venue", "notes", "round", "broadcasts", "highlights", "groups", "format", "groupings", "type", "$ref"}
        if extra_keys:
            print(f"    EXTRA KEYS: {extra_keys}")
