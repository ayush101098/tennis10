"""Parse Flashscore's live feed to understand data format for ALL tours."""
import requests
import re

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "*/*",
    "X-Fsign": "SW9D1eZo",
}

r = requests.get("https://flashscore.com/x/feed/f_2_0_1_en_1", headers=headers, timeout=15)
print(f"Status: {r.status_code} | Length: {len(r.text)}")

raw = r.text

# Flashscore uses a custom delimiter format:
# SA÷<sport> — tournament section header
# ~ZA÷ — tournament name/details
# ¬~AA÷ — match ID
# Records separated by ¬ (pilcrow) and fields by ÷ (division sign)

# Split into sections by SA÷ (sport marker)
sections = raw.split("SA÷")
print(f"\nTotal sections: {len(sections)}")

# Collect unique tournament types
tour_types = set()
all_tournaments = []

for sec in sections[1:]:  # Skip first empty
    # Tournament name is after ZA÷ 
    za_match = re.search(r"ZA÷([^¬]+)", sec)
    if za_match:
        tournament = za_match.group(1)
        all_tournaments.append(tournament)
        # Extract tour type (before colon)
        if ":" in tournament:
            tour_type = tournament.split(":")[0].strip()
            tour_types.add(tour_type)

print(f"\nUnique tour types: {len(tour_types)}")
for t in sorted(tour_types):
    count = sum(1 for x in all_tournaments if x.startswith(t))
    print(f"  {t} ({count} tournaments)")

# Show sample tournaments for each type
print(f"\n{'='*70}")
print("SAMPLE TOURNAMENTS PER TYPE")
print("="*70)
for tour_type in sorted(tour_types):
    samples = [t for t in all_tournaments if t.startswith(tour_type)][:3]
    print(f"\n  {tour_type}:")
    for s in samples:
        print(f"    → {s}")

# Now parse individual matches
print(f"\n{'='*70}")
print("PARSING INDIVIDUAL MATCHES")
print("="*70)

# Each match starts with ¬~AA÷ (match ID)
# Key fields:
# AA÷ = match ID
# AD÷ = unix timestamp
# AE÷ = player 1 name
# AF÷ = player 2 name  
# AG÷ = ???
# BA÷ = set 1 home score
# BB÷ = set 1 away score
# etc.

match_count = 0
tour_match_counts = {}

for sec in sections[1:]:
    za_match = re.search(r"ZA÷([^¬]+)", sec)
    tournament = za_match.group(1) if za_match else "Unknown"
    
    # Count matches in this section
    matches = sec.split("~AA÷")
    n_matches = len(matches) - 1  # First is header
    
    if ":" in tournament:
        tour_type = tournament.split(":")[0].strip()
    else:
        tour_type = tournament
    
    tour_match_counts[tour_type] = tour_match_counts.get(tour_type, 0) + n_matches
    match_count += n_matches

print(f"Total matches across all tours: {match_count}")
print(f"\nMatches per tour type:")
for tour, count in sorted(tour_match_counts.items(), key=lambda x: -x[1]):
    print(f"  {tour}: {count} matches")

# Parse a few actual matches to see field structure
print(f"\n{'='*70}")
print("SAMPLE MATCH DATA (first 5 matches)")
print("="*70)

parsed_count = 0
for sec in sections[1:]:
    za_match = re.search(r"ZA÷([^¬]+)", sec)
    tournament = za_match.group(1) if za_match else "Unknown"
    
    matches = sec.split("~AA÷")
    for m_raw in matches[1:]:  # Skip header
        if parsed_count >= 5:
            break
        
        # Parse fields
        fields = {}
        for field in m_raw.split("¬"):
            if "÷" in field:
                key, val = field.split("÷", 1)
                fields[key] = val
        
        # Key fields
        match_id = fields.get("AA", fields.get("~AA", "?"))
        p1 = fields.get("AE", "?")
        p2 = fields.get("AF", "?")
        status = fields.get("AB", "?")
        
        # Set scores
        sets = []
        for i in range(1, 6):
            s1_key = chr(ord('B') + (i-1)*2 - 1) + chr(ord('A'))  # BA, DA, FA...
            s2_key = chr(ord('B') + (i-1)*2 - 1) + chr(ord('B'))
            # Actually it's BA/BB, DA/DB, FA/FB...
            pass
        
        set_keys = [("BA","BB"), ("DA","DB"), ("FA","FB"), ("HA","HB"), ("JA","JB")]
        for s1k, s2k in set_keys:
            s1 = fields.get(s1k)
            s2 = fields.get(s2k)
            if s1 is not None and s2 is not None:
                sets.append(f"{s1}-{s2}")
        
        print(f"\n  Tournament: {tournament}")
        print(f"  Match: {p1} vs {p2}")
        print(f"  Status: {status}")
        print(f"  Sets: {' '.join(sets) if sets else 'N/A'}")
        print(f"  Fields: {list(fields.keys())[:20]}")
        print(f"  All: {dict(list(fields.items())[:15])}")
        
        parsed_count += 1
    
    if parsed_count >= 5:
        break

print(f"\n✅ Done!")
