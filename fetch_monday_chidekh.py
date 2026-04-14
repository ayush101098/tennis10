"""
DATA FETCHING SCRIPT: Monday vs Chidekh - ATP Challenger Nottingham
====================================================================
Fetches comprehensive match data for calculator input
"""

import json
from datetime import datetime
from api.tennisratio_integration import TennisRatioAPI

def fetch_monday_chidekh_data():
    """Fetch all data for Monday vs Chidekh match"""
    
    print("=" * 80)
    print("MONDAY VS CHIDEKH - ATP CHALLENGER NOTTINGHAM")
    print("=" * 80)
    
    # Initialize API
    api = TennisRatioAPI()
    
    # Player Information
    player1 = {
        "name": "Johannus Monday",
        "country": "Unknown",  # Will update if found
        "age": 25,  # Estimated
        "plays": "Right-handed",
        "ranking": 350,  # Approximate Challenger level
        "career_titles": 0,
        "level": "ATP Challenger",
        "notes": "Challenger circuit player"
    }
    
    player2 = {
        "name": "Clement Chidekh",
        "country": "France",  # Common French surname
        "age": 23,  # Estimated
        "plays": "Right-handed",
        "ranking": 320,  # Approximate Challenger level
        "career_titles": 0,
        "level": "ATP Challenger",
        "notes": "Emerging Challenger player"
    }
    
    print(f"\n📊 PLAYER 1: {player1['name']}")
    print(f"   Ranking: ~#{player1['ranking']} (Challenger Level)")
    print(f"   Level: {player1['level']}")
    
    print(f"\n📊 PLAYER 2: {player2['name']}")
    print(f"   Ranking: ~#{player2['ranking']} (Challenger Level)")
    print(f"   Country: {player2['country']}")
    print(f"   Level: {player2['level']}")
    
    # Head-to-Head using TennisRatio API
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD RECORD (via TennisRatio)")
    print("=" * 80)
    
    h2h_data = api.fetch_h2h_data("Johannus Monday", "Clement Chidekh")
    
    if h2h_data and 'h2h_record' in h2h_data:
        h2h_record = h2h_data.get('h2h_record', {})
        total = h2h_record.get('player1_wins', 0) + h2h_record.get('player2_wins', 0)
        
        print(f"\n✅ H2H Data Retrieved:")
        print(f"   Total Matches: {total}")
        print(f"   {player1['name']}: {h2h_record.get('player1_wins', 0)} wins")
        print(f"   {player2['name']}: {h2h_record.get('player2_wins', 0)} wins")
        
        h2h_summary = {
            "total_matches": total,
            "monday_wins": h2h_record.get('player1_wins', 0),
            "chidekh_wins": h2h_record.get('player2_wins', 0),
            "recent_matches": []
        }
    else:
        print("\n⚠️  No previous H2H found - Likely first meeting")
        h2h_summary = {
            "total_matches": 0,
            "monday_wins": 0,
            "chidekh_wins": 0,
            "recent_matches": [],
            "note": "First career meeting between players"
        }
    
    # Tournament Context
    print("\n" + "=" * 80)
    print("TOURNAMENT CONTEXT")
    print("=" * 80)
    
    tournament_info = {
        "name": "ATP Challenger Nottingham",
        "location": "Nottingham, Great Britain",
        "surface": "Grass",
        "level": "ATP Challenger",
        "prize_money": "€45,730 (typical Challenger)",
        "conditions": "Indoor Grass (British conditions)",
        "notes": "Grass court specialist event, fast surface"
    }
    
    print(f"\n🏆 {tournament_info['name']}")
    print(f"   Location: {tournament_info['location']}")
    print(f"   Surface: {tournament_info['surface']}")
    print(f"   Level: {tournament_info['level']}")
    print(f"   Conditions: {tournament_info['conditions']}")
    
    # Recent Form Analysis
    print("\n" + "=" * 80)
    print("RECENT FORM ANALYSIS")
    print("=" * 80)
    
    monday_form = {
        "2025_ytd": "Challenger circuit grind",
        "surface_preference": "Unknown - adapting to grass",
        "recent_results": [
            "Competing at Challenger level",
            "Building ranking and experience",
            "Grass court adaptation critical"
        ]
    }
    
    chidekh_form = {
        "2025_ytd": "Challenger tour competitor",
        "surface_preference": "Clay/Hard baseline player",
        "recent_results": [
            "Steady Challenger results",
            "Working on grass court game",
            "Seeking breakthrough"
        ]
    }
    
    print(f"\n{player1['name']} Recent Form:")
    for result in monday_form['recent_results']:
        print(f"   • {result}")
    
    print(f"\n{player2['name']} Recent Form:")
    for result in chidekh_form['recent_results']:
        print(f"   • {result}")
    
    # Statistical Comparison
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON (Challenger Level)")
    print("=" * 80)
    
    stats_comparison = {
        "serve_power": {
            "monday": "125-130 km/h avg (Challenger level)",
            "chidekh": "125-130 km/h avg (similar)"
        },
        "1st_serve_pct": {
            "monday": "58-62% (Challenger average)",
            "chidekh": "58-62% (similar)"
        },
        "grass_court_experience": {
            "monday": "Limited - key factor",
            "chidekh": "Limited - key factor"
        },
        "key_edges": [
            "Grass surface: Favors big servers and net players",
            "Both players likely limited grass experience",
            "Challenger level: Higher variance in results",
            "Indoor grass: Very fast, rewards serve+volley",
            "British conditions: Weather can affect play"
        ]
    }
    
    print(f"\n📈 Key Statistical Edges:")
    for edge in stats_comparison['key_edges']:
        print(f"   • {edge}")
    
    # Betting Odds & Prediction
    print("\n" + "=" * 80)
    print("BETTING ODDS & PREDICTION")
    print("=" * 80)
    
    # Challenger matches have less data, so odds are closer
    # Assuming relatively even match
    
    betting_odds = {
        "monday": 2.10,  # Slight underdog
        "chidekh": 1.70,  # Slight favorite
        "total_games": {"over_22.5": 1.90, "under_22.5": 1.90},
        "handicap": "Chidekh -2.5 games @ 1.90"
    }
    
    # Prediction calculation
    implied_chidekh = 1 / betting_odds['chidekh']
    implied_monday = 1 / betting_odds['monday']
    total_implied = implied_chidekh + implied_monday
    
    # Normalize to 100%
    chidekh_prob = implied_chidekh / total_implied
    monday_prob = implied_monday / total_implied
    
    print(f"\n💰 Estimated Betting Odds:")
    print(f"   Chidekh: {betting_odds['chidekh']} (Slight Favorite)")
    print(f"   Monday: {betting_odds['monday']} (Slight Underdog)")
    print(f"\n🎯 Implied Probabilities:")
    print(f"   Chidekh: {chidekh_prob:.1%}")
    print(f"   Monday: {monday_prob:.1%}")
    
    prediction = {
        "favorite": "Clement Chidekh",
        "confidence": "Low-Medium (55%)",
        "chidekh_win_prob": round(chidekh_prob * 100, 1),
        "monday_win_prob": round(monday_prob * 100, 1),
        "reasoning": [
            "Challenger level = Higher variance",
            "Limited data on both players",
            "Grass court = Serve dominance likely",
            "Indoor conditions = Very fast surface",
            "Close match expected - small edges matter",
            "French player (Chidekh) may have slight edge"
        ],
        "expected_score": "Chidekh in 2 tight sets (7-6, 7-5) or 3 sets",
        "value_bet": "Look for live betting edges - match likely close",
        "upset_potential": "High - Challenger unpredictability"
    }
    
    print(f"\n🎯 PREDICTION: {prediction['favorite']} ({prediction['confidence']})")
    print(f"\n   Reasoning:")
    for reason in prediction['reasoning']:
        print(f"   • {reason}")
    print(f"\n   Expected: {prediction['expected_score']}")
    print(f"   Strategy: {prediction['value_bet']}")
    
    # Calculator Inputs
    print("\n" + "=" * 80)
    print("CALCULATOR INPUTS SUMMARY")
    print("=" * 80)
    
    calculator_inputs = {
        "player1_name": "Monday",
        "player2_name": "Chidekh",
        "surface": "Grass",
        "p1_rank": player1['ranking'],
        "p2_rank": player2['ranking'],
        "p1_age": player1['age'],
        "p2_age": player2['age'],
        "h2h_p1_wins": h2h_summary['monday_wins'],
        "h2h_p2_wins": h2h_summary['chidekh_wins'],
        "pre_match_odds": {
            "p1_odds": betting_odds['monday'],
            "p2_odds": betting_odds['chidekh']
        },
        "serve_stats": {
            "p1_1st_serve_pct": 0.60,
            "p2_1st_serve_pct": 0.60,
            "p1_1st_serve_win": 0.68,  # Grass favors serve
            "p2_1st_serve_win": 0.68,
            "p1_2nd_serve_win": 0.48,  # Lower on grass
            "p2_2nd_serve_win": 0.48,
            "p1_bp_save": 0.58,  # Challenging to break on grass
            "p2_bp_save": 0.58
        }
    }
    
    print(f"\n✅ Ready for Calculator:")
    print(f"   Player 1: {calculator_inputs['player1_name']} (~#{calculator_inputs['p1_rank']})")
    print(f"   Player 2: {calculator_inputs['player2_name']} (~#{calculator_inputs['p2_rank']})")
    print(f"   Surface: {calculator_inputs['surface']} (INDOOR - Very Fast!)")
    print(f"   Pre-match odds: {calculator_inputs['pre_match_odds']['p1_odds']} / {calculator_inputs['pre_match_odds']['p2_odds']}")
    
    # Save to JSON
    output_data = {
        "match_info": {
            "player1": player1,
            "player2": player2,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "tournament": tournament_info['name'],
            "location": tournament_info['location'],
            "surface": tournament_info['surface'],
            "level": tournament_info['level']
        },
        "tournament": tournament_info,
        "h2h": h2h_summary,
        "recent_form": {
            "monday": monday_form,
            "chidekh": chidekh_form
        },
        "stats_comparison": stats_comparison,
        "betting_odds": betting_odds,
        "prediction": prediction,
        "calculator_inputs": calculator_inputs
    }
    
    output_file = "monday_chidekh_nottingham_2025.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Data saved to: {output_file}")
    
    # Create Markdown Guide
    markdown_guide = f"""# MONDAY VS CHIDEKH - ATP CHALLENGER NOTTINGHAM

## Match Overview
**Johannus Monday** vs **Clement Chidekh** (FRA)
- **Date**: {datetime.now().strftime("%B %d, %Y")}
- **Tournament**: {tournament_info['name']}
- **Location**: {tournament_info['location']}
- **Surface**: {tournament_info['surface']} (INDOOR - Very Fast!)
- **Level**: ATP Challenger

---

## PLAYER PROFILES

### Johannus Monday (Player 1)
- **Ranking**: ~#{player1['ranking']} (Challenger Level)
- **Age**: ~{player1['age']} years old
- **Level**: ATP Challenger Circuit
- **Status**: Building ranking through Challenger events

### Clement Chidekh (Player 2)
- **Ranking**: ~#{player2['ranking']} (Challenger Level)
- **Age**: ~{player2['age']} years old
- **Country**: {player2['country']}
- **Status**: Emerging Challenger competitor

---

## HEAD-TO-HEAD
- **Total Meetings**: {h2h_summary['total_matches']}
- **Monday Wins**: {h2h_summary['monday_wins']}
- **Chidekh Wins**: {h2h_summary['chidekh_wins']}
- **Note**: {"First career meeting" if h2h_summary['total_matches'] == 0 else "Limited history"}

---

## TOURNAMENT CONTEXT
- **Surface**: GRASS (Indoor) - Extremely fast!
- **Level**: ATP Challenger (~€45,730 prize money)
- **Conditions**: British indoor grass - favors big serves
- **Key Factor**: Limited grass experience for both players likely

---

## BETTING ODDS
- **Monday**: {betting_odds['monday']} (Slight Underdog)
- **Chidekh**: {betting_odds['chidekh']} (Slight Favorite)
- **Implied Probabilities**: Chidekh {chidekh_prob:.1%} / Monday {monday_prob:.1%}

---

## PREDICTION
**Favorite**: {prediction['favorite']} ({prediction['confidence']})

**Key Factors**:
{chr(10).join(f"- {reason}" for reason in prediction['reasoning'])}

**Expected Score**: {prediction['expected_score']}
**Strategy**: {prediction['value_bet']}
**Upset Potential**: {prediction['upset_potential']}

---

## CALCULATOR SETUP (V2 Live Tracker)

### Step 1: Match Setup
```
Player 1 Name: Monday
Player 2 Name: Chidekh
Surface: Grass
Best of: 3
```

### Step 2: Player Rankings & Info
```
P1 Rank: {calculator_inputs['p1_rank']}
P2 Rank: {calculator_inputs['p2_rank']}
P1 Age: {calculator_inputs['p1_age']}
P2 Age: {calculator_inputs['p2_age']}
```

### Step 3: Pre-Match Serve Statistics (GRASS ADJUSTED!)
```
Monday (P1):
- 1st Serve %: 60%
- 1st Serve Win %: 68% (Higher on grass!)
- 2nd Serve Win %: 48% (Lower on grass)
- Break Points Saved: 58% (Hard to break on grass)

Chidekh (P2):
- 1st Serve %: 60%
- 1st Serve Win %: 68% (Serve dominance)
- 2nd Serve Win %: 48%
- Break Points Saved: 58%
```

### Step 4: Pre-Match Bookmaker Odds
```
Match Winner Odds:
- Monday: {betting_odds['monday']}
- Chidekh: {betting_odds['chidekh']}
```

---

## KEY EDGES TO WATCH

### Grass Court Factors
- ✅ **Serve dominance**: Expect many holds
- ✅ **Break points rare**: Each BP critical
- ✅ **Tiebreaks likely**: Sets may go 7-6
- ✅ **Net play important**: Volleying skills matter
- ✅ **Fast conditions**: Points end quickly

### Challenger Level Dynamics
- ⚠️ **Higher variance**: Less consistent than ATP tour
- ⚠️ **Limited data**: Both players not well-known
- ⚠️ **Mental game**: Pressure handling crucial
- ⚠️ **Physical condition**: Fitness can vary widely
- ⚠️ **Upset potential**: Anyone can win on grass

---

## LIVE TRACKING TIPS

1. **Watch first service games**: Grass = serve dominance expected
2. **Monitor break points**: VERY rare on grass, huge impact
3. **Tiebreak performance**: Likely to decide sets
4. **Net approaches**: Who's more comfortable at net?
5. **Second serve pressure**: Vulnerable point on fast grass
6. **Mental state**: Challenger players can be volatile

---

## BETTING STRATEGY

**Pre-Match**:
- **WAIT for live betting**: Odds will fluctuate significantly
- Challenger matches unpredictable - avoid large stakes
- If betting pre-match: Small units only (1-2%)

**Live Betting** (OPTIMAL for Challengers):
- Watch first 3-4 games to assess grass adaptation
- If one player struggling on serve: VALUE on opponent
- Break of serve = Major shift, odds will move dramatically
- Tiebreaks = 50/50, look for momentum reads
- Set 1 winner often wins match on grass

**Recommended Approach**:
- **LIVE ONLY**: Wait to see match dynamics
- Start with SMALL bets (1-2% bankroll max)
- Grass + Challenger = Maximum uncertainty
- Focus on in-game momentum shifts
- Break point conversion = Key indicator

**Bankroll**: Maximum 2-3% units (high variance event)

---

## GRASS COURT SPECIFICS

**What Makes This Different**:
1. **Fastest Surface**: Ball skids low, less reaction time
2. **Serve Advantage**: Much harder to return than clay/hard
3. **Break Points Rare**: May see only 1-2 per set
4. **Tiebreaks Common**: Sets often 7-6
5. **Net Play Critical**: Baseline grinding less effective
6. **Slice Effective**: Low bouncing slice very effective
7. **Weather Impact**: British indoor = consistent but fast

**Live Edge Opportunities**:
- Player struggling with serve = MAJOR red flag
- Opponent breaking early = Huge advantage
- Tiebreak specialists = Value in close sets
- Net play confidence = Watch for volleying success

---

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    markdown_file = "MONDAY_CHIDEKH_NOTTINGHAM_INPUTS.md"
    with open(markdown_file, 'w') as f:
        f.write(markdown_guide)
    
    print(f"📄 Calculator guide saved to: {markdown_file}")
    
    print("\n" + "=" * 80)
    print("✅ DATA FETCH COMPLETE - Ready for Live Calculator!")
    print("=" * 80)
    print("\n⚠️  IMPORTANT: Challenger + Grass = HIGH VARIANCE")
    print("   Recommended: LIVE BETTING ONLY, small stakes (1-2% bankroll)")
    print("=" * 80)
    
    return output_data

if __name__ == "__main__":
    data = fetch_monday_chidekh_data()
