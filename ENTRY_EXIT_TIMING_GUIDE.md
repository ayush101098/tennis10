# ⏱️ DELTA-NEUTRAL: ENTRY & EXIT TIMING GUIDE

## WHEN TO ENTER (Rule E1: Value Detection)

### Preconditions
```
✓ Score is 0-0 (game just starting)
✓ Break odds must be 2.80 to 3.40
✓ No existing position in current game
```

### Value Validation
```
1. TRUE PROBABILITY CALCULATION
   P(break) = function(
       - serve % (player 1)
       - return % (player 2)
       - ranking difference
       - momentum
   )
   
   Range: 10% - 55% (very conservative)

2. IMPLIED PROBABILITY FROM ODDS
   P(implied) = 1 / break_odds
   
   Example: 3.20 odds → 31.2% implied

3. EDGE CALCULATION
   EV% = (True_P × (odds - 1)) - (1 - True_P)
   
   If True_P = 35%, Odds = 3.20:
   EV% = (0.35 × 2.20) - 0.65 = -0.28 = -28%

4. ENTRY THRESHOLD
   Only enter if EV% ≥ +2.00%
   
   Interpretation:
   - +2.00% = break-even after fees
   - +5.00% = good value
   - +10.00%+ = excellent value (rare)
```

### Entry Decision Tree
```
Is score 0-0? 
  NO  → HOLD (wait for next game)
  YES → Continue

Are break odds in [2.80, 3.40]?
  NO  → HOLD (odds outside range)
  YES → Continue

Calculate true P(break) from stats
  Result: TP%

Calculate implied P from odds
  Result: IP% = 1 / odds

Calculate edge:
  EV% = (TP% × (odds-1)) - (1-TP%)

Is EV% ≥ +2.00%?
  NO  → SKIP (no value, market better than model)
  YES → ENTRY ✅
       Place $50 break bet
       Delta = +1.0 (aggressive)
```

---

## WHEN TO EXIT

### 1. EDGE DETERIORATION EXIT

**Monitor**: Every odds update

```python
def should_exit():
    current_edge = calculate_edge(current_odds, true_P)
    
    if current_edge < -1.0%:
        return True  # Exit signal
    
    return False
```

**Trigger Conditions**:
```
Before Entry:
  Entry edge: +5.00%
  
During Game (score progression):
  15-0: Current edge: +3.50% → Hold
  30-0: Current edge: +1.50% → Hold  
  30-15: Current edge: -0.50% → EXIT ⚠️
  
Interpretation:
  - Server is stronger than expected
  - Market odds adjusted in server's favor
  - No longer profitable to hold break bet
  - Close position to minimize further loss
```

### 2. EMERGENCY EXIT (Rule R1)

**Trigger**: Break odds > 8.0 without break point reached

```
Condition: break_odds > 8.0 AND game_state NOT IN [S4, S8]

Meaning: Server completely dominating
         But market still pricing break at extreme odds
         Suggests odds error or system malfunction

Action: Close both positions immediately
        (break bet + any existing hedge)
```

### 3. HEDGE ALTERNATIVE (Instead of Exit)

Three situations where hedge replaces exit:

#### H1: Server Dominance Hedge
```
Trigger: 3 consecutive points won by server + hold_odds ≤ 1.25

Meaning: Server very strong
         Market pricing hold as very likely
         Full hedge available at good odds

Action: Full hedge = $stake × entry_odds / hold_odds
```

#### H2: Deuce Full Hedge (BEST STATE)
```
Trigger: Score reaches 40-40 (deuce)

Meaning: Game most uncertain
         Break and hold equally likely
         Perfect time to neutralize

Action: Full hedge = $stake × entry_odds / hold_odds
        Delta: +1.0 → 0.0 (neutral)
```

#### H3: Partial Hedge (Missed Break Point)
```
Trigger: State = S7 (break point saved) 
         AND odds increased ≥ 40%

Example:
  After break point saved:
  - Entry odds were 3.20
  - Current odds: 3.20 × 1.40 = 4.48
  
Meaning: Break no longer likely
         But odds haven't normalized
         Small hedge recommended

Action: Partial hedge = 0.5 × (stake × entry_odds / hold_odds)
        Delta: +1.0 → +0.5
```

---

## POSITION LIFECYCLE

```
START: Game at 0-0
│
├─ Check entry conditions
│  ├─ Odds outside [2.80-3.40]? → SKIP to next game
│  ├─ Edge < +2%? → SKIP to next game
│  └─ Edge ≥ +2%? → ENTRY ✅
│
POSITION OPEN: Long break ($50 @ odds)
│  Delta = +1.0
│
├─ Game unfolds (score updates)
│  ├─ Score 15-0, 30-0, 30-15, 30-30...
│  ├─ Monitor edge continuously
│  │
│  ├─ H2 triggered? (Score: 40-40)
│  │  └─ HEDGE: Full hedge
│  │     Delta = 0.0 (neutral)
│  │     Hold hedge position open
│  │
│  ├─ H1 triggered? (3 points + hold ≤ 1.25)
│  │  └─ HEDGE: Full hedge
│  │     Delta = 0.0 (neutral)
│  │
│  ├─ H3 triggered? (BP saved + odds jump 40%)
│  │  └─ HEDGE: Partial hedge
│  │     Delta = +0.5 (partial)
│  │
│  └─ Edge drops to -1%?
│     └─ EXIT: Close both positions
│        Minimize further loss
│
SETTLEMENT: Game ends
│
├─ Break occurs?
│  ├─ Position A: +(odds-1) × stake ✅
│  └─ Position B (if exists): -stake ❌
│
├─ Hold occurs?
│  ├─ Position A: -stake ❌
│  └─ Position B (if exists): +(odds-1) × stake ✅
│
END: Record P&L, reset for next game
```

---

## REAL-TIME MONITORING

### Dashboard Updates (every odds tick):

```
1. ENTRY GATE (Game at 0-0)
   Current odds: [BREAK] ← → [HOLD]
   Status: Outside range | In range | No edge | ✅ ENTRY
   
2. POSITION TRACKING (During game)
   Account A (Break): $50 @ 3.20
   Account B (Hedge): [None] OR $84 @ 1.90
   Delta: +1.0 OR 0.0
   
3. EDGE MONITORING
   Entry edge: +5.20%
   Current edge: +2.10% [DROP WARNING ⚠️]
   Edge threshold: -1.00%
   
4. HEDGE SIGNALS
   H1 Status: 1/3 points → Need 3 straight
   H2 Status: Current score 30-30
   H3 Status: No break point yet
```

### Key Metrics to Watch

```
Pre-Entry:
  ✓ Break odds trend (rising = smaller edge)
  ✓ Hold odds trend (falling = better hedge)
  ✓ Match context (close/one-sided)

During Position:
  ✓ Score vs model expectations
  ✓ Breakpoints reached
  ✓ Momentum shifts
  ✓ Edge erosion rate

Post-Settlement:
  ✓ P&L result
  ✓ Hedge effectiveness
  ✓ Model accuracy vs actual outcome
```

---

## EXAMPLE SCENARIOS

### Scenario 1: Value Entry + Deuce Hedge
```
T=0: Entry
  Score: 0-0
  Odds: Break 3.20 | Hold 1.85
  True P: 38%
  Implied P: 31.2%
  Edge: +16.4% ✅
  Action: ENTRY $50 @ 3.20
  
T=5min: Early Game
  Score: 15-0
  Odds: Break 3.10 | Hold 1.88
  Edge: +14.2%
  Action: HOLD
  
T=15min: Even Game
  Score: 30-30
  Odds: Break 2.80 | Hold 1.90
  Edge: +10.5%
  Action: HOLD
  
T=20min: Deuce
  Score: 40-40
  Odds: Break 2.70 | Hold 1.85
  Action: H2 TRIGGERED
         Full hedge: $50 × 3.20 / 1.85 = $86.49
  
T=25min: Settlement
  Outcome: HOLD wins
  Position A: -$50
  Position B: +$159.35
  Net: +$109.35 ROI = +73%
```

### Scenario 2: Edge Deterioration Exit
```
T=0: Entry
  Edge: +8.0%
  Position: $50 break
  
T=10min: Score 15-0
  Edge: +5.2%
  Status: Good
  
T=12min: Score 30-0
  Edge: +2.1%
  Status: Eroding, near threshold
  
T=13min: Score 30-15
  Odds jump from 2.80 to 2.45
  Edge: -1.5%
  Action: EXIT ⚠️
         Close position immediately
         Loss: -$50
         Reason: Edge turned negative
```

### Scenario 3: No Value - Skip Entry
```
T=0: Game at 0-0
  Break odds: 3.20
  Odds in range: ✓
  
Calculate edge:
  Player 1 stats: 58% serve, rank 180
  Player 2 stats: 68% serve, rank 45
  True P(break): 18% (weak server)
  Implied P: 31.2%
  Edge: -42%
  
Decision: SKIP
Reason: Market pricing is BETTER than model
        Server too weak for break value
        Wait for next game
```

---

## Summary: Entry & Exit Decision Matrix

| Situation | Signal | Action |
|-----------|--------|--------|
| 0-0, odds in range, edge ≥ +2% | E1 | ENTRY |
| 0-0, odds in range, edge < +2% | - | SKIP |
| Score progresses, edge > 0% | - | HOLD |
| Score progresses, edge < -1% | - | EXIT |
| Server 3 points straight, hold ≤ 1.25 | H1 | HEDGE |
| Score reaches 40-40 | H2 | HEDGE |
| Break point saved, odds +40% | H3 | HEDGE |
| Break odds > 8.0, no break point | R1 | EMERGENCY EXIT |

