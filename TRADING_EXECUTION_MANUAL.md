# 🎾 TENNIS IN-PLAY TRADING — EXECUTION MANUAL

## Complete Guide: Terminal Intelligence → Real Platform Execution

> **Portfolio**: 100 units | **Goal**: Profit on every event via combined position management
> **Philosophy**: Every match is ONE trade with multiple legs. We enter, scale, hedge, and exit as a combined position — never naked, never unmanaged.

---

## TABLE OF CONTENTS

1. [Core Philosophy — The Combined Event Approach](#1-core-philosophy)
2. [Portfolio Structure — 100 Unit Allocation](#2-portfolio-structure)
3. [Pre-Match: Reading the Edge Panel](#3-pre-match)
4. [Entry Rules — When & How Much](#4-entry-rules)
5. [In-Play State Machine — Signal → Action → Size](#5-in-play-state-machine)
6. [Hedge Execution — Lock Profit, Limit Loss](#6-hedge-execution)
7. [Exit Rules — Taking Profit / Cutting Loss](#7-exit-rules)
8. [Position Sizing Table — Every State](#8-position-sizing-table)
9. [Platform Execution Checklist](#9-platform-execution)
10. [Risk Rules — Non-Negotiable](#10-risk-rules)
11. [Worked Examples](#11-worked-examples)
12. [Quick Reference Card](#12-quick-reference)

---

## 1. CORE PHILOSOPHY

### The Combined Event Approach

**Every match is a single trade with multiple legs.** You don't just "back P1 and hope." You actively manage one combined position across the full event lifecycle:

```
PRE-MATCH → ENTRY → SCALE UP/DOWN → HEDGE → RE-ENTER → HEDGE → EXIT
```

The terminal gives you **three intelligence layers**:

| Layer | Source | What It Tells You |
|-------|--------|-------------------|
| **Edge Panel** | Elo model vs bookmaker odds | Pre-match value (where the edge is) |
| **Point Tracker** | Markov chain + live stats | In-play position signals at every state change |
| **Delta-Neutral Calculator** | Entry price vs current probability | Exactly how much to hedge to guarantee profit |

### Guaranteed Profit Rule

To come out profitable on EVERY event, follow this ironclad principle:

> **Enter only with edge → Scale into confirmed edge → Hedge to lock green → Never let green turn red.**

If you entered at value odds and the position moves in your favor, the hedge math GUARANTEES profit on both outcomes. The terminal calculates this for you.

---

## 2. PORTFOLIO STRUCTURE — 100 Units

### Bankroll Tiers

| Tier | Units | Purpose | Max Per Event |
|------|-------|---------|---------------|
| **A — Core** | 60 units | Main trading bankroll | 6u max initial entry |
| **B — Hedge Reserve** | 30 units | Hedging & delta-neutral exits | Used as needed |
| **C — Emergency** | 10 units | Emergency hedges / margin | Only for 🚨 signals |

### Position Sizing Rules (¼ Kelly)

The terminal calculates **Full Kelly %** from edge. You use **¼ Kelly** for safety:

| Full Kelly | ¼ Kelly | Units (of 100) | Classification |
|-----------|---------|-----------------|----------------|
| 1–4% | 0.25–1% | 0.25–1u | **MICRO** — minimum viable |
| 4–8% | 1–2% | 1–2u | **SMALL** — standard entry |
| 8–16% | 2–4% | 2–4u | **MEDIUM** — confirmed edge |
| 16–24% | 4–6% | 4–6u | **LARGE** — strong value |
| >24% | 6%+ | 6u MAX | **MAXIMUM** — capped at 6u |

**CRITICAL**: Never exceed 6 units initial entry. Never have more than 15 units exposed across all active events simultaneously.

---

## 3. PRE-MATCH — Reading the Edge Panel

### Step 1: Open the Terminal Edge Panel

Click any match → the right panel shows:

1. **MODEL PROBABILITY** — Our Elo+surface model's true probability
2. **BOOKMAKER ODDS & EDGE** — Implied probability vs model → edge %
3. **VALUE BET SIGNALS** — Color-coded: 🔥 STRONG (>5%) / ⚡ MODERATE (>2%) / ✗ Negative
4. **KELLY STAKING** — Exact unit recommendation

### Step 2: Identify Value

Read these numbers from the Edge Panel:

```
MODEL P:       68.2%         ← terminal's true probability
BOOK ODDS:     @1.85         ← what the bookmaker offers
IMPLIED P:     54.1%         ← what the odds imply (1/1.85)
EDGE:          +14.1%        ← model minus implied (68.2 - 54.1)
FAIR ODDS:     @1.47         ← what odds SHOULD be (1/0.682)
OVERROUND:     4.2%          ← bookmaker's juice
¼ KELLY:       3.8u          ← recommended stake
```

### Step 3: Decision Matrix

| Edge | Signal | Action | Size |
|------|--------|--------|------|
| **> +5%** | 🔥 STRONG | **ENTER** full ¼ Kelly | 2–6u |
| **+2% to +5%** | ⚡ MODERATE | **ENTER** half ¼ Kelly | 1–2u |
| **0% to +2%** | — | **SKIP** or tiny probe | 0–0.5u |
| **< 0%** | ✗ NEGATIVE | **NO ENTRY** | 0u |
| **< −5%** | ✗✗ | Consider **LAY** (opposing side has value) | 1–3u lay |

### Step 4: Check Context

Before entering, verify in MATCH CONTEXT section:
- **Surface**: Clay dampens favorites (more upsets). Grass amplifies (serve dominant).
- **Best of**: BO5 favors the better player significantly more than BO3.
- **Round**: Later rounds = tighter matches = smaller edges.
- **H2H/Form**: Terminal doesn't show this — check externally if edge is <3%.

---

## 4. ENTRY RULES — When & How Much

### Pre-Match Entry

| Condition | Stake | Justification |
|-----------|-------|---------------|
| Edge ≥ 5%, STRONG signal | Full ¼ Kelly (2–6u) | Clear value, model confident |
| Edge 2–5%, MODERATE signal | Half ¼ Kelly (1–2u) | Value exists but slim |
| Edge < 2% | 0u or 0.5u probe | Not enough edge to justify risk |
| Both players show value (overround very low) | Enter both sides at small size | Rare arb opportunity |

### In-Play Entry (from Point Tracker signals)

| Signal | Type | Stake from Tier A |
|--------|------|-------------------|
| **Markov P shifted >15%** — STRONG ENTRY | Match-level trend confirmed | 3–5u |
| **Markov P shifted 8–15%** — MODERATE ENTRY | Score shifting, trend emerging | 2–3u |
| **Markov P shifted 5–8%** — WEAK ENTRY | Early movement, probe only | 1–1.5u |
| **Set lead signal** — player leads in sets | Match structure advantage | 2–4u |
| **Break advantage** — net break in current set | Set structure advantage | 1.5–3u |
| **BREAK POINT** 🔴 — 30-40 or AD out | Game micro-entry (scalp) | 1–2u (SCALP) |
| **Serve dominance gap >15%** — stats signal | Systematic advantage confirmed | 2–3u |
| **DF crisis ≥4 DFs** — opponent struggling | Serve reliability broken | 1.5–2.5u |
| **Points imbalance >6%** — statistical edge | Volume-proven advantage | 1.5–3u |

### Entry Timing — When to Pull the Trigger

| Moment | Why It's Good |
|--------|---------------|
| **Start of 2nd set** (player won 1st) | Market hasn't fully adjusted. Set winner often priced too cheaply. |
| **After a break** (3-1 or 4-2 in set) | Break holder is underpriced for the set. |
| **Break point at 4-4 or 5-5** | Huge leverage moment — correct side pays big. |
| **Start of 3rd set** (split sets) | 50/50 situation — wait for model edge vs odds. |
| **Player serving for match** | If opponent's odds are very long, consider value LAY of server. |

---

## 5. IN-PLAY STATE MACHINE — Signal → Action → Size

This is the **core execution engine**. At every state change (every point), the terminal fires signals. Here's exactly what to do:

### MATCH-Level States

| State | Terminal Signal | Action | Position Size (of 100u) |
|-------|----------------|--------|------------------------|
| **Match start, pre-entry** | Edge Panel shows +5%+ | ENTER on valued side | 2–6u (¼ Kelly) |
| **0-0 first set, no edge** | HOLD — "no clear edge" | WAIT, don't force | 0u |
| **P shifted +5–8%** | WEAK ENTRY | Add 1u probe | Total: 1u |
| **P shifted +8–15%** | MODERATE ENTRY | Enter/add 2u | Total: 2–4u |
| **P shifted +15%+** | STRONG ENTRY | Full position 3–5u | Total: 3–6u |
| **Set lead (1-0 sets)** | SET LEAD ENTRY | Add 1–2u if not at max | Total: 3–6u |
| **Two-set lead (2-0)** | STRONG SET LEAD | Hold full position | Total: 4–6u |
| **Match point for your side** | MATCH POINT ENTRY | Hold + prepare exit at market | 4–6u → hedge exit |
| **Dominant >80% probability** | DOMINANT ENTRY | Take profit via hedge (green on both sides) | Begin hedge sequence |
| **Momentum reversal EXIT** | Fav trailing in points | Reduce position 50% | Cut to 2–3u |
| **Match on knife-edge 48–52%** | HEDGE signal | Hedge to lock breakeven or small green | Hedge from Tier B |

### SET-Level States

| State | Terminal Signal | Action | Position Size |
|-------|----------------|--------|---------------|
| **Break advantage 3-1, 4-2** | BREAK ADVANTAGE ENTRY | Add 1.5u on break holder | +1.5u |
| **5-3, 5-4 serving for set** | STRONG — serving for set | Hold/add 1u | +1u |
| **5-3, 5-4 returning for set** | MODERATE — close to set | Hold position, prepare hedge | 0 change |
| **Tiebreak, ≥3 mini-break lead** | TB ENTRY | Add 1u on leader | +1u |
| **Tiebreak, close (e.g. 4-3)** | TB HEDGE | Hedge 70% of set position | Hedge 70% |
| **Set level at 3-3, 4-4** | HOLD — wait for break | No action, conserve | 0 change |
| **BREAK! just happened** | STRONG ENTRY | Add 2u on break holder | +2u |
| **Break back! two breaks trading** | EXIT set position | Exit set-level adds | Remove set adds |

### GAME-Level States (Micro-Trading)

| State | Terminal Signal | Action | Position Size |
|-------|----------------|--------|---------------|
| **0-30 on server you backed** | HEDGE 60% | Place hedge bet Tier B | Hedge 60% of game adds |
| **0-40 on server you backed** | HEDGE 90% 🛡 | Emergency hedge | Hedge 90% |
| **30-40 or AD-out (break point)** | 🔴 BREAK POINT ENTRY | Scalp 1u on returner | +1u scalp |
| **Deuce after break point** | HEDGE 50% | Partial hedge, volatile | Hedge 50% |
| **Server loses grip 30-30+** | HEDGE 40% | Partial hedge | Hedge 40% |
| **Game point (40-15, 40-30)** | EXIT game position | Server stabilized, close game scalps | Close game adds |
| **Server stable (<12% break)** | EXIT | Close any game-level adds | Remove game adds |
| **Break opp >25% (Markov)** | ENTRY on returner | Scalp entry | +1u |
| **Break opp >45% (Markov)** | STRONG ENTRY on returner | Scalp entry | +1.5u |

### HEDGE-Level States (Protection)

| State | Terminal Signal | Action | Position Size |
|-------|----------------|--------|---------------|
| **P reversed >8%, now near 50%** | HEDGE up to 90% | Major hedge from Tier B | 90% of position hedged |
| **Break-back danger (30%+ opp)** | HEDGE 45–80% | Hedge set position | Hedge % per signal |
| **TB within 2 pts, no lead** | HEDGE 70% | Hedge set exposure | Hedge 70% |
| **🚨 MATCH POINT AGAINST** | HEDGE 95% | Emergency — near total hedge | 95% from Tier C |
| **Tight match (pts within 3)** | HEDGE moderate | Reduce variance | Hedge 30–50% |
| **High volatility σ>6%** | HEDGE moderate | Reduce exposure | Hedge 30–40% |

---

## 6. HEDGE EXECUTION — Lock Profit, Limit Loss

### The Core Formula

When you have a position and want to guarantee profit on both outcomes:

```
HEDGE STAKE = (Original Stake × Original Odds) / Hedge Odds

Example:
- Entered: 4u on Sinner @ 2.50 (to win 10u total return)
- Now: Sinner winning, Alcaraz odds drifted to @ 4.00
- Hedge: 10u / 4.00 = 2.5u on Alcaraz

P&L if Sinner wins:  +10.0 - 4.0 - 2.5 = +3.5u ✅
P&L if Alcaraz wins: -4.0 + 10.0 - 4.0 = +3.5u ✅ (hedge pays 2.5 × 4.0 = 10.0, minus original 4u and hedge 2.5u)

GUARANTEED +3.5u either way!
```

### Hedge Decision Tree

```
Your position is GREEN (odds moved in your favor)?
├── YES → Edge still positive?
│   ├── YES → HOLD. Don't hedge yet. Let it run.
│   └── NO → HEDGE NOW. Lock the green.
│       ├── >80% probability → Full hedge (lock profit both sides)
│       ├── 65–80% → Partial hedge (50–70% of position)
│       └── 55–65% → Small hedge (30% of position)
│
└── NO (position is RED — odds moved against you)
    ├── Terminal shows EXIT signal?
    │   ├── YES → CUT THE LOSS. Exit at market.
    │   └── NO → How red?
    │       ├── <2u loss → HOLD if edge still positive
    │       ├── 2–4u loss → REDUCE 50%
    │       └── >4u loss → EXIT (stop loss hit)
```

### When to Execute Each Hedge Type

| Hedge % | When | What to Do on Platform |
|---------|------|----------------------|
| **30–40%** | Early warning, game-level pressure | Place small counter-bet |
| **50%** | Deuce grind, break point saved | Place equal counter-bet |
| **60–70%** | Server collapse 0-30, tiebreak | Place substantial counter-bet |
| **80–90%** | Probability reversed, break-back danger | Near-full hedge |
| **95%** | 🚨 Match point against | Emergency hedge from reserve |

---

## 7. EXIT RULES — Taking Profit / Cutting Loss

### Profit Exit (Green Book)

| Situation | Action | Expected P&L |
|-----------|--------|-------------|
| **Your player wins match** | Collect full payout | +Full profit |
| **Green on both sides via hedge** | Let match finish, profit either way | +Guaranteed green |
| **>80% probability, odds too short** | Full hedge exit | +Locked profit |
| **Match point for your side** | Option: hedge for guaranteed; or ride for max | +Choice |

### Loss Exit (Red Book)

| Situation | Action | Max Loss |
|-----------|--------|----------|
| **Edge turned negative** (Kelly <0) | Exit at market | Accept current loss |
| **Stop-loss hit** (P moved >15% against) | Exit immediately | −3u max per event |
| **Momentum reversal EXIT** signal fires | Cut 50%, watch | −1.5u if cut early |
| **🚨 Match point against** | 95% hedge | −0.5u (mostly hedged) |

### Stop-Loss Rules (NON-NEGOTIABLE)

| Rule | Trigger | Action |
|------|---------|--------|
| **Per-event stop** | −4u on any single event | Exit all positions in that match |
| **Per-session stop** | −8u across all events in session | Stop trading for the day |
| **Per-week stop** | −15u for the week | Review and reset |
| **Hedge trigger** | Position −2u and getting worse | Begin hedge sequence |

---

## 8. POSITION SIZING TABLE — Every State

### Complete State × Size Reference

**Portfolio: 100 units. Tier A (60u), Tier B hedge (30u), Tier C emergency (10u).**

| Match State | Signal | Entry Side | Size (units) | Running Exposure | Source Tier |
|-------------|--------|-----------|--------------|-----------------|-------------|
| **Pre-match, edge >5%** | 🔥 STRONG | Valued player | 3–6u | 3–6u | A |
| **Pre-match, edge 2–5%** | ⚡ MOD | Valued player | 1–2u | 1–2u | A |
| **1st set, break at 3-1** | BREAK ADV | Break holder | +1.5u | 3–7.5u | A |
| **1st set, 5-3 serving** | SRV FOR SET | Set leader | +1u | 4–8.5u | A |
| **1st set won** | SET LEAD | Set winner | hold/+1u | 4–9u | A |
| **2nd set, break at 3-2** | BREAK ADV | Break holder | +1u | 5–10u | A |
| **2nd set, tiebreak 4-1** | TB ENTRY | TB leader | +1u | 5–11u | A |
| **2nd set, tiebreak 5-5** | TB HEDGE | — | hedge 70% of set adds | hedge ~2u | B |
| **Up 2 sets to 0** | DOMINANT | Hold | hold position | maintain | — |
| **2-0, >80% P** | DOMINANT | Hedge exit | full hedge | hedge all, lock green | B |
| **Break back! sets level** | EXIT set | Reduce | −2u (remove set adds) | reduce to core | — |
| **3rd set, even 3-3** | HOLD | Wait | no change | maintain core | — |
| **3rd set, break at 4-3** | BREAK ADV | Break holder | +1.5u | core + 1.5u | A |
| **3rd set, serving for match** | MATCH PT | Your player | hold + prepare hedge | prepare B | — |
| **3rd set, match point AGAINST** | 🚨 HEDGE 95% | Emergency | hedge 95% | hedge from C | C |
| **Match over, you won** | — | Collect | — | +profit | — |
| **Match over, fully hedged** | — | Collect either | — | +locked profit | — |

### Game-Level Scalp Sizing

| Game Score | Signal | Scalp Size | Max Duration |
|-----------|--------|-----------|-------------|
| **0-30** (your server) | HEDGE 60% | 1u counter | Until game ends |
| **0-40** (your server) | HEDGE 90% | 1.5u counter | Until game ends |
| **30-40** (opponent serving) | 🔴 BP ENTRY | 1u on returner | Until game ends |
| **AD-out** (opponent serving) | 🔴 BP ENTRY | 1u on returner | Until game ends |
| **Deuce** (any) | HEDGE 50% | 0.5u counter | Until game ends |
| **40-15, 40-0** (your server) | Safe → close scalps | Close game adds | Immediate |

---

## 9. PLATFORM EXECUTION CHECKLIST

### Before the Match

- [ ] Open terminal, click the match
- [ ] Read Edge Panel: MODEL P, BOOK ODDS, EDGE %, KELLY
- [ ] Check the 💎 RECOMMENDED line at the bottom of the panel
- [ ] Open your betting platform, navigate to the match
- [ ] If edge ≥ 2%: Place pre-match bet per Kelly sizing
- [ ] Record: **Player, Odds, Stake, Edge%** in your log

### During the Match (Every Point)

- [ ] Click into PointTracker (📡 live tab) for live Markov signals
- [ ] Watch for colored signal cards:
  - 🟢 Green border = **ENTRY** → Add to position
  - 🟡 Yellow border = **HOLD** → Do nothing
  - 🔴 Red border = **HEDGE/EXIT** → Reduce or hedge
  - 🛡 Shield = **HEDGE** with specific % → Place counter-bet
- [ ] Execute each signal on your platform within 5 seconds
- [ ] After each execution, note: Side, Odds, Stake

### Hedge Execution Steps

1. Signal fires: "HEDGE 60% — server collapse"
2. Calculate: 60% × your current position = hedge amount
3. On platform: Back the OTHER player at current live odds
4. **Verify P&L**: If original wins → (original payout) − (hedge stake) = profit?
5. **Verify P&L**: If hedge wins → (hedge payout) − (original stake) = profit?
6. If both green → ✅ Locked. If one red → it's a partial hedge (acceptable).

### Post-Match

- [ ] Record final P&L
- [ ] Log all entry/exit/hedge timestamps and odds
- [ ] Check if stop-losses were respected
- [ ] Update running bankroll

---

## 10. RISK RULES — NON-NEGOTIABLE

### The 10 Commandments

1. **Never exceed 6u initial entry** on any single event
2. **Never exceed 15u total exposure** across all concurrent events
3. **Always use ¼ Kelly**, never full Kelly
4. **Never enter without positive edge** (model P > implied P)
5. **Always have hedge reserve** (30u minimum in Tier B)
6. **Hedge when green**, don't wait for more green
7. **Cut when −4u** on any single event, no exceptions
8. **Stop when −8u** for the session
9. **Game-level scalps expire** when the game ends — don't carry them
10. **Log every bet** — odds, stake, edge, signal that triggered it

### Risk Per Tier

| Tier | Max Exposed | Purpose |
|------|-------------|---------|
| A (60u) | Max 15u deployed at once | Core entries |
| B (30u) | Used only for hedging | Always available for counter-bets |
| C (10u) | Used only for 🚨 emergencies | Match point against, collapse scenarios |

### Correlation Rule

**Never have >3 events running simultaneously.** Tennis matches overlap and if rain/retirement hits, you need margin.

| Events | Max Per Event | Total Max |
|--------|---------------|-----------|
| 1 event | 6u | 6u |
| 2 events | 5u each | 10u |
| 3 events | 4u each | 12u |
| 4+ events | DON'T | — |

---

## 11. WORKED EXAMPLES

### Example A: Clean Value Entry → Hedge → Guaranteed Profit

```
MATCH: Sinner vs Rune — ATP Clay, BO3

PRE-MATCH (Terminal Edge Panel):
  Model P:     72% Sinner
  Book Odds:   Sinner @ 1.75 (implied 57.1%)
  Edge:        +14.9% → 🔥 STRONG
  ¼ Kelly:     4.2u
  
ACTION: Enter 4u on Sinner @ 1.75 (from Tier A)
  Potential payout: 4 × 1.75 = 7.0u return

SET 1 — Sinner breaks, leads 4-2:
  Signal: BREAK ADVANTAGE · MODERATE
  Markov P: 81% Sinner
  Rune odds drifted: @ 5.50
  ACTION: Hold. Edge still strong.

SET 1 — Sinner wins 6-3:
  Signal: SET LEAD · Sinner at 88%
  Rune odds: @ 8.00
  ACTION: Begin hedge consideration.
  
  HEDGE CALC:
    Hedge stake = (4 × 1.75) / 8.00 = 0.875u → round to 1u on Rune @ 8.00
    If Sinner wins: 7.0 - 4.0 - 1.0 = +2.0u ✅
    If Rune wins:   8.0 - 4.0 - 1.0 = +3.0u ✅
    
  ACTION: Place 1u on Rune @ 8.00 (from Tier B)
  
RESULT: Guaranteed +2.0u to +3.0u regardless of outcome! 🎯

ACTUAL: Sinner wins 6-3, 6-4 → Collect +2.0u net
```

### Example B: Wrong Entry → Damage Control → Small Loss

```
MATCH: Pegula vs Sabalenka — WTA Hard, BO3

PRE-MATCH:
  Model P:     55% Pegula
  Book Odds:   Pegula @ 2.10 (implied 47.6%)
  Edge:        +7.4% → 🔥 STRONG
  ¼ Kelly:     1.9u
  
ACTION: Enter 2u on Pegula @ 2.10 (Tier A)

SET 1 — Sabalenka breaks, leads 3-1:
  Signal: MOMENTUM REVERSAL EXIT · Pegula trailing in total points 12-18
  Markov P: 38% Pegula (was 55%)
  ACTION: Cut 50% → Exit 1u at current Pegula odds @ 2.80
  Loss on exited unit: Entered @ 2.10, but EXIT means accepting current market.
  Remaining: 1u still on Pegula.

SET 1 — Sabalenka wins 6-2:
  Signal: 🛡 Probability reversed · was P1 +7%, now 28%-72%
  Hedge signal: HEDGE 80%
  ACTION: Hedge 80% of remaining 1u → Place 0.8u on Sabalenka @ 1.45 (Tier B)
  
  If Pegula wins: (1u × 2.10) - 1u - 0.8u = +0.3u − 1u(already lost) = −0.7u
  If Sabalenka wins: (0.8 × 1.45) - 1u - 0.8u = 1.16 - 1.8 = −0.64u

RESULT: Capped loss at −0.7u instead of −3u. Damage controlled. ✅
(Without hedging: lost 2u + 1u = −3u. With management: −0.7u)
```

### Example C: Break Point Scalp → Quick Profit

```
IN-PLAY: Djokovic serving at 2-3, 30-40 in 2nd set

Signal: 🔴 BREAK POINT! Conv: 38% · srv 1st%: 58% · DFs: 3
Side: Opponent (returner) — back opponent for set
Markov break opp: 38%

ACTION: Scalp 1u on opponent's set at current odds (e.g. set @ 2.20)

OUTCOME A — Break happens:
  Opponent leads 4-2 with break. Set odds crash to @ 1.25.
  ACTION: Close scalp. Hedge 1u back the other way:
    Hedge: (1 × 2.20) / 1.25 = 1.76u → guaranteed profit
    Or simply cash out the position.
  PROFIT: ~+0.8u

OUTCOME B — Djokovic holds (deuce → AD-in → hold):
  Signal: Server stabilized · hold 88%
  ACTION: Exit scalp at small loss.
  LOSS: ~-0.3u (game-level scalps are sized small for this reason)
```

### Example D: Full Tiebreak Hedge Sequence

```
SET 2 — You have 3u on Player A. Set at 5-6 → Tiebreak.

TIEBREAK STARTS (0-0):
  Signal: HEDGE 70% — tiebreak coin-flip variance
  ACTION: Place 2u (70% of 3u, rounded) on Player B set @ 2.00 (Tier B)

TB 1-4 — Player B leading:
  Signal: TB ENTRY · Player B has mini-break advantage
  Your hedge is already working.
  ACTION: Hold hedge position.

TB 4-6 — Player B has set point:
  Signal: 🚨 Emergency area
  Your 2u hedge will pay 4u if B takes set.
  ACTION: Hold. Hedge is protecting you.

RESULT — Player B wins TB 7-4:
  Original 3u on A → this set lost, but match continues
  Hedge 2u on B set @ 2.00 → pays 4.0u return
  Net for set: +4.0 - 2.0 - 0 = +2.0u gain on the set (offsets the damage to match position)
  Match position: Still have 3u on A for the decider, but set hedge gave +2u buffer.
```

---

## 12. QUICK REFERENCE CARD

### Signal Color → Action

| Color | Signal | Your Action | Sizing |
|-------|--------|-------------|--------|
| 🟢 **GREEN** | ENTRY | Place bet on indicated side | Per Kelly table |
| 🟡 **YELLOW** | HOLD | Do nothing. Wait. | — |
| 🔴 **RED** | EXIT | Close position at market | Sell/close |
| 🛡 **SHIELD** | HEDGE X% | Counter-bet X% of position | From Tier B |
| 🚨 **SIREN** | EMERGENCY | Hedge 95% immediately | From Tier C |

### Edge → Stake Quick Lookup (100u bankroll)

| Edge % | Strength | Stake |
|--------|----------|-------|
| +15%+ | 🔥🔥 | 5–6u |
| +10–15% | 🔥 | 3–5u |
| +5–10% | 🔥 | 2–3u |
| +3–5% | ⚡ | 1–2u |
| +2–3% | ⚡ | 0.5–1u |
| 0–2% | — | SKIP |
| Negative | ✗ | NO ENTRY |

### Hedge Formula (Memorize This)

```
Hedge Stake = (Your Stake × Your Odds) / Current Opposite Odds

If both P&L scenarios are positive → ✅ LOCKED GREEN
If one scenario is negative → partial hedge (okay if small)
```

### Game Score → Instant Action

| Score (on YOUR server) | Risk Level | Action |
|------------------------|-----------|--------|
| 40-0, 40-15 | ✅ Safe | Hold |
| 40-30 | ⚠️ Normal | Hold |
| 30-30, Deuce | ⚠️ Elevated | Prepare hedge |
| 30-40, AD-out | 🔴 Break point | HEDGE 60% |
| 0-30 | 🔴 Trouble | HEDGE 60% |
| 0-40 | 🚨 Collapse | HEDGE 90% |

| Score (OPPONENT serving) | Opportunity | Action |
|--------------------------|------------|--------|
| 0-30, 15-30 | Building pressure | PROBE 1u on returner |
| 30-40 | 🔴 BREAK POINT | ENTER 1u scalp |
| AD-out | 🔴 BREAK POINT | ENTER 1u scalp |
| Deuce | Coin flip | Hold if in, skip if out |
| 40-0, 40-15 | No opportunity | Skip / exit game scalps |

### Daily Workflow

```
1. Open terminal → Scan all matches for EDGE ≥ 2%
2. Rank by edge size → Pick top 1-3 events  
3. Enter pre-match on strongest edges
4. Switch to PointTracker when matches go live
5. Execute signals: GREEN=enter, RED=exit, SHIELD=hedge
6. Goal: Lock green via hedging on every event
7. End session: Log all P&L, check vs stop-losses
```

---

## APPENDIX: Terminal Signals → Platform Bet Type Mapping

| Terminal Signal | Bet365 / Sportsbet | Betfair Exchange | Pinnacle |
|-----------------|-------------------|-----------------|----------|
| ENTRY on P1 ML | "Match Winner: Player 1" | BACK Player 1 | "1" moneyline |
| HEDGE on P2 ML | "Match Winner: Player 2" | BACK Player 2 | "2" moneyline |
| ENTRY set winner | "Set X Winner" | BACK set market | "Set Winner" |
| SCALP break point | "Next Game Winner" | BACK next game | "Game Winner" |
| LAY (negative edge) | — (can't lay) | LAY Player | — (can't lay) |
| FULL HEDGE exit | Back opposite side | BACK opposite | Back opposite |

> **Exchange (Betfair/Smarkets) is ideal** — you can LAY (bet against) and get tighter spreads.
> **Fixed-odds (Bet365/Pinnacle)** — you can only BACK, so hedging means backing the other side.

---

*Generated from Tennis Trading Terminal v1.0 intelligence system*
*Model: Elo + Surface + Markov Chain | Signals: 19 position signals + stats bonuses*
*Kelly: ¼ fractional | Bankroll: 100 units | Goal: Green on every event*
