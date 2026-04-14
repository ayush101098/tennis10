"""
🎾 DELTA-NEUTRAL BETTING SYSTEM WITH TRUE PROBABILITY
===================================================

Demo showing:
1. True probability calculation from player stats
2. Value bet identification
3. Entry/exit timing based on edge deterioration

CRITICAL INSIGHT: Entry only happens when true P > implied P (positive edge)

Run: python delta_neutral_with_true_p.py
"""

from delta_neutral_system import (
    DeltaNeutralBettingSystem,
    ProbabilityEngine,
    Signal
)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def demo_true_p_entry_exit():
    print_section("DELTA-NEUTRAL SYSTEM: TRUE PROBABILITY INTEGRATION")
    
    print("""
✅ KEY FEATURE: Only enter when positive expected value exists

The system now calculates:
  • True P(break) from player statistics (serve %, return %, rankings)
  • Implied P(break) from bookmaker odds
  • Edge = (True P × (odds - 1)) - (1 - True P)
  
Entry rules:
  ✓ Odds must be in [2.80, 3.40] range
  ✓ Edge must be ≥ +2.00% (statistical threshold)
  ✓ Otherwise: SKIP (no value detected)
    """)
    
    # SCENARIO 1: WEAK SERVER - NO VALUE
    print_section("SCENARIO 1: WEAK SERVER - NO VALUE")
    print("Match: Ymer (195 rank, 62% serve) vs Shevchenko (147 rank, 36% return)\n")
    
    system = DeltaNeutralBettingSystem()
    system.set_player_stats(
        p1_serve_pct=62.0, p1_return_pct=34.0,
        p2_serve_pct=65.0, p2_return_pct=36.0,
        p1_rank=195, p2_rank=147
    )
    
    system.update_score(0, 0, 0, 0)
    signal = system.process_odds(break_odds=3.20, hold_odds=1.85)
    
    print(f"Break odds: 3.20, Hold odds: 1.85")
    print(f"True break P: {system.betting_state.true_prob_break:.1%}")
    print(f"Implied P @ 3.20: {1/3.20:.1%}")
    print(f"Edge: {system.betting_state.current_edge:+.2%}")
    print(f"Result: {signal.value}\n   → SKIPPED: Negative edge (no value)\n")
    
    # SCENARIO 2: STRONG SERVER - VALUE DETECTED
    print_section("SCENARIO 2: STRONG SERVER - VALUE DETECTED ✅")
    print("Match: Top player (5 rank, 72% serve) vs Lower ranked (250 rank, 28% return)\n")
    
    system = DeltaNeutralBettingSystem()
    system.set_player_stats(
        p1_serve_pct=72.0, p1_return_pct=40.0,
        p2_serve_pct=55.0, p2_return_pct=28.0,
        p1_rank=5, p2_rank=250
    )
    
    system.update_score(0, 0, 0, 0)
    signal = system.process_odds(break_odds=2.85, hold_odds=1.50)
    
    print(f"Break odds: 2.85, Hold odds: 1.50")
    print(f"True break P: {system.betting_state.true_prob_break:.1%}")
    print(f"Implied P @ 2.85: {1/2.85:.1%}")
    print(f"Edge: {system.betting_state.current_edge:+.2%}")
    
    if signal == Signal.ENTRY:
        print(f"\n✅ ENTRY EXECUTED!")
        print(f"   Stake: $50 @ 2.85 odds")
        print(f"   Delta: {system.betting_state.delta} (Aggressive)\n")
        
        # HEDGE AT DEUCE
        print_section("CONTINUING: HEDGE AT DEUCE")
        print("Game progresses... Score reaches 40-40 (Deuce)\n")
        
        system.update_score(3, 3, 0, 0)
        deuce_signal = system.process_odds(break_odds=2.40, hold_odds=1.90)
        
        if deuce_signal == Signal.FULL_HEDGE:
            print(f"✅ H2 TRIGGERED: Deuce reached - full hedge executed")
            if system.betting_state.position_b:
                print(f"   Hedge: Hold ${system.betting_state.position_b.stake:.2f} @ 1.90 odds")
                print(f"   Delta: {system.betting_state.delta} (Now neutral)\n")
        
        # SETTLEMENT
        print_section("SETTLEMENT: Server holds game")
        pnl = system.settle_game("HOLD")
        
        print(f"Position A (Break $50 @ 2.85): {pnl['pnl_a']:+.2f}")
        if system.betting_state.position_b:
            print(f"Position B (Hold ${system.betting_state.position_b.stake:.2f} @ 1.90): {pnl['pnl_b']:+.2f}")
        print(f"\nTotal PnL: {pnl['total_pnl']:+.2f} ({pnl['roi']:+.1f}%)")
    
    # KEY INSIGHTS
    print_section("KEY INSIGHTS: ENTRY & EXIT TIMING")
    print("""
ENTRY TIMING:
  ✓ Only at 0-0 score
  ✓ Odds in [2.80, 3.40] range  
  ✓ Positive edge (true P > implied P by 2%+)
  ✗ Skip negative edge (market better than model)

EXIT TIMING:
  • Monitor edge during game (check_exit_opportunity)
  • Exit if edge drops below -1%
  • Or hedge when conditions allow (H1, H2, H3)

HEDGE ENTRY:
  H1: Server dominance (3 straight + hold odds ≤ 1.25)
  H2: Deuce reached (40-40) ← Best hedge state
  H3: Break point missed (odds jumped ≥40%)

POSITION SIZING:
  • Entry: $50 at break odds
  • Full hedge: S₂ = S₁ × O₁ / O₂
  • Partial hedge: 50% of full

DELTA TRACKING:
  +1.0 = Aggressive (break only)
  +0.5 = Partial (50% hedged)
   0.0 = Delta neutral (fully hedged)

RISK MANAGEMENT:
  • Emergency exit if odds > 8.0
  • Full hedge at critical states
  • Only enter with positive EV
  • Deterministic rules (no emotions)
    """)

if __name__ == "__main__":
    demo_true_p_entry_exit()
