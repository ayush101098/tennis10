"""
🎯 MULTI-MATCH SIMULATOR - TARGET: $1000 → $5000
===================================================

Simulates multiple matches with Markov chain probabilities
and profitable betting until target is reached.
"""

import numpy as np
from profitable_simulator import ProfitableSimulator
import random

def run_multiple_matches(starting_bankroll=1000, target=5000):
    """Run multiple matches until target is reached or bankrupt"""
    
    # Player pool (mix of styles)
    players = [
        ("Novak Djokovic", "Rafael Nadal"),
        ("Carlos Alcaraz", "Jannik Sinner"),
        ("Daniil Medvedev", "Alexander Zverev"),
        ("Stefanos Tsitsipas", "Andrey Rublev"),
        ("Taylor Fritz", "Tommy Paul"),
        ("Holger Rune", "Grigor Dimitrov"),
        ("Casper Ruud", "Hubert Hurkacz"),
    ]
    
    bankroll = starting_bankroll
    match_count = 0
    total_bets = 0
    wins = 0
    losses = 0
    
    print(f"\n{'='*70}")
    print(f"🎯 MULTI-MATCH BETTING SIMULATION")
    print(f"{'='*70}")
    print(f"Starting Bankroll: ${bankroll:,.0f}")
    print(f"Target: ${target:,.0f}")
    print(f"Required Growth: {(target/bankroll - 1)*100:.0f}%")
    print(f"{'='*70}\n")
    
    while bankroll < target and bankroll >= 50:
        match_count += 1
        
        # Select random match
        p1, p2 = random.choice(players)
        surface = random.choice(['Hard', 'Clay', 'Grass'])
        
        print(f"\n{'─'*70}")
        print(f"MATCH {match_count}: {p1} vs {p2} ({surface})")
        print(f"Current Bankroll: ${bankroll:,.0f}")
        print(f"{'─'*70}")
        
        # Create simulator with current bankroll
        sim = ProfitableSimulator(bankroll=bankroll, target=target)
        sim.setup_match(p1, p2, surface=surface, best_of=3)
        
        # Optionally start from random score
        if random.random() > 0.5:
            # Start mid-match
            sim.games = [random.randint(0, 3), random.randint(0, 3)]
            print(f"Starting from: {sim.games[0]}-{sim.games[1]} games")
        
        # Run match
        sim.run()
        
        # Update bankroll
        bankroll = sim.bankroll
        total_bets += len(sim.bets)
        
        # Count W/L
        if sim.bankroll > sim.starting_bankroll:
            wins += 1
        else:
            losses += 1
        
        # Progress update
        progress = (bankroll / target) * 100
        print(f"\n📊 Progress: ${bankroll:,.0f} / ${target:,.0f} ({progress:.1f}%)")
        print(f"   Matches: {match_count} | W/L: {wins}-{losses} | Bets: {total_bets}")
        
        if match_count >= 20:
            print(f"\n⏰ Reached match limit (20 matches)")
            break
    
    # Final report
    print(f"\n{'='*70}")
    print(f"🏁 FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Starting Bankroll: ${starting_bankroll:,.0f}")
    print(f"Final Bankroll: ${bankroll:,.0f}")
    print(f"P&L: ${bankroll - starting_bankroll:+,.0f}")
    print(f"ROI: {(bankroll/starting_bankroll - 1)*100:+.1f}%")
    print(f"\nMatches Played: {match_count}")
    print(f"Match Record: {wins}W - {losses}L ({wins/(wins+losses)*100:.0f}% win rate)")
    print(f"Total Bets: {total_bets}")
    
    if bankroll >= target:
        print(f"\n🎉 TARGET ACHIEVED! ${bankroll:,.0f}")
    elif bankroll < 50:
        print(f"\n💔 BANKRUPT at ${bankroll:.0f}")
    else:
        print(f"\n📈 Profit made but target not reached")
    
    print(f"{'='*70}\n")
    
    return bankroll


if __name__ == "__main__":
    print("\n🎾 PROFITABLE TENNIS BETTING - MULTI-MATCH MODE")
    
    try:
        starting = float(input("\nStarting bankroll [$1000]: ").strip() or "1000")
        target = float(input("Target bankroll [$5000]: ").strip() or "5000")
        
        final = run_multiple_matches(starting, target)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
