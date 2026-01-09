"""
Quick Live Simulation: Liam Kilmer vs Garcia Raul
Starting at 2-2 games, Liam serving
"""

import numpy as np
from profitable_simulator import ProfitableSimulator

# Create simulator
sim = ProfitableSimulator(bankroll=1000, target=5000)

# Setup match
sim.setup_match("Liam Kilmer", "Garcia Raul", surface='Hard', best_of=3)

# Set current score: 2-2 games, Liam (player 0) serving
sim.sets = [0, 0]
sim.games = [2, 2]
sim.points = [0, 0]
sim.server = 0  # Liam serving

print(f"\n{'='*70}")
print(f"🎾 LIVE MATCH SIMULATION")
print(f"{'='*70}")
print(f"Liam Kilmer vs Garcia Raul")
print(f"Current Score: 0-0 sets, 2-2 games")
print(f"Liam Kilmer serving")
print(f"{'='*70}")
print(f"\n💰 Starting Bankroll: ${sim.bankroll:,.0f}")
print(f"🎯 Target: ${sim.target:,.0f}")
print(f"{'='*70}\n")

input("Press Enter to start simulation from current score...")

# Run simulation
sim.run()
