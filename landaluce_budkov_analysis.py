"""
Markov Chain Analysis: Martin Landaluce vs Nicolai Budkov Kjaer
"""
import numpy as np
from scipy.special import comb

def p_win_game(p_point):
    """Calculate probability of winning a service game given point win probability."""
    p = p_point
    q = 1 - p
    p_4 = p**4
    p_5 = 4 * p**4 * q
    p_6 = 10 * p**4 * q**2
    p_deuce = comb(6, 3) * p**3 * q**3
    p_win_deuce = p**2 / (p**2 + q**2)
    return p_4 + p_5 + p_6 + p_deuce * p_win_deuce

def calculate_set_probability(p_hold_a, p_hold_b, p_serve_a, p_serve_b):
    """Calculate set win probability using Markov chain."""
    states = {}
    idx = 0
    for a in range(7):
        for b in range(7):
            states[(a, b)] = idx
            idx += 1
    states['TB'] = idx; idx += 1
    states['A_wins'] = idx; idx += 1
    states['B_wins'] = idx
    n_states = idx + 1
    
    P = np.zeros((n_states, n_states))
    p_break_b = 1 - p_hold_b
    
    for state, i in states.items():
        if state in ['A_wins', 'B_wins', 'TB']:
            continue
        a, b = state
        total_games = a + b
        a_serving = (total_games % 2 == 0)
        
        if a_serving:
            p_a_wins_game = p_hold_a
        else:
            p_a_wins_game = p_break_b
        
        new_a, new_b = a + 1, b + 1
        
        if new_a >= 6 and new_a - b >= 2:
            j_a_win = states['A_wins']
        elif new_a == 6 and b == 6:
            j_a_win = states['TB']
        else:
            j_a_win = states.get((new_a, b), states['A_wins'])
        
        if new_b >= 6 and new_b - a >= 2:
            j_b_win = states['B_wins']
        elif new_b == 6 and a == 6:
            j_b_win = states['TB']
        else:
            j_b_win = states.get((a, new_b), states['B_wins'])
        
        P[i, j_a_win] += p_a_wins_game
        P[i, j_b_win] += 1 - p_a_wins_game
    
    # Tiebreak
    p_a_point_avg = (p_serve_a + (1 - p_serve_b)) / 2
    p_a_tb = p_a_point_avg ** 2 / (p_a_point_avg ** 2 + (1 - p_a_point_avg) ** 2)
    P[states['TB'], states['A_wins']] = p_a_tb
    P[states['TB'], states['B_wins']] = 1 - p_a_tb
    P[states['A_wins'], states['A_wins']] = 1
    P[states['B_wins'], states['B_wins']] = 1
    
    start = np.zeros(n_states)
    start[states[(0, 0)]] = 1
    for _ in range(100):
        start = start @ P
    return start[states['A_wins']]

def p_win_match_bo3(p_set):
    """Probability of winning best of 3 match."""
    p = p_set
    return p**2 + 2 * p**2 * (1 - p)

# ============================================================
print('='*70)
print('MARKOV CHAIN MATCH ANALYSIS')
print('Martin Landaluce vs Nicolai Budkov Kjaer')
print('='*70)

# Player stats (estimated for young ATP/Challenger players)
# Landaluce: Spanish, 20yo, former top junior, solid clay courter
landaluce_serve = 0.62   # Good serve
landaluce_return = 0.40  # Spanish return game

# Budkov Kjaer: Danish, 18yo, 2023 Junior Wimbledon champion
budkov_serve = 0.60      # Young, developing serve
budkov_return = 0.38     # Solid return

print()
print('MARTIN LANDALUCE (ESP)')
print('-'*40)
print(f'  Serve Win %:   {landaluce_serve:.0%}')
print(f'  Return Win %:  {landaluce_return:.0%}')
print(f'  Age: 20, Rank: ~150-250')
print(f'  Style: Clay court baseliner, steady')

print()
print('NICOLAI BUDKOV KJAER (DEN)')
print('-'*40)
print(f'  Serve Win %:   {budkov_serve:.0%}')
print(f'  Return Win %:  {budkov_return:.0%}')
print(f'  Age: 18, Rank: ~250-400')
print(f'  Style: All-court, 2023 Junior Wimbledon champ')

# Game level
p_land_hold = p_win_game(landaluce_serve)
p_bud_hold = p_win_game(budkov_serve)

print()
print('GAME LEVEL:')
print(f'  P(Landaluce holds) = {p_land_hold:.1%}')
print(f'  P(Budkov holds)    = {p_bud_hold:.1%}')
print(f'  P(Landaluce breaks)= {1-p_bud_hold:.1%}')
print(f'  P(Budkov breaks)   = {1-p_land_hold:.1%}')

# Set level
p_land_set = calculate_set_probability(p_land_hold, p_bud_hold, landaluce_serve, budkov_serve)

print()
print('SET LEVEL (Markov Chain):')
print(f'  P(Landaluce wins set) = {p_land_set:.1%}')
print(f'  P(Budkov wins set)    = {1-p_land_set:.1%}')

# Match level
p_land_match = p_win_match_bo3(p_land_set)

print()
print('MATCH WIN (Best of 3):')
print(f'  P(Landaluce) = {p_land_match:.1%}')
print(f'  P(Budkov)    = {1-p_land_match:.1%}')

# Bookmaker comparison
print()
print('='*70)
print('BOOKMAKER ANALYSIS')
print('(Enter your odds below if different)')
print('='*70)

# Default odds estimate - adjust with actual odds
land_odds = 1.65  # Favored slightly
bud_odds = 2.25   

p_land_implied = 1/land_odds
p_bud_implied = 1/bud_odds
overround = p_land_implied + p_bud_implied - 1
p_land_fair = p_land_implied / (p_land_implied + p_bud_implied)
p_bud_fair = p_bud_implied / (p_land_implied + p_bud_implied)

print()
print(f'Bookmaker Odds:')
print(f'  Landaluce: {land_odds}')
print(f'  Budkov:    {bud_odds}')
print(f'  Overround: {overround:.1%}')

print()
print(f'Fair Probabilities (no vig):')
print(f'  Landaluce: {p_land_fair:.1%}')
print(f'  Budkov:    {p_bud_fair:.1%}')

print()
print(f'Model Probabilities:')
print(f'  Landaluce: {p_land_match:.1%}')
print(f'  Budkov:    {1-p_land_match:.1%}')

edge_land = p_land_match - p_land_fair
edge_bud = (1 - p_land_match) - p_bud_fair

print()
print(f'EDGE CALCULATION:')
print(f'  Edge on Landaluce: {edge_land:+.1%}')
print(f'  Edge on Budkov:    {edge_bud:+.1%}')

print()
print('='*70)
print('RECOMMENDATION')
print('='*70)

if edge_land > 0.02:
    kelly = edge_land / (land_odds - 1)
    quarter_kelly = kelly * 0.25
    print(f'  BET: LANDALUCE @ {land_odds}')
    print(f'  Kelly: {kelly:.1%}, Quarter Kelly: {quarter_kelly:.1%}')
    print(f'  On $1000 bankroll: Stake ${quarter_kelly*1000:.2f}')
elif edge_bud > 0.02:
    kelly = edge_bud / (bud_odds - 1)
    quarter_kelly = kelly * 0.25
    print(f'  BET: BUDKOV @ {bud_odds}')
    print(f'  Kelly: {kelly:.1%}, Quarter Kelly: {quarter_kelly:.1%}')
    print(f'  On $1000 bankroll: Stake ${quarter_kelly*1000:.2f}')
else:
    print(f'  NO VALUE BET')
    print(f'  Both edges < 2% threshold')

print('='*70)
