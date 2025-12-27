"""
Markov Chain Match Analysis Demo
Cobolli vs Rublev (Indoor Hard)
Bookmaker Odds: Cobolli 2.0, Rublev 1.72
"""

import numpy as np
from scipy.special import comb

def p_win_game(p_point):
    """
    Probability of winning a game given point win probability.
    Uses deuce game calculation.
    """
    p = p_point
    q = 1 - p
    
    # Win in exactly 4, 5, 6 points (before deuce)
    p_4 = p**4  # 40-0, then win
    p_5 = 4 * p**4 * q  # One loss, then win at 40-15
    p_6 = 10 * p**4 * q**2  # Two losses, then win at 40-30
    
    # Reach deuce (3-3 in points won before deuce)
    p_deuce = comb(6, 3) * p**3 * q**3  # 20 ways to reach 3-3
    
    # Probability of winning from deuce
    p_win_deuce = p**2 / (p**2 + q**2)
    
    return p_4 + p_5 + p_6 + p_deuce * p_win_deuce


def calculate_set_probability(p_hold_a, p_hold_b, p_serve_a, p_serve_b):
    """
    Calculate P(player A wins set) using Markov chain.
    State: (games_A, games_B)
    """
    states = {}
    idx = 0
    
    for a in range(7):
        for b in range(7):
            states[(a, b)] = idx
            idx += 1
    
    states['TB'] = idx
    idx += 1
    states['A_wins'] = idx
    idx += 1
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
        
        # New states after this game
        new_a = a + 1
        new_b = b + 1
        
        # A wins game
        if new_a >= 6 and new_a - b >= 2:
            j_a_win = states['A_wins']
        elif new_a == 6 and b == 6:
            j_a_win = states['TB']
        else:
            j_a_win = states.get((new_a, b), states['A_wins'])
        
        # B wins game
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
    """Probability of winning best-of-3 match."""
    p = p_set
    return p**2 + 2 * p**2 * (1 - p)


def p_win_match_bo5(p_set):
    """Probability of winning best-of-5 match."""
    p = p_set
    return p**3 + 3 * p**3 * (1 - p) + 6 * p**3 * (1 - p)**2


def analyze_match(player1_name, player2_name, 
                  p1_serve_pct, p2_serve_pct,
                  p1_odds, p2_odds,
                  best_of=3):
    """
    Complete Markov chain match analysis.
    """
    print("=" * 70)
    print("MARKOV CHAIN MATCH ANALYSIS")
    print(f"{player1_name} vs {player2_name}")
    print("=" * 70)
    
    # Point level
    print(f"\n📊 PLAYER STATISTICS:")
    print(f"  {player1_name}: Serve Win% = {p1_serve_pct:.1%}")
    print(f"  {player2_name}: Serve Win% = {p2_serve_pct:.1%}")
    
    # Game level
    p1_hold = p_win_game(p1_serve_pct)
    p2_hold = p_win_game(p2_serve_pct)
    p1_break = 1 - p2_hold
    p2_break = 1 - p1_hold
    
    print(f"\n🎾 GAME LEVEL:")
    print(f"  P({player1_name} holds) = {p1_hold:.1%}")
    print(f"  P({player2_name} holds) = {p2_hold:.1%}")
    print(f"  P({player1_name} breaks) = {p1_break:.1%}")
    print(f"  P({player2_name} breaks) = {p2_break:.1%}")
    
    # Set level
    p1_set = calculate_set_probability(p1_hold, p2_hold, p1_serve_pct, p2_serve_pct)
    
    print(f"\n📊 SET LEVEL (Markov Chain):")
    print(f"  P({player1_name} wins set) = {p1_set:.1%}")
    print(f"  P({player2_name} wins set) = {1 - p1_set:.1%}")
    
    # Match level
    if best_of == 3:
        p1_match = p_win_match_bo3(p1_set)
    else:
        p1_match = p_win_match_bo5(p1_set)
    
    print(f"\n🏆 MATCH WIN (Best of {best_of}):")
    print(f"  P({player1_name}) = {p1_match:.1%}")
    print(f"  P({player2_name}) = {1 - p1_match:.1%}")
    
    # Bookmaker comparison
    p1_implied = 1 / p1_odds
    p2_implied = 1 / p2_odds
    overround = p1_implied + p2_implied - 1
    
    p1_fair = p1_implied / (p1_implied + p2_implied)
    p2_fair = p2_implied / (p1_implied + p2_implied)
    
    print(f"\n📈 BOOKMAKER COMPARISON:")
    print(f"  Odds: {player1_name} {p1_odds}, {player2_name} {p2_odds}")
    print(f"  Implied: {player1_name} {p1_implied:.1%}, {player2_name} {p2_implied:.1%}")
    print(f"  Overround: {overround:.1%}")
    print(f"  Fair: {player1_name} {p1_fair:.1%}, {player2_name} {p2_fair:.1%}")
    print(f"  Model: {player1_name} {p1_match:.1%}, {player2_name} {1 - p1_match:.1%}")
    
    # Edge
    edge1 = p1_match - p1_fair
    edge2 = (1 - p1_match) - p2_fair
    
    print(f"\n  Edge on {player1_name}: {edge1:+.1%}")
    print(f"  Edge on {player2_name}: {edge2:+.1%}")
    
    # Kelly
    if edge1 > 0.02:  # 2% minimum edge
        kelly = edge1 / (p1_odds - 1)
        print(f"\n💰 BET: {player1_name}")
        print(f"   Kelly: {kelly:.1%}, Quarter Kelly: {kelly * 0.25:.1%}")
        print(f"   On $1000: Stake ${kelly * 1000 * 0.25:.2f}")
    elif edge2 > 0.02:
        kelly = edge2 / (p2_odds - 1)
        print(f"\n💰 BET: {player2_name}")
        print(f"   Kelly: {kelly:.1%}, Quarter Kelly: {kelly * 0.25:.1%}")
        print(f"   On $1000: Stake ${kelly * 1000 * 0.25:.2f}")
    else:
        print(f"\n⚠️  NO VALUE BET (edge < 2%)")
    
    # Game-by-game table
    print(f"\n📊 SET 1 WIN PROBABILITY BY GAME SCORE:")
    print("-" * 55)
    print(f"{'Score':<10} | P({player1_name[:8]:>8}) | P({player2_name[:8]:>8})")
    print("-" * 55)
    
    scores = [(0,0), (1,0), (0,1), (1,1), (2,1), (1,2), 
              (3,2), (2,3), (4,3), (3,4), (5,4), (4,5), (5,5), (6,5), (5,6)]
    
    for g1, g2 in scores:
        lead = g1 - g2
        if g1 >= 6 and g1 - g2 >= 2:
            p_c = 1.0
        elif g2 >= 6 and g2 - g1 >= 2:
            p_c = 0.0
        else:
            p_c = p1_set + lead * 0.07
            p_c = max(0.05, min(0.95, p_c))
        
        p_match = p_win_match_bo3(p_c) if best_of == 3 else p_win_match_bo5(p_c)
        print(f"  {g1}-{g2:<6}  |   {p_match:>6.1%}     |   {1-p_match:>6.1%}")
    
    print("=" * 70)
    
    return {
        'p_player1_match': p1_match,
        'p_player1_set': p1_set,
        'edge_player1': edge1,
        'edge_player2': edge2
    }


if __name__ == "__main__":
    # Cobolli vs Rublev example
    result = analyze_match(
        player1_name="Cobolli",
        player2_name="Rublev",
        p1_serve_pct=0.63,  # Cobolli serve %
        p2_serve_pct=0.66,  # Rublev serve %
        p1_odds=2.0,
        p2_odds=1.72,
        best_of=3
    )
