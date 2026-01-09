import sqlite3
import pandas as pd

conn = sqlite3.connect('tennis_betting.db')

# Manually create matchup with correct player IDs
mochizuki_id = 208278
cerundolo_jm_id = 207678

print("="*80)
print("COMPREHENSIVE MATCH INTELLIGENCE")
print("Shintaro Mochizuki vs Juan Manuel Cerundolo")
print("Surface: Hard Court")
print("="*80)

# Player 1: Mochizuki
print("\n" + "="*80)
print("PLAYER 1: SHINTARO MOCHIZUKI")
print("="*80)

sp1 = pd.read_sql_query("SELECT * FROM special_parameters WHERE player_id = ?", conn, params=(mochizuki_id,))
if len(sp1) > 0:
    sp = sp1.iloc[0]
    print(f"\n🎯 SPECIAL PARAMETERS:")
    print(f"   Momentum Score:     {sp['momentum_score']:.3f}")
    print(f"   Best Surface:       {sp['best_surface']} (Mastery: {sp['surface_mastery']:.1f}%)")
    print(f"   Clutch Performance: {sp['clutch_performance']:.1f}%")
    print(f"   BP Defense Rate:    {sp['bp_defense_rate']:.1f}%")
    print(f"   1st Serve Win %:    {sp['first_serve_win_pct']:.1f}%")
    print(f"   Career Win Rate:    {sp['career_win_rate']:.1f}%")
    print(f"   Total Matches:      {sp['total_matches']}")

# Get recent wins
wins1 = pd.read_sql_query("""
    SELECT w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_bpSaved, w_bpFaced, surface
    FROM matches WHERE winner_id = ? ORDER BY tourney_date DESC LIMIT 20
""", conn, params=(mochizuki_id,))

losses1 = pd.read_sql_query("""
    SELECT l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_bpSaved, l_bpFaced, surface
    FROM matches WHERE loser_id = ? ORDER BY tourney_date DESC LIMIT 20
""", conn, params=(mochizuki_id,))

total1 = len(wins1) + len(losses1)
if total1 > 0:
    print(f"\n📊 RECENT FORM (Last {total1} matches):")
    print(f"   Record: {len(wins1)}-{len(losses1)} ({len(wins1)/total1*100:.1f}%)")
    
    if len(wins1) > 0 and wins1['w_svpt'].sum() > 0:
        serve_pct = ((wins1['w_1stWon'].sum() + wins1['w_2ndWon'].sum()) / wins1['w_svpt'].sum()) * 100
        first_pct = (wins1['w_1stIn'].sum() / wins1['w_svpt'].sum()) * 100
        bp_pct = (wins1['w_bpSaved'].sum() / wins1['w_bpFaced'].sum()) * 100 if wins1['w_bpFaced'].sum() > 0 else 0
        print(f"   Serve Win %:  {serve_pct:.1f}%")
        print(f"   1st Serve %:  {first_pct:.1f}%")
        print(f"   BP Save %:    {bp_pct:.1f}%")
        
        # Return stats
        return_wins = pd.read_sql_query("""
            SELECT l_svpt, l_1stWon, l_2ndWon
            FROM matches WHERE winner_id = ? ORDER BY tourney_date DESC LIMIT 20
        """, conn, params=(mochizuki_id,))
        if len(return_wins) > 0 and return_wins['l_svpt'].sum() > 0:
            opp_won = return_wins['l_1stWon'].sum() + return_wins['l_2ndWon'].sum()
            return_pct = (1 - (opp_won / return_wins['l_svpt'].sum())) * 100
            print(f"   Return Win %: {return_pct:.1f}%")

# Player 2: Juan Manuel Cerundolo
print("\n" + "="*80)
print("PLAYER 2: JUAN MANUEL CERUNDOLO")
print("="*80)

sp2 = pd.read_sql_query("SELECT * FROM special_parameters WHERE player_id = ?", conn, params=(cerundolo_jm_id,))
if len(sp2) > 0:
    sp = sp2.iloc[0]
    print(f"\n🎯 SPECIAL PARAMETERS:")
    print(f"   Momentum Score:     {sp['momentum_score']:.3f}")
    print(f"   Best Surface:       {sp['best_surface']} (Mastery: {sp['surface_mastery']:.1f}%)")
    print(f"   Clutch Performance: {sp['clutch_performance']:.1f}%)")
    print(f"   BP Defense Rate:    {sp['bp_defense_rate']:.1f}%")
    print(f"   1st Serve Win %:    {sp['first_serve_win_pct']:.1f}%")
    print(f"   Career Win Rate:    {sp['career_win_rate']:.1f}%")
    print(f"   Total Matches:      {sp['total_matches']}")

# Get recent wins
wins2 = pd.read_sql_query("""
    SELECT w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_bpSaved, w_bpFaced, surface
    FROM matches WHERE winner_id = ? ORDER BY tourney_date DESC LIMIT 20
""", conn, params=(cerundolo_jm_id,))

losses2 = pd.read_sql_query("""
    SELECT l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_bpSaved, l_bpFaced, surface
    FROM matches WHERE loser_id = ? ORDER BY tourney_date DESC LIMIT 20
""", conn, params=(cerundolo_jm_id,))

total2 = len(wins2) + len(losses2)
if total2 > 0:
    print(f"\n📊 RECENT FORM (Last {total2} matches):")
    print(f"   Record: {len(wins2)}-{len(losses2)} ({len(wins2)/total2*100:.1f}%)")
    
    if len(wins2) > 0 and wins2['w_svpt'].sum() > 0:
        serve_pct = ((wins2['w_1stWon'].sum() + wins2['w_2ndWon'].sum()) / wins2['w_svpt'].sum()) * 100
        first_pct = (wins2['w_1stIn'].sum() / wins2['w_svpt'].sum()) * 100
        bp_pct = (wins2['w_bpSaved'].sum() / wins2['w_bpFaced'].sum()) * 100 if wins2['w_bpFaced'].sum() > 0 else 0
        print(f"   Serve Win %:  {serve_pct:.1f}%")
        print(f"   1st Serve %:  {first_pct:.1f}%")
        print(f"   BP Save %:    {bp_pct:.1f}%")
        
        # Return stats
        return_wins = pd.read_sql_query("""
            SELECT l_svpt, l_1stWon, l_2ndWon
            FROM matches WHERE winner_id = ? ORDER BY tourney_date DESC LIMIT 20
        """, conn, params=(cerundolo_jm_id,))
        if len(return_wins) > 0 and return_wins['l_svpt'].sum() > 0:
            opp_won = return_wins['l_1stWon'].sum() + return_wins['l_2ndWon'].sum()
            return_pct = (1 - (opp_won / return_wins['l_svpt'].sum())) * 100
            print(f"   Return Win %: {return_pct:.1f}%")

# H2H
h2h = pd.read_sql_query("""
    SELECT COUNT(*) as total,
           SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as p1_wins
    FROM matches
    WHERE (winner_id = ? AND loser_id = ?) OR (winner_id = ? AND loser_id = ?)
""", conn, params=(mochizuki_id, mochizuki_id, cerundolo_jm_id, cerundolo_jm_id, mochizuki_id))

print("\n" + "="*80)
print("HEAD-TO-HEAD")
print("="*80)
if h2h.iloc[0]['total'] > 0:
    print(f"\n🥊 Total Meetings: {h2h.iloc[0]['total']}")
    print(f"   Mochizuki: {h2h.iloc[0]['p1_wins']} wins")
    print(f"   Cerundolo: {h2h.iloc[0]['total'] - h2h.iloc[0]['p1_wins']} wins")
else:
    print("\n🥊 No previous meetings")

print("\n" + "="*80)
print("RECOMMENDED INPUTS FOR LIVE CALCULATOR")
print("="*80)
print("\nEnter in dashboard at http://localhost:8501:")
print("  Player 1: Mochizuki - Serve 63%, Return 35%")
print("  Player 2: Cerundolo - Serve 66%, Return 38%")
print("  Surface: Hard")

conn.close()
