#!/usr/bin/env python3
"""Tier 2 momentum validation: synthetic sequences (deterministic checks) + a
real live match end-to-end through the InPlayModel."""

from execution.momentum import LiveMomentumEngine, game_win_prob


def g(server, winner, points=None):
    return {"set": 1, "game": 0, "server": server, "winner": winner, "points": points or []}


def show(label, games, sp1=0.64, sp2=0.64, fs1=0.55, fs2=0.55):
    eng = LiveMomentumEngine()
    st = eng.compute(games, sp1, sp2, fs1, fs2)
    a1, a2 = eng.apply_to_serve(sp1, sp2, st, fs1, fs2)
    print(f"\n{label}")
    print(f"  momentum_p1={st.momentum_p1:+.3f}  serve_reg_p1={st.serve_reg_p1:+.3f} "
          f"serve_reg_p2={st.serve_reg_p2:+.3f}")
    print(f"  recent p1 holds/breaks={st.recent_holds_p1}/{st.recent_breaks_p1}  "
          f"p2 holds/breaks={st.recent_holds_p2}/{st.recent_breaks_p2}")
    print(f"  serve nudge: p1 {sp1:.3f}->{a1:.3f} ({a1-sp1:+.4f})   "
          f"p2 {sp2:.3f}->{a2:.3f} ({a2-sp2:+.4f})")
    return st, (a1, a2)


print("=" * 64)
print("SANITY: hold prob from point-win-on-serve")
print("=" * 64)
for p in (0.55, 0.62, 0.64, 0.70):
    print(f"  p_pt={p:.2f} -> P(hold)={game_win_prob(p):.3f}")

print("\n" + "=" * 64)
print("SYNTHETIC SEQUENCES (both priors 0.64)")
print("=" * 64)

# A: everyone holds serve -> momentum ~0, no serve regression
seqA = [g(1, 1), g(2, 2), g(1, 1), g(2, 2), g(1, 1), g(2, 2)]
show("A. all holds (neutral)", seqA)

# B: p2 breaks p1 in the two most recent games -> p1 momentum sharply negative,
#    p1 serve regression negative
seqB = [g(1, 1), g(2, 2), g(1, 1), g(2, 2), g(1, 2), g(2, 2)]
show("B. p2 breaks p1 late (p1 collapsing on serve)", seqB)

# C: p1 breaks p2 late and holds -> p1 momentum strongly positive
seqC = [g(1, 1), g(2, 2), g(1, 1), g(2, 2), g(2, 1), g(1, 1)]
show("C. p1 breaks p2 late (p1 surging)", seqC)

# D: p1 holding but under constant break-point pressure (latent regression)
bp = [("15", "40"), ("30", "40")]  # returner reaching 40 on p1's serve
seqD = [g(1, 1, bp), g(2, 2), g(1, 1, bp), g(2, 2), g(1, 1, bp), g(2, 2)]
show("D. p1 holds but faces break points every service game", seqD)

# E: style modulation — same collapse, p1 a first-strike server vs a grinder
print("\n  -- style modulation on identical collapse (seq B) --")
show("E1. p1 first-strike (fs=0.74)", seqB, fs1=0.74)
show("E2. p1 grinder (fs=0.45)", seqB, fs1=0.45)

# ── real live match end-to-end ────────────────────────────────────────────────
print("\n" + "=" * 64)
print("LIVE MATCH END-TO-END (through InPlayModel)")
print("=" * 64)
try:
    from execution.live_odds import SofascoreOdds
    from execution.inplay import InPlayModel
    sofa = SofascoreOdds()
    evs = (sofa._get("sport/tennis/events/live") or {}).get("events") or []
    ip = InPlayModel(sofa=sofa)
    shown = 0
    for e in evs:
        h = (e.get("homeTeam") or {}).get("name", "")
        a = (e.get("awayTeam") or {}).get("name", "")
        if "/" in h or "/" in a:
            continue
        seq = sofa.game_sequence(h, a)
        if not seq or sum(1 for x in seq if x["winner"]) < 4:
            continue
        d = ip.detail(h, a)
        if not d or d.get("true_p") is None:
            continue
        m = d.get("momentum") or {}
        print(f"\n  {h} vs {a}   set {d['current_set']}  "
              f"{d['sets_p1']}-{d['sets_p2']} ({d['games_p1']}-{d['games_p2']})")
        print(f"    true_p={d['true_p']:.3f}  serve_p1={d['serve_p1']} serve_p2={d['serve_p2']}")
        if m:
            print(f"    momentum_p1={m['momentum_p1']:+.3f}  "
                  f"serve_reg_p1={m['serve_reg_p1']:+.3f} serve_reg_p2={m['serve_reg_p2']:+.3f}  "
                  f"(p1 h/b {m['recent_holds_p1']}/{m['recent_breaks_p1']}, "
                  f"p2 h/b {m['recent_holds_p2']}/{m['recent_breaks_p2']})")
        shown += 1
        if shown >= 3:
            break
    if shown == 0:
        print("  (no live match with >=4 completed games right now)")
    ip.close()
except Exception as ex:
    import traceback
    traceback.print_exc()
    print(f"  live test skipped: {ex}")
