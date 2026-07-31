"""§2 Character traits — actual minus analytically-expected, causally computed.

Each trait is (actual − expected-under-the-analytic-model), so it isolates the
situational effect rather than re-deriving "this player is good" (the spec's
computation note). All traits are Bayesian-shrunk n/(n+K) toward 0.

Honesty split — what game-level history CAN and CANNOT support:
  computable now : clutch (deciding game), deuce composure (games past 10-10),
                   comeback (from 0-2 / 1-2 down), front-running (from 2-0 up),
                   fatigue (3rd+ match of the same day)
  needs live point logs: momentum sensitivity, server-pressure hold — these
                   need point SEQUENCES, which we only accrue via live.py's
                   state logging. Slots exist; they fill as logs grow.

Same chronological engine as features.py: snapshot traits BEFORE updating with
a match's result, so the emitted per-match snapshots are leakage-free training
rows for the residual model (§3).
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from tabletennis.analytic import p_from_match_prob, p_game, p_match
from tabletennis.features import ELO_START, FeatureEngine

DB_PATH = Path(__file__).resolve().parent / "tabletennis.db"
TRAITS_PATH = Path(__file__).resolve().parent / "traits.json"

SHRINK_K = 15.0
TRAIT_NAMES = ["clutch", "deuce", "comeback", "frontrun", "fatigue"]


@lru_cache(maxsize=512)
def _p_point(match_prob_2dp: float, best_of: int) -> float:
    return p_from_match_prob(match_prob_2dp, best_of)


@dataclass
class TraitAcc:
    """Running (actual − expected) sums per situation, plus counts."""
    sums: dict = field(default_factory=lambda: {t: 0.0 for t in TRAIT_NAMES})
    ns: dict = field(default_factory=lambda: {t: 0 for t in TRAIT_NAMES})
    day_matches: dict = field(default_factory=dict)   # date -> count (fatigue)

    def shrunk(self) -> dict:
        out = {}
        for t in TRAIT_NAMES:
            n = self.ns[t]
            out[t] = round((self.sums[t] / n) * (n / (n + SHRINK_K)), 4) if n else 0.0
            out[f"{t}_n"] = n
        return out


def _elo_prob(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def compute(db_path: Path = DB_PATH, emit_states: bool = False):
    """One chronological pass. Returns (traits_by_player, state_rows).

    state_rows (when emit_states) are residual-model training rows: one per
    intermediate GAME state of every historical match, with the analytic
    baseline, elo/form features and both players' trait snapshots — all as of
    before that match. Label: did p1 win the match.
    """
    conn = sqlite3.connect(str(db_path))
    matches = conn.execute(
        "SELECT id, category_id, start_ts, p1_id, p2_id, winner, best_of "
        "FROM matches ORDER BY start_ts").fetchall()
    games_by_match = defaultdict(list)
    for mid, n, a, b in conn.execute(
            "SELECT match_id, game_no, p1_pts, p2_pts FROM games ORDER BY match_id, game_no"):
        games_by_match[mid].append((n, a, b))
    conn.close()

    eng = FeatureEngine()                      # for Elo state (same causal pass)
    acc: dict[int, TraitAcc] = defaultdict(TraitAcc)
    state_rows = []

    for mid, cat, ts, p1, p2, winner, best_of in matches:
        games = games_by_match[mid]
        if not games:
            continue
        best_of = max(best_of, 2 * len(games) - 1) if best_of < 3 else best_of
        elo1 = eng.players[(cat, p1)].elo
        elo2 = eng.players[(cat, p2)].elo
        pre = _elo_prob(elo1, elo2)
        p_pt = _p_point(round(min(max(pre, 0.05), 0.95), 2), best_of)
        won1 = winner == 1
        day = ts // 86400

        # ── emit residual-training rows (snapshot BEFORE updates) ──
        if emit_states:
            feats = eng.snapshot(cat, p1, p2, ts)
            t1, t2 = acc[p1].shrunk(), acc[p2].shrunk()
            ga = gb = 0
            for _, a, b in games[:-1]:                 # states between games
                ga, gb = ga + (a > b), gb + (b > a)
                if max(ga, gb) > best_of // 2:
                    break
                base = p_match(ga, gb, p_pt, best_of)
                state_rows.append({
                    "ts": ts, "base": base, "pre": pre, "ga": ga, "gb": gb,
                    "best_of": best_of, "elo_diff": elo1 - elo2,
                    "feats": feats,
                    "traits1": [t1[t] for t in TRAIT_NAMES],
                    "traits2": [t2[t] for t in TRAIT_NAMES],
                    "y": 1 if won1 else 0,
                })

        # ── update trait accumulators from this match's result ──
        need = best_of // 2 + 1
        n_games = len(games)
        # clutch: deciding game played?
        if n_games == best_of:
            exp = p_game(0, 0, p_pt)               # expected decider win for p1
            _, a, b = games[-1]
            acc[p1].sums["clutch"] += (1.0 if a > b else 0.0) - exp
            acc[p1].ns["clutch"] += 1
            acc[p2].sums["clutch"] += (1.0 if b > a else 0.0) - (1 - exp)
            acc[p2].ns["clutch"] += 1
        # deuce composure: any game that reached 10-10
        for _, a, b in games:
            if min(a, b) >= 10:
                from tabletennis.analytic import _deuce_win, SERVE_EDGE
                exp_d = 0.5 * (_deuce_win(min(p_pt + SERVE_EDGE, .99), max(p_pt - SERVE_EDGE, .01))
                               + _deuce_win(max(p_pt - SERVE_EDGE, .01), min(p_pt + SERVE_EDGE, .99)))
                acc[p1].sums["deuce"] += (1.0 if a > b else 0.0) - exp_d
                acc[p1].ns["deuce"] += 1
                acc[p2].sums["deuce"] += (1.0 if b > a else 0.0) - (1 - exp_d)
                acc[p2].ns["deuce"] += 1
        # comeback / front-run: track running game score
        ga = gb = 0
        seen = set()
        for _, a, b in games:
            ga, gb = ga + (a > b), gb + (b > a)
            if (ga, gb) in seen or max(ga, gb) >= need:
                continue
            seen.add((ga, gb))
            if gb - ga >= 2 or (gb - ga == 1 and gb == need - 1):   # p1 trailing hard
                exp = p_match(ga, gb, p_pt, best_of)
                acc[p1].sums["comeback"] += (1.0 if won1 else 0.0) - exp
                acc[p1].ns["comeback"] += 1
            if ga - gb >= 2:                                          # p1 front-running
                exp = p_match(ga, gb, p_pt, best_of)
                acc[p1].sums["frontrun"] += (1.0 if won1 else 0.0) - exp
                acc[p1].ns["frontrun"] += 1
            if ga - gb >= -2 and gb - ga >= 2:
                pass
        # mirror comeback/front-run for p2
        ga = gb = 0
        seen = set()
        for _, a, b in games:
            ga, gb = ga + (a > b), gb + (b > a)
            if (ga, gb) in seen or max(ga, gb) >= need:
                continue
            seen.add((ga, gb))
            if ga - gb >= 2 or (ga - gb == 1 and ga == need - 1):   # p2 trailing
                exp = 1 - p_match(ga, gb, p_pt, best_of)
                acc[p2].sums["comeback"] += (1.0 if not won1 else 0.0) - exp
                acc[p2].ns["comeback"] += 1
            if gb - ga >= 2:
                exp = 1 - p_match(ga, gb, p_pt, best_of)
                acc[p2].sums["frontrun"] += (1.0 if not won1 else 0.0) - exp
                acc[p2].ns["frontrun"] += 1
        # fatigue: 3rd+ match of the same calendar day
        for pid, won, exp in ((p1, won1, pre), (p2, not won1, 1 - pre)):
            k = acc[pid].day_matches.get(day, 0)
            if k >= 2:
                acc[pid].sums["fatigue"] += (1.0 if won else 0.0) - exp
                acc[pid].ns["fatigue"] += 1
            acc[pid].day_matches[day] = k + 1

        eng.update(cat, p1, p2, ts, winner,
                   sum(a > b for _, a, b in games), sum(b > a for _, a, b in games))

    traits = {pid: a.shrunk() for pid, a in acc.items()}
    return traits, state_rows


def save(db_path: Path = DB_PATH) -> dict:
    traits, _ = compute(db_path)
    conn = sqlite3.connect(str(db_path))
    names = dict(conn.execute("SELECT id, name FROM players"))
    conn.close()
    out = {str(pid): {"name": names.get(pid, "?"), **t} for pid, t in traits.items()}
    TRAITS_PATH.write_text(json.dumps(out))
    print(f"traits for {len(out)} players → {TRAITS_PATH.name}")
    return out


if __name__ == "__main__":
    out = save()
    ranked = [(v["clutch"], v["clutch_n"], v["name"]) for v in out.values() if v["clutch_n"] >= 15]
    ranked.sort(reverse=True)
    print("\nmost clutch (≥15 deciders, actual−expected):")
    for s, n, name in ranked[:5]:
        print(f"  {name:24} {s:+.3f}  (n={n})")
    print("least clutch:")
    for s, n, name in ranked[-5:]:
        print(f"  {name:24} {s:+.3f}  (n={n})")
