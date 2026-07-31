"""§1 Analytical baseline — exact live win probability from score state.

Pure math, no training data, sub-millisecond after memoisation: the auditable
layer the character model corrects. Race-to-11, win-by-2, serve alternating
every 2 points (every point past 10-10, handled in closed form).

Design notes vs the spec:
  • Sofascore's TT feed exposes no server identity (firstToServe=None), so all
    game probabilities average the two first-server assumptions. TT serve
    advantage is small (SERVE_EDGE≈0.03 vs ~0.15 in tennis) so the spread
    between the two assumptions is well under a point of win probability.
  • The per-point probability p is derived by INVERTING the pre-match model:
    bisection finds the p whose analytic match probability equals the Elo/model
    pre-match probability. That forces the analytic layer to be consistent with
    the pre-match layer at 0-0 — they can't disagree before a point is played.
"""

from __future__ import annotations

from functools import lru_cache

SERVE_EDGE = 0.03   # tour-average point-win bump on own serve (TT is near-neutral)
TARGET = 11


def _deuce_win(ps: float, pr: float) -> float:
    """P(win from 10-10) — serve alternates each point, so every 2-point block
    contains one serve + one return point. Closed-form geometric solution."""
    both = ps * pr
    split = ps * (1 - pr) + (1 - ps) * pr
    return both / (1 - split) if split < 1 else 0.5


@lru_cache(maxsize=200_000)
def _game(a: int, b: int, p: float, first_server_me: bool) -> float:
    """P(I win the game) from points (a=mine, b=theirs), given who served first."""
    if a >= TARGET and a - b >= 2:
        return 1.0
    if b >= TARGET and b - a >= 2:
        return 0.0
    ps, pr = min(p + SERVE_EDGE, 0.99), max(p - SERVE_EDGE, 0.01)
    if a >= 10 and b >= 10:
        return _deuce_win(ps, pr)
    serving = ((a + b) // 2) % 2 == (0 if first_server_me else 1)
    pp = ps if serving else pr
    return pp * _game(a + 1, b, p, first_server_me) + (1 - pp) * _game(a, b + 1, p, first_server_me)


def p_game(a: int, b: int, p: float) -> float:
    """P(win game) from point score (a, b) — server unknown, average both."""
    p = round(min(max(p, 0.05), 0.95), 4)   # quantise for cache hits
    return 0.5 * (_game(a, b, p, True) + _game(a, b, p, False))


@lru_cache(maxsize=50_000)
def _match(ga: int, gb: int, pg: float, best_of: int) -> float:
    need = best_of // 2 + 1
    if ga >= need:
        return 1.0
    if gb >= need:
        return 0.0
    return pg * _match(ga + 1, gb, pg, best_of) + (1 - pg) * _match(ga, gb + 1, pg, best_of)


def p_match(ga: int, gb: int, p: float, best_of: int = 5) -> float:
    """P(win match) from game score, both games from scratch."""
    pg = round(p_game(0, 0, p), 4)
    return _match(ga, gb, pg, best_of)


def p_match_live(ga: int, gb: int, pa: int, pb: int, p: float, best_of: int = 5) -> float:
    """P(win match) from full live state: games (ga,gb) + current-game points (pa,pb)."""
    pg_now = p_game(pa, pb, p)
    return (pg_now * p_match(ga + 1, gb, p, best_of)
            + (1 - pg_now) * p_match(ga, gb + 1, p, best_of))


def p_from_match_prob(match_prob: float, best_of: int = 5, tol: float = 1e-4) -> float:
    """Invert: find per-point p whose analytic P(match at 0-0) equals match_prob.
    Monotone in p → bisection. This anchors the live engine to the pre-match
    model so both layers agree before the first point."""
    match_prob = min(max(match_prob, 0.02), 0.98)
    lo, hi = 0.20, 0.80
    for _ in range(40):
        mid = (lo + hi) / 2
        if p_match(0, 0, mid, best_of) < match_prob:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


if __name__ == "__main__":
    # sanity: symmetric at p=0.5; monotone; live states behave
    print("p=0.5  P(match 0-0)      =", round(p_match(0, 0, 0.5), 4), "(expect 0.5)")
    print("p=0.55 P(match 0-0)      =", round(p_match(0, 0, 0.55), 4))
    print("p=0.5  up 2-0 games      =", round(p_match(2, 0, 0.5), 4))
    print("p=0.5  down 0-2, 10-5 up =", round(p_match_live(0, 2, 10, 5, 0.5), 4))
    print("p=0.5  9-9 in decider    =", round(p_match_live(2, 2, 9, 9, 0.5), 4), "(expect ~0.5)")
    print("invert 0.70 →", round(p_from_match_prob(0.70), 4),
          "→ forward:", round(p_match(0, 0, p_from_match_prob(0.70)), 4))
