"""Opportunity scanner — rank what is worth looking at, across every match.

WHY RANKING BEATS A BET/NO-BET FLAG
    A binary gate answers "is this tradeable?" and stops. With forty live
    matches across three market levels that leaves a hundred-plus rows all
    labelled the same, and the user still has to decide which to open first.

    The ranking quantity is the one the codebase already computes:

        EdgeScore = (P_model - P_market) / sigma

    `edgescore.score_edge` has produced exactly this since before the live
    engine existed. Dividing by sigma is what makes a 4% edge on a settled,
    well-sourced match outrank an 11% edge on a thin market with one estimate
    behind it — a raw-edge sort puts those in the wrong order, and the wrong
    order is what makes a scanner untrustworthy.

WHAT IS DELIBERATELY EXCLUDED
    Anything the publish gate would refuse: degraded feed, stale price, no
    model opinion, absurd edge. A scanner that lists opportunities the system
    would not let you take is a list of disappointments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from execution.live.feed import Health, worst
from execution.live.marketlag import Divergence


@dataclass
class Opportunity:
    """One ranked row of the scanner."""

    match_id: str
    label: str                    # "Fritz vs Shelton"
    market: str                   # "match" | "set" | "game"
    selection: str
    model_p: float
    market_p: float
    edge: float
    edge_score: float             # the ranking quantity
    sigma: float
    confidence: float
    health: Health
    liquidity: Optional[float] = None
    lag: Optional[str] = None     # Divergence, when the market has not caught up
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def as_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "label": self.label,
            "market": self.market,
            "selection": self.selection,
            "model_p": round(self.model_p, 4),
            "market_p": round(self.market_p, 4),
            "edge": round(self.edge, 4),
            "edge_score": round(self.edge_score, 2),
            "sigma": round(self.sigma, 4),
            "confidence": round(self.confidence, 3),
            "health": self.health.value,
            "liquidity": self.liquidity,
            "lag": self.lag,
            "ts_ms": self.ts_ms,
        }


@dataclass
class ScanFilter:
    """The filters the terminal exposes. All optional; all AND-ed."""

    min_edge: float = 0.0
    min_edge_score: float = 0.0
    prob_range: tuple = (0.0, 1.0)
    markets: Optional[frozenset] = None
    tours: Optional[frozenset] = None
    min_liquidity: Optional[float] = None
    live_only: bool = False
    lag_only: bool = False
    # Health floor. Defaults to what the publish gate already requires, so the
    # scanner cannot surface something the system would refuse to trade.
    allow_health: frozenset = frozenset({Health.LIVE, Health.DELAYED})

    def accepts(self, o: Opportunity, *, tour: str = "", is_live: bool = True) -> bool:
        if o.health not in self.allow_health:
            return False
        if o.edge < self.min_edge:
            return False
        if o.edge_score < self.min_edge_score:
            return False
        lo, hi = self.prob_range
        if not (lo <= o.model_p <= hi):
            return False
        if self.markets is not None and o.market not in self.markets:
            return False
        if self.tours is not None and tour not in self.tours:
            return False
        if self.min_liquidity is not None:
            # Unknown liquidity is treated as failing the filter, not passing
            # it: a size requirement that silently admits unmeasured markets is
            # not a size requirement.
            if o.liquidity is None or o.liquidity < self.min_liquidity:
                return False
        if self.live_only and not is_live:
            return False
        if self.lag_only and o.lag != Divergence.LAG.value:
            return False
        return True


class Scanner:
    """Builds the ranked list from whatever the runtime currently holds."""

    def __init__(self, *, scorer: Optional[Callable] = None):
        # Injected for tests; defaults to the real uncertainty engine.
        self._scorer = scorer

    def _score(self, p_model: float, p_market: float, estimates: dict, is_live: bool):
        if self._scorer is not None:
            return self._scorer(p_model, p_market, estimates, is_live)
        from execution.edgescore import score_edge
        est = dict(estimates)
        est.setdefault("inplay", p_model)
        est.setdefault("market", p_market)
        return score_edge(p_model, p_market, est, is_live, source="inplay")

    def evaluate(self, *, match_id: str, label: str, market: str, selection: str,
                 model_p: Optional[float], market_p: Optional[float],
                 feed_health: Health, odds_health: Health,
                 estimates: Optional[dict] = None, is_live: bool = True,
                 liquidity: Optional[float] = None,
                 lag: Optional[Divergence] = None) -> Optional[Opportunity]:
        """One (match, market, selection) into an Opportunity, or None."""
        if model_p is None or market_p is None:
            return None
        scored = self._score(model_p, market_p, estimates or {}, is_live)
        edge = model_p - market_p
        grade = getattr(scored, "grade", "red")
        escore = float(getattr(scored, "edge_score", 0.0))
        return Opportunity(
            match_id=match_id, label=label, market=market, selection=selection,
            model_p=model_p, market_p=market_p, edge=edge,
            edge_score=escore, sigma=float(getattr(scored, "sigma", 0.0)),
            confidence={"green": 0.8, "amber": 0.5, "red": 0.2}.get(grade, 0.2),
            health=worst(feed_health, odds_health),
            liquidity=liquidity,
            lag=lag.value if lag is not None else None,
        )

    @staticmethod
    def rank(opportunities: Iterable[Opportunity], *,
             filt: Optional[ScanFilter] = None,
             tour_of: Optional[Callable] = None,
             limit: int = 50) -> list:
        """Filter and sort. Highest EdgeScore first.

        Ties break on raw edge, then on liquidity — so between two equally
        significant opportunities the bigger and more executable one leads.
        """
        f = filt or ScanFilter()
        out = [o for o in opportunities
               if f.accepts(o, tour=(tour_of(o) if tour_of else ""))]
        out.sort(key=lambda o: (o.edge_score, o.edge, o.liquidity or 0.0), reverse=True)
        return out[:limit]

    @staticmethod
    def summary(opportunities: Iterable[Opportunity]) -> dict:
        """Top-line numbers for the terminal header."""
        ops = list(opportunities)
        if not ops:
            return {"count": 0, "best_edge": None, "best_score": None, "lagging": 0}
        return {
            "count": len(ops),
            "best_edge": round(max(o.edge for o in ops), 4),
            "best_score": round(max(o.edge_score for o in ops), 2),
            "lagging": sum(1 for o in ops if o.lag == Divergence.LAG.value),
        }
