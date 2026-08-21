"""
Uncertainty and EdgeScore — the gate that asks "how sure are we?" before sizing.

WHY
  The calibration report is blunt about the failure mode: the engine is
  over-confident precisely where it bets most. In the 60-90% buckets it predicts
  ~75-90% and wins ~50-62%, Brier lands at 0.35 (worse than the 0.25 an
  always-50% guess scores) and ROI is negative. A flat `edge > 5%` rule cannot
  see that, because a 9-point edge off a confident-but-wrong probability looks
  exactly like a 9-point edge off a solid one.

  EdgeScore divides the edge by how much we actually trust the number:

      EdgeScore = (p_model - p_market) / sigma

  A 9-point edge with sigma=3 scores 3.0 and is worth having. The same 9-point
  edge with sigma=9 scores 1.0 and is noise wearing a result's clothing.

WHERE SIGMA COMES FROM
  `intel.py` already computes up to four independent estimates of P(p1 wins) —
  the in-play Markov engine, the SX exchange fair price, the Sofascore line, and
  the pre-match model — then picks ONE by priority and throws the rest away.
  That discarded disagreement is the cheapest honest uncertainty signal
  available, so this module keeps it.

  Two corrections make it meaningful rather than decorative:

  1. REGIME. Pre-match and live estimates must never be pooled. Mid-match, a
     pre-match line legitimately disagrees with the in-play engine because the
     score moved — that is staleness, not model uncertainty, and averaging it in
     would inflate sigma exactly when the live engine is most trustworthy. Only
     sources valid for the current regime are pooled.

  2. CALIBRATION FLOOR. Sources agreeing does not make them right — they share
     inputs (the in-play engine reads the Sofascore scoreboard; the line and the
     exchange are both market-derived), so their spread understates true error.
     The measured over-confidence is therefore added as a floor that peaks in
     the 60-90% band where the report shows the losses concentrating.

  Sigma here is a dispersion heuristic, NOT a posterior standard deviation. It
  is not from a Gaussian Process or any generative model of belief. Treated as
  what it is — a relative "how much do my independent reads diverge" score — it
  is useful. Treated as a calibrated credible interval, it would be a lie.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from statistics import pstdev
from typing import Dict, Optional

# Which estimates are meaningful in which regime. An estimate that reflects the
# live score can be trusted mid-match; one computed before the first ball cannot.
LIVE_SOURCES = ("inplay", "sxbet")
PREMATCH_SOURCES = ("model", "sofascore", "sxbet")

# Floor on sigma when we have only ONE usable estimate. A single source has zero
# measurable disagreement, which must not be read as certainty — it is an absence
# of evidence about uncertainty, and the two are opposites.
SIGMA_SOLO = float(os.getenv("TRADING_SIGMA_SOLO", "0.10"))

# Absolute floor even with several agreeing sources; they share inputs.
SIGMA_FLOOR = float(os.getenv("TRADING_SIGMA_FLOOR", "0.04"))

# Calibration penalty. From the settled-bet report: predictions of ~0.75-0.90 in
# this band actually win ~0.50-0.62, i.e. a miss on the order of 0.15-0.25. We
# carry a conservative fraction of that as irreducible error.
CALIB_BAND = (0.60, 0.90)
CALIB_SIGMA = float(os.getenv("TRADING_SIGMA_CALIB", "0.12"))
CALIB_SIGMA_BASE = float(os.getenv("TRADING_SIGMA_CALIB_BASE", "0.05"))

# EdgeScore thresholds for the traffic light (see §20 of the design).
GREEN_SCORE = float(os.getenv("TRADING_EDGESCORE_GREEN", "2.0"))
AMBER_SCORE = float(os.getenv("TRADING_EDGESCORE_AMBER", "1.0"))
MIN_ABS_EDGE = float(os.getenv("TRADING_MIN_ABS_EDGE", "0.02"))


# The calibration penalty is a property of OUR engine, not of every number that
# reaches this module. The report measured over-confidence in `true_p` produced
# by the in-play Markov engine and the pre-match model. The SX quote is a
# de-vigged two-sided exchange book — a sharp market price that our calibration
# error says nothing about. Charging it the same penalty would throw away the
# best signal in the stack for a sin it did not commit.
MODEL_SOURCES = ("inplay", "model")      # ours: carries the measured miscalibration
MARKET_SOURCES = ("sxbet", "sofascore")  # market-derived: priced by other people


def calibration_sigma(p: float, source: Optional[str] = None) -> float:
    """Irreducible error implied by the calibration report.

    Peaks across the 60-90% band, where settled bets show ~0.75-0.90 predictions
    winning ~0.50-0.62. Market-derived sources get the small base term instead:
    they may be wrong, but they are not wrong in the way we measured.
    """
    if source in MARKET_SOURCES:
        return CALIB_SIGMA_BASE

    lo, hi = CALIB_BAND
    q = max(p, 1.0 - p)          # distance from a coin flip, folded to one side
    if lo <= q <= hi:
        return CALIB_SIGMA
    if q < lo:
        # Near 50/50 the engine is not the problem; the edge just is not there.
        return CALIB_SIGMA_BASE
    # Beyond 90%: few settled bets, so widen rather than pretend to know.
    return CALIB_SIGMA


@dataclass
class Uncertainty:
    sigma: float
    n_sources: int
    regime: str                       # "live" | "prematch"
    spread: float                     # raw disagreement between sources
    used: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict:
        return {"sigma": round(self.sigma, 4), "n_sources": self.n_sources,
                "regime": self.regime, "spread": round(self.spread, 4),
                "sources": {k: round(v, 4) for k, v in self.used.items()},
                "note": self.note}


def estimate_uncertainty(estimates: Dict[str, Optional[float]],
                         p_used: float,
                         is_live: bool,
                         source: Optional[str] = None) -> Uncertainty:
    """Combine source disagreement with the calibration floor.

    `estimates` maps source name -> P(player 1 wins), None where unavailable.
    `p_used` is the probability actually being traded on, and `source` is where
    it came from — which decides whether the calibration penalty applies.
    """
    allowed = LIVE_SOURCES if is_live else PREMATCH_SOURCES
    used = {k: float(v) for k, v in estimates.items()
            if v is not None and k in allowed}

    regime = "live" if is_live else "prematch"
    calib = calibration_sigma(p_used, source)

    if len(used) >= 2:
        vals = list(used.values())
        spread = max(vals) - min(vals)
        # Population sd: we are describing these estimates, not sampling a
        # larger population of them.
        disagree = pstdev(vals)
        note = f"{len(used)} {regime} sources"
    elif len(used) == 1:
        spread = 0.0
        disagree = SIGMA_SOLO
        note = f"single {regime} source — no disagreement signal, floor applied"
    else:
        spread = 0.0
        disagree = SIGMA_SOLO * 1.5
        note = f"no {regime} source — sigma is a guess"

    # Independent-ish error terms add in quadrature. They are not truly
    # independent (shared inputs), so this is a floor on the real uncertainty.
    sigma = max((disagree ** 2 + calib ** 2) ** 0.5, SIGMA_FLOOR)
    return Uncertainty(sigma=sigma, n_sources=len(used), regime=regime,
                       spread=spread, used=used, note=note)


@dataclass
class Scored:
    edge: float
    edge_score: float
    sigma: float
    grade: str                      # "green" | "amber" | "red"
    reasons: list
    uncertainty: Uncertainty

    @property
    def tradeable(self) -> bool:
        return self.grade == "green"

    def as_dict(self) -> dict:
        return {"edge": round(self.edge, 4), "edge_score": round(self.edge_score, 2),
                "sigma": round(self.sigma, 4), "grade": self.grade,
                "reasons": self.reasons, "uncertainty": self.uncertainty.as_dict()}


def score_edge(p_model: float, p_market: float,
               estimates: Dict[str, Optional[float]],
               is_live: bool,
               source: Optional[str] = None,
               liquidity_ok: bool = True,
               stale: bool = False) -> Scored:
    """Grade one opportunity. `p_model` and `p_market` are for the SAME side."""
    unc = estimate_uncertainty(estimates, p_model, is_live, source)
    edge = p_model - p_market
    escore = edge / unc.sigma if unc.sigma > 0 else 0.0

    reasons = []
    grade = "green"

    if abs(edge) < MIN_ABS_EDGE:
        grade = "red"
        reasons.append(f"edge {edge:+.1%} below {MIN_ABS_EDGE:.0%} floor")
    if escore < AMBER_SCORE:
        grade = "red"
        reasons.append(f"EdgeScore {escore:.2f} < {AMBER_SCORE} — edge is inside the noise")
    elif escore < GREEN_SCORE and grade != "red":
        grade = "amber"
        reasons.append(f"EdgeScore {escore:.2f} — real but not clean; wait for confirmation")

    if unc.n_sources < 2:
        # Not fatal on its own, but it means sigma is a floor, not a measurement.
        if grade == "green":
            grade = "amber"
        reasons.append(unc.note)
    if stale:
        grade = "red"
        reasons.append("scoreboard contradicts the pick — source is stale")
    if not liquidity_ok:
        grade = "red"
        reasons.append("insufficient liquidity")

    if grade == "green" and not reasons:
        reasons.append(f"EdgeScore {escore:.2f} on {unc.n_sources} {unc.regime} sources")

    return Scored(edge=edge, edge_score=escore, sigma=unc.sigma,
                  grade=grade, reasons=reasons, uncertainty=unc)


def size_multiplier(scored: Scored) -> float:
    """Fraction of the Kelly stake this confidence justifies.

    Kelly assumes the probability is correct. Ours measurably is not, so the
    stake is scaled by confidence rather than bet in full — the standard
    fractional-Kelly response to parameter uncertainty.
    """
    if scored.grade == "red":
        return 0.0
    if scored.grade == "amber":
        return 0.25
    # Green: ramp from half to full Kelly between the green and 2x-green marks.
    span = max(1e-9, GREEN_SCORE)
    extra = min(1.0, max(0.0, (scored.edge_score - GREEN_SCORE) / span))
    return 0.5 + 0.5 * extra
