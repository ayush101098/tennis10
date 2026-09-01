"""Calibration: recording it honestly, then measuring and correcting it.

WHY THIS MODULE EXISTS RATHER THAN A FITTED CONSTANT
    The model is known to be over-confident. `execution/calibrate.py` says so
    ("predicts 80%+, wins ~60%") and an independent measurement against live
    markets put it ~13 points from the market per leg and ~14 points high on
    favourites. The obvious next step is to fit a correction on settled bets.

    That was attempted on `tennis_betting.db:trade_log` — 8,218 settled rows —
    and the data does not support it. The evidence:

        corr(true_p, market_price)          +0.15   (should be ~+0.8 if both
                                                     describe the same side)
        reliability, p in 0.0-0.1           88% won
        reliability, p in 0.9-1.0           64% won

    A prediction of 0.05 winning 88% of the time is not a miscalibrated model,
    it is a mislabelled column. Testing all four orientation conventions
    (flip probability / flip price / both / neither) across match, set1 and
    set2 markets produced no consistent convention: set markets correlate
    -0.62 as stored and +0.38 under a different flip, i.e. the log mixes
    conventions across market types and across code versions.

    Fitting on that yields an authoritative-looking number that means nothing,
    and it would then be wired into the live signal gate where it silently
    corrupts every signal. So this module does two things instead:

      1. `CalibrationRecorder` writes observations with ONE unambiguous
         orientation, enforced at write time, so the dataset that does not
         currently exist starts existing.
      2. `reliability`, `brier`, `PlattCalibrator` and `IsotonicCalibrator`
         measure and correct — and are tested against synthetic data with a
         KNOWN injected bias, so the machinery is verified even though no real
         dataset has accumulated yet.

    The honest status is: the correction is ready, the data is not.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "tennis_points.db"


@dataclass(frozen=True)
class Observation:
    """One (prediction, outcome) pair, with an orientation that cannot drift.

    THE INVARIANT: `p_model` and `p_market` are BOTH the probability of
    `selection` winning, and `won` is whether `selection` actually won. There
    is no "player1" convention, no side column to misread later. Everything is
    expressed from the point of view of the thing being predicted.

    That is the single design decision this module exists to enforce, because
    its absence is what made 8,218 rows of history unusable.
    """

    match_id: str
    market: str                  # "match" | "set1" | ...
    selection: str               # the competitor this probability is ABOUT
    p_model: float
    won: bool
    p_market: Optional[float] = None
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    source: str = "inplay"

    def __post_init__(self):
        if not 0.0 <= self.p_model <= 1.0:
            raise ValueError(f"p_model out of range: {self.p_model}")
        if self.p_market is not None and not 0.0 <= self.p_market <= 1.0:
            raise ValueError(f"p_market out of range: {self.p_market}")
        if not self.selection:
            raise ValueError("selection is required — it is what defines the orientation")


class CalibrationRecorder:
    """Append-only store of observations.

    Predictions are written when made and settled later, because a prediction
    recorded after the outcome is known is worthless — and indistinguishable
    from an honest one once it is in the table.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS calibration_obs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id    TEXT NOT NULL,
        market      TEXT NOT NULL,
        selection   TEXT NOT NULL,
        p_model     REAL NOT NULL,
        p_market    REAL,
        source      TEXT,
        predicted_ms INTEGER NOT NULL,
        settled_ms  INTEGER,
        won         INTEGER,
        UNIQUE(match_id, market, selection, predicted_ms)
    );
    CREATE INDEX IF NOT EXISTS idx_calib_settled ON calibration_obs(settled_ms);
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB)
        con = self._con()
        con.executescript(self.SCHEMA)
        con.commit()
        con.close()

    def _con(self):
        return sqlite3.connect(self.db_path)

    def predict(self, *, match_id: str, market: str, selection: str,
                p_model: float, p_market: Optional[float] = None,
                source: str = "inplay", ts_ms: Optional[int] = None) -> None:
        """Record a prediction BEFORE the outcome is known."""
        ts = ts_ms if ts_ms is not None else int(time.time() * 1000)
        con = self._con()
        try:
            con.execute(
                "INSERT OR IGNORE INTO calibration_obs "
                "(match_id, market, selection, p_model, p_market, source, predicted_ms) "
                "VALUES (?,?,?,?,?,?,?)",
                (match_id, market, selection, float(p_model),
                 None if p_market is None else float(p_market), source, ts))
            con.commit()
        finally:
            con.close()

    def settle(self, *, match_id: str, market: str, winner: str,
               ts_ms: Optional[int] = None) -> int:
        """Settle every open prediction on this market.

        `winner` is the selection that won; every recorded selection is scored
        against it. Settling by naming the WINNER rather than by passing a
        boolean per row removes the last place an orientation could be
        inverted by a caller.
        """
        ts = ts_ms if ts_ms is not None else int(time.time() * 1000)
        con = self._con()
        try:
            cur = con.execute(
                "UPDATE calibration_obs SET won = (selection = ?), settled_ms = ? "
                "WHERE match_id = ? AND market = ? AND settled_ms IS NULL",
                (winner, ts, match_id, market))
            con.commit()
            return cur.rowcount
        finally:
            con.close()

    def settled(self, *, since_ms: int = 0, market: Optional[str] = None) -> list:
        con = self._con()
        try:
            q = ("SELECT match_id, market, selection, p_model, p_market, won, settled_ms, source "
                 "FROM calibration_obs WHERE settled_ms IS NOT NULL AND settled_ms >= ?")
            args = [since_ms]
            if market:
                q += " AND market = ?"
                args.append(market)
            rows = con.execute(q + " ORDER BY settled_ms", args).fetchall()
        finally:
            con.close()
        return [Observation(match_id=r[0], market=r[1], selection=r[2], p_model=r[3],
                            p_market=r[4], won=bool(r[5]), ts_ms=r[6], source=r[7] or "")
                for r in rows]

    def count(self) -> tuple[int, int]:
        con = self._con()
        try:
            total = con.execute("SELECT COUNT(*) FROM calibration_obs").fetchone()[0]
            done = con.execute(
                "SELECT COUNT(*) FROM calibration_obs WHERE settled_ms IS NOT NULL").fetchone()[0]
        finally:
            con.close()
        return total, done


# ── measurement ──────────────────────────────────────────────────────────────

@dataclass
class Bucket:
    lo: float
    hi: float
    n: int
    predicted: float
    actual: float

    @property
    def gap(self) -> float:
        return self.actual - self.predicted


def reliability(obs: Sequence[Observation], bins: int = 10) -> list:
    """Reliability diagram: what we said vs what happened.

    Buckets carry the MEAN PREDICTION inside them rather than the bin midpoint.
    With a skewed prediction distribution the midpoint misstates the gap — a
    bucket of 0.9-1.0 predictions averaging 0.97 is a different claim from one
    averaging 0.91, and using 0.95 for both hides it.
    """
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [o for o in obs if lo <= o.p_model < hi or (b == bins - 1 and o.p_model == 1.0)]
        if not sel:
            continue
        out.append(Bucket(
            lo=lo, hi=hi, n=len(sel),
            predicted=sum(o.p_model for o in sel) / len(sel),
            actual=sum(1 for o in sel if o.won) / len(sel),
        ))
    return out


def brier(obs: Sequence[Observation]) -> float:
    """Mean squared error of the probabilities. Lower is better; 0.25 is the
    score of always saying 50%."""
    if not obs:
        return float("nan")
    return sum((o.p_model - (1.0 if o.won else 0.0)) ** 2 for o in obs) / len(obs)


def log_loss(obs: Sequence[Observation], eps: float = 1e-6) -> float:
    """Punishes confident errors far harder than Brier, which is the failure
    mode this model actually has."""
    if not obs:
        return float("nan")
    t = 0.0
    for o in obs:
        p = min(1 - eps, max(eps, o.p_model))
        t += -(math.log(p) if o.won else math.log(1 - p))
    return t / len(obs)


def expected_calibration_error(obs: Sequence[Observation], bins: int = 10) -> float:
    """Average |predicted - actual|, weighted by bucket population."""
    bs = reliability(obs, bins)
    n = sum(b.n for b in bs)
    if not n:
        return float("nan")
    return sum(abs(b.gap) * b.n for b in bs) / n


# ── correction ───────────────────────────────────────────────────────────────

class PlattCalibrator:
    """Logistic recalibration on the log-odds: p' = sigmoid(a * logit(p) + b).

    `a < 1` shrinks confidence, which is the correction this model needs. Two
    parameters only — with a few thousand observations anything richer fits
    noise, and a calibrator that overfits is worse than none because it looks
    principled.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = a, b

    @staticmethod
    def _logit(p: float, eps: float = 1e-6) -> float:
        p = min(1 - eps, max(eps, p))
        return math.log(p / (1 - p))

    def apply(self, p: float) -> float:
        z = self.a * self._logit(p) + self.b
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, obs: Sequence[Observation], *, iters: int = 400, lr: float = 0.05) -> "PlattCalibrator":
        """Gradient descent on log loss. Plain Python: a few hundred iterations
        over a few thousand points is milliseconds, and it keeps this module
        importable without numpy on a minimal deployment."""
        if not obs:
            return self
        xs = [self._logit(o.p_model) for o in obs]
        ys = [1.0 if o.won else 0.0 for o in obs]
        n = len(xs)
        for _ in range(iters):
            ga = gb = 0.0
            for x, y in zip(xs, ys):
                pred = 1.0 / (1.0 + math.exp(-(self.a * x + self.b)))
                err = pred - y
                ga += err * x
                gb += err
            self.a -= lr * ga / n
            self.b -= lr * gb / n
        return self

    def as_dict(self) -> dict:
        return {"kind": "platt", "a": round(self.a, 6), "b": round(self.b, 6)}


class IsotonicCalibrator:
    """Monotone step-function calibration by pool-adjacent-violators.

    Strictly more expressive than Platt — it can fix a kink Platt cannot — and
    correspondingly hungrier for data. Prefer Platt below a few thousand
    observations; this exists for when the recorder has run long enough.
    """

    def __init__(self, xs: Optional[list] = None, ys: Optional[list] = None):
        self.xs = xs or []
        self.ys = ys or []

    def fit(self, obs: Sequence[Observation]) -> "IsotonicCalibrator":
        pts = sorted(((o.p_model, 1.0 if o.won else 0.0) for o in obs), key=lambda t: t[0])
        if not pts:
            return self
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w = [1.0] * len(ys)
        # Pool adjacent violators: merge any pair that breaks monotonicity.
        i = 0
        while i < len(ys) - 1:
            if ys[i] <= ys[i + 1]:
                i += 1
                continue
            tot = w[i] + w[i + 1]
            ys[i] = (ys[i] * w[i] + ys[i + 1] * w[i + 1]) / tot
            w[i] = tot
            del ys[i + 1], w[i + 1], xs[i + 1]
            if i > 0:
                i -= 1
        self.xs, self.ys = xs, ys
        return self

    def apply(self, p: float) -> float:
        if not self.xs:
            return p
        if p <= self.xs[0]:
            return self.ys[0]
        if p >= self.xs[-1]:
            return self.ys[-1]
        lo, hi = 0, len(self.xs) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.xs[mid] <= p:
                lo = mid
            else:
                hi = mid
        x0, x1 = self.xs[lo], self.xs[hi]
        y0, y1 = self.ys[lo], self.ys[hi]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (p - x0) / (x1 - x0)

    def as_dict(self) -> dict:
        return {"kind": "isotonic", "points": len(self.xs)}


@dataclass
class CalibrationReport:
    n_train: int
    n_test: int
    brier_raw: float
    brier_cal: float
    logloss_raw: float
    logloss_cal: float
    ece_raw: float
    ece_cal: float
    model: dict
    honest: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


# Below this, a fitted calibration is noise wearing a lab coat.
MIN_OBSERVATIONS = 500


def fit_and_evaluate(obs: Sequence[Observation], *, test_fraction: float = 0.3,
                     bins: int = 10) -> CalibrationReport:
    """Fit on the earlier portion, evaluate on the later one.

    A TIME split, not a random one. Random splitting leaks: matches from the
    same tournament, sometimes the same player-week, land on both sides and the
    calibrator scores itself on conditions it has already seen. Time order is
    also how it will actually be used — fit on the past, apply to the future.
    """
    ordered = sorted(obs, key=lambda o: o.ts_ms)
    n = len(ordered)
    cut = int(n * (1 - test_fraction))
    train, test = ordered[:cut], ordered[cut:]

    cal = PlattCalibrator().fit(train)
    cal_test = [Observation(match_id=o.match_id, market=o.market, selection=o.selection,
                            p_model=cal.apply(o.p_model), won=o.won,
                            p_market=o.p_market, ts_ms=o.ts_ms, source=o.source)
                for o in test]

    honest = n >= MIN_OBSERVATIONS
    note = "" if honest else (
        f"only {n} observations — below the {MIN_OBSERVATIONS} floor. Treat these "
        "numbers as a smoke test of the machinery, not as a calibration.")

    return CalibrationReport(
        n_train=len(train), n_test=len(test),
        brier_raw=brier(test), brier_cal=brier(cal_test),
        logloss_raw=log_loss(test), logloss_cal=log_loss(cal_test),
        ece_raw=expected_calibration_error(test, bins),
        ece_cal=expected_calibration_error(cal_test, bins),
        model=cal.as_dict(), honest=honest, note=note,
    )


def format_reliability(obs: Sequence[Observation], bins: int = 10) -> str:
    """The diagram, as text. The most useful single artefact here: a fitted
    parameter hides which band is wrong, and the band is what you act on."""
    bs = reliability(obs, bins)
    if not bs:
        return "no observations"
    out = [f"{'bucket':>12} {'n':>7} {'predicted':>10} {'actual':>9} {'gap':>9}"]
    for b in bs:
        out.append(f"  {b.lo:.1f}-{b.hi:.1f} {b.n:9d} {b.predicted:10.1%} "
                   f"{b.actual:9.1%} {b.gap:+9.1%}")
    out.append(f"\n  ECE {expected_calibration_error(obs, bins):.1%}   "
               f"Brier {brier(obs):.4f}   log-loss {log_loss(obs):.4f}   n={len(obs)}")
    return "\n".join(out)
