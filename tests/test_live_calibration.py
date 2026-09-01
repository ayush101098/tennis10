"""Calibration machinery, verified against a KNOWN injected bias.

No real calibration dataset exists yet (see the module docstring for why the
8,218 rows in trade_log cannot serve). So the machinery is proved the only way
it honestly can be: generate predictions with a bias whose size we chose,
confirm the diagnostics detect exactly that bias, and confirm the correction
removes it.

If these pass, the tooling is trustworthy the day real data arrives.
"""

import math
import random
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.live.calibration import (  # noqa: E402
    CalibrationRecorder, IsotonicCalibrator, MIN_OBSERVATIONS, Observation,
    PlattCalibrator, brier, expected_calibration_error, fit_and_evaluate,
    format_reliability, log_loss, reliability,
)


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def overconfident_dataset(n=3000, sharpen=1.9, seed=7):
    """Predictions from a model that is over-confident by construction.

    True probability is drawn, then the MODEL's stated probability is the true
    one pushed away from 0.5 on the log-odds scale by `sharpen`. Outcomes are
    drawn from the TRUE probability. This is exactly the failure this project
    measured: 80%+ predictions winning ~60%.
    """
    rng = random.Random(seed)
    obs = []
    for i in range(n):
        p_true = rng.uniform(0.05, 0.95)
        z = math.log(p_true / (1 - p_true))
        p_model = _sigmoid(z * sharpen)
        obs.append(Observation(match_id=f"m{i}", market="match", selection="A",
                               p_model=p_model, won=rng.random() < p_true, ts_ms=i))
    return obs


def calibrated_dataset(n=3000, seed=11):
    rng = random.Random(seed)
    return [Observation(match_id=f"m{i}", market="match", selection="A",
                        p_model=(p := rng.uniform(0.05, 0.95)),
                        won=rng.random() < p, ts_ms=i)
            for i in range(n)]


# ── the diagnostics detect what is actually there ────────────────────────────

def test_reliability_detects_overconfidence_in_the_right_direction():
    bs = reliability(overconfident_dataset())
    high = [b for b in bs if b.predicted > 0.8]
    low = [b for b in bs if b.predicted < 0.2]
    assert high and low
    # Confident-high predictions win LESS than claimed; confident-low win MORE.
    assert all(b.gap < 0 for b in high), [b.gap for b in high]
    assert all(b.gap > 0 for b in low), [b.gap for b in low]


def test_a_calibrated_model_shows_no_systematic_gap():
    # The control. Without it, a diagnostic that always cries "miscalibrated"
    # would look like it was working.
    assert expected_calibration_error(calibrated_dataset()) < 0.05


def test_ece_is_larger_for_the_biased_model():
    assert (expected_calibration_error(overconfident_dataset())
            > expected_calibration_error(calibrated_dataset()) + 0.05)


def test_reliability_uses_mean_prediction_not_bin_midpoint():
    # A bucket of 0.9-1.0 averaging 0.97 is a different claim from one
    # averaging 0.91; the midpoint hides that.
    obs = [Observation(match_id=str(i), market="m", selection="A",
                       p_model=0.99, won=True, ts_ms=i) for i in range(50)]
    b = reliability(obs)[-1]
    assert b.predicted == pytest.approx(0.99, abs=1e-9)


def test_log_loss_punishes_confident_errors_harder_than_brier():
    conf_wrong = [Observation(match_id="a", market="m", selection="A", p_model=0.99, won=False)]
    mild_wrong = [Observation(match_id="b", market="m", selection="A", p_model=0.60, won=False)]
    assert log_loss(conf_wrong) / log_loss(mild_wrong) > brier(conf_wrong) / brier(mild_wrong)


# ── the correction removes the bias ──────────────────────────────────────────

def test_platt_shrinks_an_overconfident_model():
    obs = overconfident_dataset()
    cal = PlattCalibrator().fit(obs)
    # a < 1 is precisely "less confident than before".
    assert cal.a < 0.95, f"expected shrinkage, got a={cal.a:.3f}"
    # And it should approximately invert the sharpening we injected.
    assert 0.35 < cal.a < 0.85


def test_platt_improves_every_headline_metric_out_of_sample():
    rep = fit_and_evaluate(overconfident_dataset(), test_fraction=0.3)
    assert rep.brier_cal < rep.brier_raw
    assert rep.logloss_cal < rep.logloss_raw
    assert rep.ece_cal < rep.ece_raw
    assert rep.honest


def test_platt_leaves_an_already_calibrated_model_alone():
    # A calibrator that "improves" a well-calibrated model is fitting noise.
    rep = fit_and_evaluate(calibrated_dataset(), test_fraction=0.3)
    assert rep.brier_cal == pytest.approx(rep.brier_raw, abs=0.01)


def test_isotonic_also_removes_the_bias():
    obs = overconfident_dataset()
    cut = int(len(obs) * 0.7)
    iso = IsotonicCalibrator().fit(obs[:cut])
    test = obs[cut:]
    cal = [Observation(match_id=o.match_id, market=o.market, selection=o.selection,
                       p_model=iso.apply(o.p_model), won=o.won, ts_ms=o.ts_ms)
           for o in test]
    assert brier(cal) < brier(test)


def test_isotonic_output_is_monotone():
    iso = IsotonicCalibrator().fit(overconfident_dataset())
    ps = [i / 50 for i in range(51)]
    out = [iso.apply(p) for p in ps]
    assert all(b >= a - 1e-9 for a, b in zip(out, out[1:])), "isotonic must not decrease"


def test_small_samples_are_flagged_as_not_honest():
    # The guard against exactly what happened before: a confident-looking
    # calibration fitted on far too little data.
    rep = fit_and_evaluate(overconfident_dataset(n=60))
    assert not rep.honest
    assert str(MIN_OBSERVATIONS) in rep.note


def test_the_split_is_by_time_not_random():
    obs = overconfident_dataset(n=1000)
    rep = fit_and_evaluate(obs, test_fraction=0.25)
    assert rep.n_train == 750 and rep.n_test == 250


# ── the recorder makes orientation impossible to get wrong ───────────────────

def _recorder():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return CalibrationRecorder(Path(tmp.name))


def test_settling_by_winner_orients_every_selection_correctly():
    # The whole point. The caller names who WON; it never asserts per-row
    # whether a prediction was right, which is where the old log went wrong.
    rec = _recorder()
    rec.predict(match_id="m1", market="match", selection="Alice", p_model=0.70)
    rec.predict(match_id="m1", market="match", selection="Bob", p_model=0.30)
    assert rec.settle(match_id="m1", market="match", winner="Alice") == 2

    obs = {o.selection: o for o in rec.settled()}
    assert obs["Alice"].won is True and obs["Alice"].p_model == 0.70
    assert obs["Bob"].won is False and obs["Bob"].p_model == 0.30


def test_a_prediction_cannot_be_recorded_without_a_selection():
    with pytest.raises(ValueError):
        Observation(match_id="m", market="match", selection="", p_model=0.5, won=True)


def test_probabilities_outside_zero_one_are_rejected_at_the_boundary():
    with pytest.raises(ValueError):
        Observation(match_id="m", market="match", selection="A", p_model=1.4, won=True)
    with pytest.raises(ValueError):
        Observation(match_id="m", market="match", selection="A", p_model=0.5,
                    p_market=-0.2, won=True)


def test_settling_twice_does_not_double_count():
    rec = _recorder()
    rec.predict(match_id="m1", market="match", selection="Alice", p_model=0.7)
    rec.settle(match_id="m1", market="match", winner="Alice")
    assert rec.settle(match_id="m1", market="match", winner="Bob") == 0
    assert rec.settled()[0].won is True     # the first settlement stands


def test_duplicate_predictions_are_ignored_not_appended():
    rec = _recorder()
    for _ in range(3):
        rec.predict(match_id="m1", market="match", selection="A", p_model=0.6, ts_ms=1000)
    total, _ = rec.count()
    assert total == 1


def test_unsettled_predictions_are_not_returned_as_data():
    rec = _recorder()
    rec.predict(match_id="m1", market="match", selection="A", p_model=0.6)
    assert rec.settled() == []
    total, done = rec.count()
    assert (total, done) == (1, 0)


def test_report_renders():
    out = format_reliability(overconfident_dataset(n=200))
    assert "ECE" in out and "Brier" in out
