"""Calibration error, reliability bins, AUROC and the accept-threshold rule."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.eval.calibration import auroc, choose_accept, ece, reliability  # noqa: E402


def test_ece_is_zero_for_a_perfectly_calibrated_set_and_large_for_an_overconfident_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 40000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)
    assert ece(p, y) < 0.02
    assert ece(np.full(1000, 0.95), np.zeros(1000)) == pytest.approx(0.95)  # says 95%, is always wrong
    assert ece(np.array([]), np.array([])) != ece(np.array([]), np.array([]))  # nan for no data


def test_ece_weights_bins_by_how_many_predictions_they_hold():
    # 90 predictions at 0.9 (all correct: gap 0.1) and 10 at 0.1 (all correct: gap 0.9): 0.9*0.1 + 0.1*0.9
    p = np.array([0.9] * 90 + [0.1] * 10)
    assert ece(p, np.ones(100)) == pytest.approx(0.9 * 0.1 + 0.1 * 0.9)


def test_reliability_bins_report_mean_confidence_accuracy_and_count():
    p = np.array([0.05, 0.07, 0.95, 0.97, 0.99])
    y = np.array([0, 1, 1, 1, 0], float)
    bins = reliability(p, y)
    assert [b["n"] for b in bins] == [2, 3]
    assert bins[0]["accuracy"] == pytest.approx(0.5) and bins[1]["accuracy"] == pytest.approx(2 / 3)
    assert bins[1]["mean_conf"] == pytest.approx(np.mean([0.95, 0.97, 0.99]))


def test_auroc_is_one_for_a_perfect_ranking_and_nan_for_one_class():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1], float)) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1], float)) == 0.0
    assert np.isnan(auroc(np.array([0.5, 0.6]), np.array([1, 1], float)))


def test_choose_accept_picks_the_smallest_threshold_meeting_the_precision_target():
    conf = np.concatenate([np.linspace(0.5, 0.9, 100), np.linspace(0.9, 1.0, 100)])
    ok = np.concatenate([np.zeros(100), np.ones(100)])  # right only above 0.9
    t = choose_accept(conf, ok, 0.99)
    assert 0.89 < t < 0.95
    assert ok[conf >= t].mean() >= 0.99


def test_choose_accept_returns_one_when_no_threshold_reaches_the_target():
    conf = np.linspace(0, 1, 200)
    assert choose_accept(conf, np.zeros(200), 0.99) == 1.0


def test_choose_accept_ignores_thresholds_supported_by_fewer_than_twenty_fields():
    conf = np.array([0.99] * 5 + [0.5] * 95)  # 5 perfect fields at the top, rest wrong
    ok = np.array([1.0] * 5 + [0.0] * 95)
    assert choose_accept(conf, ok, 0.99) == 1.0  # 5 fields are not evidence enough
