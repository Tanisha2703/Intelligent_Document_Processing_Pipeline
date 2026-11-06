"""The isotonic calibration map."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.pipeline.calibration import IsotonicMap  # noqa: E402


def test_fit_recovers_a_known_calibration_curve():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, p.size) < p**2).astype(float)  # true P(correct) = p^2
    m = IsotonicMap.fit(p, y)
    for x in (0.3, 0.6, 0.9):
        assert m(x) == pytest.approx(x**2, abs=0.05)


def test_map_is_monotone_and_clamps_outside_its_knots():
    m = IsotonicMap([0.2, 0.5, 0.9], [0.1, 0.4, 0.95])
    assert m(0.0) == pytest.approx(0.1) and m(1.0) == pytest.approx(0.95)
    assert m(0.3) <= m(0.6) <= m(0.8)


def test_decreasing_knots_are_rejected():
    with pytest.raises(ValueError):
        IsotonicMap([0.1, 0.5, 0.9], [0.5, 0.4, 0.9])


def test_empty_map_is_rejected():
    with pytest.raises(ValueError):
        IsotonicMap([], [])


def test_round_trip_through_a_dict():
    m = IsotonicMap([0.1, 0.5, 0.9], [0.05, 0.5, 0.97])
    back = IsotonicMap.from_dict(m.to_dict())
    assert back(0.7) == pytest.approx(m(0.7), abs=1e-5)
