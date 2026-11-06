"""
Confidence calibration.

A model's token probability is a ranking signal, not a probability. `IsotonicMap` turns it into P(field is correct),
fitted on held-out labelled data (the dev split) as a monotone step function; applying it needs only numpy.
"""

import numpy as np


class IsotonicMap:
    """Monotone non-decreasing map from a raw confidence in [0, 1] to P(correct), piecewise linear between knots."""

    def __init__(self, xs, ys):
        order = np.argsort(xs)
        self.xs = np.asarray(xs, float)[order]
        self.ys = np.asarray(ys, float)[order]
        if len(self.xs) == 0:
            raise ValueError("an IsotonicMap needs at least one knot")
        if np.any(np.diff(self.ys) < -1e-12):
            raise ValueError("calibration knots must be non-decreasing")

    @classmethod
    def fit(cls, p, y) -> "IsotonicMap":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            np.asarray(p, float), np.asarray(y, float)
        )
        return cls(iso.X_thresholds_, iso.y_thresholds_)

    def __call__(self, p) -> float:
        return float(np.interp(float(p), self.xs, self.ys))  # clamps outside the knots

    def to_dict(self) -> dict:
        return {"xs": [round(float(x), 6) for x in self.xs], "ys": [round(float(y), 6) for y in self.ys]}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicMap":
        return cls(d["xs"], d["ys"])
