"""
Drift monitor for the confidence signal.

The cascade's router depends on calibration fitted on a reference set. If production documents change (a new
scanner, a new vendor template, a degraded batch), the distribution of the router's score shifts BEFORE anyone
has labelled a single document, which makes it the earliest warning available without ground truth.

PSI (population stability index) between the reference scores and a rolling window of live scores:
  < 0.10 stable, 0.10 - 0.25 moderate shift, > 0.25 large shift (the usual rule-of-thumb bands).
The escalation rate is tracked next to it, because it is what a shift costs in money.
"""

from collections import deque

import numpy as np


def psi(reference, current, bins: int = 10, eps: float = 1e-4, bias_correct: bool = True) -> float:
    """
    Population stability index with bin edges taken from the reference's quantiles.

    PSI is biased upward on small samples: even two draws from the SAME distribution give about
    (bins - 1) * (1/n_ref + 1/n_cur), which for a 100-observation window already sits in the "moderate shift" band.
    `bias_correct` subtracts that expectation (floored at 0) so a short window does not raise false alarms.
    """
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # degenerate reference (constant): any change is a shift
        return 0.0 if np.allclose(ref.mean(), cur.mean()) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    v = float(np.sum((q - p) * np.log(q / p)))
    if bias_correct:
        v = max(v - (len(edges) - 2) * (1 / len(ref) + 1 / len(cur)), 0.0)
    return v


class DriftMonitor:
    """Population-stability drift of the model's expected errors per document against the dev-set reference."""

    def __init__(self, reference_scores, window: int = 200, min_window: int = 50, kappa: float | None = None):
        self.reference = np.asarray(reference_scores, float)
        self.window, self.min_window, self.kappa = window, min_window, kappa
        self._scores: deque = deque(maxlen=window)

    def observe(self, expected_errors: float) -> None:
        self._scores.append(float(expected_errors))

    def status(self) -> dict:
        """PSI against the reference and a level (stable / moderate / large); 'insufficient_data' until min_window documents."""
        n = len(self._scores)
        out = {"n_observed": n, "window": self.window, "psi": None, "level": "insufficient_data"}
        if n < self.min_window:
            return out
        cur = np.array(self._scores)
        v = psi(self.reference, cur)
        out.update(
            psi=round(v, 4),
            level="stable" if v < 0.10 else "moderate" if v < 0.25 else "large",
            mean_score=round(float(cur.mean()), 4),
            reference_mean_score=round(float(self.reference.mean()), 4),
        )
        if self.kappa is not None:
            out["escalation_rate"] = round(float((cur >= self.kappa).mean()), 4)
            out["reference_escalation_rate"] = round(float((self.reference >= self.kappa).mean()), 4)
        return out
