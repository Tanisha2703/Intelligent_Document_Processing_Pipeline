"""Calibration metrics and the accept-threshold rule."""

import numpy as np
from sklearn.metrics import roc_auc_score


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: bin-size-weighted gap between mean confidence and accuracy."""
    if len(p) == 0:
        return float("nan")
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(
        sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    )


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return [
        {"mean_conf": float(p[idx == b].mean()), "accuracy": float(y[idx == b].mean()), "n": int((idx == b).sum())}
        for b in range(bins)
        if (idx == b).any()
    ]


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def choose_accept(conf: np.ndarray, ok: np.ndarray, target: float) -> float:
    """Smallest probability threshold whose DEV precision on accepted fields reaches `target` (1.0 if none does).

    A threshold must be supported by at least 20 accepted fields, so a handful of lucky fields cannot set it."""
    for t in np.unique(conf):
        sel = conf >= t
        if sel.sum() >= 20 and ok[sel].mean() >= target:
            return float(t)
    return 1.0
