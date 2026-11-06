"""
Regenerate the README figures from the raw results (results/raw): nothing here is typed in by hand.

    python benchmarks/make_figures.py

Style: light chart surface, recessive hairline grid, thin marks, series colours from a validated categorical order.
Values are always also printed in ink or tabulated in docs/RESULTS.md.
"""

import json  # noqa: F401
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
RAW, FIG = ROOT / "results" / "raw", ROOT / "results" / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
    }
)


def _save(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


REAL_SYSTEMS = [
    ("zero_shot_7b", "7B zero-shot", MUTED),
    ("zero_shot_3b", "3B zero-shot", AXIS),
    ("fine_tuned_3b", "3B fine-tuned", BLUE),
    ("fine_tuned_7b", "7B fine-tuned", ORANGE),
]


def fig_real_sroie_accuracy():
    """Field accuracy on the 361 real SROIE test receipts, overall and on shops never seen in fine-tuning."""
    d = (
        json.loads((RAW / "real_compare.json").read_text(encoding="utf-8"))["systems"]
        if (RAW / "real_compare.json").exists()
        else None
    )
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.38
    for gi, grp in enumerate(("all", "unseen_shops")):
        for i, (k, _, col) in enumerate(REAL_SYSTEMS):
            if k not in d:
                continue
            g = d[k][grp]
            x = i + (gi - 0.5) * w
            ax.bar(x, g["field_accuracy"], w * 0.92, color=col, alpha=1.0 if gi == 0 else 0.55, label=None)
            ax.errorbar(
                x,
                g["field_accuracy"],
                yerr=[[g["field_accuracy"] - g["ci95"][0]], [g["ci95"][1] - g["field_accuracy"]]],
                color=INK,
                lw=1,
                capsize=2.5,
            )
            ax.text(x, g["ci95"][1] + 0.006, f"{g['field_accuracy']:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(REAL_SYSTEMS)), [n for _, n, _ in REAL_SYSTEMS])
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("field accuracy (exact match)")
    ax.set_title(
        "Real receipts (SROIE): left bar all test, right bar unseen shops; 95% bootstrap",
        fontsize=10,
        color=INK2,
        loc="left",
    )
    _save(fig, "real_sroie_accuracy")


def fig_real_risk_coverage():
    """Accuracy of auto-accepted fields against the share of fields auto-accepted, on real SROIE test receipts."""
    import numpy as np
    from analyze_real import SYSTEMS, load

    from src.eval.receipts import ETYPE, correct
    from src.pipeline.calibration import IsotonicMap

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, name, col in [r for r in REAL_SYSTEMS if r[0] != "zero_shot_3b"]:
        test_tag, dev_tag = SYSTEMS[k]
        test, dev = load(test_tag), load(dev_tag)
        if not (test and dev):
            continue

        def samples(recs):
            x = np.array([(recs[f]["confidence"].get(c) or {}).get("min_prob", 0.0) for f in recs for c in ETYPE])
            y = np.array(
                [float(correct(c, recs[f]["truth"][c], recs[f]["fields"].get(c))) for f in recs for c in ETYPE]
            )
            return x, y

        xd, yd = samples(dev)
        xt, yt = samples(test)
        m = IsotonicMap.fit(xd, yd)
        order = np.argsort(-np.array([m(v) for v in xt]), kind="stable")
        cov = np.arange(1, len(yt) + 1) / len(yt)
        ax.plot(cov, np.cumsum(yt[order]) / np.arange(1, len(yt) + 1), color=col, lw=1.8, label=name)
    ax.axhline(0.99, color=GRID, lw=1)
    ax.set_xlabel("share of fields auto-accepted (most confident first)")
    ax.set_ylabel("accuracy of the accepted fields")
    ax.set_ylim(0.8, 1.005)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _save(fig, "real_risk_coverage")


FIGS = {"real_sroie_accuracy": fig_real_sroie_accuracy, "real_risk_coverage": fig_real_risk_coverage}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        FIGS[name]()
