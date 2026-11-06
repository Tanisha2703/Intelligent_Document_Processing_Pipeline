"""
CORD receipts: zero-shot vs fine-tuned VLM on three amounts (total, subtotal, tax) and the menu (item-level F1).
Inputs: results/raw/vlm_<tag>.jsonl written by `modal_run.py --suite real --data data/real/cord_test --tag <tag>`.

    python benchmarks/analyze_cord.py --out results/raw/cord_compare.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.cord import SCALARS, menu_prf, scalar_correct, score_menu  # noqa: E402

SYSTEMS = {
    "zero_shot_7b": "real_zs7b_cord_test",
    "zero_shot_3b": "real_zs3b_cord_test",
    "fine_tuned_3b": "real_ft3b_cord_test",
    "fine_tuned_7b": "real_ft7b_cord_test",
}
PRICE_PER_S = 0.00039392  # A10 on Modal, USD per second


def load(tag):
    p = ROOT / "results" / "raw" / f"vlm_{tag}.jsonl"
    if not p.exists():
        return None
    return {json.loads(x)["id"].split("|")[-1]: json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()}


def per_receipt(r):
    sc = {f: scalar_correct(r["truth"].get(f), r["fields"].get(f)) for f in SCALARS}
    return sc, score_menu(r["truth"].get("menu", []), r["fields"].get("menu", []))


def boot(fn, rows, n=2000, seed=0):
    rng = random.Random(seed)
    v = sorted(fn([rng.choice(rows) for _ in rows]) for _ in range(n))
    return [v[int(0.025 * n)], v[int(0.975 * n)]]


def summarize(recs):
    rows = [per_receipt(r) for r in recs.values()]
    scal = lambda rs: float(np.mean([np.mean(list(s.values())) for s, _ in rs]))  # noqa: E731
    f1 = lambda rs: menu_prf([m for _, m in rs])["f1"]  # noqa: E731
    out = {
        "n": len(rows),
        "scalar_accuracy": scal(rows),
        "scalar_accuracy_ci95": boot(scal, rows),
        "per_scalar": {f: float(np.mean([s[f] for s, _ in rows])) for f in SCALARS},
        "menu": menu_prf([m for _, m in rows]),
        "menu_f1_ci95": boot(f1, rows),
        "receipts_all_scalars_right": float(np.mean([all(s.values()) for s, _ in rows])),
        "receipts_menu_perfect": float(np.mean([m["matched"] == m["n_true"] == m["n_pred"] for _, m in rows])),
        "seconds_per_page": float(np.mean([r["item_seconds"] for r in recs.values()])),
    }
    out["usd_per_1000_pages_at_a10_price"] = out["seconds_per_page"] * PRICE_PER_S * 1000
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, rows_by = {"_run": {"test": "CORD-v2 test split", "price_per_second": PRICE_PER_S}, "systems": {}}, {}
    for k, tag in SYSTEMS.items():
        recs = load(tag)
        if recs:
            res["systems"][k], rows_by[k] = summarize(recs)
    ref = rows_by.get("zero_shot_7b")
    for k, rows in rows_by.items():
        if ref and k != "zero_shot_7b" and len(rows) == len(ref):
            d_s = [
                np.mean(list(s.values())) - np.mean(list(r0.values()))
                for (s, _), (r0, _) in zip(rows, ref, strict=True)
            ]
            rng = random.Random(0)
            m = sorted(np.mean([rng.choice(d_s) for _ in d_s]) for _ in range(2000))
            res["systems"][k]["scalar_accuracy_vs_zero_shot_7b"] = {
                "diff": float(np.mean(d_s)),
                "ci95": [float(m[50]), float(m[1949])],
            }
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, s in res["systems"].items():
        print(
            k,
            "scalars",
            round(s["scalar_accuracy"], 3),
            {f: round(v, 3) for f, v in s["per_scalar"].items()},
            "menu F1",
            round(s["menu"]["f1"], 3),
            "P",
            round(s["menu"]["precision"], 3),
            "R",
            round(s["menu"]["recall"], 3),
        )


if __name__ == "__main__":
    main()
