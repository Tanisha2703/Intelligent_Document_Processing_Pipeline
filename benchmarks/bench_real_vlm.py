"""
Score a VLM's output on the real SROIE receipts (results/raw/vlm_real_*.jsonl, produced by
`modal_run.py --suite real --data data/real/sroie_test --tag <tag>`).

A field is correct when its normalised value equals the label's (dates and amounts via src/utils/normalize.py).
Because the dataset's own labels contain typos (a printed "81750" labelled "B1750"), text fields are also scored
"near exact": within 2 characters of edit distance after normalisation. Both are reported.

    python benchmarks/bench_real_vlm.py --out results/raw/real_vlm.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.calibration import ece  # noqa: E402
from src.eval.receipts import ETYPE, correct  # noqa: E402


def conf(r, field):
    c = r["confidence"].get(field) or {}
    return c.get("mean_prob", 0.0) if isinstance(c, dict) else float(c)


def boot(xs, n_boot=2000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n_boot))
    return [m[int(0.025 * n_boot)], m[int(0.975 * n_boot)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/raw/vlm_real_Qwen2.5-VL-7B-Instruct_sroie_test.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.loads(x) for x in (ROOT / a.inp).read_text(encoding="utf-8").splitlines()]
    out = {"_run": {"input": a.inp, "n_receipts": len(recs)}, "fields": {}}
    for f in ETYPE:
        ex = [correct(f, r["truth"][f], r["fields"].get(f)) for r in recs]
        nr = [correct(f, r["truth"][f], r["fields"].get(f), near=True) for r in recs]
        out["fields"][f] = {
            "exact": sum(ex) / len(ex),
            "exact_ci95": boot([float(x) for x in ex]),
            "near_exact": sum(nr) / len(nr),
            "missing": sum(not r["fields"].get(f) for r in recs) / len(recs),
        }
    allx = [correct(f, r["truth"][f], r["fields"].get(f)) for r in recs for f in ETYPE]
    alln = [correct(f, r["truth"][f], r["fields"].get(f), near=True) for r in recs for f in ETYPE]
    out["all_fields"] = {
        "exact": sum(allx) / len(allx),
        "exact_ci95": boot([float(x) for x in allx]),
        "near_exact": sum(alln) / len(alln),
        "n_fields": len(allx),
    }
    out["receipts_all_fields_exact"] = sum(
        all(correct(f, r["truth"][f], r["fields"].get(f)) for f in ETYPE) for r in recs
    ) / len(recs)
    out["receipts_all_fields_near_exact"] = sum(
        all(correct(f, r["truth"][f], r["fields"].get(f), near=True) for f in ETYPE) for r in recs
    ) / len(recs)
    probs = [conf(r, f) for r in recs for f in ETYPE]
    out["raw_confidence"] = {
        "ece": ece(np.array(probs), np.array(allx, float)),
        "mean_confidence": sum(probs) / len(probs),
        "accuracy": sum(allx) / len(allx),
    }
    out["seconds_per_page"] = sum(r["item_seconds"] for r in recs) / len(recs)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
