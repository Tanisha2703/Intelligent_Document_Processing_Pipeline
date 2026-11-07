"""
XFUND forms in 7 languages (real documents): zero-shot vs fine-tuned Qwen2.5-VL-3B, per language, as key-value pair F1.

Inputs: results/raw/vlm_<tag>.jsonl from `experiments/serving/modal_run.py --suite real --data data/real/xfund_test --tag <tag> [--adapter ...]`.
Reported: pair precision / recall / F1 (lenient = accent-folded fuzzy match on key and value; exact = normalised equality),
per language and overall, with a bootstrap over forms, and the paired difference per form.

    python experiments/evaluation/analyze_xfund.py --out results/raw/xfund_compare.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.xfund import match_pairs, prf  # noqa: E402

SYSTEMS = {
    "zero_shot_3b": "real_zs3b_xfund_test",
    "fine_tuned_3b": "real_ft3b_xfund_xfund_test",
    "zero_shot_7b": "real_zs7b_xfund_test",
}
LANGS = ["zh", "ja", "es", "fr", "it", "de", "pt"]
PRICE_PER_S = 0.00039392


def load(tag):
    p = ROOT / "results" / "raw" / f"vlm_{tag}.jsonl"
    if not p.exists():
        return None
    return {json.loads(x)["id"].split("|")[-1]: json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()}


def scored(recs, labels):
    out = {}
    for f, r in recs.items():
        out[f] = {
            **match_pairs(r["truth"]["pairs"], r["fields"].get("pairs", [])),
            "lang": labels[f]["lang"],
            "hit_cap": r["n_output_tokens"] >= 2040,
            "seconds": r["item_seconds"],
        }
    return out


def boot(rows, mode, n=2000, seed=0):
    rng = random.Random(seed)
    v = sorted(prf([rng.choice(rows) for _ in rows], mode)["f1"] for _ in range(n))
    return [v[int(0.025 * n)], v[int(0.975 * n)]]


def summary(rows):
    s = {"n_forms": len(rows), "n_true_pairs": sum(r["n_true"] for r in rows)}
    for mode in ("lenient", "exact"):
        s[mode] = {**prf(rows, mode), "f1_ci95": boot(rows, mode)}
    s["seconds_per_form"] = float(np.mean([r["seconds"] for r in rows]))
    s["output_hit_token_cap"] = int(sum(r["hit_cap"] for r in rows))
    return s


def paired(a, b, files, seed=0):
    def f1(r):
        return 0.0 if not r["n_true"] else (2 * r["lenient"] / (r["n_true"] + r["n_pred"]) if r["n_pred"] else 0.0)

    d = [f1(b[f]) - f1(a[f]) for f in files]
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return {"mean_per_form_f1_diff": float(np.mean(d)), "ci95": [m[50], m[1949]], "n": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    labels = {
        x["file"]: x
        for x in json.loads((ROOT / "data" / "real" / "xfund_test" / "labels.json").read_text(encoding="utf-8"))
    }
    res = {
        "_run": {"test": "XFUND val split, 15 forms per language (seeded sample)", "price_per_second": PRICE_PER_S},
        "systems": {},
    }
    S = {}
    for k, tag in SYSTEMS.items():
        recs = load(tag)
        if not recs:
            continue
        S[k] = scored(recs, labels)
        small = [f for f in S[k] if len(labels[f]["entities"]["pairs"]) <= 60]
        res["systems"][k] = {
            "all": summary(list(S[k].values())),
            "max_60_pairs": summary([S[k][f] for f in small]),
            "by_language": {
                lg: summary([r for r in S[k].values() if r["lang"] == lg])
                for lg in LANGS
                if any(r["lang"] == lg for r in S[k].values())
            },
        }
        res["systems"][k]["usd_per_1000_forms_at_a10_price"] = (
            res["systems"][k]["all"]["seconds_per_form"] * PRICE_PER_S * 1000
        )
    if "zero_shot_3b" in S and "fine_tuned_3b" in S:
        files = sorted(set(S["zero_shot_3b"]) & set(S["fine_tuned_3b"]))
        res["fine_tuned_3b_minus_zero_shot_3b"] = {
            "all": paired(S["zero_shot_3b"], S["fine_tuned_3b"], files),
            **{
                lg: paired(S["zero_shot_3b"], S["fine_tuned_3b"], [f for f in files if labels[f]["lang"] == lg])
                for lg in LANGS
            },
        }
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, s in res["systems"].items():
        print(
            k,
            "all F1",
            round(s["all"]["lenient"]["f1"], 3),
            {lg: round(v["lenient"]["f1"], 2) for lg, v in s["by_language"].items()},
        )


if __name__ == "__main__":
    main()
