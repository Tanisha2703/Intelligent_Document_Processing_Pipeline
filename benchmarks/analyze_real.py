"""
Real receipts (SROIE): compare the zero-shot and fine-tuned VLMs, overall and on shops never seen in fine-tuning,
and calibrate the fine-tuned model's confidence on a company-disjoint DEV set. Everything reported is on the test
split; calibration and the accept threshold are chosen on dev only.

Inputs (written by `modal_run.py --suite real --data <folder> --tag <tag>`):
    results/raw/vlm_<tag>.jsonl for the systems named in SYSTEMS below.

    python benchmarks/analyze_real.py --out results/raw/real_compare.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.calibration import choose_accept, ece  # noqa: E402
from src.eval.receipts import ETYPE, content_match, correct  # noqa: E402
from src.pipeline.calibration import IsotonicMap  # noqa: E402

# name -> (test file tag, dev file tag or None)
SYSTEMS = {
    "zero_shot_7b": ("real_Qwen2.5-VL-7B-Instruct_sroie_test", "real_zs7b_sroie_dev"),
    "zero_shot_3b": ("real_zs3b_sroie_test", "real_zs3b_sroie_dev"),
    "fine_tuned_3b": ("real_ft3b_sroie_test", "real_ft3b_sroie_dev"),
    "fine_tuned_7b": ("real_ft7b_sroie_test", "real_ft7b_sroie_dev"),
}
PRICE_PER_S = 0.00039392  # A10 on Modal, USD per second (modal_common.py)


def load(tag):
    p = ROOT / "results" / "raw" / f"vlm_{tag}.jsonl"
    if not p.exists():
        return None
    recs = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        recs[r["id"].split("|")[-1]] = r
    return recs


def fields_ok(r, near=False):
    return [correct(f, r["truth"][f], r["fields"].get(f), near=near) for f in ETYPE]


def fields_content(r):
    return [content_match(f, r["truth"][f], r["fields"].get(f)) for f in ETYPE]


def boot_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return [m[int(0.025 * n)], m[int(0.975 * n)]]


def summary(recs, files):
    rs = [recs[f] for f in files if f in recs]
    per_field = {f: float(np.mean([correct(f, r["truth"][f], r["fields"].get(f)) for r in rs])) for f in ETYPE}
    ok = [float(np.mean(fields_ok(r))) for r in rs]
    return {
        "n": len(rs),
        "per_field_exact": per_field,
        "field_accuracy": float(np.mean(ok)),
        "ci95": boot_ci(ok),
        "near_exact_accuracy": float(np.mean([np.mean(fields_ok(r, near=True)) for r in rs])),
        "per_field_content": {
            f: float(np.mean([content_match(f, r["truth"][f], r["fields"].get(f)) for r in rs])) for f in ETYPE
        },
        "field_accuracy_content": float(np.mean([np.mean(fields_content(r)) for r in rs])),
        "receipts_all_fields_exact": float(np.mean([all(fields_ok(r)) for r in rs])),
        "seconds_per_page": float(np.mean([r["item_seconds"] for r in rs])),
    }


def paired(a, b, files, seed=0, metric=None):
    metric = metric or fields_ok
    d = [float(np.mean(metric(b[f])) - np.mean(metric(a[f]))) for f in files if f in a and f in b]
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return {"diff": float(np.mean(d)), "ci95": [m[50], m[1949]], "n": len(d)}


OPERATING_THRESHOLDS = (
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
)  # the grid was fixed before looking at test; the shipped default (0.80) was adopted afterwards


def operating_point(pd_, yd, pt, yt, thr, fields):
    """What accepting every field with calibrated probability >= thr does: dev numbers (where a threshold is chosen)
    and test numbers (what happened when it was applied unchanged)."""
    sd, st = pd_ >= thr, pt >= thr
    return {
        "threshold": thr,
        "dev_auto_accepted": float(sd.mean()),
        "dev_precision_of_accepted": float(yd[sd].mean()) if sd.any() else None,
        "test_auto_accepted": float(st.mean()),
        "test_precision_of_accepted": float(yt[st].mean()) if st.any() else None,
        "test_wrong_auto_accepted_per_1000_fields": float(((yt == 0) & st).sum() / len(yt) * 1000),
        "test_errors_caught_by_review": float(((~st) & (yt == 0)).sum() / max((yt == 0).sum(), 1)),
        "test_per_field": {
            f: {
                "accepted": int((st & (fields == f)).sum()),
                "wrong_accepted": int((st & (fields == f) & (yt == 0)).sum()),
            }
            for f in ETYPE
        },
    }


def calibrate(dev, test, feature="min_prob", target=0.99):
    def samples(recs):
        x = np.array(
            [(recs[f]["confidence"].get(k) or {}).get(feature, 0.0) for f in recs for k in ETYPE]
        )  # no value -> probability 0
        y = np.array([float(correct(k, recs[f]["truth"][k], recs[f]["fields"].get(k))) for f in recs for k in ETYPE])
        return x, y

    xd, yd = samples(dev)
    xt, yt = samples(test)
    test_fields = np.array([k for f in test for k in ETYPE])  # the field name of each row of xt / yt
    m = IsotonicMap.fit(xd, yd)
    pd_, pt = np.array([m(v) for v in xd]), np.array([m(v) for v in xt])
    thr = choose_accept(pd_, yd, target)
    acc = pt >= thr
    out = {
        "feature": feature,
        "dev_fields": len(xd),
        "test_fields": len(xt),
        "test_accuracy": float(yt.mean()),
        "ece_raw_test": ece(xt, yt),
        "ece_calibrated_test": ece(pt, yt),
        "target_precision": target,
        "accept_threshold_chosen_on_dev": thr,
        "test_auto_accepted": float(acc.mean()),
        "test_precision_of_accepted": float(yt[acc].mean()) if acc.any() else None,
        "test_errors_caught_by_review": float(((~acc) & (yt == 0)).sum() / max((yt == 0).sum(), 1)),
        "map": m.to_dict(),
        "operating_points": [operating_point(pd_, yd, pt, yt, t, test_fields) for t in OPERATING_THRESHOLDS],
    }
    # risk-coverage points on test
    order = np.argsort(-pt)
    cum = np.cumsum(yt[order]) / np.arange(1, len(yt) + 1)
    out["coverage_at_precision"] = {
        str(t): float((cum >= t).nonzero()[0].max() + 1) / len(yt) if (cum >= t).any() else 0.0
        for t in (0.99, 0.98, 0.95)
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = json.loads((ROOT / "data" / "real" / "split_info.json").read_text(encoding="utf-8"))
    unseen = set(info["test_unseen_shop_files"])
    res = {
        "_run": {"test": "SROIE test split", "unseen_shop_receipts": len(unseen), "price_per_second": PRICE_PER_S},
        "systems": {},
    }
    loaded = {k: (load(t), load(d) if d else None) for k, (t, d) in SYSTEMS.items()}
    ref = loaded["zero_shot_7b"][0]
    for k, (test, dev) in loaded.items():
        if not test:
            continue
        files = sorted(test)
        seen = [f for f in files if f not in unseen]
        s = {
            "all": summary(test, files),
            "seen_shops": summary(test, seen),
            "unseen_shops": summary(test, sorted(unseen & set(files))),
        }
        s["usd_per_1000_pages_at_a10_price"] = s["all"]["seconds_per_page"] * PRICE_PER_S * 1000
        if k != "zero_shot_7b" and ref:
            s["paired_vs_zero_shot_7b"] = {
                "all": paired(ref, test, files),
                "unseen_shops": paired(ref, test, sorted(unseen)),
            }
        if dev:
            s["calibration"] = calibrate(dev, test)
        res["systems"][k] = s
    # same-size comparisons: what fine-tuning itself buys, apart from model size
    for ft, zs in (("fine_tuned_3b", "zero_shot_3b"), ("fine_tuned_7b", "zero_shot_7b")):
        if ft in res["systems"] and loaded[zs][0] and loaded[ft][0]:
            files = sorted(loaded[ft][0])
            res["systems"][ft]["paired_vs_same_size_zero_shot"] = {
                "all": paired(loaded[zs][0], loaded[ft][0], files),
                "unseen_shops": paired(loaded[zs][0], loaded[ft][0], sorted(unseen)),
                "all_content_metric": paired(loaded[zs][0], loaded[ft][0], files, metric=fields_content),
            }
    if "fine_tuned_7b" in res["systems"] and loaded["fine_tuned_3b"][0] and loaded["fine_tuned_7b"][0]:
        res["systems"]["fine_tuned_7b"]["paired_vs_fine_tuned_3b"] = {
            "all": paired(loaded["fine_tuned_3b"][0], loaded["fine_tuned_7b"][0], sorted(loaded["fine_tuned_7b"][0]))
        }
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, s in res["systems"].items():
        print(
            k,
            {g: round(s[g]["field_accuracy"], 3) for g in ("all", "seen_shops", "unseen_shops")},
            "cal" if "calibration" in s else "",
        )


if __name__ == "__main__":
    main()
