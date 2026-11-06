"""
Is the fine-tuned model's CORD score inflated by memorisation or by labelling conventions?

Two checks on the 100 CORD test receipts (results/raw/vlm_real_{zs7b,ft7b}_cord_test.jsonl):
  1. Near-duplicates: a test receipt whose sorted item-name list is >= 90% similar (rapidfuzz token_sort_ratio) to a
     training receipt's. Scores are split by near-duplicate / not.
  2. Absent fields: many receipts have no subtotal or tax. Predicting nothing for them counts as correct, so
     accuracy is also reported on the receipts where the field is actually present.

    python benchmarks/check_cord_overlap.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.cord import SCALARS, menu_prf, scalar_correct, score_menu  # noqa: E402

DATA = ROOT / "data" / "real"
RAW = ROOT / "results" / "raw"


def sig(x):
    return " | ".join(sorted(m["name"].lower() for m in x["entities"]["menu"]))


def load(tag):
    return {
        json.loads(x)["id"].split("|")[-1]: json.loads(x)
        for x in (RAW / f"vlm_{tag}.jsonl").read_text(encoding="utf-8").splitlines()
    }


def main(threshold: float = 90.0):
    train = json.loads((DATA / "cord_train" / "labels.json").read_text(encoding="utf-8"))
    test = json.loads((DATA / "cord_test" / "labels.json").read_text(encoding="utf-8"))
    train_sigs = [sig(x) for x in train]
    dup = {x["file"] for x in test if sig(x) and max(fuzz.token_sort_ratio(sig(x), t) for t in train_sigs) >= threshold}
    out = {
        "threshold": threshold,
        "n_train_receipts": len(train),
        "near_duplicate_test_receipts": len(dup),
        "field_presence_in_test": {f: sum(bool(r["entities"].get(f)) for r in test) for f in SCALARS},
        "systems": {},
    }
    for tag, name in (("real_zs7b_cord_test", "zero_shot_7b"), ("real_ft7b_cord_test", "fine_tuned_7b")):
        recs = load(tag)
        o = {}
        for grp, files in (("near_duplicate", dup), ("not_near_duplicate", set(recs) - dup), ("all", set(recs))):
            rs = [recs[f] for f in files]
            pres = [scalar_correct(r["truth"][k], r["fields"].get(k)) for r in rs for k in SCALARS if r["truth"].get(k)]
            o[grp] = {
                "n": len(rs),
                "scalar_accuracy": float(
                    np.mean(
                        [np.mean([scalar_correct(r["truth"].get(k), r["fields"].get(k)) for k in SCALARS]) for r in rs]
                    )
                ),
                "scalar_accuracy_where_field_present": float(np.mean(pres)),
                "menu_f1": menu_prf([score_menu(r["truth"].get("menu", []), r["fields"].get("menu", [])) for r in rs])[
                    "f1"
                ],
            }
        o["left_empty_correctly_when_absent"] = {
            k: float(np.mean([not recs[f]["fields"].get(k) for f in recs if not recs[f]["truth"].get(k)]))
            for k in SCALARS
        }
        out["systems"][name] = o
    (RAW / "cord_overlap_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
