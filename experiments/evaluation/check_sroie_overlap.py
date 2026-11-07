"""
Are the SROIE test scores inflated by receipts that also appear in training?

A test receipt counts as a duplicate when its (company, date, total) labels equal those of a fine-tuning or dev
receipt (SROIE contains some re-scanned receipts under different ids). Scores are reported on all test receipts and
without the duplicates, for every system.

    python experiments/evaluation/check_sroie_overlap.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "evaluation"))
from analyze_real import SYSTEMS, load  # noqa: E402

from src.eval.receipts import ETYPE, correct  # noqa: E402

DATA = ROOT / "data" / "real"


def labels(name):
    return json.loads((DATA / name / "labels.json").read_text(encoding="utf-8"))


def key(e):
    return (" ".join(e["company"].upper().split()), e["date"], e["total"])


def main():
    seen = {key(x["entities"]) for x in labels("sroie_ft") + labels("sroie_dev")}
    dup = {x["file"] for x in labels("sroie_test") if key(x["entities"]) in seen}
    out = {"duplicate_test_receipts": len(dup), "test_receipts": len(labels("sroie_test")), "systems": {}}
    for name, (tag, _) in SYSTEMS.items():
        recs = load(tag)
        if not recs:
            continue

        def acc(files, recs=recs):
            return float(
                np.mean(
                    [np.mean([correct(f, recs[x]["truth"][f], recs[x]["fields"].get(f)) for f in ETYPE]) for x in files]
                )
            )

        out["systems"][name] = {
            "all": acc(sorted(recs)),
            "without_duplicates": acc(sorted(set(recs) - dup)),
            "duplicates_only": acc(sorted(dup & set(recs))),
        }
    (ROOT / "results" / "raw" / "sroie_overlap_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
