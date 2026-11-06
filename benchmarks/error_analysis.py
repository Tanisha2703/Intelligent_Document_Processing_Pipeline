"""
Where do a fine-tuned model's remaining SROIE errors come from? Uses the stored model outputs only (no GPU).

  1. what kind of error each wrong field is (a near miss, extra text, a different value, ...)
  2. for near-miss addresses: do the edits touch a digit, or only letters and punctuation?
  3. accuracy by the receipt's original size: are the heavily downscaled receipts the ones that go wrong?

    python benchmarks/error_analysis.py --out results/raw/error_analysis.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.receipts import ETYPE, NORM, content_match, correct  # noqa: E402

MODEL_LIMIT_MP = 1.4  # the pixel budget the model sees (src/api/images.py MAX_PIXELS)


def load(tag):
    p = ROOT / "results" / "raw" / f"vlm_{tag}.jsonl"
    return {json.loads(x)["id"].split("|")[-1]: json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()}


def error_kind(field, truth, pred):
    if not pred:
        return "empty"
    a, b = NORM[ETYPE[field]](truth), NORM[ETYPE[field]](pred)
    if correct(field, truth, pred, near=True):
        return "near_miss_within_2_chars"
    if a and b and a in b:
        return "extra_text_in_prediction"
    if a and b and b in a:
        return "prediction_truncated"
    ratio = fuzz.ratio(a, b)
    return "similar_85_plus" if ratio >= 85 else "partly_similar_60_85" if ratio >= 60 else "different"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="real_ft7b_sroie_test")
    a = ap.parse_args()
    recs = load(a.tag)
    n = len(recs)
    out = {"system_tag": a.tag, "receipts": n, "fields": {}, "near_miss_addresses": {}, "by_original_size": {}}

    for f in ETYPE:
        kinds = Counter()
        for r in recs.values():
            t, p = r["truth"][f], r["fields"].get(f)
            if not correct(f, t, p):
                kinds[error_kind(f, t, p)] += 1
        out["fields"][f] = {
            "right_strict": n - sum(kinds.values()),
            "wrong": sum(kinds.values()),
            "wrong_but_content_matches": sum(
                1
                for r in recs.values()
                if not correct(f, r["truth"][f], r["fields"].get(f))
                and content_match(f, r["truth"][f], r["fields"].get(f))
            ),
            "kinds": dict(kinds.most_common()),
        }

    touches_digit, letters_only, subs = 0, 0, Counter()
    for r in recs.values():
        t, p = r["truth"]["address"], r["fields"].get("address")
        if not p or correct("address", t, p) or not correct("address", t, p, near=True):
            continue
        x, y = NORM["text"](t), NORM["text"](p)
        digit = False
        for tag, i, j in Levenshtein.editops(x, y).as_list():
            cx, cy = x[i] if i < len(x) else "", y[j] if j < len(y) else ""
            digit |= any(c.isdigit() for c in cx + cy)
            if tag == "replace":
                subs[f"{cx}->{cy}"] += 1
        touches_digit += digit
        letters_only += not digit
    out["near_miss_addresses"] = {
        "count": touches_digit + letters_only,
        "edit_touches_a_digit": touches_digit,
        "letters_or_punctuation_only": letters_only,
        "most_common_substitutions": subs.most_common(8),
    }

    rows = []
    for f, r in recs.items():
        w, h = Image.open(ROOT / "data" / "real" / "sroie_test" / f).size
        rows.append(
            (
                w * h / 1e6,
                [correct(k, r["truth"][k], r["fields"].get(k)) for k in ETYPE],
                [correct(k, r["truth"][k], r["fields"].get(k), near=True) for k in ETYPE],
            )
        )
    addr = list(ETYPE).index("address")
    for lo, hi, name in (
        (0, MODEL_LIMIT_MP, "at_most_1.4_MP_not_downscaled"),
        (MODEL_LIMIT_MP, 5, "1.4_to_5_MP"),
        (5, 1e9, "over_5_MP_heavily_downscaled"),
    ):
        sel = [x for x in rows if lo < x[0] <= hi]
        out["by_original_size"][name] = {
            "receipts": len(sel),
            "field_accuracy": float(np.mean([np.mean(x[1]) for x in sel])),
            "address_exact": float(np.mean([x[1][addr] for x in sel])),
            "address_within_2_chars": float(np.mean([x[2][addr] for x in sel])),
        }
    out["share_of_test_receipts_downscaled"] = float(np.mean([x[0] > MODEL_LIMIT_MP for x in rows]))

    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {k: out[k] for k in ("near_miss_addresses", "by_original_size", "share_of_test_receipts_downscaled")},
            indent=1,
        )
    )
    for f, v in out["fields"].items():
        print(f, "wrong", v["wrong"], "of which content matches", v["wrong_but_content_matches"], v["kinds"])


if __name__ == "__main__":
    main()
