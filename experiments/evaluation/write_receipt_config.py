"""
Write the calibration files the receipt service loads (configs/receipt_calibration*.json) from results/raw/real_compare.json.

The map was fitted on the company-disjoint SROIE DEV receipts only (analyze_real.py). `accept` is the calibrated
probability a field needs to be auto-accepted: a product choice (default 0.80), not a fitted value, so the config also
records what it does (`test_*`, applied unchanged to the SROIE test split) and the strict alternative that reached 99%
precision on dev (`strict`). Whoever deploys it can override it with IDP_ACCEPT_THRESHOLD.

    python experiments/evaluation/write_receipt_config.py                 # accept = 0.80
    python experiments/evaluation/write_receipt_config.py --accept 0.90
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "evaluation"))
from analyze_real import SYSTEMS, load  # noqa: E402

from src.eval.receipts import ETYPE  # noqa: E402
from src.pipeline.calibration import IsotonicMap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept", type=float, default=0.80, help="calibrated probability needed to auto-accept a field")
    a = ap.parse_args()
    d = json.loads((ROOT / "results" / "raw" / "real_compare.json").read_text(encoding="utf-8"))["systems"]
    for key, adapter, default in (("fine_tuned_7b", "ft7b", True), ("fine_tuned_3b", "ft3b", False)):
        c = d[key]["calibration"]
        op = next((o for o in c["operating_points"] if abs(o["threshold"] - a.accept) < 1e-9), None)
        if op is None:
            raise SystemExit(
                f"--accept must be one of the measured operating points: {[o['threshold'] for o in c['operating_points']]}"
            )
        cfg = {
            "model": {"fine_tuned_7b": "Qwen/Qwen2.5-VL-7B-Instruct", "fine_tuned_3b": "Qwen/Qwen2.5-VL-3B-Instruct"}[
                key
            ],
            "adapter": adapter,
            "feature": c["feature"],
            "map": c["map"],
            "accept": a.accept,
            "fitted_on": f"SROIE dev, {c['dev_fields']} fields from company-disjoint receipts",
            "dev_precision_of_accepted": op["dev_precision_of_accepted"],
            "test_auto_accepted": op["test_auto_accepted"],
            "test_precision_of_accepted": op["test_precision_of_accepted"],
            "test_errors_caught_by_review": op["test_errors_caught_by_review"],
            "test_ece_calibrated": c["ece_calibrated_test"],
            "strict": {
                "accept": c["accept_threshold_chosen_on_dev"],
                "target_precision_on_dev": c["target_precision"],
                "test_auto_accepted": c["test_auto_accepted"],
                "test_precision_of_accepted": c["test_precision_of_accepted"],
            },
        }
        # reference distribution for the drift monitor: each dev receipt's expected number of wrong fields
        m = IsotonicMap.from_dict(c["map"])
        dev = load(SYSTEMS[key][1])
        cfg["reference_scores"] = [
            round(sum(1.0 - m((r["confidence"].get(f) or {}).get(c["feature"], 0.0)) for f in ETYPE), 4)
            for r in dev.values()
        ]
        out = ROOT / "configs" / f"receipt_calibration_{adapter}.json"
        out.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        if default:
            (ROOT / "configs" / "receipt_calibration.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        print(
            out.name,
            "accept",
            round(cfg["accept"], 4),
            "test auto",
            round(cfg["test_auto_accepted"], 3),
            "precision",
            round(cfg["test_precision_of_accepted"], 3),
        )


if __name__ == "__main__":
    main()
