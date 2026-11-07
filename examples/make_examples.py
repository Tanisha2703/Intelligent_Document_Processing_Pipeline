"""
Build the recorded examples the Streamlit demo shows without a GPU: two real SROIE test receipts, the fine-tuned 7B's
stored output for each, passed through the same ReceiptExtractor (calibration and accept threshold) as the live service.

    python examples/make_examples.py            # needs data/real/sroie_test and results/raw/vlm_real_ft7b_sroie_test.jsonl

Receipts: SROIE (ICDAR 2019), CC-BY-4.0, https://rrc.cvc.uab.es/?ch=13
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.receipts import DEFAULT_CONFIG, ReceiptExtractor  # noqa: E402

FILES = ["0003.jpg", "0010.jpg"]


def main():
    recs = {
        json.loads(x)["id"].split("|")[-1]: json.loads(x)
        for x in (ROOT / "results" / "raw" / "vlm_real_ft7b_sroie_test.jsonl").read_text(encoding="utf-8").splitlines()
    }
    out = ROOT / "examples" / "recorded"
    out.mkdir(parents=True, exist_ok=True)
    examples = []
    for f in FILES:
        r = recs[f]
        ex = ReceiptExtractor.from_config(
            lambda png, dt, r=r: {"fields": r["fields"], "confidence": r["confidence"]}, DEFAULT_CONFIG
        )
        res = ex.extract(b"")
        shutil.copy(ROOT / "data" / "real" / "sroie_test" / f, out / f"sroie_{f}")
        examples.append(
            {
                "image": f"sroie_{f}",
                "recorded_output": {k: v for k, v in res.items() if k != "seconds"},
                "ground_truth": r["truth"],
                "model": "Qwen2.5-VL-7B + LoRA (ft7b), SROIE test split, not used in training",
            }
        )
    (out / "examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", len(examples), "examples to", out)


if __name__ == "__main__":
    main()
