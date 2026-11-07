"""
Extract a receipt with the fine-tuned model deployed on Modal and print fields, probabilities and review flags.

    modal deploy experiments/serving/modal_run.py                       # once
    IDP_VLM_APP=idp-vlm python examples/demo_receipt.py path/to/receipt.jpg

The first call after idle waits for a cold GPU container (about a minute); later calls take a few seconds.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.api.backends import modal_receipts  # noqa: E402
from src.api.images import prepare_image  # noqa: E402


def main(path: str):
    ex = modal_receipts()
    if ex is None:
        raise SystemExit("set IDP_VLM_APP to the deployed Modal app name (idp-vlm)")
    res = ex.extract(prepare_image(Path(path).read_bytes()))
    for f, v in res["fields"].items():
        print(f"{f:8s} {v['status']:6s} p={v['probability']:.3f}  {v['value']}")
    print(f"needs review: {res['review_fields'] or 'nothing'}   ({res['seconds']:.1f} s)")
    if "--json" in sys.argv:
        print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
