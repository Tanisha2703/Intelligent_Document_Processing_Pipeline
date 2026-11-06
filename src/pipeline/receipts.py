"""
Receipt extraction with a fine-tuned vision-language model, calibrated confidence and a review flag.

  image -> VLM (fine-tuned, JSON out) -> per-field calibrated P(correct) -> auto-accept above a threshold, else review

The calibration (an isotonic map from the model's value-token probability to P(correct)) and the accept threshold are
fitted on a labelled DEV set that shares no shop with the fine-tuning data (benchmarks/analyze_real.py writes
configs/receipt_calibration.json). `vlm` is any callable (png_bytes, doc_type) -> {"fields": {...}, "confidence": {...}};
benchmarks/modal_run.py's VLM class (with `adapter` set) is the real one, a stub is enough for tests.
"""

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from src.extract.vlm import SCHEMAS
from src.pipeline.calibration import IsotonicMap
from src.utils.normalize import normalize_value

DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "receipt_calibration.json"


class ReceiptExtractor:
    """Fields from a receipt with a calibrated probability each.

    `vlm(png, doc_type)` returns {fields, confidence}. A field is auto-accepted when its calibrated probability is
    at least `accept`; everything else, including a missing field, needs a person.
    """

    def __init__(
        self,
        vlm: Callable,
        calibration: IsotonicMap,
        accept: float,
        feature: str = "min_prob",
        doc_type: str = "scanned_receipt",
    ):
        self.vlm, self.cal, self.accept, self.feature, self.doc_type = vlm, calibration, accept, feature, doc_type
        self.schema = SCHEMAS[doc_type]

    @classmethod
    def from_config(cls, vlm: Callable, path: str | Path = DEFAULT_CONFIG) -> "ReceiptExtractor":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        accept = float(os.environ.get("DOCINTEL_ACCEPT_THRESHOLD") or d["accept"])
        return cls(vlm, IsotonicMap.from_dict(d["map"]), accept, d.get("feature", "min_prob"))

    def extract(self, png: bytes) -> dict:
        """Extract one receipt: per-field value, normalised value, probability and status, plus the review list and the
        expected number of wrong fields (the drift monitor's signal).
        """
        t0 = time.perf_counter()
        out = self.vlm(png, self.doc_type)
        fields = {}
        for f, etype in self.schema.items():
            v = out["fields"].get(f)
            c = (out.get("confidence") or {}).get(f)
            # a field the model did not produce, or produced without a measurable probability, can never be auto-accepted
            p = float(self.cal(c[self.feature])) if (v and c and self.feature in c) else 0.0
            fields[f] = {
                "value": v,
                "normalized": normalize_value(etype, v) if v else None,
                "probability": p,
                "status": "auto" if v and p >= self.accept else "review",
            }
        review = [f for f, x in fields.items() if x["status"] == "review"]
        # the expected number of wrong fields on this receipt: the signal the drift monitor watches
        expected = sum(1.0 - x["probability"] for x in fields.values())
        return {
            "doc_type": self.doc_type,
            "fields": fields,
            "needs_review": bool(review),
            "review_fields": review,
            "expected_errors": expected,
            "seconds": time.perf_counter() - t0,
        }
