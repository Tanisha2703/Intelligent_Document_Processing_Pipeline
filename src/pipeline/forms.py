"""
Form extraction (the XFUND adapter): a list of (label, value) pairs.

This model has no per-pair confidence (its answer is a list, not fixed fields), so nothing is calibrated and every pair
is flagged for review. On the 105 XFUND test forms its pair F1 was 0.44: it is a starting point for a person, not a
result to trust.
"""

import time
from collections.abc import Callable


class FormExtractor:
    """Label/value pairs from a form. There is no per-pair confidence, so every pair is flagged for review."""

    def __init__(self, vlm: Callable, doc_type: str = "xfund_form"):
        self.vlm, self.doc_type = vlm, doc_type

    def extract(self, png: bytes) -> dict:
        """Run the form model on a PNG; same result shape as ReceiptExtractor.extract, plus the raw `pairs`."""
        t0 = time.perf_counter()
        pairs = self.vlm(png, self.doc_type)["fields"].get("pairs", [])
        fields = {
            f"{i}: {p['key']}": {"value": p["value"], "probability": 0.0, "status": "review", "normalized": None}
            for i, p in enumerate(pairs)
        }
        return {
            "doc_type": self.doc_type,
            "pairs": pairs,
            "fields": fields,
            "needs_review": True,
            "review_fields": list(fields),
            "expected_errors": float(len(pairs)),
            "seconds": time.perf_counter() - t0,
        }
