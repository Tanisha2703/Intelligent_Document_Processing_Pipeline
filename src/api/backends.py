"""
Where the models actually run. Each function returns None when its backend is not configured, so the service can start
(and answer 503 for the endpoints that need it) on a machine with no GPU account.

  DOCINTEL_VLM_APP   name of the Modal app deployed from benchmarks/modal_run.py (receipt and form extraction)
  DOCINTEL_DOCTYPE   "1" loads the DiT document-type classifier on CPU (needs torch and transformers)
"""

import json
import os
from pathlib import Path

from src.pipeline.forms import FormExtractor
from src.pipeline.receipts import DEFAULT_CONFIG, ReceiptExtractor

FORM_MODEL, FORM_ADAPTER, FORM_MAX_NEW_TOKENS = "Qwen/Qwen2.5-VL-3B-Instruct", "ft3b_xfund", 2048


def _modal_vlm(model_id: str, adapter: str, **item_options):
    """A callable (png, doc_type) -> {fields, confidence} backed by the deployed Modal class."""
    import modal

    instance = modal.Cls.from_name(os.environ["DOCINTEL_VLM_APP"], "VLM")(model_id=model_id, adapter=adapter)

    def call(png: bytes, doc_type: str) -> dict:
        result = instance.extract.remote([{"id": "api", "doc_type": doc_type, "image": png, **item_options}])[0]
        return {"fields": result["fields"], "confidence": result["confidence"]}

    return call


def modal_receipts(cfg_path: Path = DEFAULT_CONFIG) -> ReceiptExtractor | None:
    """Receipt extractor with the fine-tuned adapter named in the calibration config."""
    if not os.environ.get("DOCINTEL_VLM_APP") or not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return ReceiptExtractor.from_config(_modal_vlm(cfg["model"], cfg["adapter"]), cfg_path)


def modal_forms() -> FormExtractor | None:
    """Form extractor with the XFUND adapter."""
    if not os.environ.get("DOCINTEL_VLM_APP"):
        return None
    return FormExtractor(_modal_vlm(FORM_MODEL, FORM_ADAPTER, max_new_tokens=FORM_MAX_NEW_TOKENS))


def doctype_service():
    """The document-type classifier, or None unless DOCINTEL_DOCTYPE=1."""
    if os.environ.get("DOCINTEL_DOCTYPE") != "1":
        return None
    from src.doctype.service import DocTypeService

    return DocTypeService.load()


def reference_scores(cfg_path: Path = DEFAULT_CONFIG) -> list[float]:
    """The dev-set confidence scores the drift monitor compares live traffic against."""
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))["reference_scores"]
    except (OSError, KeyError, ValueError):
        return [0.0, 1.0]
