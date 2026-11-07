"""
HTTP service: document routing, receipt extraction with calibrated probabilities, a review queue and a drift monitor.

  POST /document                    upload any document image: classify its type, route it, extract if a model supports it
  POST /receipt                     upload a receipt image; fields with calibrated probabilities and review flags
  GET  /review/queue                documents with fields waiting for a human
  POST /review/{doc}/{field}        a reviewer confirms or corrects one field (the corrected value is logged)
  GET  /monitor                     drift of the confidence distribution against the dev-set reference, review statistics
  GET  /health                      (open; everything else needs the API key or a local client, see src/utils/guard.py)

    IDP_VLM_APP=idp-vlm IDP_DOCTYPE=1 uvicorn "src.api.app:create_app" --factory --port 8000

The models run elsewhere (src/api/backends.py): extraction on Modal, document types on CPU. A backend that is not
configured makes the endpoints that need it answer 503; the review and monitor endpoints always work.
"""

from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.api import backends
from src.api.images import prepare_image, read_upload
from src.monitor.drift import DriftMonitor
from src.monitor.review import ReviewStore
from src.pipeline.forms import FormExtractor
from src.pipeline.receipts import ReceiptExtractor
from src.utils.guard import DailyCap, cap_from_env, check_access

ROOT = Path(__file__).parents[2]


class Correction(BaseModel):
    """A reviewer's answer for one field: the right value (None if the field is empty)."""

    value: str | None
    reviewer: str = "anonymous"


def require_access(request: Request) -> None:
    """Everything except /health needs the API key, or a client on the same machine when no key is configured."""
    reason = check_access(
        request.headers.get("x-api-key"),
        request.client.host if request.client else None,
        request.headers.get("x-forwarded-for"),
    )
    if reason:
        raise HTTPException(401, reason)


def lazy(provided, factory: Callable):
    """`provided` if given, else the result of `factory()`, built on first use and then kept."""
    cell: list = []

    def get():
        if provided is not None:
            return provided
        if not cell:
            cell.append(factory())
        return cell[0]

    return get


def create_app(
    receipts: ReceiptExtractor | None = None,
    review_path: str | Path | None = None,
    reference_scores=None,
    max_file_mb: float = 50.0,
    doctype=None,
    forms: FormExtractor | None = None,
    max_gpu_calls_per_day: int | None = None,
) -> FastAPI:
    """Build the service. Every model argument can be injected (the tests do); otherwise it comes from `backends`."""
    app = FastAPI(title="Intelligent Document Processing Pipeline", version="3.0")
    store = ReviewStore(review_path or ROOT / "data" / "review" / "review.jsonl")
    monitor = DriftMonitor(reference_scores if reference_scores is not None else backends.reference_scores())
    cap = DailyCap(max_gpu_calls_per_day) if max_gpu_calls_per_day is not None else cap_from_env()
    receipt_extractor = lazy(receipts, backends.modal_receipts)
    form_extractor = lazy(forms, backends.modal_forms)
    classifier = lazy(doctype, backends.doctype_service)
    extractors = {"receipt": receipt_extractor, "form": form_extractor}  # route name -> extractor

    def extract(extractor, png: bytes) -> tuple[dict, str | None]:
        """Run an extractor (one GPU call), record its confidence for the monitor, and queue it for review if needed."""
        if not cap.take():
            raise HTTPException(429, f"daily limit of {cap.limit} GPU calls reached; it resets at 00:00 UTC")
        try:
            result = extractor.extract(png)
        except Exception as exc:
            raise HTTPException(422, f"could not process the document: {type(exc).__name__}: {exc}") from exc
        monitor.observe(result["expected_errors"])
        fields = {name: {**value, "source": "vlm"} for name, value in result["fields"].items()}
        doc_id = store.enqueue(None, result["doc_type"], fields) if result["needs_review"] else None
        return {**result, "fields": fields}, doc_id

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_configured": receipt_extractor() is not None,
            "doctype_configured": classifier() is not None,
        }

    @app.post("/document", dependencies=[Depends(require_access)])
    def document(file: UploadFile = File(...)):
        """Classify an uploaded document image, route it, and extract its fields if an extractor supports its type.

        Types with no validated extractor, and types the classifier is unsure of, go to a person (queued for review).
        """
        clf = classifier()
        if clf is None:
            raise HTTPException(
                503, "no document-type model configured; set IDP_DOCTYPE=1 (needs torch and transformers)"
            )
        data = read_upload(file, max_file_mb)
        try:
            png = prepare_image(data)
            kind = clf.classify(png)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"could not classify the document: {type(exc).__name__}: {exc}") from exc
        out = {
            "doc_type": kind["doc_type"],
            "type_confidence": kind["confidence"],
            "route": kind["route"],
            "reason": kind["reason"],
            "result": None,
            "needs_review": True,
            "doc_id": None,
        }

        get_extractor = extractors.get(kind["route"])
        extractor = get_extractor() if get_extractor else None
        if get_extractor and extractor is None:
            out["reason"] += "; no extraction model is configured, so a person has to read it"
        if extractor is not None:
            out["result"], out["doc_id"] = extract(extractor, png)
            out["needs_review"] = out["result"]["needs_review"]
        else:  # a person decides: queue the predicted type so a reviewer can correct it
            out["doc_id"] = store.enqueue(
                None,
                "unrouted",
                {
                    "document_type": {
                        "value": kind["doc_type"],
                        "probability": kind["confidence"],
                        "source": "doctype",
                        "status": "review",
                    }
                },
            )
        return out

    @app.post("/receipt", dependencies=[Depends(require_access)])
    def receipt(file: UploadFile = File(...)):
        """Extract a receipt image: each field with a calibrated probability and an auto / review decision."""
        extractor = receipt_extractor()
        if extractor is None:
            raise HTTPException(503, "no model configured; set IDP_VLM_APP to the deployed Modal app")
        try:
            png = prepare_image(read_upload(file, max_file_mb))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"could not read the image: {type(exc).__name__}: {exc}") from exc
        result, doc_id = extract(extractor, png)
        return {**result, "doc_id": doc_id}

    @app.get("/review/queue", dependencies=[Depends(require_access)])
    def queue():
        return {"pending": store.pending(), **store.stats()}

    @app.post("/review/{doc_id}/{field}", dependencies=[Depends(require_access)])
    def resolve(doc_id: str, field: str, body: Correction):
        try:
            store.resolve(doc_id, field, body.value, body.reviewer)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return store.stats()

    @app.get("/monitor", dependencies=[Depends(require_access)])
    def monitor_status():
        return {"drift": monitor.status(), "review": store.stats()}

    return app
