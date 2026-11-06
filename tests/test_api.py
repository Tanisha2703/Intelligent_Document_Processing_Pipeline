"""The HTTP service, with a stub model behind ReceiptExtractor (no GPU, no network)."""

import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.api.app import create_app  # noqa: E402
from src.pipeline.calibration import IsotonicMap  # noqa: E402
from src.pipeline.receipts import ReceiptExtractor  # noqa: E402

ID = IsotonicMap([0, 1], [0, 1])
FIELDS = {"company": "ABC SDN BHD", "date": "15/01/2019", "address": "1 JALAN X", "total": "12.50"}
CONFIDENT = {"company": 0.99, "date": 0.99, "address": 0.98, "total": 0.99}


def png_bytes():
    b = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(b, "PNG")
    return b.getvalue()


def client(tmp_path, conf=CONFIDENT, fields=FIELDS, ref=None, boom=False):
    def vlm(png, doc_type):
        if boom:
            raise RuntimeError("GPU unavailable")
        return {"fields": fields, "confidence": {k: {"min_prob": v} for k, v in conf.items()}}

    app = create_app(
        ReceiptExtractor(vlm, ID, accept=0.95),
        tmp_path / "review.jsonl",
        reference_scores=ref if ref is not None else [0.1, 0.2, 0.3, 0.4],
    )
    return TestClient(app)


def post(c, name="r.png", data=None):
    return c.post("/receipt", files={"file": (name, data or png_bytes(), "application/octet-stream")})


def test_health_reports_whether_a_model_is_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCINTEL_VLM_APP", raising=False)
    assert client(tmp_path).get("/health").json() == {
        "status": "ok",
        "model_configured": True,
        "doctype_configured": False,
    }
    plain = TestClient(create_app(None, tmp_path / "r2.jsonl"))
    assert plain.get("/health").json()["model_configured"] is False


def test_confident_receipt_needs_no_review_and_amounts_are_normalised(tmp_path):
    r = post(client(tmp_path), "r.png").json()
    assert r["needs_review"] is False and r["doc_id"] is None
    assert r["fields"]["total"]["normalized"] == "12.50"
    assert set(r["fields"]) == {"company", "date", "address", "total"}


def test_uncertain_field_is_queued_and_can_be_corrected(tmp_path):
    c = client(tmp_path, {**CONFIDENT, "address": 0.5})
    r = post(c, "r.jpg").json()
    assert r["review_fields"] == ["address"] and r["doc_id"]
    q = c.get("/review/queue").json()
    assert q["pending"][0]["fields"].keys() == {"address"} and q["pending_docs"] == 1
    assert c.post(f"/review/{r['doc_id']}/address", json={"value": "1 JALAN Y", "reviewer": "asha"}).status_code == 200
    assert c.get("/review/queue").json()["pending"] == []
    assert c.get("/monitor").json()["review"]["resolved_fields"] == 1


def test_correcting_an_unknown_field_is_404(tmp_path):
    c = client(tmp_path)
    assert c.post("/review/nope/address", json={"value": "x"}).status_code == 404


def test_unsupported_oversized_and_corrupt_uploads_are_rejected(tmp_path):
    c = client(tmp_path)
    assert post(c, "r.pdf", b"x").status_code == 415
    assert post(c, "r.png", b"not an image").status_code == 422
    big = TestClient(
        create_app(
            ReceiptExtractor(lambda p, d: {"fields": FIELDS, "confidence": {}}, ID, 0.95),
            tmp_path / "r3.jsonl",
            max_file_mb=0.0001,
        )
    )
    assert post(big, "r.png", b"x" * 1000).status_code == 413


def test_suffix_check_ignores_case(tmp_path):
    assert post(client(tmp_path), "SCAN.PNG").status_code == 200


def test_a_model_failure_becomes_422_not_500(tmp_path):
    assert post(client(tmp_path, boom=True)).status_code == 422


def test_no_model_configured_gives_503(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCINTEL_VLM_APP", raising=False)
    plain = TestClient(create_app(None, tmp_path / "r.jsonl"))
    assert post(plain).status_code == 503


def test_monitor_flags_a_shift_after_enough_receipts(tmp_path):
    ref = [0.05 + 0.01 * i for i in range(60)]  # a reference of confident receipts
    c = client(tmp_path, dict.fromkeys(CONFIDENT, 0.2), ref=ref)  # live traffic the model is unsure about
    assert c.get("/monitor").json()["drift"]["level"] == "insufficient_data"
    for _ in range(60):
        post(c)
    d = c.get("/monitor").json()["drift"]
    assert d["level"] == "large" and d["mean_score"] > d["reference_mean_score"]
