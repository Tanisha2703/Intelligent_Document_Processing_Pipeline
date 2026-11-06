"""Document-type stage: the probe, the routing rules and the /document endpoint (stubs: no model, no network)."""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.api.app import create_app  # noqa: E402
from src.doctype.classifier import CLASSES, Probe, softmax  # noqa: E402
from src.doctype.router import ROUTES, route  # noqa: E402
from src.doctype.service import DocTypeService  # noqa: E402
from src.pipeline.calibration import IsotonicMap  # noqa: E402
from src.pipeline.forms import FormExtractor  # noqa: E402
from src.pipeline.receipts import ReceiptExtractor  # noqa: E402

ID = IsotonicMap([0, 1], [0, 1])
D = 4


def probe_for(cls: str, strength: float = 8.0, accept: float = 0.8) -> Probe:
    """A probe whose embedding axis 0 votes for `cls` (embedding [1,0,0,0] -> that class)."""
    W = np.zeros((len(CLASSES), D))
    W[CLASSES.index(cls), 0] = strength
    return Probe(W, np.zeros(len(CLASSES)), ID, accept)


def test_there_are_seventeen_classes_and_receipt_is_the_extra_one():
    assert len(CLASSES) == 17 and CLASSES[-1] == "receipt" and "invoice" in CLASSES


def test_softmax_sums_to_one_and_is_stable_for_large_logits():
    p = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert p.sum() == pytest.approx(1.0) and np.isfinite(p).all()


def test_probe_predicts_the_voted_class_with_a_confidence():
    r = probe_for("receipt").predict(np.array([[1.0, 0, 0, 0]]))
    assert r["doc_type"] == "receipt" and r["confidence"] > 0.9
    assert sum(r["scores"].values()) == pytest.approx(1.0)


def test_probe_round_trips_through_its_config(tmp_path):
    p = probe_for("form")
    np.savez(tmp_path / "w.npz", W=p.W, b=p.b)
    (tmp_path / "doctype.json").write_text(
        json.dumps({"probe_file": "w.npz", "map": ID.to_dict(), "accept": 0.8, "classes": CLASSES})
    )
    q = Probe.load(tmp_path / "doctype.json")
    assert q.predict(np.array([[1.0, 0, 0, 0]]))["doc_type"] == "form" and q.accept == 0.8


@pytest.mark.parametrize(
    "doc_type,expected",
    [
        ("receipt", "receipt"),
        ("form", "form"),
        ("questionnaire", "form"),
        ("invoice", "human"),
        ("scientific report", "human"),
    ],
)
def test_only_measured_types_are_routed_to_an_extractor(doc_type, expected):
    out = route({"doc_type": doc_type, "confidence": 0.99}, accept=0.8)
    assert out["route"] == expected
    if expected == "human":
        assert "no extractor supports" in out["reason"]


def test_an_unsure_type_goes_to_a_person_even_if_the_type_is_routable():
    out = route({"doc_type": "receipt", "confidence": 0.5}, accept=0.8)
    assert out["route"] == "human" and "unsure" in out["reason"]


def test_routes_only_name_extractors_that_exist():
    assert set(ROUTES.values()) == {"receipt", "form"}


# ---- the endpoint
def png():
    b = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(b, "PNG")
    return b.getvalue()


class StubEmbedder:
    def __init__(self, vec):
        self.vec = np.array([vec], dtype=np.float32)

    def __call__(self, images):
        return self.vec, np.zeros((1, 16), np.float32)


FIELDS = {"company": "ABC", "date": "15/01/2019", "address": "1 JALAN X", "total": "12.50"}
CONF = {"company": 0.99, "date": 0.99, "address": 0.98, "total": 0.99}


def client(tmp_path, doc_class, vec=(1.0, 0, 0, 0), accept=0.8, with_models=True):
    svc = DocTypeService(StubEmbedder(vec), probe_for(doc_class, accept=accept))
    receipts = (
        ReceiptExtractor(
            lambda p, d: {"fields": FIELDS, "confidence": {k: {"min_prob": v} for k, v in CONF.items()}}, ID, 0.95
        )
        if with_models
        else None
    )
    forms = (
        FormExtractor(lambda p, d: {"fields": {"pairs": [{"key": "Name", "value": "Ana"}]}, "confidence": {}})
        if with_models
        else None
    )
    return TestClient(
        create_app(receipts, tmp_path / "review.jsonl", reference_scores=[0.1, 0.2], doctype=svc, forms=forms)
    )


def post(c, name="d.png"):
    return c.post("/document", files={"file": (name, png(), "application/octet-stream")})


def test_a_receipt_is_classified_routed_and_extracted(tmp_path):
    r = post(client(tmp_path, "receipt")).json()
    assert r["doc_type"] == "receipt" and r["route"] == "receipt"
    assert r["result"]["fields"]["total"]["normalized"] == "12.50" and r["needs_review"] is False


def test_a_form_is_routed_to_the_form_extractor_and_every_pair_is_flagged(tmp_path):
    c = client(tmp_path, "form")
    r = post(c).json()
    assert r["route"] == "form" and r["needs_review"] is True and r["doc_id"]
    assert r["result"]["pairs"] == [{"key": "Name", "value": "Ana"}]
    assert all(v["status"] == "review" for v in r["result"]["fields"].values())
    assert c.get("/review/queue").json()["pending_docs"] == 1


def test_an_unsupported_type_goes_to_a_person_with_the_predicted_type_queued(tmp_path):
    c = client(tmp_path, "invoice")
    r = post(c).json()
    assert r["route"] == "human" and r["result"] is None and r["needs_review"] is True
    q = c.get("/review/queue").json()["pending"][0]
    assert q["doc_type"] == "unrouted" and q["fields"]["document_type"]["value"] == "invoice"


def test_an_unsure_classification_goes_to_a_person(tmp_path):
    r = post(client(tmp_path, "receipt", vec=(0.05, 0, 0, 0))).json()  # weak vote: low probability
    assert r["route"] == "human"


def test_missing_extraction_model_is_explained_not_hidden(tmp_path):
    r = post(client(tmp_path, "receipt", with_models=False)).json()
    assert r["route"] == "receipt" and r["result"] is None and "no extraction model is configured" in r["reason"]


def test_document_endpoint_validates_uploads_and_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCINTEL_DOCTYPE", raising=False)
    c = client(tmp_path, "receipt")
    assert c.post("/document", files={"file": ("d.pdf", b"x", "application/pdf")}).status_code == 415
    assert c.post("/document", files={"file": ("d.png", b"not an image", "image/png")}).status_code == 422
    assert post(TestClient(create_app(None, tmp_path / "r2.jsonl"))).status_code == 503
    assert c.get("/health").json()["doctype_configured"] is True


def test_document_endpoint_refuses_an_image_with_too_many_pixels(tmp_path):
    big = io.BytesIO()
    Image.new("1", (7000, 7000)).save(big, "PNG")
    r = client(tmp_path, "receipt").post("/document", files={"file": ("huge.png", big.getvalue(), "image/png")})
    assert r.status_code == 413
