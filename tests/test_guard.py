"""Access control and the daily GPU-call cap: a public deployment must not let strangers spend GPU money."""

import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.api.app import create_app  # noqa: E402
from src.pipeline.calibration import IsotonicMap  # noqa: E402
from src.pipeline.receipts import ReceiptExtractor  # noqa: E402
from src.utils.guard import DailyCap, check_access  # noqa: E402

FIELDS = {"company": "ABC", "date": "15/01/2019", "address": "1 X", "total": "12.50"}


def png():
    b = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(b, "PNG")
    return b.getvalue()


def app(tmp_path, cap=None, client=("testclient", 5000)):
    calls = []

    def vlm(p, d):
        calls.append(1)
        return {"fields": FIELDS, "confidence": {k: {"min_prob": 0.99} for k in FIELDS}}

    a = create_app(
        ReceiptExtractor(vlm, IsotonicMap([0, 1], [0, 1]), 0.95),
        tmp_path / "r.jsonl",
        reference_scores=[0.1, 0.2],
        max_gpu_calls_per_day=cap,
    )
    return TestClient(a, client=client), calls


def post(c, **kw):
    return c.post("/receipt", files={"file": ("r.png", png(), "image/png")}, **kw)


def test_without_a_key_only_the_local_machine_is_let_in(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCINTEL_API_KEY", raising=False)
    c, calls = app(tmp_path, client=("203.0.113.9", 5000))
    assert post(c).status_code == 401 and not calls
    assert c.get("/monitor").status_code == 401 and c.get("/review/queue").status_code == 401
    assert c.get("/health").status_code == 200  # health stays open


def test_a_proxy_forwarding_header_does_not_unlock_the_local_exception(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCINTEL_API_KEY", raising=False)
    c, calls = app(tmp_path)  # looks local, but a proxy says otherwise
    assert post(c, headers={"x-forwarded-for": "203.0.113.9"}).status_code == 401 and not calls
    assert post(c).status_code == 200


def test_with_a_key_the_right_one_is_required_from_anywhere(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCINTEL_API_KEY", "s3cret")
    c, calls = app(tmp_path, client=("203.0.113.9", 5000))
    assert post(c).status_code == 401 and post(c, headers={"x-api-key": "nope"}).status_code == 401 and not calls
    assert post(c, headers={"x-api-key": "s3cret"}).status_code == 200 and len(calls) == 1


def test_the_daily_cap_stops_gpu_calls_with_429(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCINTEL_API_KEY", raising=False)
    c, calls = app(tmp_path, cap=2)
    assert [post(c).status_code for _ in range(3)] == [200, 200, 429] and len(calls) == 2


def test_dailycap_resets_on_a_new_day():
    from datetime import date

    cap = DailyCap(1)
    assert cap.take() and not cap.take()
    cap._day = date(2000, 1, 1)
    assert cap.take()


def test_check_access_rules(monkeypatch):
    monkeypatch.delenv("DOCINTEL_API_KEY", raising=False)
    assert check_access(None, "127.0.0.1", None) is None and check_access(None, "::1", None) is None
    assert check_access(None, "10.0.0.5", None) and check_access("k", "10.0.0.5", None)
