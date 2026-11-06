"""The Streamlit demo, driven with streamlit's AppTest (no browser, no GPU)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).parents[1] / "app.py")


@pytest.fixture(autouse=True)
def fresh_cache():
    import streamlit as st

    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def run(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCINTEL_REVIEW_PATH", str(tmp_path / "review.jsonl"))
    monkeypatch.delenv("DOCINTEL_VLM_APP", raising=False)
    return AppTest.from_file(APP, default_timeout=30).run()


def test_app_starts_and_shows_a_recorded_example(tmp_path, monkeypatch):
    at = run(tmp_path, monkeypatch)
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "accepted" in html and "review" in html  # status chips for the recorded receipt
    assert any("A person should check" in w.value for w in at.warning)


def test_second_example_is_selectable(tmp_path, monkeypatch):
    at = run(tmp_path, monkeypatch)
    at.selectbox[0].select(1).run()
    assert not at.exception


def test_upload_mode_explains_how_to_enable_the_model_when_none_is_configured(tmp_path, monkeypatch):
    at = run(tmp_path, monkeypatch)
    at.radio[0].set_value("Upload a receipt").run()
    assert not at.exception
    assert any("Modal" in i.value for i in at.info)


def test_review_tab_lists_queued_fields_and_saves_a_correction(tmp_path, monkeypatch):
    from src.monitor.review import ReviewStore

    store = ReviewStore(tmp_path / "review.jsonl")
    doc = store.enqueue(
        None, "scanned_receipt", {"company": {"value": "ABC", "probability": 0.5, "source": "vlm", "status": "review"}}
    )
    at = run(tmp_path, monkeypatch)
    assert not at.exception
    assert any(m.label == "Documents waiting" and m.value == "1" for m in at.metric)
    at.text_input[0].set_value("ABC SDN BHD").run()
    next(b for b in at.button if b.label == "Save").click().run()
    assert not at.exception
    rec = [json.loads(x) for x in (tmp_path / "review.jsonl").read_text(encoding="utf-8").splitlines()][-1]
    assert (
        rec["kind"] == "resolved"
        and rec["value"] == "ABC SDN BHD"
        and rec["doc_id"] == doc
        and rec["confirmed"] is False
    )
