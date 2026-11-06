import json

from src.pipeline.calibration import IsotonicMap
from src.pipeline.receipts import ReceiptExtractor

IDENT = IsotonicMap([0.0, 1.0], [0.0, 1.0])


def stub(fields, conf):
    return lambda png, doc_type: {"fields": fields, "confidence": {k: {"min_prob": v} for k, v in conf.items()}}


def test_confident_fields_are_auto_and_uncertain_ones_go_to_review():
    ex = ReceiptExtractor(
        stub(
            {"company": "ABC", "date": "01/02/2019", "address": "X", "total": "9.50"},
            {"company": 0.99, "date": 0.97, "address": 0.6, "total": 0.995},
        ),
        IDENT,
        accept=0.95,
    )
    r = ex.extract(b"png")
    assert {f: x["status"] for f, x in r["fields"].items()} == {
        "company": "auto",
        "date": "auto",
        "address": "review",
        "total": "auto",
    }
    assert r["needs_review"] and r["review_fields"] == ["address"]


def test_missing_field_is_never_auto_even_with_a_stray_confidence():
    ex = ReceiptExtractor(
        stub(
            {"company": "ABC", "date": None, "address": "", "total": "1.00"},
            {"company": 0.99, "date": 0.99, "address": 0.99, "total": 0.99},
        ),
        IDENT,
        accept=0.5,
    )
    r = ex.extract(b"png")
    assert r["fields"]["date"]["status"] == "review" and r["fields"]["address"]["status"] == "review"
    assert r["fields"]["date"]["probability"] == 0.0


def test_value_without_confidence_goes_to_review():
    ex = ReceiptExtractor(
        lambda png, dt: {
            "fields": {"company": "A", "date": "1/1/2019", "address": "B", "total": "2"},
            "confidence": {},
        },
        IDENT,
        0.5,
    )
    assert ex.extract(b"x")["review_fields"] == ["company", "date", "address", "total"]


def test_calibration_map_is_applied_before_the_threshold():
    squash = IsotonicMap([0.0, 0.9, 1.0], [0.0, 0.1, 0.5])  # a raw 0.95 is worth only 0.3
    ex = ReceiptExtractor(
        stub(
            {"company": "A", "date": "1/1/2019", "address": "B", "total": "2"},
            {"company": 0.95, "date": 0.95, "address": 0.95, "total": 0.95},
        ),
        squash,
        accept=0.5,
    )
    assert ex.extract(b"x")["needs_review"] and all(
        f["status"] == "review" for f in ex.extract(b"x")["fields"].values()
    )


def test_total_is_normalised_and_config_round_trips(tmp_path):
    ex = ReceiptExtractor(
        stub(
            {"company": "A", "date": "15/01/2019", "address": "B", "total": "1,193.00"},
            dict.fromkeys(("company", "date", "address", "total"), 0.99),
        ),
        IDENT,
        accept=0.9,
    )
    assert ex.extract(b"x")["fields"]["total"]["normalized"] == "1193.00"
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"map": IDENT.to_dict(), "accept": 0.9, "feature": "min_prob"}), encoding="utf-8")
    ex2 = ReceiptExtractor.from_config(
        stub(
            {"company": "A", "date": "1/1/2019", "address": "B", "total": "2"},
            dict.fromkeys(("company", "date", "address", "total"), 0.95),
        ),
        cfg,
    )
    assert not ex2.extract(b"x")["needs_review"]


def test_accept_threshold_can_be_overridden_from_the_environment(tmp_path, monkeypatch):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"map": IDENT.to_dict(), "accept": 0.8, "feature": "min_prob"}), encoding="utf-8")
    vlm = stub(
        {"company": "A", "date": "1/1/2019", "address": "B", "total": "2"},
        dict.fromkeys(("company", "date", "address", "total"), 0.85),
    )
    monkeypatch.delenv("DOCINTEL_ACCEPT_THRESHOLD", raising=False)
    assert not ReceiptExtractor.from_config(vlm, cfg).extract(b"x")["needs_review"]  # 0.85 >= the configured 0.80
    monkeypatch.setenv("DOCINTEL_ACCEPT_THRESHOLD", "0.95")
    strict = ReceiptExtractor.from_config(vlm, cfg)
    assert strict.accept == 0.95 and strict.extract(b"x")["needs_review"]  # the strict setting reviews them
