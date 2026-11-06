"""Drift monitor (PSI) and the review store."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.monitor.drift import DriftMonitor, psi
from src.monitor.review import ReviewStore


def test_psi_is_near_zero_for_the_same_distribution_and_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.gamma(2.0, 0.5, 5000)
    same = rng.gamma(2.0, 0.5, 5000)
    small = rng.gamma(2.0, 0.6, 5000)
    large = rng.gamma(2.0, 1.5, 5000)
    assert psi(ref, same) < 0.05
    assert psi(ref, same) < psi(ref, small) < psi(ref, large)
    assert psi(ref, large) > 0.25


def test_psi_has_no_small_sample_false_alarms():
    # raw PSI of two draws from the SAME distribution grows as the window shrinks; the corrected value stays low
    rng = np.random.default_rng(7)
    ref = rng.gamma(2.0, 0.5, 5000)
    raw = [psi(ref, rng.gamma(2.0, 0.5, 100), bias_correct=False) for _ in range(40)]
    fixed = [psi(ref, rng.gamma(2.0, 0.5, 100)) for _ in range(40)]
    assert np.mean(raw) > 0.07 > np.mean(fixed)
    assert max(fixed) < 0.25


def test_psi_with_a_constant_reference_flags_any_change():
    assert psi([1.0] * 100, [1.0] * 50) == 0.0
    assert psi([1.0] * 100, [2.0] * 50) == 1.0


def test_monitor_waits_for_enough_data_then_reports_levels():
    rng = np.random.default_rng(1)
    ref = rng.gamma(2.0, 0.5, 2000)
    m = DriftMonitor(ref, window=300, min_window=100, kappa=1.0)
    for x in rng.gamma(2.0, 0.5, 60):
        m.observe(x)
    assert m.status()["level"] == "insufficient_data"
    for x in rng.gamma(2.0, 0.5, 240):
        m.observe(x)
    assert m.status()["level"] == "stable"


def test_a_degraded_batch_trips_the_monitor_and_raises_the_escalation_rate():
    rng = np.random.default_rng(2)
    ref = rng.gamma(2.0, 0.5, 2000)
    m = DriftMonitor(ref, window=200, min_window=100, kappa=1.0)
    for x in rng.gamma(2.0, 0.5, 200):
        m.observe(x)
    healthy = m.status()
    for x in rng.gamma(2.0, 1.5, 200):  # e.g. a batch of phone photos: expected errors jump
        m.observe(x)
    bad = m.status()
    assert bad["level"] == "large" and bad["psi"] > healthy["psi"]
    assert bad["escalation_rate"] > bad["reference_escalation_rate"]


def test_window_forgets_old_observations():
    ref = np.random.default_rng(3).gamma(2.0, 0.5, 2000)
    m = DriftMonitor(ref, window=100, min_window=50)
    for _ in range(100):
        m.observe(10.0)
    assert m.status()["level"] == "large"
    for x in np.random.default_rng(4).gamma(2.0, 0.5, 100):
        m.observe(x)
    assert m.status()["level"] == "stable"


FIELDS = {
    "name": {"value": "Amit", "probability": 0.99, "source": "vlm", "status": "auto"},
    "dob": {"value": "01/02/1990", "probability": 0.4, "source": "vlm", "status": "review"},
    "id_number": {"value": None, "probability": 0.0, "source": None, "status": "review"},
}


def test_only_review_fields_are_queued_and_resolution_removes_them(tmp_path):
    s = ReviewStore(tmp_path / "review.jsonl")
    doc = s.enqueue("d1", "id_card", FIELDS)
    pend = s.pending()
    assert [p["doc_id"] for p in pend] == ["d1"] and set(pend[0]["fields"]) == {"dob", "id_number"}
    s.resolve("d1", "dob", "01/02/1990")
    assert set(s.pending()[0]["fields"]) == {"id_number"}
    s.resolve("d1", "id_number", "ABCDE1234F")
    assert s.pending() == [] and doc == "d1"


def test_corrections_record_whether_the_model_was_right(tmp_path):
    s = ReviewStore(tmp_path / "review.jsonl")
    s.enqueue("d1", "id_card", FIELDS)
    s.resolve("d1", "dob", "01/02/1990")  # confirms the proposal
    s.resolve("d1", "id_number", "ABCDE1234F")  # fills a missing value
    c = {r["field"]: r for r in s.corrections()}
    assert c["dob"]["confirmed"] is True and c["id_number"]["confirmed"] is False
    assert s.stats() == {"pending_docs": 0, "resolved_fields": 2, "model_was_right": 1, "model_was_wrong": 1}


def test_resolving_an_unqueued_field_is_an_error(tmp_path):
    s = ReviewStore(tmp_path / "review.jsonl")
    s.enqueue("d1", "id_card", FIELDS)
    with pytest.raises(KeyError):
        s.resolve("d1", "name", "x")  # name was auto-accepted, never queued
    with pytest.raises(KeyError):
        s.resolve("nope", "dob", "x")


def test_a_torn_last_line_is_skipped_not_fatal(tmp_path):
    s = ReviewStore(tmp_path / "review.jsonl")
    s.enqueue("d1", "id_card", FIELDS)
    with (tmp_path / "review.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"kind": "resolved", "doc_i')  # crash mid-write
    assert [p["doc_id"] for p in s.pending()] == ["d1"]


def test_unicode_values_survive_the_round_trip(tmp_path):
    s = ReviewStore(tmp_path / "review.jsonl")
    s.enqueue(
        "d2", "id_card", {"name": {"value": "राजेश कुमार", "probability": 0.3, "source": "vlm", "status": "review"}}
    )
    assert s.pending()[0]["fields"]["name"]["value"] == "राजेश कुमार"
