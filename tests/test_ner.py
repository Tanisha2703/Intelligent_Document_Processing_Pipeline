"""
Tests for NER model, postprocessor, and entity definitions.

Run: pytest tests/test_ner.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


# ---------------------------------------------------------------------------
# Entity definitions
# ---------------------------------------------------------------------------

class TestEntityDefinitions:
    def test_entity_types_count(self):
        from src.ner.entities import ENTITY_TYPES
        assert len(ENTITY_TYPES) == 8

    def test_iob2_labels_count(self):
        from src.ner.entities import IOB2_LABELS
        # O + 2 * 8 types = 17
        assert len(IOB2_LABELS) == 17

    def test_iob2_starts_with_O(self):
        from src.ner.entities import IOB2_LABELS
        assert IOB2_LABELS[0] == "O"

    def test_label2id_and_id2label_consistent(self):
        from src.ner.entities import LABEL2ID, ID2LABEL, IOB2_LABELS
        assert len(LABEL2ID) == len(IOB2_LABELS)
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label

    def test_all_entity_types_in_colors(self):
        from src.ner.entities import ENTITY_TYPES, ENTITY_COLORS
        for etype in ENTITY_TYPES:
            assert etype in ENTITY_COLORS, f"{etype} missing from ENTITY_COLORS"

    def test_num_labels(self):
        from src.ner.entities import NUM_LABELS
        assert NUM_LABELS == 17


# ---------------------------------------------------------------------------
# Postprocessor — regex patterns
# ---------------------------------------------------------------------------

class TestRegexPatterns:
    def test_pan_detection(self):
        from src.ner.postprocessor import extract_regex_entities
        text = "PAN card: ABCDE1234F of the applicant"
        entities = extract_regex_entities(text)
        types = [e["type"] for e in entities]
        assert "ID_NUMBER" in types
        id_ents = [e for e in entities if e["type"] == "ID_NUMBER"]
        assert any("ABCDE1234F" in e["value"] for e in id_ents)

    def test_aadhaar_detection(self):
        from src.ner.postprocessor import extract_regex_entities
        text = "Aadhaar Number: 1234 5678 9012"
        entities = extract_regex_entities(text)
        id_ents = [e for e in entities if e["type"] == "ID_NUMBER"]
        assert len(id_ents) >= 1

    def test_phone_detection(self):
        from src.ner.postprocessor import extract_regex_entities
        text = "Contact: +91 9876543210"
        entities = extract_regex_entities(text)
        phone_ents = [e for e in entities if e["type"] == "PHONE"]
        assert len(phone_ents) >= 1

    def test_email_detection(self):
        from src.ner.postprocessor import extract_regex_entities
        text = "Email: john.doe@example.com"
        entities = extract_regex_entities(text)
        email_ents = [e for e in entities if e["type"] == "EMAIL"]
        assert len(email_ents) >= 1
        assert "john.doe@example.com" in email_ents[0]["value"]

    def test_amount_rupee_detection(self):
        from src.ner.postprocessor import extract_regex_entities
        text = "Total amount: ₹50,000"
        entities = extract_regex_entities(text)
        amt_ents = [e for e in entities if e["type"] == "AMOUNT"]
        assert len(amt_ents) >= 1

    def test_amount_rs_detection(self):
        from src.ner.postprocessor import extract_regex_entities
        text = "Payment of Rs. 1,23,456 received"
        entities = extract_regex_entities(text)
        amt_ents = [e for e in entities if e["type"] == "AMOUNT"]
        assert len(amt_ents) >= 1

    def test_date_detection(self):
        from src.ner.postprocessor import extract_regex_entities
        text = "Dated: 15/03/2024"
        entities = extract_regex_entities(text)
        date_ents = [e for e in entities if e["type"] == "DATE"]
        assert len(date_ents) >= 1

    def test_iso_date_detection(self):
        from src.ner.postprocessor import extract_regex_entities
        text = "Date: 2024-03-15"
        entities = extract_regex_entities(text)
        date_ents = [e for e in entities if e["type"] == "DATE"]
        assert len(date_ents) >= 1


# ---------------------------------------------------------------------------
# Postprocessor — entity merge logic
# ---------------------------------------------------------------------------

class TestEntityMerge:
    def _make_entity(self, etype, value, start, end, conf=0.9, source="model"):
        return {"type": etype, "value": value, "start": start, "end": end,
                "confidence": conf, "source": source}

    def test_no_overlap_keeps_both(self):
        from src.ner.postprocessor import merge_entities
        a = self._make_entity("PERSON", "Ram Kumar", 0, 9)
        b = self._make_entity("ORG", "Acme Corp", 15, 24)
        merged = merge_entities([a], [b], threshold=0.0)
        assert len(merged) == 2

    def test_same_span_same_type_boosts_confidence(self):
        from src.ner.postprocessor import merge_entities
        a = self._make_entity("PHONE", "+91 9876543210", 0, 15, conf=0.85)
        b = self._make_entity("PHONE", "+91 9876543210", 0, 15, conf=0.88, source="regex")
        merged = merge_entities([a], [b], threshold=0.0)
        assert len(merged) == 1
        assert merged[0]["confidence"] >= 0.85

    def test_overlap_different_type_keeps_higher_confidence(self):
        from src.ner.postprocessor import merge_entities
        a = self._make_entity("PERSON", "ABCDE1234F", 0, 10, conf=0.60)
        b = self._make_entity("ID_NUMBER", "ABCDE1234F", 0, 10, conf=0.88, source="regex")
        merged = merge_entities([a], [b], threshold=0.0)
        assert len(merged) == 1
        assert merged[0]["type"] == "ID_NUMBER"

    def test_confidence_filter_works(self):
        from src.ner.postprocessor import filter_by_confidence
        ents = [
            {"type": "PERSON", "value": "X", "confidence": 0.9},
            {"type": "PERSON", "value": "Y", "confidence": 0.5},
        ]
        filtered = filter_by_confidence(ents, threshold=0.70)
        assert len(filtered) == 1
        assert filtered[0]["value"] == "X"

    def test_deduplicate_removes_same_value_type(self):
        from src.ner.postprocessor import deduplicate_entities
        ents = [
            {"type": "PERSON", "value": "Ram Kumar", "confidence": 0.9},
            {"type": "PERSON", "value": "Ram Kumar", "confidence": 0.85},
        ]
        deduped = deduplicate_entities(ents)
        assert len(deduped) == 1
        assert deduped[0]["confidence"] == 0.9


# ---------------------------------------------------------------------------
# NER model (integration — requires internet for first download)
# ---------------------------------------------------------------------------

class TestNERModel:
    @pytest.mark.slow
    def test_returns_list(self):
        from src.ner.model import NERModel
        model = NERModel()
        result = model.predict("Ram Kumar paid Rs. 50,000 to Acme Corp on 15/03/2024.")
        assert isinstance(result, list)

    @pytest.mark.slow
    def test_entities_have_required_keys(self):
        from src.ner.model import NERModel
        model = NERModel()
        entities = model.predict("Call me at +91 9876543210 or email test@example.com")
        for ent in entities:
            assert "type" in ent
            assert "value" in ent
            assert "confidence" in ent

    @pytest.mark.slow
    def test_regex_entities_always_present(self):
        """Even without model, regex should catch structured entities."""
        from src.ner.model import NERModel
        model = NERModel()
        # PAN card — regex should definitely catch this
        entities = model.predict("PAN: ABCDE1234F")
        id_ents = [e for e in entities if e["type"] == "ID_NUMBER"]
        assert len(id_ents) >= 1

    @pytest.mark.slow
    def test_empty_text_returns_empty_list(self):
        from src.ner.model import NERModel
        model = NERModel()
        assert model.predict("") == []

    @pytest.mark.slow
    def test_hindi_text(self):
        from src.ner.model import NERModel
        model = NERModel()
        # Regex should catch Aadhaar even in Hindi context
        entities = model.predict("आधार संख्या: 1234 5678 9012")
        assert isinstance(entities, list)
