"""
Tests for JSON/CSV/Excel export functions.

Run: pytest tests/test_export.py -v
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


# ---------------------------------------------------------------------------
# Fixture: minimal pipeline result dict
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_result():
    return {
        "ocr_text": "Invoice No. INV-12345\nDate: 15/03/2024\nAmount: ₹50,000",
        "ocr_confidence": 94.5,
        "page_count": 1,
        "language": {"primary": "english", "hindi_ratio": 0.0, "english_ratio": 1.0},
        "entities": [
            {"type": "ID_NUMBER", "value": "INV-12345", "confidence": 0.92, "source": "regex", "start": 12, "end": 21},
            {"type": "DATE", "value": "15/03/2024", "confidence": 0.89, "source": "regex", "start": 28, "end": 38},
            {"type": "AMOUNT", "value": "₹50,000", "confidence": 0.96, "source": "regex", "start": 47, "end": 54},
        ],
        "entities_by_type": {
            "PERSON": [],
            "ORG": [],
            "LOCATION": [],
            "DATE": [{"type": "DATE", "value": "15/03/2024", "confidence": 0.89}],
            "AMOUNT": [{"type": "AMOUNT", "value": "₹50,000", "confidence": 0.96}],
            "ID_NUMBER": [{"type": "ID_NUMBER", "value": "INV-12345", "confidence": 0.92}],
            "PHONE": [],
            "EMAIL": [],
        },
        "entity_count": 3,
        "classification": {
            "category": "invoice",
            "display": "🧾 Invoice",
            "confidence": 0.95,
            "all_scores": {"invoice": 0.95, "receipt": 0.03, "id_card": 0.01},
        },
        "processing_time_s": 3.2,
        "confidence_threshold": 0.70,
        "word_data": [],
    }


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

class TestJSONExport:
    def test_returns_string(self, sample_result):
        from src.utils.export import to_json
        result = to_json(sample_result)
        assert isinstance(result, str)

    def test_valid_json(self, sample_result):
        from src.utils.export import to_json
        result = to_json(sample_result)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_has_entities(self, sample_result):
        from src.utils.export import to_json
        parsed = json.loads(to_json(sample_result))
        assert "entities" in parsed
        assert len(parsed["entities"]) == 3

    def test_has_classification(self, sample_result):
        from src.utils.export import to_json
        parsed = json.loads(to_json(sample_result))
        assert "classification" in parsed
        assert parsed["classification"]["category"] == "invoice"

    def test_has_exported_at(self, sample_result):
        from src.utils.export import to_json
        parsed = json.loads(to_json(sample_result))
        assert "exported_at" in parsed

    def test_unicode_preserved(self, sample_result):
        from src.utils.export import to_json
        result = to_json(sample_result)
        assert "₹50,000" in result  # Unicode rupee symbol preserved


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestCSVExport:
    def test_returns_string(self, sample_result):
        from src.utils.export import to_csv
        result = to_csv(sample_result)
        assert isinstance(result, str)

    def test_has_header(self, sample_result):
        from src.utils.export import to_csv
        csv_str = to_csv(sample_result)
        assert "entity_type" in csv_str
        assert "value" in csv_str
        assert "confidence" in csv_str

    def test_row_count(self, sample_result):
        from src.utils.export import to_csv
        import io
        csv_str = to_csv(sample_result)
        df = pd.read_csv(io.StringIO(csv_str))
        assert len(df) == 3  # 3 entities

    def test_empty_entities(self):
        from src.utils.export import to_csv
        result = {"entities": []}
        csv_str = to_csv(result)
        assert "entity_type" in csv_str  # header still present

    def test_entity_values_present(self, sample_result):
        from src.utils.export import to_csv
        import io
        csv_str = to_csv(sample_result)
        df = pd.read_csv(io.StringIO(csv_str))
        values = list(df["value"])
        assert "INV-12345" in values


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

class TestExcelExport:
    def test_returns_bytes(self, sample_result):
        from src.utils.export import to_excel_bytes
        result = to_excel_bytes(sample_result)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_has_three_sheets(self, sample_result):
        from src.utils.export import to_excel_bytes
        import io
        excel_bytes = to_excel_bytes(sample_result)
        xl = pd.ExcelFile(io.BytesIO(excel_bytes))
        assert "Summary" in xl.sheet_names
        assert "Entities" in xl.sheet_names
        assert "Raw Text" in xl.sheet_names

    def test_summary_sheet_content(self, sample_result):
        from src.utils.export import to_excel_bytes
        import io
        excel_bytes = to_excel_bytes(sample_result)
        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Summary")
        fields = list(df["Field"])
        assert "Document Category" in fields
        assert "OCR Confidence (%)" in fields

    def test_entities_sheet_count(self, sample_result):
        from src.utils.export import to_excel_bytes
        import io
        excel_bytes = to_excel_bytes(sample_result)
        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Entities")
        assert len(df) == 3

    def test_raw_text_sheet(self, sample_result):
        from src.utils.export import to_excel_bytes
        import io
        excel_bytes = to_excel_bytes(sample_result)
        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Raw Text")
        assert "OCR Extracted Text" in df.columns
        text = str(df.iloc[0, 0])
        assert "INV-12345" in text
