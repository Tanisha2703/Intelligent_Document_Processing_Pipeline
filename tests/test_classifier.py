"""
Tests for the document classifier.

Run: pytest tests/test_classifier.py -v
pytest tests/test_classifier.py -v -m "not slow"  # skip model tests
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

VALID_CATEGORIES = [
    "invoice", "receipt", "id_card", "legal_document",
    "medical_report", "bank_statement", "govt_form", "letter",
]


class TestDocumentClassifier:
    @pytest.mark.slow
    def test_returns_dict(self):
        from src.classifier.model import DocumentClassifier
        clf = DocumentClassifier()
        result = clf.predict("Invoice No. INV-12345 Total: ₹50,000")
        assert isinstance(result, dict)

    @pytest.mark.slow
    def test_category_is_valid(self):
        from src.classifier.model import DocumentClassifier
        clf = DocumentClassifier()
        result = clf.predict("Invoice No. INV-12345 Total: ₹50,000")
        assert result["category"] in VALID_CATEGORIES or result["category"] == "unknown"

    @pytest.mark.slow
    def test_confidence_in_range(self):
        from src.classifier.model import DocumentClassifier
        clf = DocumentClassifier()
        result = clf.predict("Invoice No. INV-12345 Total: ₹50,000")
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.slow
    def test_all_scores_present(self):
        from src.classifier.model import DocumentClassifier
        clf = DocumentClassifier()
        result = clf.predict("Invoice No. INV-12345 Total: ₹50,000")
        assert "all_scores" in result
        # Zero-shot may have all, fine-tuned should too
        assert isinstance(result["all_scores"], dict)

    @pytest.mark.slow
    def test_empty_text_returns_unknown(self):
        from src.classifier.model import DocumentClassifier
        clf = DocumentClassifier()
        result = clf.predict("")
        assert result["category"] == "unknown"
        assert result["confidence"] == 0.0

    @pytest.mark.slow
    def test_invoice_text(self):
        from src.classifier.model import DocumentClassifier
        clf = DocumentClassifier()
        text = (
            "Tax Invoice No. INV-12345 GSTIN: 22AAAAA0000A1Z5 "
            "Date: 15/03/2024 To: Ram Kumar "
            "Total Amount: ₹50,000 CGST: ₹4,500 SGST: ₹4,500"
        )
        result = clf.predict(text)
        # Should classify as invoice with reasonable confidence
        assert result["category"] in VALID_CATEGORIES
        assert result["confidence"] > 0.0

    @pytest.mark.slow
    def test_fallback_zero_shot_works_without_finetuned(self):
        """When fine-tuned model path doesn't exist, should fall back to zero-shot."""
        from src.classifier.model import DocumentClassifier
        clf = DocumentClassifier(config={
            "use_finetuned": True,
            "finetuned_model_path": "nonexistent/path",
            "fallback_model": "facebook/bart-large-mnli",
            "max_token_length": 512,
            "categories": VALID_CATEGORIES,
        })
        # Model should have fallen back to zero-shot
        assert clf._is_zero_shot is True
        result = clf.predict("Invoice for services rendered")
        assert result["category"] in VALID_CATEGORIES

    def test_display_name_not_empty(self):
        from src.classifier.model import CATEGORY_DISPLAY
        for cat in VALID_CATEGORIES:
            assert cat in CATEGORY_DISPLAY
            assert len(CATEGORY_DISPLAY[cat]) > 0
