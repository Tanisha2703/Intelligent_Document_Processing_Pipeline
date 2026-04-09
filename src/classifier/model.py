"""
Document Classifier — Fine-tuned XLM-RoBERTa for document type classification.

Classifies documents into 8 categories:
  invoice, receipt, id_card, legal_document,
  medical_report, bank_statement, govt_form, letter

Loads fine-tuned model from models/classifier/ if available.
Falls back to zero-shot (facebook/bart-large-mnli) if not found.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 8 document categories
CATEGORIES = [
    "invoice",
    "receipt",
    "id_card",
    "legal_document",
    "medical_report",
    "bank_statement",
    "govt_form",
    "letter",
]

CATEGORY_DISPLAY = {
    "invoice": "🧾 Invoice",
    "receipt": "🏷️ Receipt",
    "id_card": "🪪 ID Card",
    "legal_document": "⚖️ Legal Document",
    "medical_report": "🏥 Medical Report",
    "bank_statement": "🏦 Bank Statement",
    "govt_form": "📋 Government Form",
    "letter": "✉️ Letter",
}


class DocumentClassifier:
    """XLM-RoBERTa sequence classifier for document type detection."""

    def __init__(self, config: Optional[dict] = None):
        from src.utils.config import get_config

        cfg = config or get_config().get("classifier", {})
        self.use_finetuned = cfg.get("use_finetuned", False)
        self.finetuned_path = cfg.get("finetuned_model_path", "models/classifier")
        self.fallback_model = cfg.get("fallback_model", "facebook/bart-large-mnli")
        self.max_token_length = cfg.get("max_token_length", 512)
        self.categories = cfg.get("categories", CATEGORIES)

        self._pipeline = None
        self._is_zero_shot = False
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict:
        """
        Classify document text into one of the 8 categories.

        Uses first max_token_length tokens (fine-tuned) or first 1024 chars (zero-shot).

        Args:
            text: OCR-extracted document text

        Returns:
            dict:
                category     — predicted category string
                display      — human-readable label with icon
                confidence   — float 0.0–1.0
                all_scores   — dict mapping category → confidence
        """
        if not text or not text.strip():
            return self._empty_result()

        try:
            if self._is_zero_shot:
                return self._zero_shot_predict(text)
            else:
                return self._finetuned_predict(text)
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            return self._empty_result()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load fine-tuned or fallback zero-shot model."""
        from transformers import pipeline as hf_pipeline

        model_path = None

        if self.use_finetuned:
            fpath = Path(self.finetuned_path)
            if fpath.exists() and any(fpath.iterdir()):
                model_path = str(fpath)
                logger.info("Classifier: loading fine-tuned model from %s", model_path)
            else:
                logger.warning(
                    "Classifier: fine-tuned model not found at %s, using zero-shot fallback",
                    self.finetuned_path,
                )

        if model_path is not None:
            try:
                self._pipeline = hf_pipeline(
                    "text-classification",
                    model=model_path,
                    device=-1,
                    top_k=None,  # Return all scores
                )
                self._is_zero_shot = False
                logger.info("Classifier: fine-tuned model loaded")
                return
            except Exception as exc:
                logger.error("Failed to load fine-tuned classifier: %s", exc)

        # Fallback: zero-shot classification
        logger.info("Classifier: loading zero-shot model: %s", self.fallback_model)
        try:
            self._pipeline = hf_pipeline(
                "zero-shot-classification",
                model=self.fallback_model,
                device=-1,
            )
            self._is_zero_shot = True
            logger.info("Classifier: zero-shot model loaded")
        except Exception as exc:
            logger.error("Failed to load zero-shot classifier: %s", exc)
            self._pipeline = None

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _finetuned_predict(self, text: str) -> dict:
        """Run fine-tuned sequence classifier."""
        # Truncate to avoid token limit issues (rough char heuristic: 3 chars/token)
        max_chars = self.max_token_length * 3
        truncated = text[:max_chars]

        results = self._pipeline(truncated)

        # HuggingFace returns list of list when top_k=None
        if results and isinstance(results[0], list):
            scores = results[0]
        else:
            scores = results

        # Build all_scores dict
        all_scores = {item["label"]: round(item["score"], 4) for item in scores}
        best = max(scores, key=lambda x: x["score"])

        category = best["label"]
        confidence = round(best["score"], 4)

        return {
            "category": category,
            "display": CATEGORY_DISPLAY.get(category, category),
            "confidence": confidence,
            "all_scores": all_scores,
        }

    def _zero_shot_predict(self, text: str) -> dict:
        """Run zero-shot classification with facebook/bart-large-mnli."""
        # Limit text for zero-shot (slower model)
        truncated = text[:1500]

        result = self._pipeline(
            truncated,
            candidate_labels=self.categories,
            hypothesis_template="This document is a {}.",
        )

        labels = result["labels"]
        scores = result["scores"]

        all_scores = {
            label: round(score, 4) for label, score in zip(labels, scores)
        }
        best_label = labels[0]
        best_score = scores[0]

        return {
            "category": best_label,
            "display": CATEGORY_DISPLAY.get(best_label, best_label),
            "confidence": round(best_score, 4),
            "all_scores": all_scores,
        }

    def _empty_result(self) -> dict:
        """Return a safe empty result when classification cannot be performed."""
        return {
            "category": "unknown",
            "display": "❓ Unknown",
            "confidence": 0.0,
            "all_scores": {cat: 0.0 for cat in self.categories},
        }
