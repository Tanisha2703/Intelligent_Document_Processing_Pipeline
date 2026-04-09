"""
NER Model — Fine-tuned XLM-RoBERTa for Named Entity Recognition.

Loads fine-tuned model from models/ner/ if available.
Falls back to Davlan/xlm-roberta-base-ner-hrl (pretrained) if not found.

Handles long text by chunking at line breaks (max 1000 chars per chunk).
Post-processes with regex for structured entity types.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class NERModel:
    """XLM-RoBERTa NER inference wrapper with regex post-processing."""

    def __init__(self, config: Optional[dict] = None):
        from src.utils.config import get_config

        cfg = config or get_config().get("ner", {})
        self.use_finetuned = cfg.get("use_finetuned", False)
        self.finetuned_path = cfg.get("finetuned_model_path", "models/ner")
        self.fallback_model = cfg.get("fallback_model", "Davlan/xlm-roberta-base-ner-hrl")
        self.confidence_threshold = cfg.get("confidence_threshold", 0.70)
        self.max_chunk_length = cfg.get("max_chunk_length", 1000)

        self._pipeline = None
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> list[dict]:
        """
        Run NER on input text and return entity list.

        Long texts are split into chunks at line breaks, processed separately,
        and merged with adjusted character offsets.

        Args:
            text: OCR-extracted text (may be multi-page)

        Returns:
            List of entity dicts:
                {type, value, start, end, confidence, source}
        """
        if not text or not text.strip():
            return []

        # Split into chunks to stay within token limits
        chunks = self._chunk_text(text)
        all_model_entities: list[dict] = []

        for chunk_text, chunk_offset in chunks:
            chunk_entities = self._run_inference(chunk_text)
            # Adjust offsets to be relative to the full text
            for ent in chunk_entities:
                ent["start"] += chunk_offset
                ent["end"] += chunk_offset
            all_model_entities.extend(chunk_entities)

        # Regex post-processing on the full text
        from src.ner.postprocessor import (
            extract_regex_entities,
            merge_entities,
            deduplicate_entities,
        )
        regex_entities = extract_regex_entities(text)

        # Merge model + regex predictions
        merged = merge_entities(all_model_entities, regex_entities, self.confidence_threshold)
        return deduplicate_entities(merged)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load fine-tuned or fallback model into a HuggingFace pipeline."""
        from transformers import pipeline as hf_pipeline

        model_path = None

        if self.use_finetuned:
            fpath = Path(self.finetuned_path)
            if fpath.exists() and any(fpath.iterdir()):
                model_path = str(fpath)
                logger.info("NER: loading fine-tuned model from %s", model_path)
            else:
                logger.warning(
                    "NER: fine-tuned model not found at %s, falling back to pretrained",
                    self.finetuned_path,
                )

        if model_path is None:
            model_path = self.fallback_model
            logger.info("NER: loading pretrained fallback model: %s", model_path)

        try:
            self._pipeline = hf_pipeline(
                task="ner",
                model=model_path,
                aggregation_strategy="simple",
                device=-1,  # CPU; will use GPU if device=0 and CUDA available
            )
            logger.info("NER model loaded successfully")
        except Exception as exc:
            logger.error("Failed to load NER model: %s", exc)
            self._pipeline = None

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _run_inference(self, text: str) -> list[dict]:
        """Run the HuggingFace NER pipeline on a single chunk."""
        if self._pipeline is None:
            logger.warning("NER pipeline not loaded — skipping model inference")
            return []

        try:
            raw_entities = self._pipeline(text)
        except Exception as exc:
            logger.error("NER inference error: %s", exc)
            return []

        return self._normalize_entities(raw_entities, text)

    def _normalize_entities(self, raw: list, text: str) -> list[dict]:
        """
        Convert HuggingFace pipeline output to our entity format.

        HuggingFace returns dicts with keys: entity_group, score, word, start, end.
        We remap entity_group using PRETRAINED_LABEL_MAP if using fallback model.
        """
        from src.ner.entities import ENTITY_TYPES, PRETRAINED_LABEL_MAP

        entities = []
        for item in raw:
            label = item.get("entity_group", item.get("entity", "O"))

            # Strip B-/I- prefix if present (aggregation_strategy="simple" gives group names)
            label = label.replace("B-", "").replace("I-", "")

            # Map pretrained model labels to our label scheme
            label = PRETRAINED_LABEL_MAP.get(label, label)

            if label not in ENTITY_TYPES:
                continue  # Unknown or "O" — skip

            entities.append(
                {
                    "type": label,
                    "value": item.get("word", "").strip(),
                    "start": item.get("start", 0),
                    "end": item.get("end", 0),
                    "confidence": round(float(item.get("score", 0.0)), 4),
                    "source": "model",
                }
            )
        return entities

    def _chunk_text(self, text: str) -> list[tuple[str, int]]:
        """
        Split text into chunks at line breaks, respecting max_chunk_length.

        Returns:
            List of (chunk_text, offset_in_full_text) tuples.
        """
        chunks: list[tuple[str, int]] = []
        current_chunk: list[str] = []
        current_len = 0
        current_offset = 0
        char_pos = 0

        lines = text.split("\n")
        for line in lines:
            line_len = len(line) + 1  # +1 for the newline

            if current_len + line_len > self.max_chunk_length and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append((chunk_text, current_offset))
                current_offset = char_pos
                current_chunk = []
                current_len = 0

            current_chunk.append(line)
            current_len += line_len
            char_pos += line_len

        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append((chunk_text, current_offset))

        return chunks if chunks else [(text, 0)]
