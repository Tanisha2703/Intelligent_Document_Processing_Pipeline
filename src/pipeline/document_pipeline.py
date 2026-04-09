"""
Document Pipeline — End-to-end orchestrator.

Stages:
  1. Document ingestion (file path or bytes)
  2. Image preprocessing (OCR accuracy boost)
  3. OCR extraction (Tesseract hin+eng)
  4a. NER — fine-tuned XLM-RoBERTa + regex post-processing
  4b. Classification — fine-tuned XLM-RoBERTa (runs in parallel with 4a)
  5. Structured output (JSON/CSV/Excel export)

Fail-soft: if NER fails → regex only; if classifier fails → "unknown".
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    End-to-end document processing pipeline.

    Usage:
        pipeline = DocumentPipeline()
        result = pipeline.process(file_path="invoice.pdf")
        print(result["entities"])
        pipeline.export_json(result)
    """

    def __init__(self, config: dict | None = None):
        from src.utils.config import get_config

        self._cfg = config or get_config()
        self._ocr = None
        self._ner = None
        self._classifier = None
        self._initialized = False

    def initialize(self) -> None:
        """Load all models. Call this once before processing documents."""
        if self._initialized:
            return

        logger.info("Initializing DocumentPipeline...")
        from src.ocr.engine import OCREngine
        from src.ner.model import NERModel
        from src.classifier.model import DocumentClassifier

        self._ocr = OCREngine(self._cfg.get("ocr"))
        self._ner = NERModel(self._cfg.get("ner"))
        self._classifier = DocumentClassifier(self._cfg.get("classifier"))
        self._initialized = True
        logger.info("Pipeline initialized")

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def process(
        self,
        file_path: Union[str, Path, None] = None,
        file_bytes: Union[bytes, None] = None,
        file_type: str = "pdf",
        confidence_threshold: float | None = None,
    ) -> dict:
        """
        Process a document through the full pipeline.

        Provide either file_path OR (file_bytes + file_type).

        Args:
            file_path            — path to PDF or image file
            file_bytes           — raw file bytes
            file_type            — "pdf", "png", "jpg", "tiff"
            confidence_threshold — override config threshold (0.0–1.0)

        Returns:
            Full structured result dict (see _build_result for schema).
        """
        if not self._initialized:
            self.initialize()

        threshold = (
            confidence_threshold
            or self._cfg.get("ner", {}).get("confidence_threshold", 0.70)
        )
        start_time = time.perf_counter()

        # Stage 1 + 2 + 3: OCR
        ocr_result = self._run_ocr(file_path, file_bytes, file_type)
        text = ocr_result.get("text", "")

        # Stage 4a + 4b: NER + Classification in parallel
        run_parallel = self._cfg.get("pipeline", {}).get("parallel_ner_classifier", True)
        ner_entities, classification = self._run_ner_and_classify(text, run_parallel)

        # Language detection
        from src.ocr.language_detect import detect_language

        language_info = detect_language(text)

        elapsed = round(time.perf_counter() - start_time, 2)

        return self._build_result(
            ocr_result=ocr_result,
            text=text,
            entities=ner_entities,
            classification=classification,
            language_info=language_info,
            processing_time=elapsed,
            confidence_threshold=threshold,
        )

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_json(self, result: dict) -> str:
        """Export result as JSON string."""
        from src.utils.export import to_json
        return to_json(result)

    def export_csv(self, result: dict) -> str:
        """Export entities as CSV string."""
        from src.utils.export import to_csv
        return to_csv(result)

    def export_excel(self, result: dict) -> bytes:
        """Export result as Excel bytes (3 sheets: Summary, Entities, Raw Text)."""
        from src.utils.export import to_excel_bytes
        return to_excel_bytes(result)

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _run_ocr(
        self,
        file_path,
        file_bytes,
        file_type: str,
    ) -> dict:
        """Stage 1–3: ingest → preprocess → OCR."""
        try:
            if file_path is not None:
                return self._ocr.extract_from_file(file_path)
            elif file_bytes is not None:
                return self._ocr.extract_from_bytes(file_bytes, file_type)
            else:
                raise ValueError("Provide either file_path or file_bytes")
        except Exception as exc:
            logger.error("OCR stage failed: %s", exc)
            return {"text": "", "word_data": [], "avg_confidence": 0.0, "page_count": 0}

    def _run_ner_and_classify(
        self, text: str, parallel: bool
    ) -> tuple[list, dict]:
        """Stage 4a + 4b: NER and classification (optionally parallel)."""
        ner_entities: list = []
        classification: dict = {}

        if parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_ner = executor.submit(self._safe_ner, text)
                future_cls = executor.submit(self._safe_classify, text)
                ner_entities = future_ner.result()
                classification = future_cls.result()
        else:
            ner_entities = self._safe_ner(text)
            classification = self._safe_classify(text)

        return ner_entities, classification

    def _safe_ner(self, text: str) -> list:
        """NER with fail-soft: returns regex-only entities if model fails."""
        try:
            return self._ner.predict(text)
        except Exception as exc:
            logger.error("NER failed, falling back to regex only: %s", exc)
            from src.ner.postprocessor import extract_regex_entities, filter_by_confidence
            threshold = self._cfg.get("ner", {}).get("confidence_threshold", 0.70)
            return filter_by_confidence(extract_regex_entities(text), threshold)

    def _safe_classify(self, text: str) -> dict:
        """Classification with fail-soft: returns unknown if model fails."""
        try:
            return self._classifier.predict(text)
        except Exception as exc:
            logger.error("Classification failed, returning unknown: %s", exc)
            return {"category": "unknown", "display": "❓ Unknown", "confidence": 0.0, "all_scores": {}}

    def _build_result(
        self,
        ocr_result: dict,
        text: str,
        entities: list,
        classification: dict,
        language_info: dict,
        processing_time: float,
        confidence_threshold: float,
    ) -> dict:
        """Stage 5: assemble structured output dict."""
        # Group entities by type
        from src.ner.entities import ENTITY_TYPES
        from src.ner.postprocessor import deduplicate_entities

        entities = deduplicate_entities(entities)
        entities_by_type: dict[str, list] = {etype: [] for etype in ENTITY_TYPES}
        for ent in entities:
            etype = ent.get("type", "")
            if etype in entities_by_type:
                entities_by_type[etype].append(ent)

        return {
            # Raw OCR output
            "ocr_text": text,
            "ocr_confidence": round(ocr_result.get("avg_confidence", 0.0), 1),
            "page_count": ocr_result.get("page_count", 1),
            "word_data": ocr_result.get("word_data", []),

            # Language detection
            "language": language_info,

            # Named entities (flat list + grouped by type)
            "entities": entities,
            "entities_by_type": entities_by_type,
            "entity_count": len(entities),

            # Document classification
            "classification": classification,

            # Processing metadata
            "processing_time_s": processing_time,
            "confidence_threshold": confidence_threshold,
        }
