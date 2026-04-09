"""
OCR Engine — wraps Tesseract for multilingual document extraction.

Supports PDF (via pdf2image) and image files (PNG, JPG, TIFF).
Returns full text, per-word confidence scores, and bounding boxes.
"""
import io
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import pytesseract
from PIL import Image

from src.ocr.preprocessor import ImagePreprocessor
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class OCREngine:
    """Tesseract OCR wrapper supporting PDF and image inputs."""

    def __init__(self, config: dict | None = None):
        cfg = config or get_config()["ocr"]
        self.languages = cfg.get("languages", "hin+eng")
        self.oem = cfg.get("oem", 3)
        self.psm = cfg.get("psm", 6)
        self.confidence_threshold = cfg.get("confidence_threshold", 30)
        self.dpi = cfg.get("dpi", 300)
        self.preprocessor = ImagePreprocessor()

        # Windows: allow TESSERACT_CMD env var to point to binary
        tesseract_cmd = os.getenv("TESSERACT_CMD", "")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self._tesseract_config = (
            f"--oem {self.oem} --psm {self.psm} -l {self.languages}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_file(self, file_path: Union[str, Path]) -> dict:
        """
        Extract text from a file path (PDF or image).

        Returns:
            dict with keys:
                text          — full extracted string
                pages         — list of per-page dicts
                word_data     — list of {word, confidence, bbox}
                avg_confidence — float (0–100)
                page_count    — int
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            images = self._pdf_to_images(file_path)
        else:
            images = [Image.open(file_path).convert("RGB")]

        return self._process_images(images)

    def extract_from_bytes(self, data: bytes, file_type: str) -> dict:
        """
        Extract text from raw bytes.

        Args:
            data      — file bytes
            file_type — "pdf", "png", "jpg", "jpeg", "tiff", "tif"

        Returns:
            Same dict structure as extract_from_file.
        """
        file_type = file_type.lower().lstrip(".")
        if file_type == "pdf":
            images = self._pdf_bytes_to_images(data)
        else:
            images = [Image.open(io.BytesIO(data)).convert("RGB")]

        return self._process_images(images)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pdf_to_images(self, pdf_path: Path) -> list:
        """Convert PDF pages to PIL Images at self.dpi."""
        try:
            from pdf2image import convert_from_path

            poppler_path = os.getenv("POPPLER_PATH") or None
            return convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                fmt="png",
                poppler_path=poppler_path,
            )
        except Exception as exc:
            logger.error("PDF conversion failed: %s", exc)
            raise

    def _pdf_bytes_to_images(self, data: bytes) -> list:
        """Convert PDF bytes to PIL Images at self.dpi."""
        try:
            from pdf2image import convert_from_bytes

            poppler_path = os.getenv("POPPLER_PATH") or None
            return convert_from_bytes(
                data,
                dpi=self.dpi,
                fmt="png",
                poppler_path=poppler_path,
            )
        except Exception as exc:
            logger.error("PDF bytes conversion failed: %s", exc)
            raise

    def _process_images(self, images: list) -> dict:
        """Run preprocessing + Tesseract on each image, aggregate results."""
        cfg = get_config()
        max_pages = cfg.get("pipeline", {}).get("max_pages", 20)
        images = images[:max_pages]

        all_text_parts = []
        all_word_data = []
        pages = []

        for page_idx, pil_image in enumerate(images):
            preprocessed = self.preprocessor.process(pil_image)
            page_result = self._run_tesseract(preprocessed, page_idx)
            all_text_parts.append(page_result["text"])
            all_word_data.extend(page_result["word_data"])
            pages.append(page_result)
            logger.debug(
                "Page %d: %d chars, confidence=%.1f%%",
                page_idx + 1,
                len(page_result["text"]),
                page_result["avg_confidence"],
            )

        full_text = "\n\n".join(all_text_parts)
        confidences = [w["confidence"] for w in all_word_data if w["confidence"] >= 0]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            "text": full_text,
            "pages": pages,
            "word_data": all_word_data,
            "avg_confidence": avg_conf,
            "page_count": len(images),
        }

    def _run_tesseract(self, image: Image.Image, page_index: int) -> dict:
        """Run Tesseract on a single preprocessed PIL image."""
        try:
            data = pytesseract.image_to_data(
                image,
                config=self._tesseract_config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            logger.error("Tesseract failed on page %d: %s", page_index + 1, exc)
            return {
                "text": "",
                "word_data": [],
                "avg_confidence": 0.0,
                "page": page_index + 1,
            }

        words = []
        text_parts = []
        for i, word in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if conf < 0:
                # Tesseract returns -1 for non-word rows (blocks/paragraphs)
                continue
            word = word.strip()
            if not word:
                continue
            if conf < self.confidence_threshold:
                continue

            bbox = {
                "x": data["left"][i],
                "y": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
                "page": page_index + 1,
            }
            words.append({"word": word, "confidence": conf, "bbox": bbox})
            text_parts.append(word)

        # Reconstruct text preserving line structure from Tesseract block_num / line_num
        full_text = pytesseract.image_to_string(
            image, config=self._tesseract_config
        ).strip()

        confidences = [w["confidence"] for w in words]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            "text": full_text,
            "word_data": words,
            "avg_confidence": avg_conf,
            "page": page_index + 1,
        }
