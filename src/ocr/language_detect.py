"""
Language Detection — identifies Hindi, English, or mixed text.

Two-level detection:
  Level 1: Unicode range check (Devanagari U+0900–U+097F) — fast, character-level
  Level 2: langdetect library for paragraph-level statistical detection

Returns: "hindi", "english", or "mixed" with percentage breakdown.
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Unicode range for Devanagari script (used for Hindi)
_DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
# Basic Latin + Latin Extended A/B (English alphabet)
_LATIN_RANGE = re.compile(r"[A-Za-z]")

# Thresholds for classification
_HINDI_DOMINANT_THRESHOLD = 0.70   # >70% Devanagari chars → Hindi
_ENGLISH_DOMINANT_THRESHOLD = 0.70  # >70% Latin chars → English


class LanguageDetector:
    """Detects language of OCR-extracted text (Hindi, English, or mixed)."""

    def detect(self, text: str) -> dict:
        """
        Detect the primary language and character ratios.

        Args:
            text: Extracted OCR text string

        Returns:
            dict:
                primary       — "hindi", "english", or "mixed"
                hindi_ratio   — float 0.0–1.0
                english_ratio — float 0.0–1.0
                method        — "unicode" or "langdetect"
        """
        if not text or not text.strip():
            return {
                "primary": "unknown",
                "hindi_ratio": 0.0,
                "english_ratio": 0.0,
                "method": "none",
            }

        # Level 1: Character-level Unicode analysis
        result = self._unicode_detect(text)

        # Level 2: If unicode inconclusive (both < 50%), use langdetect
        if result["primary"] == "mixed" and len(text) > 50:
            langdetect_result = self._langdetect_classify(text)
            if langdetect_result:
                result["langdetect_primary"] = langdetect_result
                result["method"] = "unicode+langdetect"

        return result

    def _unicode_detect(self, text: str) -> dict:
        """Level 1: Count Devanagari vs Latin characters."""
        # Count only script characters (ignore digits, punctuation, spaces)
        devanagari_chars = len(_DEVANAGARI_RANGE.findall(text))
        latin_chars = len(_LATIN_RANGE.findall(text))
        total_script_chars = devanagari_chars + latin_chars

        if total_script_chars == 0:
            return {
                "primary": "unknown",
                "hindi_ratio": 0.0,
                "english_ratio": 0.0,
                "method": "unicode",
            }

        hindi_ratio = devanagari_chars / total_script_chars
        english_ratio = latin_chars / total_script_chars

        if hindi_ratio >= _HINDI_DOMINANT_THRESHOLD:
            primary = "hindi"
        elif english_ratio >= _ENGLISH_DOMINANT_THRESHOLD:
            primary = "english"
        else:
            primary = "mixed"

        return {
            "primary": primary,
            "hindi_ratio": round(hindi_ratio, 3),
            "english_ratio": round(english_ratio, 3),
            "method": "unicode",
        }

    def _langdetect_classify(self, text: str) -> Optional[str]:
        """
        Level 2: Use the langdetect library for statistical language detection.
        Falls back gracefully if langdetect is unavailable or fails.
        """
        try:
            from langdetect import detect, LangDetectException

            lang_code = detect(text[:2000])  # Limit for speed
            if lang_code == "hi":
                return "hindi"
            elif lang_code in ("en",):
                return "english"
            else:
                return lang_code
        except Exception as exc:
            logger.debug("langdetect failed: %s", exc)
            return None


def detect_language(text: str) -> dict:
    """Module-level convenience function."""
    return LanguageDetector().detect(text)
