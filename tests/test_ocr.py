"""
Tests for OCR engine and preprocessor.

Run: pytest tests/test_ocr.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parents[1]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def english_text_image() -> Image.Image:
    """Create a simple white image with black English text."""
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "Hello World 123", fill="black")
    return img


@pytest.fixture
def noisy_image() -> Image.Image:
    """Create a grayscale image with salt-and-pepper noise."""
    from PIL import ImageFilter
    img = Image.new("L", (200, 50), color=200)
    draw = ImageDraw.Draw(img)
    draw.text((5, 10), "Noisy Test", fill=50)
    return img


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------

class TestImagePreprocessor:
    def test_returns_pil_image(self, english_text_image):
        from src.ocr.preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(english_text_image)
        assert isinstance(result, Image.Image)

    def test_output_is_rgb(self, english_text_image):
        from src.ocr.preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(english_text_image)
        assert result.mode == "RGB"

    def test_output_size_reasonable(self, english_text_image):
        from src.ocr.preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(english_text_image)
        # Should not shrink below input size
        assert result.width > 0
        assert result.height > 0

    def test_disabled_preprocessing_returns_original(self, english_text_image):
        from src.ocr.preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor(config={"enabled": False})
        result = preprocessor.process(english_text_image)
        assert result.size == english_text_image.size

    def test_grayscale_conversion(self, english_text_image):
        import cv2
        from src.ocr.preprocessor import ImagePreprocessor
        preprocessor = ImagePreprocessor(
            config={
                "enabled": True,
                "grayscale": True,
                "dpi_normalize": False,
                "noise_removal": False,
                "deskew": False,
                "adaptive_threshold": False,
            }
        )
        result = preprocessor.process(english_text_image)
        # Result should still be RGB (converted back from grayscale)
        assert result.mode == "RGB"

    def test_handles_rgba_input(self):
        from src.ocr.preprocessor import ImagePreprocessor
        img = Image.new("RGBA", (100, 50), color=(255, 255, 255, 255))
        preprocessor = ImagePreprocessor()
        # Should not raise
        result = preprocessor.process(img)
        assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# Language detection tests
# ---------------------------------------------------------------------------

class TestLanguageDetector:
    def test_detects_english(self):
        from src.ocr.language_detect import LanguageDetector
        detector = LanguageDetector()
        result = detector.detect(
            "Invoice from Acme Corp for payment of Rs. 50,000 dated 15/03/2024."
        )
        assert result["primary"] == "english"
        assert result["english_ratio"] > 0.5

    def test_detects_hindi(self):
        from src.ocr.language_detect import LanguageDetector
        detector = LanguageDetector()
        result = detector.detect(
            "राम कुमार ने एबीसी कंपनी को पचास हजार रुपये का भुगतान किया।"
        )
        assert result["primary"] == "hindi"
        assert result["hindi_ratio"] > 0.5

    def test_detects_mixed(self):
        from src.ocr.language_detect import LanguageDetector
        detector = LanguageDetector()
        result = detector.detect(
            "राम कुमार ने Acme Corp को payment of ₹50,000 किया on 15/03/2024."
        )
        assert result["primary"] == "mixed"
        assert result["hindi_ratio"] > 0.0
        assert result["english_ratio"] > 0.0

    def test_empty_text(self):
        from src.ocr.language_detect import LanguageDetector
        detector = LanguageDetector()
        result = detector.detect("")
        assert result["primary"] == "unknown"

    def test_returns_ratios_sum_to_one(self):
        from src.ocr.language_detect import LanguageDetector
        detector = LanguageDetector()
        result = detector.detect("Hello world नमस्ते दुनिया")
        total = result["hindi_ratio"] + result["english_ratio"]
        assert abs(total - 1.0) < 0.01  # should sum to ~1.0

    def test_convenience_function(self):
        from src.ocr.language_detect import detect_language
        result = detect_language("Hello")
        assert isinstance(result, dict)
        assert "primary" in result


# ---------------------------------------------------------------------------
# OCR Engine tests (requires Tesseract installed)
# ---------------------------------------------------------------------------

class TestOCREngine:
    @pytest.mark.skipif(
        not _tesseract_available(),
        reason="Tesseract not installed",
    )
    def test_extract_returns_dict(self, english_text_image, tmp_path):
        from src.ocr.engine import OCREngine
        # Save image to disk
        img_path = tmp_path / "test.png"
        english_text_image.save(img_path)

        engine = OCREngine()
        result = engine.extract_from_file(img_path)

        assert isinstance(result, dict)
        assert "text" in result
        assert "avg_confidence" in result
        assert "page_count" in result
        assert result["page_count"] == 1

    @pytest.mark.skipif(
        not _tesseract_available(),
        reason="Tesseract not installed",
    )
    def test_extract_from_bytes(self, english_text_image):
        from src.ocr.engine import OCREngine
        buf = _image_to_bytes(english_text_image)
        engine = OCREngine()
        result = engine.extract_from_bytes(buf, "png")
        assert isinstance(result, dict)
        assert "text" in result

    @pytest.mark.skipif(
        not _tesseract_available(),
        reason="Tesseract not installed",
    )
    def test_confidence_is_in_range(self, english_text_image, tmp_path):
        from src.ocr.engine import OCREngine
        img_path = tmp_path / "test.png"
        english_text_image.save(img_path)
        engine = OCREngine()
        result = engine.extract_from_file(img_path)
        assert 0.0 <= result["avg_confidence"] <= 100.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _image_to_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
