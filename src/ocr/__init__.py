"""OCR sub-package: engine, preprocessor, language detection."""
from .engine import OCREngine
from .preprocessor import ImagePreprocessor
from .language_detect import LanguageDetector

__all__ = ["OCREngine", "ImagePreprocessor", "LanguageDetector"]
