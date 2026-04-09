"""
Download pretrained fallback models from HuggingFace Hub.
Also verifies Tesseract and Hindi language pack are installed.

Usage:
    python scripts/download_pretrained.py
"""
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def check_tesseract() -> bool:
    """Verify Tesseract OCR is installed and Hindi language data is available."""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        logger.info("✅ Tesseract version: %s", version)
    except Exception as exc:
        logger.error("❌ Tesseract not found: %s", exc)
        logger.error("   Install: sudo apt-get install tesseract-ocr tesseract-ocr-hin")
        return False

    try:
        import pytesseract
        langs = pytesseract.get_languages(config="")
        if "hin" in langs:
            logger.info("✅ Hindi language data (hin) available")
        else:
            logger.error("❌ Hindi language data not found. Available: %s", langs)
            logger.error("   Install: sudo apt-get install tesseract-ocr-hin")
            return False
        if "eng" in langs:
            logger.info("✅ English language data (eng) available")
    except Exception as exc:
        logger.warning("Could not verify language packs: %s", exc)

    return True


def check_poppler() -> bool:
    """Verify Poppler (pdfinfo) is installed for PDF processing."""
    try:
        result = subprocess.run(["pdfinfo", "--version"], capture_output=True, text=True)
        version = (result.stdout or result.stderr).strip().split("\n")[0]
        logger.info("✅ Poppler: %s", version)
        return True
    except FileNotFoundError:
        logger.warning("⚠️  Poppler (pdfinfo) not found — PDF processing will not work")
        logger.warning("   Linux: sudo apt-get install poppler-utils")
        logger.warning("   macOS: brew install poppler")
        logger.warning("   Windows: Download from https://github.com/oschwartz10612/poppler-windows")
        return False


def download_ner_fallback() -> None:
    """Download Davlan/xlm-roberta-base-ner-hrl (NER fallback)."""
    model_name = "Davlan/xlm-roberta-base-ner-hrl"
    logger.info("Downloading NER fallback model: %s", model_name)
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        logger.info("✅ NER fallback model downloaded successfully")
    except Exception as exc:
        logger.error("❌ Failed to download NER fallback: %s", exc)


def download_classifier_fallback() -> None:
    """Download facebook/bart-large-mnli (zero-shot classification fallback)."""
    model_name = "facebook/bart-large-mnli"
    logger.info("Downloading classifier fallback model: %s (this is large ~1.6GB)", model_name)
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("✅ Classifier fallback model downloaded successfully")
    except Exception as exc:
        logger.error("❌ Failed to download classifier fallback: %s", exc)


def main():
    logger.info("=" * 50)
    logger.info("Checking system dependencies...")
    logger.info("=" * 50)

    tess_ok = check_tesseract()
    popl_ok = check_poppler()

    logger.info("=" * 50)
    logger.info("Downloading pretrained fallback models...")
    logger.info("(Models are cached in HuggingFace cache — only downloaded once)")
    logger.info("=" * 50)

    download_ner_fallback()
    download_classifier_fallback()

    logger.info("=" * 50)
    if tess_ok:
        logger.info("✅ All checks passed. Ready to run: streamlit run app.py")
    else:
        logger.warning("⚠️  Fix Tesseract installation before running the pipeline")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
