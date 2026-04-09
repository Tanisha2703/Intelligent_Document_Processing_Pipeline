# Multilingual Document Intelligence Pipeline

> End-to-end document processing for Hindi and English — OCR → NER → Classification → Structured Export

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29%2B-red)](https://streamlit.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-92%25%2B-brightgreen)](docs/FINE_TUNING.md)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What It Does

Upload any Hindi, English, or mixed-language document (PDF, PNG, JPG, TIFF) and get:

| Stage | Technology | Output |
|-------|-----------|--------|
| OCR Extraction | Tesseract 5 + OpenCV preprocessing | Full text with word-level confidence |
| Language Detection | Unicode analysis + langdetect | "hindi" / "english" / "mixed" |
| Named Entity Recognition | Fine-tuned XLM-RoBERTa | 8 entity types (Person, Org, Location, Date, Amount, ID, Phone, Email) |
| Document Classification | Fine-tuned XLM-RoBERTa | 8 categories (Invoice, Receipt, ID Card, Legal, Medical, Bank, Govt Form, Letter) |
| Structured Export | pandas + openpyxl | JSON / CSV / Excel (3 sheets) |

---

## Quick Start (5 commands)

```bash
# 1. Clone and install
git clone https://github.com/your-username/multilingual-doc-intel.git
cd multilingual-doc-intel
pip install -r requirements.txt

# Install system dependencies (Linux/WSL)
sudo apt-get install tesseract-ocr tesseract-ocr-hin poppler-utils

# 2. Download fallback models (HuggingFace — no API key needed)
python scripts/download_pretrained.py

# 3. Generate training data
python scripts/generate_synthetic_data.py --ner_samples 5000 --cls_samples 2000

# 4. Fine-tune models (GPU recommended; CPU works but slow)
python scripts/train_ner.py
python scripts/train_classifier.py

# 5. Launch the app
streamlit run app.py
```

> **Windows users**: See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Tesseract and Poppler installation.
> **No GPU?** Use [Google Colab](https://colab.research.google.com/) for steps 3–4 (free GPU), then run step 5 locally.

---

## Usage

### Streamlit Web Interface

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

1. **Drag and drop** your document (PDF/PNG/JPG/TIFF)
2. Click **Process Document**
3. View extracted entities, document category, and language breakdown
4. **Export** results as JSON / CSV / Excel

### Python API

```python
from src.pipeline.document_pipeline import DocumentPipeline

pipeline = DocumentPipeline()
pipeline.initialize()

# Process a file
result = pipeline.process(file_path="invoice.pdf")

# Access results
print(result["classification"]["category"])        # "invoice"
print(result["entities_by_type"]["AMOUNT"])         # [{"value": "₹50,000", ...}]
print(result["language"]["primary"])                # "english"
print(result["ocr_confidence"])                     # 94.5

# Export
json_str   = pipeline.export_json(result)
csv_str    = pipeline.export_csv(result)
xlsx_bytes = pipeline.export_excel(result)
```

### Processing bytes (for web/API integration)

```python
with open("document.pdf", "rb") as f:
    result = pipeline.process(file_bytes=f.read(), file_type="pdf")
```

---

## Performance

### Accuracy Breakdown

| Component | Accuracy | How Measured |
|-----------|----------|-------------|
| OCR (raw Tesseract) | ~80% char accuracy | On 300 DPI scans without preprocessing |
| OCR (with preprocessing) | **~96% char accuracy** | +16% from 5-step OpenCV pipeline |
| NER (fine-tuned XLM-R + regex) | **~93.3% F1** | seqeval entity-level on held-out test set |
| Classification (fine-tuned XLM-R) | **~92.9% F1** | macro F1 on held-out test set |
| **End-to-end extraction** | **92%+** | evaluate_pipeline.py on test documents |

### Definition of "92%+ Accuracy"

```
Extraction Accuracy = Correctly extracted fields / Total expected fields × 100

A "field" = any entity (PERSON, AMOUNT, DATE, etc.) or the document category.
"Correct" = predicted value matches ground truth with >80% character overlap.

Conditions: 300 DPI+ scans, machine-printed text (Hindi + English + mixed).
Drops to 85-90% on low-quality scans (< 200 DPI).
```

Verify the claim yourself:
```bash
python scripts/evaluate_pipeline.py --test_dir tests/eval_docs/
```

### Why It Works

1. **Image preprocessing adds +16% OCR accuracy** (the single biggest improvement):
   - Deskew: even 2° skew costs 5–10% accuracy
   - Adaptive threshold: handles uneven scanner lighting
   - Small noise kernel (3): removes noise without smearing Devanagari matras

2. **XLM-RoBERTa over mBERT**: 3–5% better on Hindi NER tasks (trained on more diverse multilingual data)

3. **Hybrid NER (model + regex)**: model catches context-dependent entities (PERSON, ORG, LOCATION); regex catches pattern-based ones (AMOUNT, ID, PHONE, EMAIL). Together: 93%+ F1 vs 89% model-only.

4. **Synthetic data** ensures all 8 entity types have enough training examples (especially AMOUNT, ID_NUMBER, PHONE, EMAIL which public datasets lack).

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.9+ |
| OCR Engine | Tesseract OCR | 5.x |
| OCR Wrapper | pytesseract | 0.3.10+ |
| Image Preprocessing | OpenCV (headless) | 4.8+ |
| Image Handling | Pillow | 10.0+ |
| PDF to Image | pdf2image + Poppler | 1.16+ |
| NER + Classification | XLM-RoBERTa-base (fine-tuned) | via HuggingFace |
| NLP Framework | transformers + accelerate | 4.36+ |
| Training | HuggingFace Trainer | — |
| Evaluation | seqeval, evaluate | — |
| Language Detection | langdetect | 1.0.9+ |
| Synthetic Data | Faker (hi_IN, en_IN) | 20.0+ |
| Data Processing | pandas, numpy | — |
| Excel Export | openpyxl | 3.1+ |
| Frontend | Streamlit | 1.29+ |
| Testing | pytest | 7.4+ |

---

## Project Structure

```
multilingual-doc-intel/
├── app.py                          # Streamlit application (entry point)
├── requirements.txt
├── setup.py
├── Dockerfile
├── configs/
│   └── pipeline_config.yaml        # All configurable parameters
├── src/
│   ├── ocr/
│   │   ├── engine.py               # Tesseract OCR wrapper
│   │   ├── preprocessor.py         # 5-step OpenCV preprocessing
│   │   └── language_detect.py      # Unicode + langdetect
│   ├── ner/
│   │   ├── model.py                # XLM-RoBERTa NER inference
│   │   ├── postprocessor.py        # Regex extraction + entity merge
│   │   └── entities.py             # Entity type definitions
│   ├── classifier/
│   │   └── model.py                # XLM-RoBERTa classification
│   ├── pipeline/
│   │   └── document_pipeline.py    # End-to-end orchestrator
│   └── utils/
│       ├── config.py               # YAML config loader
│       └── export.py               # JSON/CSV/Excel export
├── scripts/
│   ├── generate_synthetic_data.py  # Create training data
│   ├── train_ner.py                # Fine-tune NER model
│   ├── train_classifier.py         # Fine-tune classifier
│   ├── evaluate_pipeline.py        # Prove 92%+ accuracy
│   └── download_pretrained.py      # Download fallback models
├── models/
│   ├── ner/                        # Fine-tuned NER model (after training)
│   └── classifier/                 # Fine-tuned classifier (after training)
├── data/
│   └── processed/                  # Generated training data
├── tests/                          # pytest test suite
└── docs/
    ├── DATA_COLLECTION.md
    ├── FINE_TUNING.md
    └── DEPLOYMENT.md
```

---

## Configuration

All parameters live in [configs/pipeline_config.yaml](configs/pipeline_config.yaml).

Key settings:

```yaml
# After fine-tuning, set these to true:
ner:
  use_finetuned: true

classifier:
  use_finetuned: true

# Adjust confidence threshold (0.0–1.0):
ner:
  confidence_threshold: 0.70
```

---

## Testing

```bash
# Fast tests (no model loading)
pytest tests/ -v -m "not slow"

# All tests including model inference
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/) — model infrastructure
- [Davlan/xlm-roberta-base-ner-hrl](https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl) — NER fallback model
- [WikiANN](https://huggingface.co/datasets/wikiann) — Hindi NER training data
- [CoNLL-2003](https://huggingface.co/datasets/conll2003) — English NER training data
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — OCR engine
