# Fine-Tuning Guide

## Prerequisites

### Packages
```bash
pip install -r requirements.txt
```

### Hardware
| Setup | NER Training Time | Classifier Training Time |
|-------|------------------|------------------------|
| GPU (T4/V100) | ~30–60 minutes | ~20–40 minutes |
| CPU only | 4–10 hours | 3–6 hours |
| Google Colab (free GPU) | ~45 minutes | ~30 minutes |

### Disk Space
- `xlm-roberta-base` model download: ~1.1 GB
- Trained NER model: ~1.1 GB
- Trained classifier: ~1.1 GB
- Training data: ~50 MB

---

## Step 1: Generate Training Data

```bash
python scripts/generate_synthetic_data.py --ner_samples 5000 --cls_samples 2000
```

Expected output:
```
2024-03-15 10:00:00 INFO Generating synthetic NER sentences...
100%|████████████████| 5000/5000 [00:45<00:00, 111.2it/s]
2024-03-15 10:00:45 INFO Generated 5000 synthetic NER sentences
2024-03-15 10:00:45 INFO Loading WikiANN Hindi dataset...
2024-03-15 10:01:30 INFO Loaded 2000 WikiANN Hindi sentences
2024-03-15 10:01:30 INFO Loading CoNLL-2003 English dataset...
2024-03-15 10:02:00 INFO Loaded 2000 CoNLL-2003 sentences
2024-03-15 10:02:00 INFO Total NER sentences: 9000
2024-03-15 10:02:00 INFO NER data saved — train:7200 val:900 test:900
2024-03-15 10:02:00 INFO Generating classification data...
100%|████████████████| 8/8 [00:05<00:00,  1.6it/s]
2024-03-15 10:02:05 INFO Classification data saved — train:1600 val:200 test:200
```

---

## Step 2: Fine-Tune NER Model

```bash
python scripts/train_ner.py
```

### Hyperparameters Explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `base_model` | xlm-roberta-base | Best multilingual accuracy for Hindi+English |
| `learning_rate` | 2e-5 | Standard for BERT-family fine-tuning; lower → no forgetting, higher → unstable |
| `batch_size` | 16 | Fits in 8GB GPU VRAM; larger → more stable gradients |
| `epochs` | 10 (max) | With early stopping at patience=3, usually stops at 5–7 |
| `max_length` | 256 | NER is local — 256 tokens covers most sentences |
| `warmup_ratio` | 0.1 | First 10% of steps use linear LR warmup — prevents early gradient explosion |
| `weight_decay` | 0.01 | L2 regularization — prevents overfitting on synthetic data |
| `fp16` | True (GPU) | Half-precision training: 2× speed, 2× memory savings |
| `early_stopping_patience` | 3 | Stop if val F1 doesn't improve for 3 eval steps |
| `metric_for_best_model` | `overall_f1` | Entity-level F1 (seqeval) — stricter than token-level accuracy |

### Subword Label Alignment (CRITICAL)

XLM-RoBERTa uses sentencepiece tokenization. A single Hindi word like "राजेश" may become multiple subword tokens: ["▁राजेश"] or ["▁राज", "ेश"].

**Rule**: Only the FIRST subword of each word gets the real NER label. All continuation subwords get `-100` (ignored by the loss function).

```
Word:     राजेश          कुमार
Subwords: [▁राज][ेश]    [▁कुम][ार]
Labels:   B-PERSON -100   I-PERSON -100
```

This is implemented in `NERDataset._encode()` using HuggingFace's `word_ids()` method.

### Good Signs During Training

```
Epoch 1: loss=0.8, eval_overall_f1=0.45  ← learning has started
Epoch 3: loss=0.4, eval_overall_f1=0.72  ← rapid improvement
Epoch 5: loss=0.25, eval_overall_f1=0.88 ← good plateau
Epoch 6: loss=0.22, eval_overall_f1=0.89 ← still improving
Epoch 7: loss=0.21, eval_overall_f1=0.89 ← flat = early stop soon
```

### Warning Signs

- `eval_overall_f1` stuck at 0.0 → check label alignment, verify IOB2 format
- Loss increases after epoch 3 → reduce `learning_rate` to 1e-5
- CUDA OOM → reduce `batch_size` to 8 and set `fp16=True`
- All predictions are "O" → check that training data has entity examples

### Expected Output

```
NER fine-tuning complete!
Final metrics:
  overall_precision: 0.924
  overall_recall:    0.938
  overall_f1:        0.931
  per-entity:
    PERSON:    F1=0.94  P=0.93  R=0.95
    ORG:       F1=0.91  P=0.90  R=0.92
    LOCATION:  F1=0.93  P=0.92  R=0.94
    DATE:      F1=0.97  P=0.97  R=0.97
    AMOUNT:    F1=0.98  P=0.97  R=0.99
    ID_NUMBER: F1=0.95  P=0.94  R=0.96
    PHONE:     F1=0.97  P=0.97  R=0.97
    EMAIL:     F1=0.99  P=0.99  R=0.99
```

---

## Step 3: Fine-Tune Classifier

```bash
python scripts/train_classifier.py
```

### Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `learning_rate` | 3e-5 | Slightly higher than NER — classification is simpler, converges faster |
| `epochs` | 8 (max) | Simpler task needs fewer epochs |
| `max_length` | 512 | Classification needs full document context |
| `metric_for_best_model` | `f1_macro` | Unweighted F1 across all 8 categories |

### Expected Output

```
Classifier training complete!
Final metrics:
  f1_macro: 0.929
  
  Per-category classification report:
                    precision  recall  f1-score
  invoice              0.94      0.93      0.94
  receipt              0.92      0.91      0.92
  id_card              0.96      0.97      0.97
  legal_document       0.88      0.90      0.89
  medical_report       0.93      0.92      0.93
  bank_statement       0.95      0.94      0.95
  govt_form            0.90      0.91      0.91
  letter               0.92      0.93      0.93
```

---

## Step 4: Update Config and Run App

After training completes, update `configs/pipeline_config.yaml`:

```yaml
ner:
  use_finetuned: true    # ← change from false

classifier:
  use_finetuned: true    # ← change from false
```

Then launch the app:
```bash
streamlit run app.py
```

---

## Step 5: Evaluate the Full Pipeline

```bash
# Create test documents with ground truth JSON files in tests/eval_docs/
python scripts/evaluate_pipeline.py --test_dir tests/eval_docs/
```

See [scripts/evaluate_pipeline.py](../scripts/evaluate_pipeline.py) for the expected JSON format.

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in configs/pipeline_config.yaml:
training:
  ner:
    batch_size: 8     # was 16
    fp16: true        # must be true with small batch
```

### Low F1 Score (< 0.70)
1. Check data format: `head -20 data/processed/ner_train.conll`
2. Verify blank line separation between sentences
3. Check that labels are IOB2 (O, B-TYPE, I-TYPE — not just TYPE)
4. Increase `ner_samples`: `--ner_samples 10000`

### Training Data Format Issue
```bash
# Validate CoNLL format
python -c "
from scripts.train_ner import read_conll
from pathlib import Path
tokens, labels = read_conll(Path('data/processed/ner_train.conll'))
print(f'Sentences: {len(tokens)}')
print(f'First sentence: {list(zip(tokens[0], labels[0]))}')
"
```

### Google Colab Setup
```python
# In a Colab cell:
!git clone https://github.com/your-username/multilingual-doc-intel.git
%cd multilingual-doc-intel
!pip install -r requirements.txt
!python scripts/generate_synthetic_data.py --ner_samples 5000 --cls_samples 2000
!python scripts/train_ner.py
!python scripts/train_classifier.py

# Download the models to your local machine:
# In Colab Files panel: compress models/ and download
```
