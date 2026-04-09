# Data Collection Guide

## Overview of Data Needs

| Task | Minimum Data | What We Use |
|------|-------------|-------------|
| NER training | 8,000+ sentences | WikiANN + CoNLL-2003 + Faker synthetic |
| Classification training | 2,000+ documents | Faker synthetic (250 per category × 8 categories) |
| NER validation | 10% of train | Auto-split |
| NER test (accuracy proof) | 10% of train | Auto-split |

The generate_synthetic_data.py script automates everything. Manual collection is only needed if you want to push accuracy above 95%.

---

## Public Datasets (Downloaded Automatically)

### 1. WikiANN Hindi — NER
- **Source**: HuggingFace `datasets.load_dataset("wikiann", "hi")`
- **Size**: ~10,000 sentences
- **Labels**: PER, ORG, LOC (mapped to our PERSON, ORG, LOCATION)
- **Language**: Hindi (Devanagari script)
- **Download command**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("wikiann", "hi")
  ```

### 2. CoNLL-2003 English — NER
- **Source**: HuggingFace `datasets.load_dataset("conll2003")`
- **Size**: ~14,000 sentences (train split)
- **Labels**: PER, ORG, LOC, MISC (MISC mapped to ORG)
- **Language**: English
- **Download command**:
  ```python
  from datasets import load_dataset
  ds = load_dataset("conll2003")
  ```

**Note**: Both datasets use IOB2 format but with different label names. The generate script maps them to our 17-label scheme automatically.

---

## Synthetic Data Generation

### Why Synthetic Data?

Public datasets (WikiANN, CoNLL-2003) don't have:
- AMOUNT entities (₹50,000, Rs. 1,23,456)
- ID_NUMBER entities (PAN cards, Aadhaar, GST numbers)
- PHONE entities (+91 Indian mobile format)
- EMAIL entities
- Mixed Hindi-English sentences

Faker-generated data fills these gaps.

### Generation Process (Step by Step)

```bash
python scripts/generate_synthetic_data.py --ner_samples 5000 --cls_samples 2000
```

**Step 1**: Choose a random template from 40 templates (Hindi/English/mixed)

```
Template: "{PERSON} ने {ORG} को {AMOUNT} का भुगतान {DATE} को किया।"
```

**Step 2**: Fill placeholders with Faker-generated values

```
PERSON   → Faker("hi_IN").name()  → "राजेश कुमार"
ORG      → Faker("en_IN").company() → "TechSoft Solutions"
AMOUNT   → random.choice(["₹", "Rs. "]) + "{N:,}" → "₹50,000"
DATE     → date.strftime("%d/%m/%Y") → "15/03/2024"
```

**Step 3**: Tokenize and assign IOB2 labels

```
Token       Label
─────────── ──────────
राजेश       B-PERSON
कुमार       I-PERSON
ने          O
TechSoft    B-ORG
Solutions   I-ORG
को          O
₹50,000     B-AMOUNT
का          O
...
```

**Step 4**: Download WikiANN Hindi and CoNLL-2003 from HuggingFace and merge

**Step 5**: Shuffle and split 80/10/10 → train/val/test

### Data Format Examples

**CoNLL/IOB2 format** (NER training data):
```
राजेश B-PERSON
कुमार I-PERSON
ने O
TechSoft B-ORG
Solutions I-ORG
को O
₹50,000 B-AMOUNT
का O
भुगतान O
किया O
। O

Ram B-PERSON
Kumar I-PERSON
paid O
₹50,000 B-AMOUNT
to O
Acme B-ORG
Corp I-ORG
. O
```

Sentences are separated by blank lines.

**CSV format** (Classification training data):
```csv
text,label
"Tax Invoice No. INV-12345 GSTIN: 22AAAAA0000A1Z5 Date: 15/03/2024 ...",invoice
"Payment Receipt Received from: Ram Kumar Amount: ₹50,000 ...",receipt
"भारत सरकार आधार कार्ड नाम: राजेश कुमार जन्म तिथि: 15/03/1990 ...",id_card
```

---

## Manual Annotation Guide

For higher accuracy (95%+), supplement synthetic data with manually annotated real documents.

### Recommended Tool: Label Studio

```bash
pip install label-studio
label-studio start
# Open http://localhost:8080
```

**Setup for NER**:
1. Create new project → NLP → Named Entity Recognition
2. Configure label set: PERSON, ORG, LOCATION, DATE, AMOUNT, ID_NUMBER, PHONE, EMAIL
3. Upload your documents (or paste OCR text)
4. Annotate by selecting text → choosing label
5. Export as CoNLL format

**Annotation Guidelines**:
- PERSON: full names only ("राम कुमार" not just "राम")
- ORG: company names, government departments, banks
- LOCATION: cities, states, addresses (not countries alone unless specific)
- AMOUNT: include currency symbol and all digits ("₹50,000" not just "50,000")
- ID_NUMBER: include the full identifier (all digits/letters)
- DATE: full date string as written in document
- Do NOT annotate partial/truncated text

---

## How to Combine Sources and Split

```python
# Already handled by generate_synthetic_data.py
# Manual combination:

import random
from pathlib import Path

def merge_and_split(sources, train_ratio=0.8, val_ratio=0.1):
    combined = []
    for source in sources:
        combined.extend(source)
    random.shuffle(combined)
    n = len(combined)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return combined[:t], combined[t:t+v], combined[t+v:]
```

---

## Pre-Training Checklist

Before running `train_ner.py`:

- [ ] `data/processed/ner_train.conll` exists and has content
- [ ] `data/processed/ner_val.conll` exists and has content
- [ ] Sentences are separated by blank lines
- [ ] Each line is `TOKEN LABEL` (space-separated)
- [ ] Labels use IOB2 scheme (O, B-PERSON, I-PERSON, ...)
- [ ] Training set has at least 500 examples per entity type

Before running `train_classifier.py`:

- [ ] `data/processed/cls_train.csv` exists
- [ ] CSV has columns: `text`, `label`
- [ ] All 8 categories present in label column
- [ ] At least 200 examples per category
- [ ] No null values in text or label columns

Check with:
```python
import pandas as pd
df = pd.read_csv("data/processed/cls_train.csv")
print(df["label"].value_counts())
print(df.isnull().sum())
```
