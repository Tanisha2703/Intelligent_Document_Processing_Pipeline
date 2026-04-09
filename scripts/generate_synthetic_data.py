"""
Generate Synthetic Training Data for NER and Classification Models.

Usage:
    python scripts/generate_synthetic_data.py --ner_samples 5000 --cls_samples 2000

Outputs:
    data/processed/ner_train.conll   (80%)
    data/processed/ner_val.conll     (10%)
    data/processed/ner_test.conll    (10%)
    data/processed/cls_train.csv     (80%)
    data/processed/cls_val.csv       (10%)
    data/processed/cls_test.csv      (10%)

Data sources:
  1. WikiANN Hindi dataset (via HuggingFace datasets)
  2. CoNLL-2003 English dataset (via HuggingFace datasets)
  3. Faker-generated synthetic sentences (Hindi, English, mixed templates)
"""
import argparse
import logging
import random
import re
import sys
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
from faker import Faker
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity value generators
# ---------------------------------------------------------------------------
_fake_hi = Faker("hi_IN")
_fake_en = Faker("en_IN")


def _random_amount() -> str:
    n = random.randint(100, 999999)
    n_fmt = f"{n:,}"
    prefix = random.choice(["₹", "Rs. ", "Rs ", "INR ", "रु. "])
    return prefix + n_fmt


def _random_date() -> str:
    from datetime import date, timedelta
    base = date(2015, 1, 1)
    delta = random.randint(0, 3650)
    d = base + timedelta(days=delta)
    fmt = random.choice([
        d.strftime("%d/%m/%Y"),
        d.strftime("%d-%m-%Y"),
        d.strftime("%d %B %Y"),
        d.strftime("%Y-%m-%d"),
    ])
    return fmt


def _random_pan() -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(random.choices(letters, k=5))
        + "".join(random.choices("0123456789", k=4))
        + random.choice(letters)
    )


def _random_aadhaar() -> str:
    return " ".join(
        ["".join(random.choices("0123456789", k=4)) for _ in range(3)]
    )


def _random_invoice_no() -> str:
    prefix = random.choice(["INV", "BILL", "RCP", "REF"])
    num = random.randint(10000, 99999)
    return f"{prefix}-{num}"


def _random_id() -> str:
    return random.choice([_random_pan, _random_aadhaar, _random_invoice_no])()


def _random_phone() -> str:
    digits = "".join(random.choices("6789", k=1) + random.choices("0123456789", k=9))
    fmt = random.choice([f"+91 {digits}", f"+91-{digits}", digits, f"0{digits}"])
    return fmt


def _random_person() -> str:
    return random.choice([_fake_hi.name, _fake_en.name])()


def _random_org() -> str:
    return random.choice([_fake_hi.company, _fake_en.company])()


def _random_location() -> str:
    return random.choice([_fake_hi.city, _fake_en.city])()


def _random_email() -> str:
    return _fake_en.email()


# Placeholder → generator mapping
_GENERATORS = {
    "PERSON": _random_person,
    "ORG": _random_org,
    "LOCATION": _random_location,
    "AMOUNT": _random_amount,
    "DATE": _random_date,
    "ID_NUMBER": _random_id,
    "PHONE": _random_phone,
    "EMAIL": _random_email,
}

# ---------------------------------------------------------------------------
# Sentence templates
# ---------------------------------------------------------------------------
_HINDI_TEMPLATES = [
    "{PERSON} ने {ORG} को {AMOUNT} का भुगतान {DATE} को किया।",
    "{PERSON} का पता {LOCATION} है और फोन नंबर {PHONE} है।",
    "दिनांक {DATE} को {PERSON} ने {ORG} में आवेदन दिया।",
    "श्री {PERSON} का पैन नंबर {ID_NUMBER} है।",
    "{PERSON} का ईमेल {EMAIL} और मोबाइल {PHONE} है।",
    "{ORG} द्वारा {PERSON} को {AMOUNT} का बिल जारी किया गया।",
    "{PERSON} निवासी {LOCATION} ने {ORG} में {AMOUNT} जमा किया।",
    "आधार संख्या {ID_NUMBER} धारक {PERSON} का खाता {ORG} में है।",
    "चालान संख्या {ID_NUMBER} दिनांक {DATE} को {ORG} द्वारा जारी।",
    "प्रिय {PERSON}, आपका बकाया {AMOUNT} है, कृपया {DATE} तक भुगतान करें।",
    "{DATE} को {LOCATION} में {ORG} की बैठक हुई।",
    "{PERSON} को {ORG} ने {LOCATION} में नियुक्त किया।",
    "{PERSON} का आधार {ID_NUMBER} है और ईमेल {EMAIL} है।",
    "{ORG} ने {DATE} को {PERSON} को {AMOUNT} की राशि दी।",
    "{PERSON} जी {LOCATION} से हैं और उनका फोन {PHONE} है।",
]

_ENGLISH_TEMPLATES = [
    "{PERSON} paid {AMOUNT} to {ORG} on {DATE}.",
    "The office of {ORG} is located at {LOCATION}.",
    "Contact {PERSON} at {EMAIL} or {PHONE}.",
    "{PERSON} with PAN {ID_NUMBER} filed return on {DATE}.",
    "Invoice from {ORG} for {AMOUNT} dated {DATE}.",
    "{PERSON} residing at {LOCATION} holds account at {ORG}.",
    "Dear {PERSON}, your outstanding balance is {AMOUNT}.",
    "Receipt No. {ID_NUMBER}: {PERSON} paid {AMOUNT} on {DATE}.",
    "Aadhaar No. {ID_NUMBER} belongs to {PERSON} from {LOCATION}.",
    "{PERSON} sent {AMOUNT} to {ORG} via NEFT on {DATE}.",
    "Dr. {PERSON} from {ORG}, {LOCATION}, prescribed on {DATE}.",
    "{PERSON}'s phone: {PHONE}, email: {EMAIL}.",
    "{ORG} issued challan {ID_NUMBER} for {AMOUNT} on {DATE}.",
    "Payment of {AMOUNT} received from {PERSON} on {DATE} at {LOCATION}.",
    "{PERSON} joined {ORG} on {DATE} and works at {LOCATION}.",
]

_MIXED_TEMPLATES = [
    "{PERSON} ने {ORG} को payment of {AMOUNT} किया on {DATE}.",
    "{PERSON} ka PAN number {ID_NUMBER} hai, email: {EMAIL}.",
    "Invoice {ID_NUMBER} from {ORG}: total {AMOUNT}, date {DATE}.",
    "{PERSON} ji ka phone {PHONE} hai, {LOCATION} mein rehte hain.",
    "Dear {PERSON}, {AMOUNT} ka bill {DATE} ko {ORG} ne bheja.",
    "{ORG} is located in {LOCATION} aur {DATE} ko founded hua.",
    "{PERSON} ne {AMOUNT} transfer kiya {ORG} ko on {DATE}.",
]

_ALL_TEMPLATES = _HINDI_TEMPLATES + _ENGLISH_TEMPLATES + _MIXED_TEMPLATES


# ---------------------------------------------------------------------------
# IOB2 label generation
# ---------------------------------------------------------------------------

def _fill_template(template: str) -> tuple[str, dict[str, str]]:
    """Fill a template with random entity values. Returns (text, {placeholder: value})."""
    placeholders = re.findall(r"\{(\w+)\}", template)
    values = {}
    for ph in placeholders:
        if ph in _GENERATORS:
            values[ph] = _GENERATORS[ph]()

    text = template
    for ph, val in values.items():
        text = text.replace(f"{{{ph}}}", val, 1)
    return text, values


def _tokenize_simple(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for IOB2 labeling."""
    return re.findall(r'\S+', text)


def _generate_iob2_sentence(template: str) -> list[tuple[str, str]]:
    """
    Generate a list of (word, IOB2_label) pairs from a template.

    For each entity span in the filled text, assign B- to the first token
    and I- to subsequent tokens. Everything else is O.
    """
    filled_text, values = _fill_template(template)
    tokens = _tokenize_simple(filled_text)

    # Find character positions of each entity value in the filled text
    entity_spans: list[tuple[int, int, str]] = []  # (start, end, type)
    search_pos = 0
    for ph, val in values.items():
        idx = filled_text.find(val, search_pos)
        if idx >= 0:
            entity_spans.append((idx, idx + len(val), ph))
            search_pos = idx + len(val)

    # Build char → label mapping
    char_labels = ["O"] * len(filled_text)
    for start, end, etype in entity_spans:
        for i in range(start, end):
            if i == start:
                char_labels[i] = f"B-{etype}"
            else:
                char_labels[i] = f"I-{etype}"

    # Map tokens to labels by checking the first character of each token
    result = []
    pos = 0
    for token in tokens:
        # Find this token's start position in the full text
        tok_start = filled_text.find(token, pos)
        if tok_start < 0:
            result.append((token, "O"))
            continue
        label = char_labels[tok_start]
        result.append((token, label))
        pos = tok_start + len(token)

    return result


# ---------------------------------------------------------------------------
# WikiANN / CoNLL-2003 integration
# ---------------------------------------------------------------------------

def _load_wikiann_hindi(max_samples: int = 2000) -> list[list[tuple[str, str]]]:
    """Download and convert WikiANN Hindi to our IOB2 format."""
    logger.info("Loading WikiANN Hindi dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikiann", "hi", trust_remote_code=True)
        sentences = []
        label_map = {0: "O", 1: "B-PERSON", 2: "I-PERSON", 3: "B-ORG", 4: "I-ORG",
                     5: "B-LOCATION", 6: "I-LOCATION"}
        for split in ["train", "validation"]:
            for example in ds[split]:
                tokens = example["tokens"]
                labels = [label_map.get(t, "O") for t in example["ner_tags"]]
                sentences.append(list(zip(tokens, labels)))
                if len(sentences) >= max_samples:
                    break
            if len(sentences) >= max_samples:
                break
        logger.info("Loaded %d WikiANN Hindi sentences", len(sentences))
        return sentences
    except Exception as exc:
        logger.warning("Could not load WikiANN Hindi: %s", exc)
        return []


def _load_conll2003(max_samples: int = 2000) -> list[list[tuple[str, str]]]:
    """Download and convert CoNLL-2003 English to our IOB2 format."""
    logger.info("Loading CoNLL-2003 English dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("conll2003", trust_remote_code=True)
        # CoNLL tag map: PER→PERSON, LOC→LOCATION, ORG→ORG, MISC→ORG
        raw_labels = ds["train"].features["ner_tags"].feature.names
        # e.g. ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        our_map = {}
        for i, lbl in enumerate(raw_labels):
            if lbl == "O":
                our_map[i] = "O"
            else:
                prefix, tag = lbl.split("-", 1)
                tag = tag.replace("PER", "PERSON").replace("LOC", "LOCATION").replace("MISC", "ORG")
                our_map[i] = f"{prefix}-{tag}"

        sentences = []
        for example in ds["train"]:
            tokens = example["tokens"]
            labels = [our_map.get(t, "O") for t in example["ner_tags"]]
            sentences.append(list(zip(tokens, labels)))
            if len(sentences) >= max_samples:
                break
        logger.info("Loaded %d CoNLL-2003 sentences", len(sentences))
        return sentences
    except Exception as exc:
        logger.warning("Could not load CoNLL-2003: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Classification data
# ---------------------------------------------------------------------------

_CLS_TEMPLATES = {
    "invoice": (
        "Tax Invoice No. {ID_NUMBER} GSTIN: {gstin} Date: {DATE} "
        "Bill To: {PERSON} Address: {LOCATION} "
        "Item Description: Professional Services "
        "HSN Code: 998313 Qty: 1 Rate: {AMOUNT} Amount: {AMOUNT} "
        "CGST 9%: {tax} SGST 9%: {tax} Total Amount: {AMOUNT} "
        "Bank Details: {ORG} A/C {id2} IFSC: HDFC0001234"
    ),
    "receipt": (
        "Payment Receipt Received from: {PERSON} "
        "Amount: {AMOUNT} Date: {DATE} "
        "Mode of Payment: {mode} Reference No: {ID_NUMBER} "
        "For: {ORG} Authorized Signatory: {PERSON2}"
    ),
    "id_card": (
        "भारत सरकार Government of India "
        "आधार कार्ड Aadhaar Card "
        "नाम Name: {PERSON} "
        "जन्म तिथि Date of Birth: {DATE} "
        "लिंग Gender: Male "
        "पता Address: {LOCATION} "
        "आधार संख्या Aadhaar No: {aadhaar} "
    ),
    "legal_document": (
        "AGREEMENT made on {DATE} between {PERSON} (hereinafter 'Party A') "
        "residing at {LOCATION} and {ORG} (hereinafter 'Party B') "
        "WHEREAS Party A agrees to provide services for consideration of {AMOUNT} "
        "This agreement shall be governed by the laws of India. "
        "IN WITNESS WHEREOF the parties have executed this Agreement on {DATE2}."
    ),
    "medical_report": (
        "Laboratory Report Patient Name: {PERSON} "
        "Patient ID: {ID_NUMBER} Date: {DATE} "
        "Referred by: Dr. {PERSON2} Hospital: {ORG} "
        "Test: Complete Blood Count (CBC) "
        "Hemoglobin: 13.5 g/dL WBC: 8000 cells/μL "
        "Platelet Count: 2.5 lakhs/μL "
        "Result: Normal Range Report validated by: {PERSON3}"
    ),
    "bank_statement": (
        "Account Statement Name: {PERSON} "
        "Account No: {ID_NUMBER} IFSC: SBIN0001234 "
        "Branch: {LOCATION} Period: {DATE} to {DATE2} "
        "Opening Balance: {AMOUNT} "
        "Date | Description | Debit | Credit | Balance "
        "{DATE3} | NEFT from {ORG} | | {AMOUNT2} | {AMOUNT3} "
        "Closing Balance: {AMOUNT4}"
    ),
    "govt_form": (
        "Form No. 16 Certificate of Tax Deducted at Source "
        "Name and Address of Deductor: {ORG} TAN: {ID_NUMBER} "
        "Name of Employee: {PERSON} PAN: {pan} "
        "Assessment Year: 2024-25 Period: {DATE} to {DATE2} "
        "Total Tax Deducted: {AMOUNT} Total Tax Deposited: {AMOUNT2}"
    ),
    "letter": (
        "Date: {DATE} To, {PERSON} {ORG} {LOCATION} "
        "Subject: {subject} "
        "Dear Sir/Madam, "
        "With reference to the above subject, we wish to bring to your kind attention "
        "that our company {ORG2} has been experiencing issues regarding payment of {AMOUNT}. "
        "We request you to kindly look into the matter at the earliest. "
        "Thanking you, Yours faithfully, {PERSON2} {ORG2}"
    ),
}

_SUBJECTS = [
    "Non-payment of dues", "Request for refund", "Application for loan",
    "Complaint regarding service", "Request for information",
    "Notice of termination", "Offer letter",
]
_MODES = ["UPI", "NEFT", "RTGS", "Cash", "Cheque", "Demand Draft"]


def _gen_cls_row(category: str) -> str:
    """Generate one fake document text for the given category."""
    tmpl = _CLS_TEMPLATES.get(category, "{PERSON} {ORG} {AMOUNT} {DATE}")

    # Fill placeholders
    vals = {
        "PERSON": _random_person(),
        "PERSON2": _random_person(),
        "PERSON3": _random_person(),
        "ORG": _random_org(),
        "ORG2": _random_org(),
        "LOCATION": _random_location(),
        "AMOUNT": _random_amount(),
        "AMOUNT2": _random_amount(),
        "AMOUNT3": _random_amount(),
        "AMOUNT4": _random_amount(),
        "DATE": _random_date(),
        "DATE2": _random_date(),
        "DATE3": _random_date(),
        "ID_NUMBER": _random_id(),
        "PHONE": _random_phone(),
        "EMAIL": _random_email(),
        "gstin": (
            str(random.randint(10, 35))
            + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
            + "".join(random.choices("0123456789", k=4))
            + "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=1))
            + str(random.randint(1, 9))
            + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            + str(random.randint(1, 9))
        ),
        "aadhaar": _random_aadhaar(),
        "pan": _random_pan(),
        "id2": "".join(random.choices("0123456789", k=11)),
        "tax": _random_amount(),
        "mode": random.choice(_MODES),
        "subject": random.choice(_SUBJECTS),
    }
    text = tmpl
    for k, v in vals.items():
        text = text.replace(f"{{{k}}}", v)
    return text


# ---------------------------------------------------------------------------
# Data splitting and CoNLL writing
# ---------------------------------------------------------------------------

def _split(data: list, train=0.8, val=0.1):
    """Split list into train/val/test."""
    random.shuffle(data)
    n = len(data)
    t = int(n * train)
    v = int(n * val)
    return data[:t], data[t:t+v], data[t+v:]


def _write_conll(sentences: list[list[tuple[str, str]]], path: Path) -> None:
    """Write list of token-label pairs to CoNLL format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for token, label in sentence:
                f.write(f"{token} {label}\n")
            f.write("\n")


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(ner_samples: int = 5000, cls_samples: int = 2000, seed: int = 42) -> None:
    random.seed(seed)
    Faker.seed(seed)

    output_dir = Path(__file__).parents[1] / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- NER DATA --------------------------------------------------------
    logger.info("Generating synthetic NER sentences...")
    synthetic_sentences = []
    for _ in tqdm(range(ner_samples), desc="Synthetic NER"):
        template = random.choice(_ALL_TEMPLATES)
        try:
            pairs = _generate_iob2_sentence(template)
            if pairs:
                synthetic_sentences.append(pairs)
        except Exception:
            pass

    logger.info("Generated %d synthetic NER sentences", len(synthetic_sentences))

    # Load public datasets (optional — gracefully skip if unavailable)
    wikiann = _load_wikiann_hindi(max_samples=2000)
    conll = _load_conll2003(max_samples=2000)

    all_ner = synthetic_sentences + wikiann + conll
    random.shuffle(all_ner)
    logger.info("Total NER sentences: %d", len(all_ner))

    ner_train, ner_val, ner_test = _split(all_ner)
    _write_conll(ner_train, output_dir / "ner_train.conll")
    _write_conll(ner_val, output_dir / "ner_val.conll")
    _write_conll(ner_test, output_dir / "ner_test.conll")
    logger.info("NER data saved — train:%d val:%d test:%d", len(ner_train), len(ner_val), len(ner_test))

    # ---- CLASSIFICATION DATA ---------------------------------------------
    logger.info("Generating classification data...")
    per_category = cls_samples // len(_CLS_TEMPLATES)
    cls_rows = []
    for category in tqdm(_CLS_TEMPLATES, desc="Classification"):
        for _ in range(per_category):
            text = _gen_cls_row(category)
            cls_rows.append({"text": text, "label": category})

    random.shuffle(cls_rows)
    cls_train, cls_val, cls_test = _split(cls_rows)
    _write_csv(cls_train, output_dir / "cls_train.csv")
    _write_csv(cls_val, output_dir / "cls_val.csv")
    _write_csv(cls_test, output_dir / "cls_test.csv")
    logger.info("Classification data saved — train:%d val:%d test:%d",
                len(cls_train), len(cls_val), len(cls_test))

    logger.info("Data generation complete. Files in: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--ner_samples", type=int, default=5000,
                        help="Number of synthetic NER sentences to generate")
    parser.add_argument("--cls_samples", type=int, default=2000,
                        help="Number of classification documents to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(ner_samples=args.ner_samples, cls_samples=args.cls_samples, seed=args.seed)
