"""
NER Post-Processor — Regex extraction + entity merge + deduplication.

Regex patterns cover structured fields that the model may miss:
  AMOUNT:    ₹ amounts, Rs., INR, रु.
  ID_NUMBER: PAN, Aadhaar, GST numbers
  PHONE:     Indian mobile numbers (+91 format)
  EMAIL:     Standard email format
  DATE:      DD/MM/YYYY and variants, plus Hindi month names

Merge logic:
  Same span + same type   → boost confidence
  Overlap + different type → keep higher confidence
  No overlap              → keep both
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex Patterns
# ---------------------------------------------------------------------------
_PATTERNS: dict[str, list[re.Pattern]] = {
    "AMOUNT": [
        # ₹1,23,456  ₹50000.50
        re.compile(r"₹\s*[\d,]+(?:\.\d{1,2})?", re.UNICODE),
        # Rs. 50,000  Rs 1000
        re.compile(r"Rs\.?\s*[\d,]+(?:\.\d{1,2})?", re.UNICODE),
        # INR 50000
        re.compile(r"INR\s*[\d,]+(?:\.\d{1,2})?", re.UNICODE),
        # रु. 5000  रुपये 1000
        re.compile(r"रु\.?\s*[\d,]+(?:\.\d{1,2})?", re.UNICODE),
    ],
    "ID_NUMBER": [
        # PAN card: ABCDE1234F
        re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
        # Aadhaar: 1234 5678 9012 (12 digits in groups of 4)
        re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
        # GST: 22AAAAA0000A1Z5
        re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z\d][A-Z]\d\b"),
        # Invoice numbers: INV-12345, BILL/001/2024
        re.compile(r"\b(?:INV|BILL|RCP|REF|TXN)[-/]\w{3,10}\b", re.IGNORECASE),
    ],
    "PHONE": [
        # +91 9876543210 | +91-9876543210
        re.compile(r"\+91[-\s]?[6-9]\d{9}\b"),
        # 09876543210 (leading 0)
        re.compile(r"\b0[6-9]\d{9}\b"),
        # 9876543210 (bare 10 digits starting with 6-9)
        re.compile(r"\b[6-9]\d{9}\b"),
    ],
    "EMAIL": [
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ],
    "DATE": [
        # DD/MM/YYYY | DD-MM-YYYY | DD.MM.YYYY
        re.compile(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b"),
        # YYYY-MM-DD (ISO)
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
        # DD Mon YYYY | DD Month YYYY
        re.compile(
            r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
            r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
            r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b",
            re.IGNORECASE,
        ),
        # Hindi months: जनवरी, फरवरी, etc.
        re.compile(
            r"\b\d{1,2}\s+(?:जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|"
            r"अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)\s+\d{4}\b",
            re.UNICODE,
        ),
    ],
}

# Regex source confidence (slightly lower than fine-tuned model)
_REGEX_CONFIDENCE = 0.88


def extract_regex_entities(text: str) -> list[dict]:
    """
    Run all regex patterns over text and return matched entities.

    Returns:
        List of entity dicts:
            {type, value, start, end, confidence, source}
    """
    entities = []
    for entity_type, patterns in _PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                entities.append(
                    {
                        "type": entity_type,
                        "value": match.group(0).strip(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": _REGEX_CONFIDENCE,
                        "source": "regex",
                    }
                )
    return entities


def merge_entities(
    model_entities: list[dict],
    regex_entities: list[dict],
    confidence_threshold: float = 0.70,
) -> list[dict]:
    """
    Merge model-predicted and regex-extracted entities.

    Rules:
      1. Same span + same type  → keep one, boost confidence by averaging
      2. Overlap + diff type    → keep the one with higher confidence
      3. No overlap             → keep both
    """
    all_entities = list(model_entities) + list(regex_entities)

    # Sort by start position for overlap detection
    all_entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))

    merged: list[dict] = []
    for entity in all_entities:
        if not merged:
            merged.append(dict(entity))
            continue

        last = merged[-1]
        overlap = entity["start"] < last["end"]

        if not overlap:
            merged.append(dict(entity))
            continue

        # Overlapping entities
        same_type = entity["type"] == last["type"]

        if same_type:
            # Boost confidence (average) and extend span to cover both
            last["confidence"] = min(
                1.0,
                (last["confidence"] + entity["confidence"]) / 2 + 0.05,
            )
            last["end"] = max(last["end"], entity["end"])
            # Prefer the longer value
            if len(entity.get("value", "")) > len(last.get("value", "")):
                last["value"] = entity["value"]
            if entity.get("source") == "regex" and last.get("source") != "regex":
                last["source"] = "model+regex"
        else:
            # Different types — keep the higher-confidence one
            if entity["confidence"] > last["confidence"]:
                merged[-1] = dict(entity)

    return filter_by_confidence(merged, confidence_threshold)


def filter_by_confidence(
    entities: list[dict], threshold: float = 0.70
) -> list[dict]:
    """Remove entities below confidence threshold."""
    return [e for e in entities if e.get("confidence", 0) >= threshold]


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """
    Remove duplicate entities (same value + same type).
    Keeps the one with highest confidence.
    """
    seen: dict[tuple, dict] = {}
    for entity in entities:
        key = (entity["type"], entity["value"].strip().lower())
        if key not in seen or entity["confidence"] > seen[key]["confidence"]:
            seen[key] = entity
    return list(seen.values())
