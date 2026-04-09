"""
Entity type definitions for the NER model.

Defines the 8 entity types, IOB2 label scheme (17 labels),
label↔ID mappings, fallback model label mapping, and UI colors.
"""

# --- 8 Entity Types ---
ENTITY_TYPES = [
    "PERSON",
    "ORG",
    "LOCATION",
    "DATE",
    "AMOUNT",
    "ID_NUMBER",
    "PHONE",
    "EMAIL",
]

# --- IOB2 Labels (17 total: O + B-/I- for each of 8 types) ---
IOB2_LABELS = ["O"] + [
    f"{prefix}-{etype}"
    for etype in ENTITY_TYPES
    for prefix in ("B", "I")
]
# Result: ['O', 'B-PERSON', 'I-PERSON', 'B-ORG', 'I-ORG', ...]

NUM_LABELS = len(IOB2_LABELS)  # 17

# --- Label ↔ ID mappings ---
LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(IOB2_LABELS)}
ID2LABEL: dict[int, str] = {idx: label for idx, label in enumerate(IOB2_LABELS)}

# --- Mapping from Davlan/xlm-roberta-base-ner-hrl labels → our labels ---
# Davlan model uses PER, LOC, ORG labels
PRETRAINED_LABEL_MAP: dict[str, str] = {
    "PER": "PERSON",
    "B-PER": "B-PERSON",
    "I-PER": "I-PERSON",
    "LOC": "LOCATION",
    "B-LOC": "B-LOCATION",
    "I-LOC": "I-LOCATION",
    "ORG": "ORG",
    "B-ORG": "B-ORG",
    "I-ORG": "I-ORG",
    "O": "O",
    # Davlan also has MISC — map to ORG as closest
    "MISC": "ORG",
    "B-MISC": "B-ORG",
    "I-MISC": "I-ORG",
}

# --- UI Colors for entity highlighting (hex) ---
ENTITY_COLORS: dict[str, str] = {
    "PERSON":    "#AED6F1",  # Light blue
    "ORG":       "#A9DFBF",  # Light green
    "LOCATION":  "#FAD7A0",  # Light orange
    "DATE":      "#D7BDE2",  # Light purple
    "AMOUNT":    "#F9E79F",  # Light yellow
    "ID_NUMBER": "#FADBD8",  # Light red/pink
    "PHONE":     "#A8E6CF",  # Mint
    "EMAIL":     "#FFDAC1",  # Peach
}

# --- Entity display names for UI ---
ENTITY_DISPLAY_NAMES: dict[str, str] = {
    "PERSON":    "Person",
    "ORG":       "Organisation",
    "LOCATION":  "Location",
    "DATE":      "Date",
    "AMOUNT":    "Amount",
    "ID_NUMBER": "ID / Reference No.",
    "PHONE":     "Phone",
    "EMAIL":     "Email",
}

# --- Entity icons for UI ---
ENTITY_ICONS: dict[str, str] = {
    "PERSON":    "👤",
    "ORG":       "🏢",
    "LOCATION":  "📍",
    "DATE":      "📅",
    "AMOUNT":    "💰",
    "ID_NUMBER": "🔢",
    "PHONE":     "📞",
    "EMAIL":     "✉️",
}
