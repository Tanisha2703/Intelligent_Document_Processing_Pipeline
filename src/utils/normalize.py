"""
Field-value normalization.

Used by the evaluator (so "₹50,000/-" and "Rs. 50000" count as the same amount)
and by the extractor (so exported values are canonical).
"""

import re
import unicodedata
from decimal import Decimal, InvalidOperation

_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿"))  # ZWSP, ZWNJ, ZWJ, BOM
_MONTHS = {
    m: i + 1
    for i, names in enumerate(
        [
            ("january", "jan", "जनवरी"),
            ("february", "feb", "फरवरी"),
            ("march", "mar", "मार्च"),
            ("april", "apr", "अप्रैल"),
            ("may", "मई"),
            ("june", "jun", "जून"),
            ("july", "jul", "जुलाई"),
            ("august", "aug", "अगस्त"),
            ("september", "sep", "sept", "सितंबर"),
            ("october", "oct", "अक्टूबर"),
            ("november", "nov", "नवंबर"),
            ("december", "dec", "दिसंबर"),
        ]
    )
    for m in names
}


def norm_text(s: str) -> str:
    """NFC, drop zero-width characters, ASCII digits, collapse whitespace."""
    s = unicodedata.normalize("NFC", s or "").translate(_INVISIBLE).translate(_DEVA_DIGITS)
    return re.sub(r"\s+", " ", s).strip()


def norm_name(s: str) -> str:
    s = norm_text(s).casefold()
    return re.sub(r"^(shri|smt|mr|mrs|ms|dr|श्री|श्रीमती)\.?\s+", "", s)


def norm_amount(s: str) -> str:
    """'₹1,23,456.50', 'Rs. 123456.5/-', 'INR 123456.50' -> '123456.50'."""
    t = norm_text(s)
    t = re.sub(r"(?i)(₹|rs\.?|inr|रु\.?|रुपये|/-)", " ", t)
    t = t.replace(",", "").strip()
    m = re.search(r"\d+(?:\.\d+)?", t)
    if not m:
        return t
    try:
        return str(Decimal(m.group(0)).quantize(Decimal("0.01")))
    except InvalidOperation:
        return t


def norm_date(s: str) -> str:
    """Day-first numeric dates and 'DD Month YYYY' (English or Hindi month) -> 'YYYY-MM-DD'."""
    t = norm_text(s).casefold()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})", t)
        if m:
            d, mo, y = map(int, m.groups())
            y += 2000 if y < 100 else 0
        else:
            m = re.fullmatch(r"(\d{1,2})\s+(\S+?),?\s+(\d{4})", t)
            if not m or m.group(2) not in _MONTHS:
                return t
            d, mo, y = int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3))
    return f"{y:04d}-{mo:02d}-{d:02d}"


def norm_id(s: str) -> str:
    """Identifiers: alphanumerics only, upper-case ('1234 5678 9012' == '123456789012')."""
    return re.sub(r"[^0-9A-Za-z]", "", norm_text(s)).upper()


def norm_phone(s: str) -> str:
    digits = re.sub(r"\D", "", norm_text(s))
    return digits[-10:] if len(digits) >= 10 else digits


def norm_email(s: str) -> str:
    return norm_text(s).casefold()


_BY_ENTITY = {
    "PERSON": norm_name,
    "ORG": norm_name,
    "LOCATION": norm_name,
    "AMOUNT": norm_amount,
    "DATE": norm_date,
    "ID_NUMBER": norm_id,
    "PHONE": norm_phone,
    "EMAIL": norm_email,
}


def normalize_value(entity_type: str, value: str) -> str:
    """Canonical form of a value of the given entity type."""
    return _BY_ENTITY.get(entity_type, norm_text)(value)
