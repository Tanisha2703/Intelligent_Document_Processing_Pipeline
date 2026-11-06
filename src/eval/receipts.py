"""Scoring for SROIE receipts: company, date, address, total.

A field is correct when its normalised value equals the label's (dates and amounts via src/utils/normalize.py).
SROIE's own labels contain typos (a printed "81750" labelled "B1750"), so text fields can also be scored
"near exact": within 2 characters of edit distance after normalisation. Both are reported.
"""

from rapidfuzz.distance import Levenshtein

from src.utils.normalize import norm_amount, norm_date, norm_text

ETYPE = {"company": "text", "address": "text", "date": "date", "total": "amount"}
NORM = {"text": norm_text, "date": norm_date, "amount": norm_amount}


def correct(field: str, truth: str, pred: str | None, near: bool = False) -> bool:
    """Is `pred` right for `field` after normalising both (dates and amounts by value, text by case and spacing)?
    With near=True a text field within 2 edits also counts; a missing prediction is never right.
    """
    if not pred:
        return False
    n = NORM[ETYPE[field]]
    a, b = n(truth), n(pred)
    if a == b:
        return True
    return near and ETYPE[field] == "text" and Levenshtein.distance(a, b) <= 2
