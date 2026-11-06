"""
Scoring for CORD receipts: three scalar amounts and a menu (list of items).

Amounts are compared by their digits only: CORD prints Indonesian amounts like "60.000" (sixty thousand), where the
dot is a thousands separator, so a numeric parser that reads it as 60.0 would be wrong. Two amounts are equal when
their digit strings are equal.

Menu items are matched one-to-one (greedy, best first). An item matches when its price digits are equal and its name
is close (rapidfuzz token_sort_ratio >= `name_threshold`, case-insensitive); the quantity is not required to match,
because CORD's quantity strings are inconsistent ("2", "2x", "x2"). Precision = matched / predicted items,
recall = matched / true items.
"""

import re

from rapidfuzz import fuzz

SCALARS = ("total_price", "subtotal_price", "tax_price")


def digits(s) -> str:
    return re.sub(r"\D", "", s or "")


def scalar_correct(truth, pred) -> bool:
    """Both absent counts as correct (the receipt has no such field and the model did not invent one)."""
    return digits(truth) == digits(pred) and (bool(truth) == bool(pred) or digits(truth) == digits(pred) == "")


def score_menu(truth: list[dict], pred: list[dict], name_threshold: float = 80.0) -> dict:
    """Match predicted menu items to true ones one-to-one: same price digits and a fuzzy name score >= name_threshold."""
    cands = []
    for i, t in enumerate(truth):
        for j, p in enumerate(pred):
            if digits(t.get("price")) != digits(p.get("price")) or not digits(t.get("price")):
                continue
            sc = fuzz.token_sort_ratio((t.get("name") or "").lower(), (p.get("name") or "").lower())
            if sc >= name_threshold:
                cands.append((sc, i, j))
    used_t, used_p, matched = set(), set(), 0
    for _, i, j in sorted(cands, reverse=True):
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        matched += 1
    return {"matched": matched, "n_true": len(truth), "n_pred": len(pred)}


def menu_prf(rows: list[dict]) -> dict:
    """Micro-averaged precision, recall and F1 over the per-receipt counts from score_menu."""
    m = sum(r["matched"] for r in rows)
    nt, npred = sum(r["n_true"] for r in rows), sum(r["n_pred"] for r in rows)
    p, r_ = (m / npred if npred else 0.0), (m / nt if nt else 0.0)
    return {
        "precision": p,
        "recall": r_,
        "f1": (2 * p * r_ / (p + r_)) if p + r_ else 0.0,
        "n_true_items": nt,
        "n_pred_items": npred,
    }
