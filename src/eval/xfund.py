"""
Scoring for XFUND forms: a form is a set of (key, value) pairs (a printed field label and what was filled in next to it).

Pairs are matched one-to-one, best first. Two texts are compared after casefolding, dropping punctuation, the Unicode
replacement character (the source annotations contain some) and extra whitespace.
  exact    both key and value equal after that normalisation
  lenient  both key and value at least `threshold` similar (rapidfuzz ratio) after folding accents, which forgives a
           missed accent or a stray character in a long value but not a different word
Precision = matched / predicted pairs, recall = matched / true pairs.
"""

import re
import unicodedata

from rapidfuzz import fuzz


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold().replace("�", "")
    s = "".join(c for c in s if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", " ", s).strip()


def fold(s: str) -> str:
    """Accent-insensitive form (Latin scripts); CJK characters have no combining marks and are unchanged."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def match_pairs(truth: list[dict], pred: list[dict], threshold: float = 85.0) -> dict:
    """Match predicted (key, value) pairs to true ones one-to-one, exactly and leniently (fuzzy ratio >= threshold)."""
    t = [(norm(p["key"]), norm(p["value"])) for p in truth]
    q = [(norm(p.get("key", "")), norm(p.get("value", ""))) for p in pred]
    res = {"n_true": len(t), "n_pred": len(q)}
    for mode in ("exact", "lenient"):
        cands = []
        for i, (tk, tv) in enumerate(t):
            for j, (qk, qv) in enumerate(q):
                if mode == "exact":
                    if tk == qk and tv == qv:
                        cands.append((100.0, i, j))
                else:
                    sk, sv = fuzz.ratio(fold(tk), fold(qk)), fuzz.ratio(fold(tv), fold(qv))
                    if sk >= threshold and sv >= threshold:
                        cands.append((sk + sv, i, j))
        used_t, used_q, m = set(), set(), 0
        for _, i, j in sorted(cands, reverse=True):
            if i in used_t or j in used_q:
                continue
            used_t.add(i)
            used_q.add(j)
            m += 1
        res[mode] = m
    return res


def prf(rows: list[dict], mode: str = "lenient") -> dict:
    m, nt, nq = sum(r[mode] for r in rows), sum(r["n_true"] for r in rows), sum(r["n_pred"] for r in rows)
    p, r_ = (m / nq if nq else 0.0), (m / nt if nt else 0.0)
    return {"precision": p, "recall": r_, "f1": 2 * p * r_ / (p + r_) if p + r_ else 0.0}
