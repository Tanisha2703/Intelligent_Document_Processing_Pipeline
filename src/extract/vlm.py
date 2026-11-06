"""
Vision-language-model extraction: prompts, JSON parsing and per-field confidence for receipts and forms.

No torch import here, so it is unit-testable anywhere; the model runs in benchmarks/modal_run.py.

Confidence: the model emits JSON token by token. For each field we find the character span of its value in
the generated text, map it to the tokens covering that span, and use the probability the model gave those
tokens (geometric mean, and the minimum). A VLM that is unsure of a digit assigns it low probability, which is
the signal that is calibrated into a probability; the text alone carries no such signal.

The prompt wording is part of the trained adapters' contract: the models were fine-tuned and evaluated with exactly this
text, so changing it would change their behaviour.
"""

import json
import math
import re

# Document types: SROIE receipts, CORD receipts (scalar fields only; the menu is a list handled by _parse_cord)
# and XFUND forms (a list of key-value pairs handled by _parse_xfund, no per-field confidences).
SCHEMAS = {
    "scanned_receipt": {"company": "ORG", "date": "DATE", "address": "TEXT", "total": "AMOUNT"},
    "cord_receipt": {"total_price": "AMOUNT", "subtotal_price": "AMOUNT", "tax_price": "AMOUNT"},
    "xfund_form": {},
}


def _schema(doc_type: str) -> dict:
    return SCHEMAS[doc_type]


DOC_NAMES = {"scanned_receipt": "shop receipt", "cord_receipt": "shop receipt"}

FIELD_HINTS = {
    "total_price": "final total amount paid, as printed",
    "subtotal_price": "subtotal before tax, as printed, or null",
    "tax_price": "tax amount, as printed, or null",
    "company": "name of the shop or company that issued the receipt",
    "address": "street address of the shop, as printed",
    "total": "final total amount paid (the grand total, not a subtotal or a tax line)",
    "date": "document date",
}


CORD_PROMPT = (
    "This is a scanned shop receipt. Extract its contents and answer with ONLY a JSON object with exactly these keys:\n"
    "{\n"
    '  "total_price": final total amount paid, as printed,\n'
    '  "subtotal_price": subtotal before tax, as printed, or null,\n'
    '  "tax_price": tax amount, as printed, or null,\n'
    '  "menu": [{"name": item name, "quantity": count as printed, "price": line price as printed}, ...]\n'
    "}\n"
    "Copy every value exactly as printed (keep digits and punctuation such as the thousands separator). "
    "Use null for a field that is not on the receipt and an empty list if there are no items."
)


XFUND_PROMPT = (
    "This is a scanned form. It may be in any language. List every field that has been filled in: for each one give the "
    "field label exactly as printed (key) and the value written or typed for it (value), copying both exactly, in the "
    "original language and script. Answer with ONLY a JSON object of the form "
    '{"pairs": [{"key": "...", "value": "..."}, ...]}. Skip labels that have no value.'
)


def build_prompt(doc_type: str) -> str:
    """The instruction the model sees for a document type. Frozen: the adapters were trained on these exact strings."""
    if doc_type == "cord_receipt":
        return CORD_PROMPT
    if doc_type == "xfund_form":
        return XFUND_PROMPT
    keys = "\n".join(f'  "{f}": {FIELD_HINTS[f]}' for f in _schema(doc_type))
    return (
        f"This is a scanned {DOC_NAMES[doc_type]}. It may be in Hindi, English, or both.\n"
        "Extract the following fields and answer with ONLY a JSON object using exactly these keys:\n"
        f"{{\n{keys}\n}}\n"
        "Copy every value exactly as printed on the document (keep the original script, digits and punctuation). "
        "Use null for a field that is not on the document."
    )


def _parse_cord(text: str) -> dict:
    """CORD answer: scalar fields as strings or None, and `menu` as a list of {name, quantity, price} strings."""
    start = text.find("{")
    body = text[start:] if start >= 0 else ""
    end = body.rfind("}")
    data = {}
    try:
        data = json.loads(body[: end + 1] if end >= 0 else body)
    except json.JSONDecodeError:
        for f in ("total_price", "subtotal_price", "tax_price"):
            mm = re.search(rf'"{f}"\s*:\s*("((?:[^"\\]|\\.)*)"|null|[\d.,]+)', body)
            if mm:
                data[f] = None if mm.group(1) == "null" else (mm.group(2) if mm.group(2) is not None else mm.group(1))
        items = []  # salvage the complete {...} item objects of a truncated menu list
        for m in re.finditer(r"\{[^{}]*\}", body[body.find('"menu"') :] if '"menu"' in body else ""):
            try:
                items.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                continue
        data["menu"] = items
    out = {}
    for f in ("total_price", "subtotal_price", "tax_price"):
        v = data.get(f)
        out[f] = None if v is None or str(v).strip().lower() in ("", "null", "none") else str(v).strip()
    menu = data.get("menu") if isinstance(data.get("menu"), list) else []
    out["menu"] = [
        {k: ("" if it.get(k) is None else str(it.get(k)).strip()) for k in ("name", "quantity", "price")}
        for it in menu
        if isinstance(it, dict)
    ]
    return out


def _parse_xfund(text: str) -> dict:
    """XFUND answer: {"pairs": [{"key", "value"}, ...]}; complete pair objects are salvaged from truncated output."""
    start = text.find("{")
    body = text[start:] if start >= 0 else ""
    end = body.rfind("}")
    try:
        data = json.loads(body[: end + 1] if end >= 0 else body)
        items = data.get("pairs", []) if isinstance(data, dict) else []
    except json.JSONDecodeError:
        items = []
        for m in re.finditer(r"\{[^{}]*\}", body):
            try:
                items.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                continue
    pairs = [
        {"key": str(it.get("key", "")).strip(), "value": str(it.get("value", "")).strip()}
        for it in items
        if isinstance(it, dict) and str(it.get("key", "")).strip() and str(it.get("value", "")).strip()
    ]
    return {"pairs": pairs}


def parse_json(text: str, doc_type: str) -> dict[str, str | None]:
    """The first {...} object in the model output, restricted to the schema's keys, values as strings."""
    if doc_type == "xfund_form":
        return _parse_xfund(text)
    if doc_type == "cord_receipt":
        return _parse_cord(text)
    fields = list(_schema(doc_type))
    start = text.find("{")
    data = {}
    if start >= 0:
        body = text[start:]  # may have no closing brace: generation hit the token limit
        end = body.rfind("}")
        try:
            data = json.loads(body[: end + 1] if end >= 0 else body)
        except json.JSONDecodeError:
            data = {}
            for (
                f
            ) in fields:  # tolerate truncated / slightly invalid JSON: pull complete "key": "value" pairs one by one
                mm = re.search(rf'"{f}"\s*:\s*("((?:[^"\\]|\\.)*)"|null|[\d.]+)', body)
                if mm:
                    data[f] = (
                        None if mm.group(1) == "null" else (mm.group(2) if mm.group(2) is not None else mm.group(1))
                    )
    out = {}
    for f in fields:
        v = data.get(f)
        out[f] = None if v is None or str(v).strip().lower() in ("", "null", "none") else str(v).strip()
    return out


def field_confidences(generated: str, pieces: list[str], logprobs: list[float], doc_type: str) -> dict[str, dict]:
    """
    Per-field {mean_prob, min_prob, n_tokens} from per-token log-probabilities.

    pieces[i] is the text token i decodes to, so ''.join(pieces) == generated (the caller guarantees this);
    logprobs[i] is the log-probability of the token the model emitted at step i.
    """
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos)
        pos += len(p)
    out = {}
    for f in _schema(doc_type):
        m = re.search(rf'"{f}"\s*:\s*"((?:[^"\\]|\\.)*)"', generated)
        if not m:
            continue
        lo, hi = m.span(1)
        idx = [i for i, (s, p) in enumerate(zip(starts, pieces, strict=True)) if s < hi and s + len(p) > lo]
        if not idx:
            continue
        lps = [logprobs[i] for i in idx]
        out[f] = {"mean_prob": math.exp(sum(lps) / len(lps)), "min_prob": math.exp(min(lps)), "n_tokens": len(lps)}
    return out


def pieces_and_logprobs(decode, ids: list[int], logprobs: list[float]) -> tuple[list[str], list[float]]:
    """
    Align generated token ids with text pieces for `field_confidences`.

    Byte-level tokenizers split one Devanagari character across several tokens, and decoding such a token on
    its own gives the replacement character. So pieces are taken from the growing decoded prefix: a token whose
    bytes complete no character adds no text, and its log-probability is carried onto the token that completes
    it (the probability of the whole character is the product of its byte tokens' probabilities).

    decode(list_of_ids) -> str is the tokenizer's decode. Returns (pieces, logprobs), equal length, with
    ''.join(pieces) == decode(ids) minus any incomplete trailing bytes.
    """
    pieces, lps, prev, carry = [], [], "", 0.0
    for i in range(len(ids)):
        stable = decode(ids[: i + 1]).rstrip("�")  # drop an incomplete UTF-8 tail
        carry += logprobs[i]
        piece = stable[len(prev) :]
        if piece:
            pieces.append(piece)
            lps.append(carry)
            carry, prev = 0.0, stable
    if carry and lps:
        lps[-1] += carry
    return pieces, lps
