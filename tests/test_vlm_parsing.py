"""The VLM module's pure functions: prompts, JSON parsing, token-probability confidences."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.extract.vlm import SCHEMAS, build_prompt, field_confidences, parse_json, pieces_and_logprobs  # noqa: E402

RECEIPT = "scanned_receipt"


@pytest.mark.parametrize("doc_type", ["scanned_receipt", "cord_receipt"])
def test_prompt_names_every_field_of_the_schema(doc_type):
    p = build_prompt(doc_type)
    for f in SCHEMAS[doc_type]:
        assert f'"{f}"' in p
    assert "null" in p and "JSON" in p


def test_prompt_disambiguates_the_total_from_a_subtotal():
    assert "not a subtotal" in build_prompt(RECEIPT)


def test_parse_plain_json_keeps_only_schema_keys_as_strings():
    out = parse_json(
        '{"company": "ABC SDN BHD", "date": "01/02/2019", "address": null, "total": 12.5, "extra": "x"}', RECEIPT
    )
    assert out == {"company": "ABC SDN BHD", "date": "01/02/2019", "address": None, "total": "12.5"}


def test_parse_json_inside_markdown_fence_and_prose():
    txt = 'Sure! ```json\n{"company": "ABC", "date": "01/02/2019", "address": "1 Jalan X", "total": "9.50"}\n``` Hope it helps.'
    assert parse_json(txt, RECEIPT)["total"] == "9.50"


def test_parse_truncated_json_recovers_complete_pairs():
    out = parse_json('{"company": "ABC", "date": "01/02/2019", "address": "1 Jal', RECEIPT)
    assert out["company"] == "ABC" and out["date"] == "01/02/2019"


def test_parse_garbage_gives_all_none():
    assert parse_json("I cannot read this image.", RECEIPT) == dict.fromkeys(SCHEMAS[RECEIPT])


def test_blank_and_null_strings_become_none():
    out = parse_json('{"company": "", "date": "null", "address": " ", "total": "5.00"}', RECEIPT)
    assert out == {"company": None, "date": None, "address": None, "total": "5.00"}


def test_field_confidence_uses_only_the_tokens_inside_the_value():
    text = '{"company": "Amit", "date": "0102"}'
    pieces = ['{"', "company", '":', ' "', "Am", "it", '",', ' "', "date", '":', ' "', "01", "02", '"}']
    assert "".join(pieces) == text
    lp = [-0.01] * len(pieces)
    lp[11] = math.log(0.2)  # an unsure digit token inside the date's value
    conf = field_confidences(text, pieces, lp, RECEIPT)
    assert conf["company"]["min_prob"] == pytest.approx(math.exp(-0.01))
    assert conf["date"]["min_prob"] == pytest.approx(0.2)
    assert conf["date"]["mean_prob"] < conf["company"]["mean_prob"]
    assert conf["company"]["n_tokens"] == 2 and conf["date"]["n_tokens"] == 2
    assert "total" not in conf  # not generated: no confidence


def test_a_token_spanning_the_value_boundary_counts_for_the_value():
    text = '{"date": "0102"}'
    pieces = ['{"', "date", '":', ' "0', "102", '"}']  # ' "0' straddles the opening quote
    assert "".join(pieces) == text
    conf = field_confidences(text, pieces, [-0.1, -0.1, -0.1, math.log(0.5), -0.1, -0.1], RECEIPT)
    assert conf["date"]["min_prob"] == pytest.approx(0.5)


# --- byte-level token alignment (non-Latin scripts are split across several tokens) -------------
def _byte_tokens(text: str, sizes: list[int]):
    """A fake byte-level tokenizer: the UTF-8 bytes of `text` cut into chunks of the given sizes."""
    data, chunks, i = text.encode("utf-8"), [], 0
    for s in sizes:
        chunks.append(data[i : i + s])
        i += s
    chunks.append(data[i:])
    chunks = [c for c in chunks if c]
    decode = lambda ids: b"".join(chunks[j] for j in ids).decode("utf-8", errors="replace")  # noqa: E731
    return decode, list(range(len(chunks)))


def test_non_latin_text_split_across_byte_tokens_is_realigned_to_characters():
    text = '{"company": "राजेश"}'
    decode, ids = _byte_tokens(text, [2, 5, 3, 7, 2, 4, 2])  # cuts inside multi-byte characters
    assert "�" in "".join(decode([i]) for i in ids)  # the naive per-token decode really is broken
    pieces, lps = pieces_and_logprobs(decode, ids, [-0.1] * len(ids))
    assert "".join(pieces) == text and len(pieces) == len(lps)


def test_byte_fragment_log_probs_are_carried_to_the_completing_token():
    text = "अ"  # 3 UTF-8 bytes
    decode, ids = _byte_tokens(text, [1, 1])  # 3 tokens, one character
    pieces, lps = pieces_and_logprobs(decode, ids, [math.log(0.5), math.log(0.5), math.log(0.8)])
    assert pieces == ["अ"] and lps[0] == pytest.approx(math.log(0.5 * 0.5 * 0.8))


def test_confidence_for_a_non_latin_value_reflects_its_byte_tokens():
    text = '{"company": "राजेश", "date": "0102"}'
    decode, ids = _byte_tokens(text, [3, 4, 5, 2, 6, 3, 4, 1, 2, 5, 3])
    pieces, plp = pieces_and_logprobs(decode, ids, [-0.01] * len(ids))
    assert "".join(pieces) == text
    plp[next(i for i, p in enumerate(pieces) if "0" in p and i > 5)] = math.log(0.1)
    conf = field_confidences(text, pieces, plp, RECEIPT)
    assert conf["date"]["min_prob"] == pytest.approx(0.1) and conf["company"]["min_prob"] > 0.5


def test_prompts_are_frozen_because_the_adapters_were_trained_on_them():
    """Changing a prompt silently invalidates every reported number: a change here must be a deliberate retraining."""
    import hashlib

    digest = {dt: hashlib.sha256(build_prompt(dt).encode()).hexdigest()[:12] for dt in SCHEMAS}
    assert digest == {"scanned_receipt": "83a132102f1f", "cord_receipt": "3bb84974cd20", "xfund_form": "f36900496ac5"}
