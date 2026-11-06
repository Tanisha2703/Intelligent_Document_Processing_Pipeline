"""The real-receipt additions: the scanned_receipt VLM schema and the scoring used on SROIE labels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.eval import receipts as bv  # noqa: E402
from src.extract.vlm import build_prompt, parse_json  # noqa: E402


def test_receipt_prompt_asks_for_exactly_the_four_fields():
    p = build_prompt("scanned_receipt")
    for f in ("company", "date", "address", "total"):
        assert f'"{f}"' in p
    assert "shop receipt" in p


def test_receipt_json_is_parsed_with_fences_and_nulls():
    out = parse_json(
        '```json\n{"company": "ABC SDN BHD", "date": "01/02/2019", "address": null, "total": "12.50"}\n```',
        "scanned_receipt",
    )
    assert out == {"company": "ABC SDN BHD", "date": "01/02/2019", "address": None, "total": "12.50"}


def test_amounts_and_dates_compare_after_normalisation():
    assert bv.correct("total", "1,193.00", "1193.00")
    assert not bv.correct("total", "193.00", "198.00")
    assert bv.correct("date", "15/01/2019", "15-01-2019")


def test_missing_prediction_is_never_correct_even_near_exact():
    assert not bv.correct("company", "ABC", None, near=True)
    assert not bv.correct("company", "ABC", "", near=True)


def test_near_exact_tolerates_two_characters_for_text_but_not_for_numbers():
    assert not bv.correct("address", "JALAN B1750 MASAI", "JALAN 81750 MASAI")
    assert bv.correct("address", "JALAN B1750 MASAI", "JALAN 81750 MASAI", near=True)
    assert not bv.correct("address", "JALAN BAYU", "TAMAN SERI", near=True)
    assert not bv.correct("total", "193.00", "193.01", near=True)


def test_cord_prompt_and_parse_with_menu():
    assert "menu" in build_prompt("cord_receipt")
    out = parse_json(
        '{"total_price": "60.000", "subtotal_price": null, "tax_price": "5.455", '
        '"menu": [{"name": "-TICKET CP", "quantity": "2", "price": "60.000"}]}',
        "cord_receipt",
    )
    assert out["total_price"] == "60.000" and out["subtotal_price"] is None and out["tax_price"] == "5.455"
    assert out["menu"] == [{"name": "-TICKET CP", "quantity": "2", "price": "60.000"}]


def test_cord_parse_salvages_a_truncated_menu():
    txt = '{"total_price": "9.000", "subtotal_price": null, "tax_price": null, "menu": [{"name": "A", "quantity": "1", "price": "4.000"}, {"name": "B", "quant'
    out = parse_json(txt, "cord_receipt")
    assert out["total_price"] == "9.000"
    assert out["menu"] == [{"name": "A", "quantity": "1", "price": "4.000"}]


def test_cord_parse_of_garbage_is_empty_not_a_crash():
    out = parse_json("I cannot read this.", "cord_receipt")
    assert out == {"total_price": None, "subtotal_price": None, "tax_price": None, "menu": []}
