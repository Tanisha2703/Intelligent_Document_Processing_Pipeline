"""Normalisation of printed values: amounts, dates, names, ids."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.normalize import norm_amount, norm_date, norm_id, norm_name, norm_phone, norm_text


@pytest.mark.parametrize(
    "a,b",
    [
        ("₹1,23,456", "123456.00"),
        ("Rs. 50,000/-", "50000.00"),
        ("INR 50000", "50000"),
        ("रु. 5000", "5000.00"),
        ("₹१,२३,४५६.५०", "123456.50"),
    ],
)
def test_amount(a, b):
    assert norm_amount(a) == norm_amount(b)


def test_amount_distinguishes_values():
    assert norm_amount("₹50,000") != norm_amount("₹5,000")


@pytest.mark.parametrize(
    "s,iso",
    [
        ("15/03/2024", "2024-03-15"),
        ("15-03-2024", "2024-03-15"),
        ("15.03.2024", "2024-03-15"),
        ("2024-03-15", "2024-03-15"),
        ("15 March 2024", "2024-03-15"),
        ("15 मार्च 2024", "2024-03-15"),
        ("5 Sep 2024", "2024-09-05"),
        ("१५/०३/२०२४", "2024-03-15"),
    ],
)
def test_date(s, iso):
    assert norm_date(s) == iso


def test_date_is_day_first():
    assert norm_date("03/04/2024") == "2024-04-03"


def test_id_phone_name():
    assert norm_id("1234 5678 9012") == norm_id("123456789012")
    assert norm_id("inv-48213") == "INV48213"
    assert norm_phone("+91 98765 43210") == norm_phone("09876543210") == "9876543210"
    assert norm_name("श्री  राजेश कुमार") == norm_name("राजेश कुमार")
    assert norm_text("a‍b   c") == "ab c"
