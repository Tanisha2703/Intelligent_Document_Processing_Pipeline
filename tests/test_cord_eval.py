from src.eval.cord import menu_prf, scalar_correct, score_menu


def test_indonesian_thousands_separator_is_not_a_decimal():
    assert scalar_correct("60.000", "60000")
    assert not scalar_correct("60.000", "60.00")  # 6000 vs 60000


def test_absent_field_predicted_absent_is_correct_but_invented_is_not():
    assert scalar_correct(None, None)
    assert not scalar_correct(None, "5.455")
    assert not scalar_correct("5.455", None)


def test_menu_matches_one_to_one_and_needs_the_price():
    t = [
        {"name": "ES TEH", "quantity": "1", "price": "5.000"},
        {"name": "NASI GORENG", "quantity": "2", "price": "30.000"},
    ]
    p = [
        {"name": "Es Teh", "quantity": "", "price": "5.000"},
        {"name": "NASI GORENG", "quantity": "1", "price": "20.000"},
    ]
    r = score_menu(t, p)
    assert r == {"matched": 1, "n_true": 2, "n_pred": 2}


def test_duplicate_predictions_cannot_match_one_true_item_twice():
    t = [{"name": "TEA", "quantity": "1", "price": "5.000"}]
    p = [{"name": "TEA", "quantity": "1", "price": "5.000"}] * 2
    assert score_menu(t, p)["matched"] == 1
    f = menu_prf([score_menu(t, p)])
    assert f["precision"] == 0.5 and f["recall"] == 1.0


def test_empty_menus_do_not_divide_by_zero():
    assert menu_prf([score_menu([], [])])["f1"] == 0.0
