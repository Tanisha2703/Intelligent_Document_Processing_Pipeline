from src.eval.xfund import match_pairs, norm, prf
from src.extract.vlm import build_prompt, field_confidences, parse_json


def test_norm_ignores_case_punctuation_and_replacement_characters():
    assert norm("Universit�t: Wuppertal ") == norm("universitt wuppertal")


def test_exact_and_lenient_matching_one_to_one():
    t = [{"key": "Nom", "value": "Dupont"}, {"key": "Ville", "value": "Genève"}]
    p = [{"key": "nom:", "value": "Dupont"}, {"key": "Ville", "value": "Geneve"}, {"key": "Ville", "value": "Geneve"}]
    r = match_pairs(t, p)
    assert r == {"n_true": 2, "n_pred": 3, "exact": 1, "lenient": 2}
    f = prf([r])
    assert abs(f["precision"] - 2 / 3) < 1e-9 and f["recall"] == 1.0


def test_a_different_word_is_not_forgiven():
    assert match_pairs([{"key": "Name", "value": "Maria"}], [{"key": "Name", "value": "Marcos"}])["lenient"] == 0


def test_cjk_pairs_match_exactly():
    t = [{"key": "姓名", "value": "张伟"}]
    assert match_pairs(t, [{"key": "姓名", "value": "张伟"}])["exact"] == 1


def test_xfund_prompt_and_parser_including_truncation():
    assert '"pairs"' in build_prompt("xfund_form")
    ok = parse_json('{"pairs": [{"key": "A", "value": "1"}, {"key": "B", "value": ""}]}', "xfund_form")
    assert ok == {"pairs": [{"key": "A", "value": "1"}]}
    cut = parse_json('{"pairs": [{"key": "A", "value": "1"}, {"key": "B", "val', "xfund_form")
    assert cut == {"pairs": [{"key": "A", "value": "1"}]}
    assert parse_json("no json here", "xfund_form") == {"pairs": []}
    assert field_confidences("{}", ["{}"], [-0.1], "xfund_form") == {}
