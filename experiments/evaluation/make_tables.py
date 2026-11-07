"""
Fill the generated tables in docs/RESULTS.md from the raw results in results/raw/ (nothing in those tables is typed by hand).

A table lives between `<!-- BEGIN table:NAME -->` and `<!-- END table:NAME -->` markers; this script rewrites it.

    python experiments/evaluation/make_tables.py
"""

import json
import re
import statistics  # noqa: F401
from pathlib import Path

ROOT = Path(__file__).parents[2]
RAW = ROOT / "results" / "raw"


def _load(name):
    p = RAW / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def md(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = "|" + "|".join({"l": "---", "r": "---:"}[a] for a in align) + "|"
    return "\n".join(["| " + " | ".join(header) + " |", sep] + ["| " + " | ".join(r) + " |" for r in rows])


def _md(header, rows, align=None):
    return md(header, rows, align)


FT_NAMES = {
    "zero_shot_7b": "Qwen2.5-VL-7B, zero-shot",
    "zero_shot_3b": "Qwen2.5-VL-3B, zero-shot",
    "fine_tuned_3b": "**3B, fine-tuned (LoRA, 506 SROIE receipts)**",
    "fine_tuned_7b": "**7B, fine-tuned (LoRA, SROIE + CORD)**",
}


def t_real_vlm() -> str | None:
    d = _load("real_vlm.json")
    if not d:
        return None
    rows = [
        [f, f"{v['exact']:.3f}", f"[{v['exact_ci95'][0]:.3f}, {v['exact_ci95'][1]:.3f}]", f"{v['near_exact']:.3f}"]
        for f, v in d["fields"].items()
    ]
    a = d["all_fields"]
    rows.append(
        [
            "**all four fields**",
            f"**{a['exact']:.3f}**",
            f"[{a['exact_ci95'][0]:.3f}, {a['exact_ci95'][1]:.3f}]",
            f"{a['near_exact']:.3f}",
        ]
    )
    rows.append(
        [
            "receipts with every field right",
            f"{d['receipts_all_fields_exact']:.3f}",
            "",
            f"{d['receipts_all_fields_near_exact']:.3f}",
        ]
    )
    return _md(
        ["Qwen2.5-VL-7B zero-shot, 361 real receipts", "exact", "95% bootstrap", "near-exact (<= 2 characters)"], rows
    )


def t_ft_sroie() -> str | None:
    d = _load("real_compare.json")
    if not d:
        return None
    rows = []
    for k, label in FT_NAMES.items():
        s = d["systems"].get(k)
        if not s:
            continue
        a = s["all"]
        rows.append(
            [
                label,
                f"{a['field_accuracy']:.3f}",
                f"[{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]",
                f"{s['unseen_shops']['field_accuracy']:.3f}",
            ]
            + [f"{a['per_field_exact'][f]:.2f}" for f in ("company", "date", "address", "total")]
            + [f"{a['receipts_all_fields_exact']:.2f}", f"{s['usd_per_1000_pages_at_a10_price']:.2f}"]
        )
    return _md(
        [
            "system (361 real SROIE test receipts)",
            "field accuracy",
            "95% bootstrap",
            "unseen shops",
            "company",
            "date",
            "address",
            "total",
            "receipts all right",
            "USD / 1,000 pages",
        ],
        rows,
        "l" + "r" * 9,
    )


def t_ft_sroie_paired() -> str | None:
    d = _load("real_compare.json")
    if not d:
        return None
    rows = []
    for k, key, label in (
        ("fine_tuned_3b", "paired_vs_zero_shot_7b", "fine-tuned 3B minus zero-shot 7B"),
        (
            "fine_tuned_3b",
            "paired_vs_same_size_zero_shot",
            "fine-tuned 3B minus zero-shot 3B (what fine-tuning alone buys)",
        ),
        ("fine_tuned_7b", "paired_vs_same_size_zero_shot", "fine-tuned 7B minus zero-shot 7B"),
        ("fine_tuned_7b", "paired_vs_fine_tuned_3b", "fine-tuned 7B minus fine-tuned 3B"),
    ):
        for grp in ("all", "unseen_shops"):
            v = d["systems"].get(k, {}).get(key, {}).get(grp)
            if v:
                rows.append(
                    [
                        label,
                        "all 361" if grp == "all" else "135 unseen shops",
                        f"{v['diff']:+.3f}",
                        f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]",
                    ]
                )
    return _md(["paired difference in field accuracy", "receipts", "difference", "95% paired bootstrap"], rows)


def t_ft_calibration() -> str | None:
    d = _load("real_compare.json")
    if not d:
        return None
    rows = []
    for k in ("zero_shot_7b", "fine_tuned_3b", "fine_tuned_7b"):
        c = d["systems"].get(k, {}).get("calibration")
        if c:
            cov = c["coverage_at_precision"]
            rows.append(
                [
                    FT_NAMES[k],
                    f"{c['ece_raw_test']:.3f} -> {c['ece_calibrated_test']:.3f}",
                    f"{c['accept_threshold_chosen_on_dev']:.3f}",
                    f"{c['test_auto_accepted']:.0%}",
                    f"{c['test_precision_of_accepted']:.1%}",
                    f"{c['test_errors_caught_by_review']:.0%}",
                    f"{cov['0.99']:.0%} / {cov['0.98']:.0%} / {cov['0.95']:.0%}",
                ]
            )
    return _md(
        [
            "calibration fitted on 120 dev receipts, applied to test",
            "ECE raw -> calibrated",
            "accept threshold (99% precision on dev)",
            "auto-accepted",
            "precision of those",
            "errors caught by review",
            "coverage at 99 / 98 / 95% precision",
        ],
        rows,
    )


def t_ft_thresholds() -> str | None:
    """Every accept threshold we measured, for both fine-tuned models: how many fields it accepts and what that costs."""
    d = _load("real_compare.json")
    if not d:
        return None
    ops = {
        k: {o["threshold"]: o for o in d["systems"][k]["calibration"]["operating_points"]}
        for k in ("fine_tuned_7b", "fine_tuned_3b")
        if k in d["systems"]
    }
    if len(ops) < 2:
        return None
    rows = []
    for th in sorted(ops["fine_tuned_7b"]):
        a, b = ops["fine_tuned_7b"][th], ops["fine_tuned_3b"][th]
        rows.append(
            [
                f"**{th:.2f} (default)**" if abs(th - 0.8) < 1e-9 else f"{th:.2f}",
                f"{a['test_auto_accepted']:.0%}",
                f"{a['test_precision_of_accepted']:.1%}",
                f"{a['test_wrong_auto_accepted_per_1000_fields']:.0f}",
                f"{a['test_errors_caught_by_review']:.0%}",
                f"{b['test_auto_accepted']:.0%}",
                f"{b['test_precision_of_accepted']:.1%}",
                f"{b['test_wrong_auto_accepted_per_1000_fields']:.0f}",
            ]
        )
    return _md(
        [
            "accept threshold (calibrated probability)",
            "7B: fields auto-accepted",
            "7B: precision of those",
            "7B: wrong fields let through per 1,000",
            "7B: errors caught by review",
            "3B: auto-accepted",
            "3B: precision",
            "3B: wrong per 1,000",
        ],
        rows,
    )


def t_ft_content() -> str | None:
    """How much of each system's score is reading the characters, and how much is matching the labels' formatting?"""
    d = _load("real_compare.json")
    if not d:
        return None
    rows = []
    for k, label in FT_NAMES.items():
        s = d["systems"].get(k)
        if not s or "field_accuracy_content" not in s["all"]:
            continue
        a = s["all"]
        pf, pc = a["per_field_exact"], a["per_field_content"]
        rows.append(
            [
                label,
                f"{a['field_accuracy']:.3f}",
                f"{a['field_accuracy_content']:.3f}",
                f"{pf['company']:.2f} -> {pc['company']:.2f}",
                f"{pf['address']:.2f} -> {pc['address']:.2f}",
            ]
        )
    for k, name in (("fine_tuned_3b", "3B"), ("fine_tuned_7b", "7B")):
        p = d["systems"][k]["paired_vs_same_size_zero_shot"]
        e, c = p["all"], p["all_content_metric"]
        rows.append(
            [
                f"gain from fine-tuning, {name} vs same-size zero-shot",
                f"{e['diff']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]",
                f"{c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]",
                "",
                "",
            ]
        )
    return _md(
        [
            "361 SROIE test receipts",
            "field accuracy, strict (headline)",
            "field accuracy, content only",
            "company: strict -> content",
            "address: strict -> content",
        ],
        rows,
    )


def t_ft_cord() -> str | None:
    d = _load("cord_compare.json")
    o = _load("cord_overlap_check.json")
    if not (d and o):
        return None
    rows = []
    for k, label in (
        ("zero_shot_7b", "Qwen2.5-VL-7B, zero-shot"),
        ("fine_tuned_7b", "**7B, fine-tuned (SROIE + 400 CORD receipts)**"),
    ):
        s, ov = d["systems"].get(k), o["systems"].get(k)
        if s and ov:
            n, a = ov["not_near_duplicate"], ov["all"]
            rows.append(
                [
                    label,
                    f"{a['scalar_accuracy']:.3f}",
                    f"{a['scalar_accuracy_where_field_present']:.3f}",
                    f"{a['menu_f1']:.3f}",
                    f"[{s['menu_f1_ci95'][0]:.3f}, {s['menu_f1_ci95'][1]:.3f}]",
                    f"{n['scalar_accuracy_where_field_present']:.3f}",
                    f"{n['menu_f1']:.3f}",
                    f"{ov['near_duplicate']['menu_f1']:.3f}",
                ]
            )
    return _md(
        [
            "CORD test (100 real receipts)",
            "3 amounts, absent counted",
            "3 amounts, where present",
            "menu F1",
            "menu F1 95% bootstrap",
            "amounts where present, no near-duplicates",
            "menu F1, no near-duplicates",
            "menu F1, near-duplicates only",
        ],
        rows,
        "l" + "r" * 7,
    )


def t_ft_training() -> str | None:
    rows = []
    for tag, label in (("ft3b", "3B"), ("ft7b", "7B")):
        d = _load(f"ft_{tag}.json")
        if d:
            dev = [h["dev_loss"] for h in d["history"] if "dev_loss" in h]
            rows.append(
                [
                    label,
                    d["gpu"].replace("NVIDIA ", ""),
                    f"{d['train_examples']}",
                    f"{d['epochs']}",
                    f"{d['trainable_params'] / 1e6:.0f}M of {d['all_params'] / 1e9:.1f}B",
                    f"{d['train_seconds'] / 60:.0f}",
                    f"{d['peak_gpu_gb']:.0f}",
                    " -> ".join(f"{x:.4f}" for x in dev),
                ]
            )
    return _md(
        ["model", "GPU", "train examples", "epochs", "LoRA params", "minutes", "peak GB", "dev loss per epoch"], rows
    )


def t_xfund() -> str | None:
    d = _load("xfund_compare.json")
    if not d or "zero_shot_3b" not in d["systems"] or "fine_tuned_3b" not in d["systems"]:
        return None
    names = {
        "zh": "Chinese",
        "ja": "Japanese",
        "es": "Spanish",
        "fr": "French",
        "it": "Italian",
        "de": "German",
        "pt": "Portuguese",
    }
    zs, ft, pr = d["systems"]["zero_shot_3b"], d["systems"]["fine_tuned_3b"], d["fine_tuned_3b_minus_zero_shot_3b"]
    f = lambda s: f"{s['lenient']['f1']:.2f} [{s['lenient']['f1_ci95'][0]:.2f}, {s['lenient']['f1_ci95'][1]:.2f}]"  # noqa: E731
    g = lambda v: f"{v['mean_per_form_f1_diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"  # noqa: E731
    rows = [
        [
            names[lg],
            str(zs["by_language"][lg]["n_forms"]),
            f(zs["by_language"][lg]),
            f(ft["by_language"][lg]),
            g(pr[lg]),
        ]
        for lg in names
        if lg in zs["by_language"]
    ]
    rows.append(["**all 105 forms**", str(zs["all"]["n_forms"]), f(zs["all"]), f(ft["all"]), g(pr["all"])])
    rows.append(
        ["forms with <= 60 pairs", str(zs["max_60_pairs"]["n_forms"]), f(zs["max_60_pairs"]), f(ft["max_60_pairs"]), ""]
    )
    rows.append(
        [
            "precision / recall, all forms",
            "",
            f"{zs['all']['lenient']['precision']:.2f} / {zs['all']['lenient']['recall']:.2f}",
            f"{ft['all']['lenient']['precision']:.2f} / {ft['all']['lenient']['recall']:.2f}",
            "",
        ]
    )
    return _md(
        [
            "XFUND forms (pair F1, lenient)",
            "forms",
            "3B zero-shot [95% bootstrap]",
            "3B fine-tuned (250 forms) [95%]",
            "paired per-form gain [95%]",
        ],
        rows,
        "lrrrr",
    )


def t_doctype_types() -> str | None:
    d = _load("doctype_compare.json")
    if not d:
        return None
    r = d["rvlcdip_test"]
    rows = [
        [
            "DiT-base-finetuned-rvlcdip, its own 16-class head",
            f"{r['native_dit_accuracy']:.3f}",
            f"[{r['native_ci95'][0]:.3f}, {r['native_ci95'][1]:.3f}]",
        ],
        [
            "**17-class linear probe on its embeddings (adds receipt)**",
            f"**{r['probe_accuracy']:.3f}**",
            f"[{r['probe_ci95'][0]:.3f}, {r['probe_ci95'][1]:.3f}]",
        ],
    ]
    rows.append(["probe: share of RVL-CDIP documents it calls a receipt", f"{r['probe_predicted_receipt']:.3f}", ""])
    return _md(["RVL-CDIP test, 1,600 documents (100 per class)", "accuracy", "95% bootstrap"], rows)


def t_doctype_receipts() -> str | None:
    a, b = _load("doctype_compare_sroie_only.json"), _load("doctype_compare.json")
    if not a or not b:
        return None
    rows = [
        [
            "SROIE test receipts",
            str(a["sroie_test_receipts"]["n"]),
            f"{a['sroie_test_receipts']['probe_predicted_receipt']:.3f}",
            f"{b['sroie_test_receipts']['probe_predicted_receipt']:.3f}",
        ],
        [
            "CORD test receipts (a different format)",
            str(a["cord_test_receipts_unseen_format"]["n"]),
            f"{a['cord_test_receipts_unseen_format']['probe_predicted_receipt']:.3f} (never seen)",
            f"{b['cord_test_receipts']['probe_predicted_receipt']:.3f}",
        ],
        [
            "XFUND forms called a receipt (should be 0)",
            str(a["xfund_test_forms"]["n"]),
            f"{a['xfund_test_forms']['wrongly_routed_to_receipt_extractor']:.3f}",
            f"{b['xfund_test_forms']['wrongly_routed_to_receipt_extractor']:.3f}",
        ],
        [
            "XFUND forms sent to the form extractor",
            str(a["xfund_test_forms"]["n"]),
            f"{a['xfund_test_forms']['routed_to_form_extractor']:.3f}",
            f"{b['xfund_test_forms']['routed_to_form_extractor']:.3f}",
        ],
    ]
    return _md(
        [
            "share of documents",
            "n",
            "probe trained on SROIE receipts only",
            "probe trained on SROIE + CORD receipts (shipped)",
        ],
        rows,
    )


def t_doctype_routing() -> str | None:
    d = _load("doctype_compare.json")
    if not d:
        return None
    r = d["routing"]
    rows = [
        [k, str(r[k]["routed"]), f"{r[k]['precision']:.3f}", f"{r[k]['recall']:.3f}"]
        for k in ("receipt", "form", "human")
    ]
    rows.append(["receipts sent to the form extractor", str(r["receipts_sent_to_the_wrong_extractor"]), "", ""])
    for s, v in r["by_source"].items():
        rows.append(
            [
                f"{s} ({v['n']}): routed correctly / sent to a person",
                "",
                f"{v['routing_correct']:.3f}",
                f"{v['to_human']:.3f}",
            ]
        )
    return _md(
        [
            f"route (accept threshold {r['accept_threshold']:.2f}, {r['n']} test documents)",
            "routed",
            "precision",
            "recall",
        ],
        rows,
    )


TABLES = {
    fn.__name__[2:]: fn
    for fn in (
        t_real_vlm,
        t_ft_training,
        t_ft_sroie,
        t_ft_sroie_paired,
        t_ft_calibration,
        t_ft_thresholds,
        t_ft_content,
        t_ft_cord,
        t_xfund,
        t_doctype_types,
        t_doctype_receipts,
        t_doctype_routing,
    )
}


def main():
    path = ROOT / "docs" / "RESULTS.md"
    text = path.read_text(encoding="utf-8")
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- BEGIN table:{re.escape(name)} -->\n).*?(<!-- END table:{re.escape(name)} -->)", re.S)
        if not pat.search(text):
            continue
        table = fn()
        if table is None:
            print("skipped (no data):", name)
            continue
        text = pat.sub(lambda m, table=table: m.group(1) + table + "\n" + m.group(2), text)
        print("filled", name)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
