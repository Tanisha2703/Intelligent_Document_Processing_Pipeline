# Results

Every number below was produced by a script in `benchmarks/` and is stored as raw JSON in `results/raw/`. Tables are filled from those files by `benchmarks/make_tables.py`; figures are redrawn by `benchmarks/make_figures.py`. Nothing here is typed in by hand.

## Setup and data

| | |
|---|---|
| Receipts | **SROIE** (ICDAR 2019; Malaysian shop receipts; `jsdnrs/ICDAR2019-SROIE`, tagged CC-BY-4.0 on Hugging Face): 626 train, 361 test. Fields: company, date, address, total. **CORD-v2** (Indonesian receipts, `naver-clova-ix/cord-v2`): 100 test. Fields: total, subtotal, tax, menu items |
| Forms | **XFUND** (seven languages: zh, ja, es, fr, it, de, pt): 250 train, 105 test forms. Hindi is out of scope |
| Document types | **RVL-CDIP** (16 classes of scanned tobacco-industry documents, `chainyo/rvl-cdip`): 100 documents per class from the test split for evaluation, 100 per class from the val split for training and 60 per class for calibration |
| Models | Qwen2.5-VL-3B and 7B (zero-shot and LoRA fine-tuned) on Modal (A100 for training, A10 for evaluation); DiT-base (`microsoft/dit-base-finetuned-rvlcdip`) on CPU for document types |
| Statistics | Bootstrap intervals over documents; differences between systems are **paired** bootstrap intervals on the same documents. A difference is called solid only if its 95% interval excludes zero. One seed per fine-tune |
| Discipline | Calibration and thresholds are fitted on a dev set that shares no shop with the fine-tuning data and reported on a test set that was untouched until reporting. Near-duplicate checks between train and test are in 2.1 and 2.2 |

**Field accuracy** is the mean over the four SROIE fields of exact match after normalisation (dates and amounts via `src/utils/normalize.py`; text compared case- and whitespace-insensitively). The labels contain errors (a printed "81750" is labelled "B1750"), which caps measurable accuracy.

---

## 1. Zero-shot baseline

<!-- BEGIN table:real_vlm -->
| Qwen2.5-VL-7B zero-shot, 361 real receipts | exact | 95% bootstrap | near-exact (<= 2 characters) |
|---|---:|---:|---:|
| company | 0.803 | [0.762, 0.845] | 0.839 |
| address | 0.571 | [0.521, 0.620] | 0.734 |
| date | 0.939 | [0.914, 0.964] | 0.939 |
| total | 0.986 | [0.972, 0.997] | 0.986 |
| **all four fields** | **0.825** | [0.805, 0.844] | 0.875 |
| receipts with every field right | 0.512 |  | 0.668 |
<!-- END table:real_vlm -->

*`results/raw/real_vlm.json`; model output in `results/raw/vlm_real_Qwen2.5-VL-7B-Instruct_sroie_test.jsonl`. 4.1 s a page on an A10.*

- **Totals and dates are close to solved zero-shot (0.986 and 0.939).** Addresses (0.571, 0.734 near-exact) and company names (0.803) lose points to multi-line layout, abbreviations and label noise; only half the receipts have every field exactly right.
- **The room to improve is in the text fields**, which is what the next section addresses.

## 2. Fine-tuning on real receipts

The zero-shot VLM is far from solved on addresses and company names, so this section fine-tunes it. Both models are LoRA adapters (rank 16, language-model projections only, answer tokens only) on Qwen2.5-VL; training and evaluation use the same prompt, pixel budget and JSON output format as the zero-shot runs (`benchmarks/modal_finetune.py`, evaluation through `modal_run.py --adapter`).

**Data and splits.** SROIE train (626 receipts) is split into 506 for fine-tuning and a 120-receipt **dev set that shares no shop with the fine-tuning receipts** (`benchmarks/prep_real_split.py`); dev is used only to fit calibration and the accept threshold. The SROIE test split (361 receipts) is untouched until reporting, and it is also reported on the **135 test receipts from shops that never appear in fine-tuning**. The 7B was fine-tuned on the 506 SROIE receipts plus 400 CORD training receipts (half of CORD's train split; the rest were not used). One run per model, one seed.

<!-- BEGIN table:ft_training -->
| model | GPU | train examples | epochs | LoRA params | minutes | peak GB | dev loss per epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3B | A100-SXM4-80GB | 506 | 3 | 30M of 3.8B | 52 | 12 | 0.0230 -> 0.0221 -> 0.0228 |
| 7B | A100-SXM4-80GB | 906 | 2 | 40M of 8.3B | 58 | 22 | 0.0208 -> 0.0215 |
<!-- END table:ft_training -->

*Dev loss is flat after the first epoch. The Modal container reported an 80 GB A100 although I priced it at the 40 GB rate, so the A100 spend in the ledger may be low by up to about 19%. `results/raw/ft_ft3b.json`, `ft_ft7b.json`.*

### 2.1 SROIE (361 real receipts)

<!-- BEGIN table:ft_sroie -->
| system (361 real SROIE test receipts) | field accuracy | 95% bootstrap | unseen shops | company | date | address | total | receipts all right | USD / 1,000 pages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B, zero-shot | 0.825 | [0.803, 0.846] | 0.767 | 0.80 | 0.94 | 0.57 | 0.99 | 0.51 | 1.63 |
| Qwen2.5-VL-3B, zero-shot | 0.835 | [0.814, 0.855] | 0.744 | 0.80 | 0.96 | 0.61 | 0.97 | 0.53 | 1.50 |
| **3B, fine-tuned (LoRA, 506 SROIE receipts)** | 0.892 | [0.876, 0.908] | 0.824 | 0.91 | 0.97 | 0.71 | 0.98 | 0.63 | 1.23 |
| **7B, fine-tuned (LoRA, SROIE + CORD)** | 0.901 | [0.886, 0.916] | 0.843 | 0.92 | 0.98 | 0.71 | 0.99 | 0.65 | 1.64 |
<!-- END table:ft_sroie -->

<!-- BEGIN table:ft_sroie_paired -->
| paired difference in field accuracy | receipts | difference | 95% paired bootstrap |
|---|---:|---:|---:|
| fine-tuned 3B minus zero-shot 7B | all 361 | +0.067 | [+0.048, +0.086] |
| fine-tuned 3B minus zero-shot 7B | 135 unseen shops | +0.057 | [+0.028, +0.087] |
| fine-tuned 3B minus zero-shot 3B (what fine-tuning alone buys) | all 361 | +0.057 | [+0.042, +0.072] |
| fine-tuned 3B minus zero-shot 3B (what fine-tuning alone buys) | 135 unseen shops | +0.080 | [+0.052, +0.107] |
| fine-tuned 7B minus zero-shot 7B | all 361 | +0.076 | [+0.059, +0.093] |
| fine-tuned 7B minus zero-shot 7B | 135 unseen shops | +0.076 | [+0.048, +0.106] |
| fine-tuned 7B minus fine-tuned 3B | all 361 | +0.009 | [-0.002, +0.020] |
<!-- END table:ft_sroie_paired -->

*Field accuracy is the mean over the four fields of exact match after normalisation. `results/raw/real_compare.json`, from `benchmarks/analyze_real.py`.*

- **Fine-tuning on 506 receipts helps a lot and does not depend on shops seen in training.** Holding the model at 3B, it adds +0.057 field accuracy (interval +0.042 to +0.072) and +0.080 on the 135 unseen shops (+0.052 to +0.107). The fine-tuned 3B beats the zero-shot 7B by +0.067 (+0.048 to +0.086).
- **The gain is mostly company names (0.80 to 0.91) and addresses (0.61 to 0.71).** Dates and totals were already 0.94 to 0.99 zero-shot. Addresses remain the weakest field, and only about two receipts in three have every field right.
- **The 7B is not reliably better than the 3B on accuracy** (+0.009, interval -0.002 to +0.020). It costs about a third more per page. Its advantage is in calibration (2.3).
- **Duplicates check.** SROIE contains some re-scanned receipts: 11 of the 361 test receipts have the same company, date and total as a fine-tuning or dev receipt. Removing them moves the fine-tuned 7B from 0.901 to 0.899 and the fine-tuned 3B from 0.892 to 0.889 (zero-shot scores drop by a similar 0.003 to 0.004; for the fine-tuned models 0.002 to 0.003), so the leakage is real but does not explain the gain (`benchmarks/check_sroie_overlap.py`, `results/raw/sroie_overlap_check.json`). The check uses labels only; I did not compare images pixel by pixel.
- **Caveats.** One seed per model; the 7B also saw CORD, the 3B did not; the unseen-shop split is by exact company name after normalisation, so near-spelled variants of a shop could straddle the split; and SROIE labels contain typos, which caps measurable accuracy.

#### Where the gain comes from, and where the errors are left

The headline score is strict: a text field must match the label after case and spacing are normalised, so `NO.20. GROUND FLOOR` and
`NO 20, GROUND FLOOR` count as different. Fine-tuning teaches the model SROIE's punctuation and spacing conventions, which is
useful for a product with a fixed output format but is not the same as reading the receipt better. To separate the two, a second
**content** score compares company and address on letters and digits only (dates and amounts are unchanged). It was added after
seeing the errors, so read it as an explanation, not as a second headline (`benchmarks/analyze_real.py`).

<!-- BEGIN table:ft_content -->
| 361 SROIE test receipts | field accuracy, strict (headline) | field accuracy, content only | company: strict -> content | address: strict -> content |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B, zero-shot | 0.825 | 0.931 | 0.80 -> 0.92 | 0.57 -> 0.88 |
| Qwen2.5-VL-3B, zero-shot | 0.835 | 0.920 | 0.80 -> 0.90 | 0.61 -> 0.85 |
| **3B, fine-tuned (LoRA, 506 SROIE receipts)** | 0.892 | 0.939 | 0.91 -> 0.94 | 0.71 -> 0.87 |
| **7B, fine-tuned (LoRA, SROIE + CORD)** | 0.901 | 0.954 | 0.92 -> 0.95 | 0.71 -> 0.90 |
| gain from fine-tuning, 3B vs same-size zero-shot | +0.057 [+0.042, +0.072] | +0.019 [+0.008, +0.030] |  |  |
| gain from fine-tuning, 7B vs same-size zero-shot | +0.076 [+0.059, +0.093] | +0.023 [+0.013, +0.033] |  |  |
<!-- END table:ft_content -->

- **Most of the headline gain is formatting.** Against the same-size zero-shot model, fine-tuning gains +0.057 (3B) and +0.076 (7B) on the strict score but +0.019 and +0.023 on the content score. The content gain is smaller and still excludes zero (intervals above), so fine-tuning does read better, but most of the strict gain is learning the labels' conventions. Addresses show it most: the zero-shot 7B has 0.571 strict and 0.881 content, the fine-tuned 7B 0.715 and 0.900.
- **What is left in the fine-tuned 7B** (`benchmarks/error_analysis.py`, `results/raw/error_analysis.json`): of 103 wrong addresses, 67 differ only in punctuation, spacing or case and **36 are real content errors (10%)**. Of the 28 wrong company names, 10 are formatting-only. Dates have 8 errors and totals 4 (for example 19.10 read as 18.10, and 3.00 as 2.99).
- **The one-character address errors are mostly not digits.** 76 addresses are within 2 characters of the label: 51 touch only letters or punctuation (`.` against `,` is the most common edit, 12 times) and 25 touch a digit. At least one is the dataset's own typo (the label `B1750` for a printed postcode `81750`); I did not audit the rest against the images, so I cannot say how many more are label errors.
- **Resolution does not explain the errors.** 54.6% of test receipts are larger than the 1.4 MP the model sees and are downscaled, some to a 35 MP original. If downscaling caused the errors, the heavily downscaled receipts would do worst; they do best on addresses (0.982 within 2 characters for receipts over 5 MP, against 0.887 for those between 1.4 and 5 MP). I did not run a higher-resolution experiment, and this result is the reason I would not spend GPU time on one.

### 2.2 CORD (100 real receipts)

<!-- BEGIN table:ft_cord -->
| CORD test (100 real receipts) | 3 amounts, absent counted | 3 amounts, where present | menu F1 | menu F1 95% bootstrap | amounts where present, no near-duplicates | menu F1, no near-duplicates | menu F1, near-duplicates only |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B, zero-shot | 0.797 | 0.951 | 0.834 | [0.775, 0.890] | 0.949 | 0.841 | 0.795 |
| **7B, fine-tuned (SROIE + 400 CORD receipts)** | 0.960 | 0.980 | 0.921 | [0.884, 0.955] | 0.987 | 0.911 | 0.973 |
<!-- END table:ft_cord -->

*`results/raw/cord_compare.json` and `cord_overlap_check.json` (`benchmarks/check_cord_overlap.py`). Amounts: total, subtotal, tax; compared by digits only because CORD prints "60.000" for sixty thousand. Menu: items matched one-to-one on price and fuzzy name.*

- **The headline CORD score overstates the gain, for two reasons I checked.** Many receipts have no subtotal (35 of 100) or no tax (57 of 100), and leaving them empty counts as correct: the zero-shot model invents a subtotal on 97% of the receipts that have none, the fine-tuned model leaves it empty on 91%. That is learning CORD's labelling convention, which is useful but not better reading. Restricted to receipts where the field exists, amounts go from 0.949 to 0.987 (no near-duplicates).
- **There is some template memorisation on the menu.** 24 of the 100 test receipts have an item list at least 90% similar to a training receipt; on those the fine-tuned menu F1 is 0.973 against 0.911 on the other 76. The honest CORD figure is the no-near-duplicate one: **menu F1 0.911 against 0.841 zero-shot**.
- **Only 100 test receipts**, so the intervals are wide (menu F1 for the fine-tuned model: 0.884 to 0.955 over all 100). SROIE is the stronger evidence.

### 2.3 Calibrated review

The model's confidence is calibrated on 120 dev receipts that share no shop with fine-tuning, and a field is accepted
automatically when its calibrated probability reaches the **accept threshold**. The threshold is a product choice
about how many errors you can tolerate, not a fitted value, so this section shows both the strict choice (the lowest
threshold that reached 99% precision on dev) and the full trade-off.

**The strict setting (99% precision target on dev):**

<!-- BEGIN table:ft_calibration -->
| calibration fitted on 120 dev receipts, applied to test | ECE raw -> calibrated | accept threshold (99% precision on dev) | auto-accepted | precision of those | errors caught by review | coverage at 99 / 98 / 95% precision |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B, zero-shot | 0.088 -> 0.043 | 0.989 | 48% | 98.8% | 97% | 43% / 59% / 74% |
| **3B, fine-tuned (LoRA, 506 SROIE receipts)** | 0.039 -> 0.022 | 0.991 | 39% | 99.5% | 98% | 52% / 66% / 87% |
| **7B, fine-tuned (LoRA, SROIE + CORD)** | 0.039 -> 0.016 | 0.992 | 58% | 99.2% | 95% | 64% / 76% / 90% |
<!-- END table:ft_calibration -->

- **Fine-tuning improves the probabilities, not only the answers.** Calibrated on the dev receipts and applied unchanged to test, the fine-tuned 7B auto-accepts 58% of fields at 99.2% precision, against 48% at 98.8% for the zero-shot 7B and 39% at 99.5% for the fine-tuned 3B. At a 95% precision target the fine-tuned 7B can auto-accept 90%.
- **Both fine-tuned models sit within about a point of the 99% dev target**. Calibration was fitted on the same kind of document as the test; it does not transfer between document types, so refit it on a labelled sample of your own documents before relying on a threshold.
**Every threshold, applied unchanged to the 361 test receipts (1,444 fields).** The grid of thresholds was fixed before looking at test, but the default (0.80) was adopted after this table had been seen, at the owner's request. Its dev-set precision (95.6%) agrees with its test precision (96.3%), which is the reason to trust the choice, not the test result itself.

<!-- BEGIN table:ft_thresholds -->
| accept threshold (calibrated probability) | 7B: fields auto-accepted | 7B: precision of those | 7B: wrong fields let through per 1,000 | 7B: errors caught by review | 3B: auto-accepted | 3B: precision | 3B: wrong per 1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 97% | 91.9% | 78 | 21% | 97% | 91.6% | 82 |
| 0.60 | 94% | 93.6% | 60 | 40% | 87% | 95.1% | 42 |
| 0.70 | 91% | 94.8% | 47 | 52% | 87% | 95.1% | 42 |
| **0.80 (default)** | 87% | 96.3% | 33 | 67% | 82% | 96.8% | 26 |
| 0.90 | 69% | 98.6% | 10 | 90% | 73% | 97.8% | 16 |
| 0.95 | 59% | 99.2% | 5 | 95% | 51% | 99.2% | 4 |
| 0.99 | 58% | 99.2% | 5 | 95% | 39% | 99.5% | 2 |
<!-- END table:ft_thresholds -->

- **The default accept threshold is 0.80**, which sends the fine-tuned 7B's confident fields through and leaves 13% of fields for a person. On dev it showed 95.6% precision; on test it accepted **87%** of fields at **96.3% precision**, or about 33 wrong fields let through per 1,000.
- **What that costs: review catches only 67% of the model's errors at 0.80**, against 95% at the strict setting (58% accepted, 99.2% precision, 5 wrong per 1,000). One field in 27 that is accepted automatically is wrong. **Most of the leaked errors are addresses**: 35 of the 47 wrong fields accepted at 0.80, because accepted addresses are only 86% correct, against 98.1% for company names and 99.1% for dates and for totals (`test_per_field` in `results/raw/real_compare.json`).
- **Choose the threshold from what a wrong field costs you.** `DOCINTEL_ACCEPT_THRESHOLD=0.95` gives the strict behaviour for the 7B (the calibrated probability moves in steps, so every threshold from 0.95 to 0.99 behaves the same). The calibration config (`configs/receipt_calibration.json`) records both operating points.
- The 3B behaves similarly at the default (82% accepted at 0.80, 96.8% precision). The 7B's advantage shows at the strict end: at 0.99 it accepts 58% at 99.2% precision, against 39% for the 3B.
- **Consequence for the design:** the 7B's higher cost per page ($1.64 against $1.23 per 1,000) is offset by about 780 fewer fields sent to a human per 1,000 pages (42% against 61% of 4,000 fields, at the strict setting), a break-even of about 0.05 cents per review. At any realistic review cost the **fine-tuned 7B with calibrated review is the cheaper total system**; a cheap-model-then-expensive-model cascade adds nothing here because both models cost about the same.


## 3. Document type: which extractor should this go to?

A receipt extractor must not be fed a form. A DiT-base image classifier (`microsoft/dit-base-finetuned-rvlcdip`, on CPU) embeds the page; a 17-class logistic-regression probe on the 768-d embedding (the 16 RVL-CDIP classes plus **receipt**) predicts the type; an isotonic map turns its top probability into a calibrated confidence; and a router sends receipts to the receipt extractor, forms and questionnaires to the form extractor, and everything else, or anything the probe is unsure about, to a person (`src/doctype`, `benchmarks/analyze_doctype.py`). Invoices are deliberately routed to a person: no extractor here has been validated on invoices.

**Split.** The probe trains on 100 RVL-CDIP documents per class (val split) plus the SROIE fine-tuning receipts and CORD train receipts. Regularisation, calibration and the accept threshold come from a disjoint dev set (60 per class plus SROIE dev and CORD validation receipts). Everything below is on test documents the probe never saw.

<!-- BEGIN table:doctype_types -->
| RVL-CDIP test, 1,600 documents (100 per class) | accuracy | 95% bootstrap |
|---|---:|---:|
| DiT-base-finetuned-rvlcdip, its own 16-class head | 0.938 | [0.925, 0.949] |
| **17-class linear probe on its embeddings (adds receipt)** | **0.922** | [0.909, 0.936] |
| probe: share of RVL-CDIP documents it calls a receipt | 0.012 |  |
<!-- END table:doctype_types -->

<!-- BEGIN table:doctype_receipts -->
| share of documents | n | probe trained on SROIE receipts only | probe trained on SROIE + CORD receipts (shipped) |
|---|---:|---:|---:|
| SROIE test receipts | 361 | 0.989 | 0.992 |
| CORD test receipts (a different format) | 100 | 0.700 (never seen) | 0.980 |
| XFUND forms called a receipt (should be 0) | 105 | 0.095 | 0.114 |
| XFUND forms sent to the form extractor | 105 | 0.724 | 0.724 |
<!-- END table:doctype_receipts -->

<!-- BEGIN table:doctype_routing -->
| route (accept threshold 0.79, 2166 test documents) | routed | precision | recall |
|---|---:|---:|---:|
| receipt | 463 | 0.972 | 0.976 |
| form | 231 | 0.957 | 0.725 |
| human | 1472 | 0.940 | 0.989 |
| receipts sent to the form extractor | 0 |  |  |
| rvlcdip_test (1600): routed correctly / sent to a person |  | 0.966 | 0.887 |
| sroie_test (361): routed correctly / sent to a person |  | 0.975 | 0.025 |
| cord_test (100): routed correctly / sent to a person |  | 0.980 | 0.020 |
| xfund_test (105): routed correctly / sent to a person |  | 0.562 | 0.390 |
<!-- END table:doctype_routing -->

*`results/raw/doctype_compare.json` (shipped probe) and `doctype_compare_sroie_only.json` (trained without CORD).*

- **The probe is not better than DiT at the 16 classes DiT was trained on** (0.922 against 0.938, the intervals barely overlap; DiT was fine-tuned on the full RVL-CDIP training set and the probe sees 1,600 documents). What the probe adds is the receipt class and a calibrated confidence. Weakest classes: form (0.80), invoice (0.88), budget (0.89), presentation (0.89), questionnaire (0.89).
- **Receipts are found reliably when their format has been seen, and not otherwise.** Trained on SROIE only, the probe recognises 98.9% of SROIE test receipts but only **70%** of CORD test receipts (23% become "advertisement"), a format it had never seen; DiT's own head calls CORD receipts "invoice" or "advertisement" as well. Adding CORD train receipts lifts CORD test to 98%. For a new receipt format, add a few labelled receipts of it and refit; the refit takes seconds on CPU because the embeddings are cached.
- **The risky errors are forms and invoices sent to the receipt extractor.** 12 of 105 XFUND forms (11%) are called receipts, and 5 of 100 RVL-CDIP invoices. The receipt extractor would then return confident-looking fields for a document that is not a receipt. Receipts sent to the form extractor: none.
- **The type confidence is well calibrated** (ECE 0.010 raw, 0.008 calibrated, on 2,061 test documents with known labels). The accept threshold (0.79, chosen for 95% precision on dev) sends most RVL-CDIP test documents, whose types have no extractor, to a person by design.
- **Caveats.** RVL-CDIP is old, mostly black-and-white tobacco-industry scans: it says nothing about modern phone photos or other document families, and a classifier could separate "receipt" from those scans by image style rather than content. The CORD and XFUND results are the check on that shortcut: modern receipts and forms were mostly routed correctly, but 11% of forms were not. 100 documents per class, one seed.

## 4. A multilingual pilot on real forms (XFUND)

Everything above is English receipts. As a first test of other languages, the 3B model was fine-tuned and evaluated on **XFUND**, real forms in seven languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese; Hindi is out of scope). The task differs from receipts: list every filled-in field of the form as a (label, value) pair, taken from XFUND's question-to-answer links.

**Setup.** 250 training forms (about 36 per language, forms with at most 60 pairs, from the XFUND train split), 105 test forms (15 per language, a seeded sample of the XFUND val split), one seed, 2 epochs, LoRA on the 3B (`benchmarks/modal_finetune.py`, 19 minutes on an A100). Protocol fixed before any results: test on all 105 forms and also on the 97 with at most 60 pairs; 2,048 generation tokens for both systems. A pair is matched when key and value are both similar after normalisation (lenient: accent-folded, rapidfuzz ratio at least 85) or equal (exact); precision and recall are over pairs (`src/eval/xfund.py`). Intervals are bootstrap over forms.

<!-- BEGIN table:xfund -->
| XFUND forms (pair F1, lenient) | forms | 3B zero-shot [95% bootstrap] | 3B fine-tuned (250 forms) [95%] | paired per-form gain [95%] |
|---|---:|---:|---:|---:|
| Chinese | 15 | 0.26 [0.13, 0.38] | 0.44 [0.34, 0.55] | +0.234 [+0.121, +0.367] |
| Japanese | 15 | 0.19 [0.12, 0.28] | 0.31 [0.21, 0.44] | +0.165 [+0.050, +0.284] |
| Spanish | 15 | 0.42 [0.33, 0.51] | 0.43 [0.33, 0.55] | +0.031 [-0.071, +0.140] |
| French | 15 | 0.58 [0.46, 0.69] | 0.65 [0.56, 0.74] | +0.050 [-0.027, +0.123] |
| Italian | 15 | 0.39 [0.25, 0.53] | 0.44 [0.28, 0.60] | +0.058 [-0.004, +0.116] |
| German | 15 | 0.32 [0.21, 0.46] | 0.50 [0.36, 0.65] | +0.147 [+0.042, +0.266] |
| Portuguese | 15 | 0.39 [0.24, 0.61] | 0.38 [0.23, 0.60] | +0.019 [-0.158, +0.175] |
| **all 105 forms** | 105 | 0.36 [0.30, 0.42] | 0.44 [0.37, 0.50] | +0.101 [+0.055, +0.147] |
| forms with <= 60 pairs | 97 | 0.42 [0.36, 0.47] | 0.50 [0.45, 0.56] |  |
| precision / recall, all forms |  | 0.55 / 0.27 | 0.49 / 0.40 |  |
<!-- END table:xfund -->

*`results/raw/xfund_compare.json` from `benchmarks/analyze_xfund.py`; outputs in `vlm_real_zs3b_xfund_test.jsonl` and `vlm_real_ft3b_xfund_xfund_test.jsonl`. Exact-match pair F1 is 0.287 zero-shot and 0.355 fine-tuned.*

- **Fine-tuning helps, modestly, and not equally in every language.** Pair F1 goes from 0.362 to 0.438 over all 105 forms; the paired per-form gain is +0.101 (interval +0.055 to +0.147). The gain is clear in Chinese (+0.234), Japanese (+0.165) and German (+0.147), which are also the three languages where the zero-shot model scored lowest; in Spanish, French, Italian and Portuguese the interval includes zero.
- **The absolute level is low.** Even fine-tuned, precision is 0.49 and recall 0.40: it finds fewer than half of a form's fields. Forms are denser and harder than receipts (about 32 pairs per form), and one seed and 250 forms is a small fine-tune. This is evidence that fine-tuning transfers to other languages, not a usable multilingual extractor.
- **Caveats.** 15 test forms per language, so the per-language intervals are wide; the label source has some encoding damage (replacement characters) that the scorer ignores; XFUND's pairs come from question-answer links and miss values the annotators did not link; 6 of the 105 fine-tuned outputs hit the token limit (3 zero-shot); the fine-tune used forms only (no receipts), so it is a separate adapter (`ft3b_xfund`) from the receipt models; the A100 cost estimate may be low by up to about 19%.

---

## 5. Cost and what was not done

The GPU spend for fine-tuning and evaluation is recorded run by run in `results/raw/modal_spend.json`; the per-1,000-page inference costs above use the A10 list price and measured seconds per page. Not done: a second seed for any fine-tune; a validated invoice extractor; calibrated confidence for the form extractor (every form pair is flagged for review); any Hindi or other Indic-script data; photographed (non-scanned) receipts.
