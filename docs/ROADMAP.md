# Roadmap

Rule: **no number goes in the README unless a script in `benchmarks/` produced it and the raw JSON is in `results/raw/`.**

## Done
- [x] Real datasets with disjoint splits: SROIE (361 test receipts), CORD (100), XFUND (105 forms, 7 languages), RVL-CDIP (1,600 test documents, 16 classes); company-disjoint dev set; unseen-shop evaluation
- [x] Zero-shot Qwen2.5-VL 3B and 7B baselines; LoRA fine-tuning of both on Modal, paired bootstrap comparisons ([RESULTS](RESULTS.md) sections 1 and 2)
- [x] Calibrated per-field probabilities and accept threshold fitted on dev, reported on test; risk-coverage curves
- [x] Memorisation checks: SROIE re-scans, CORD near-duplicate item lists
- [x] Review queue that records corrections as labelled data; PSI drift monitor; FastAPI service; Streamlit demo; cost guard and spend ledger
- [x] Multilingual form pilot on XFUND (pair F1 0.362 to 0.438, clear gain in zh/ja/de only)
- [x] Document-type stage: DiT embeddings, 17-class probe with calibrated confidence, router, `POST /document` ([RESULTS](RESULTS.md) section 3)

## Next, in order of value
- [ ] A phone-photo test set (receipts and forms photographed at an angle) for the document-type stage and the receipt extractor; every dataset here is a flat scan
- [ ] Fix the worst routing error: forms and invoices sent to the receipt extractor (11% of XFUND forms). Candidates: a "not a receipt" margin, or training the probe on more modern forms
- [ ] A validated invoice extractor (needs field-labelled real invoices), then add `invoice` to `ROUTES`
- [ ] More seeds per fine-tune and a held-out evaluation set that nothing was tuned on
- [ ] Calibrated confidence for the form extractor, so some pairs can be auto-accepted
- [ ] Per-user accounts and a database-backed review store for the service (a shared API key, a daily GPU-call cap and a Modal container cap exist)

Hindi and other non-English-script documents beyond the XFUND languages are out of scope.
