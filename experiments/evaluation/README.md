# evaluation

One script per result in [docs/RESULTS.md](../../docs/RESULTS.md). Each writes raw JSON to `results/raw/`; `make_tables.py` and `make_figures.py` turn those files into the tables and figures, so no reported number is typed by hand.

| Script | What it does |
|---|---|
| `bench_real_vlm.py` | zero-shot scoring on SROIE |
| `analyze_real.py`, `analyze_cord.py`, `analyze_xfund.py` | per-dataset results with bootstrap intervals; calibration and accept thresholds fitted on dev |
| `error_analysis.py` | what kinds of errors remain, formatting or content, and whether image size explains them |
| `check_cord_overlap.py`, `check_sroie_overlap.py` | memorisation and duplicate checks |
| `embed_doctype.py`, `analyze_doctype.py` | DiT embeddings (CPU, cached); fit, calibrate and evaluate the document-type probe |
| `write_receipt_config.py` | writes the calibration files the service loads |
| `make_tables.py`, `make_figures.py` | regenerate the tables in RESULTS.md and the figures |

Typical order: `experiments/datasets/` to download, `experiments/training/` and `experiments/serving/` to produce model outputs, then `analyze_*`, then `make_tables.py`.
