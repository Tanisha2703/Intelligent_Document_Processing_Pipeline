# benchmarks

One script per experiment. Each writes raw JSON to `results/raw/`; `make_tables.py` and `make_figures.py` turn those into the tables in `docs/RESULTS.md` and the figures in `results/figures/`, so no reported number is typed by hand.

| Script | What it does |
|---|---|
| `fetch_real.py`, `fetch_cord.py`, `fetch_xfund.py`, `fetch_rvlcdip.py`, `prep_real_split.py` | download SROIE, CORD, XFUND and RVL-CDIP into `data/real/`; make the company-disjoint fine-tune / dev split |
| `modal_finetune.py` | LoRA fine-tune of Qwen2.5-VL on a rented GPU (cost guard, per-epoch checkpoints) |
| `modal_run.py` | run the VLM (zero-shot or with an adapter) on a dataset; also the deployed app the service calls |
| `modal_common.py` | cost guard and spend ledger shared by the Modal scripts |
| `bench_real_vlm.py` | zero-shot VLM scoring on SROIE |
| `analyze_real.py`, `analyze_cord.py`, `analyze_xfund.py` | per-dataset results with bootstrap intervals and calibration fitted on dev |
| `check_cord_overlap.py`, `check_sroie_overlap.py` | memorisation and duplicate checks |
| `write_receipt_config.py` | writes the calibration files the service loads |
| `error_analysis.py` | what kinds of errors the fine-tuned model still makes, whether they are formatting or content, and whether image size explains them; writes `results/raw/error_analysis.json` |
| `embed_doctype.py`, `analyze_doctype.py` | DiT embeddings (CPU, cached); fit, calibrate and evaluate the document-type probe, write `configs/doctype*` |
| `make_tables.py`, `make_figures.py` | regenerate the tables and figures from `results/raw/` |

`../scripts/` holds the receipt demo and the script that records the Streamlit examples.
