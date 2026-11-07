# data/

Nothing in this folder is committed except this file. The datasets are public, large, and not mine to redistribute, so the scripts download them:

| Folder (created by the script) | Script | Source | Licence / terms |
|---|---|---|---|
| `real/sroie_*` | `datasets/fetch_real.py`, `training/prep_real_split.py` | SROIE (ICDAR 2019) via `jsdnrs/ICDAR2019-SROIE` | the Hugging Face mirror is tagged CC-BY-4.0; the original competition terms may differ, so check them before redistributing |
| `real/cord_*` | `datasets/fetch_cord.py` | CORD-v2 via `naver-clova-ix/cord-v2` | CC-BY-4.0 |
| `real/xfund_*` | `datasets/fetch_xfund.py` | XFUND (zh, ja, es, fr, it, de, pt) | see the XFUND repository |
| `real/rvlcdip_*` | `datasets/fetch_rvlcdip.py` | RVL-CDIP via `chainyo/rvl-cdip` (derived from the IIT-CDIP collection) | research use; check the dataset card before reusing |
| `real/doctype_cache/` | `evaluation/embed_doctype.py` | DiT embeddings computed from the folders above | derived from the above |
| `review/review.jsonl` | the service, at runtime | corrections made in the review queue | yours |

## What is inside `data/real/` (local only; git ignores it)

The folders hold **the samples this project used, not the full datasets**. Most are a dataset's official split; two are derived by code.

| Folder | Files | What it is | Made by |
|---|---:|---|---|
| `sroie_train` | 626 | SROIE official train split | `datasets/fetch_real.py train` |
| `sroie_test` | 361 | SROIE official test split: **the main test set** | `datasets/fetch_real.py test` |
| `sroie_ft` | 506 | **copies** of 506 `sroie_train` receipts, used to fine-tune | `training/prep_real_split.py` |
| `sroie_dev` | 120 | **copies** of 120 other `sroie_train` receipts that share no shop with `sroie_ft`; used to fit calibration | `training/prep_real_split.py` |
| `cord_train`, `cord_validation`, `cord_test` | 400, 100, 100 | CORD-v2 samples (400 of its 800 train receipts) | `datasets/fetch_cord.py` |
| `xfund_train`, `xfund_dev`, `xfund_test` | 250 labelled (280 images downloaded, 30 unused), 14, 105 | XFUND forms: 40 train and 15 test per language downloaded | `datasets/fetch_xfund.py` |
| `rvlcdip_train`, `rvlcdip_dev`, `rvlcdip_test` | 1,600, 960, 1,600 | RVL-CDIP, 100 / 60 / 100 documents per class | `datasets/fetch_rvlcdip.py` |
| `doctype_cache/` | 11 | DiT embeddings of the folders above (not images) | `evaluation/embed_doctype.py` |
| `split_info.json` | 1 | which test receipts come from shops never seen in fine-tuning | `training/prep_real_split.py` |

**Known gap.** `xfund_train/labels.json` lists 250 forms, every one with at most 60 label/value pairs, and that is the set the form model was fine-tuned on. The filter that cut 280 downloaded forms down to those 250 is not in `datasets/fetch_xfund.py`: run as committed, the script labels all 280. Reproducing the exact training set needs that filter added to the script.

`sroie_ft` and `sroie_dev` duplicate about 153 MB of `sroie_train`: the split is stored as copied files, not as a list of IDs. The evaluation tags in `results/raw/` (for example `real_ft7b_sroie_dev`) refer to these folder names, so changing the layout means regenerating or renaming those too.

About 2.2 GB for everything, measured. The two sample receipts in `assets/examples/` are SROIE test receipts, included for the recorded demo under the mirror's CC-BY-4.0 tag (attribution in that folder's README).
