# data/

Nothing in this folder is committed except this file. The datasets are public, large, and not mine to redistribute, so the scripts download them:

| Folder (created by the script) | Script | Source | Licence / terms |
|---|---|---|---|
| `real/sroie_*` | `benchmarks/fetch_real.py`, `prep_real_split.py` | SROIE (ICDAR 2019) via `jsdnrs/ICDAR2019-SROIE` | the Hugging Face mirror is tagged CC-BY-4.0; the original competition terms may differ, so check them before redistributing |
| `real/cord_*` | `benchmarks/fetch_cord.py` | CORD-v2 via `naver-clova-ix/cord-v2` | CC-BY-4.0 |
| `real/xfund_*` | `benchmarks/fetch_xfund.py` | XFUND (zh, ja, es, fr, it, de, pt) | see the XFUND repository |
| `real/rvlcdip_*` | `benchmarks/fetch_rvlcdip.py` | RVL-CDIP via `chainyo/rvl-cdip` (derived from the IIT-CDIP collection) | research use; check the dataset card before reusing |
| `real/doctype_cache/` | `benchmarks/embed_doctype.py` | DiT embeddings computed from the folders above | derived from the above |
| `review/review.jsonl` | the service, at runtime | corrections made in the review queue | yours |

About 2.2 GB for everything, measured. The two sample receipts in `assets/examples/` are SROIE test receipts, included for the recorded demo under the mirror's CC-BY-4.0 tag (attribution in that folder's README).
