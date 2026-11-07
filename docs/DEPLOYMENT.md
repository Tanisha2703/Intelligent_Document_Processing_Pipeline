# Deployment

What exists: a Streamlit demo, a FastAPI service, a CPU container image, and Modal scripts that train and serve the fine-tuned model on a rented GPU. What does not: per-user accounts, multi-user storage, autoscaling. Treat this as a reference deployment, not a hardened one.

## 1. The model on Modal (GPU)

The fine-tuned model is a LoRA adapter on Qwen2.5-VL, saved on the Modal volume `idp-models` (`ft3b`, `ft7b`). The base weights are cached on `idp-hf-cache`.

```bash
pip install -r requirements.txt                      # includes the modal client
modal token new                                      # or set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
modal run experiments/serving/modal_run.py --suite fetch      # download the base weights on a CPU container (GPU billing starts later)
modal deploy experiments/serving/modal_run.py                 # app "idp-vlm"; the class takes the adapter name as a parameter
```

**Redeploy after pulling changes.** The deployed app is a snapshot of `experiments/serving/modal_run.py`: an old deployment does not know newer document types (the form route fails with `KeyError: 'xfund_form'`). On Windows, set `PYTHONUTF8=1` for `modal deploy`, or the CLI crashes printing its check mark.

`configs/receipt_calibration.json` names the model and adapter the service calls (`ft7b` by default; `receipt_calibration_ft3b.json` is the cheaper alternative) and holds the calibration map and accept threshold fitted on dev receipts. A cold GPU container made the first requests take 28 to 45 seconds in my runs, against about 3 to 4 seconds per page when warm (RESULTS.md section 1 and 2.1). Settings: `IDP_GPU` (default A10G), `IDP_MAX_PIXELS` (the image budget, 1.4 MP, which sets image tokens and therefore time), `IDP_BUDGET_USD`.

**Cost guard.** Every Modal script estimates its worst-case cost before launching and refuses to start if that exceeds the budget left after the ledger (`results/raw/modal_spend.json`). Estimates use Modal list prices and measured seconds; check the price table in `src/utils/gpu_budget.py` before relying on it.

## 2. Streamlit demo

```bash
streamlit run app.py          # http://localhost:8501
```

"Recorded examples" works with no GPU (two real receipts and the model's stored output). "Upload a receipt" needs `IDP_VLM_APP=idp-vlm` and the deployed app. The review tab shares `data/review/review.jsonl` with the HTTP service (override with `IDP_REVIEW_PATH`).

## 3. HTTP service

```bash
IDP_VLM_APP=idp-vlm uvicorn "src.api.app:create_app" --factory --host 0.0.0.0 --port 8000
curl -F file=@receipt.jpg localhost:8000/receipt
curl localhost:8000/review/queue
curl -X POST localhost:8000/review/<doc_id>/company -H 'content-type: application/json' -d '{"value": "ABC SDN BHD", "reviewer": "asha"}'
curl localhost:8000/monitor
```

| Endpoint | Purpose |
|---|---|
| `POST /receipt` | upload a receipt image (PNG, JPG, TIFF; limit 50 MB); returns each field's value, normalised value, calibrated probability and `auto` / `review` status; uncertain fields are queued |
| `POST /document` | any document image: classify the type (DiT, CPU), route it, extract if an extractor supports the type, otherwise queue it for a person. Needs `IDP_DOCTYPE=1`, torch and transformers (section 5) |
| `GET /review/queue`, `POST /review/{doc}/{field}` | the review queue; every resolution is appended to `data/review/review.jsonl` with whether the model was right |
| `GET /monitor` | PSI of the confidence distribution against the reference, review statistics |
| `GET /health` | whether a VLM and the document-type model are configured |

A processing error returns 422, not 500; an unsupported suffix returns 415 before anything is read; without `IDP_VLM_APP`, `/receipt` returns 503; without `IDP_DOCTYPE=1`, `/document` returns 503.

## 4. Container (CPU)

```bash
docker build -t idp .            # or: podman build -t idp .
docker run --rm -p 8501:8501 -e IDP_VLM_APP=idp-vlm -e MODAL_TOKEN_ID=... -e MODAL_TOKEN_SECRET=... idp
```

The image serves the Streamlit demo with the exact versions in `requirements-lock.txt`; it has no OCR engine, no torch and no model weights (the GPU model runs on Modal, and the document-type classifier is a separate optional install). Without a Modal token the demo shows the recorded examples only. The Dockerfile was written for this slimmer project; build it once in CI before relying on it.

## 5. Reproducing the training and evaluation

```bash
python experiments/datasets/fetch_real.py test && python experiments/datasets/fetch_real.py train      # SROIE
python experiments/datasets/fetch_cord.py test train                                        # CORD (falls back to parquet shards if the rows API errors)
python experiments/training/prep_real_split.py                                              # fine-tune / dev / unseen-shop split
modal volume put idp-data data/real/sroie_ft /sroie_ft
modal volume put idp-data data/real/sroie_dev /sroie_dev
export IDP_BUDGET_USD=18
IDP_FT_MODEL=Qwen/Qwen2.5-VL-3B-Instruct modal run --detach experiments/training/modal_finetune.py --tag ft3b --epochs 3
modal run experiments/serving/modal_run.py --suite real --data data/real/sroie_test --tag real_ft3b_sroie_test --adapter ft3b
python experiments/evaluation/analyze_real.py --out results/raw/real_compare.json
python experiments/evaluation/write_receipt_config.py                                         # calibration files for the service
python experiments/evaluation/make_tables.py && python experiments/evaluation/make_figures.py             # RESULTS.md tables and README figures
```

The 7B run (SROIE plus 400 CORD receipts) is the same with `IDP_FT_MODEL=Qwen/Qwen2.5-VL-7B-Instruct`, `--sets "sroie_ft,cord_train"`, `--epochs 2`; see RESULTS.md section 2.

**XFUND multilingual pilot** (RESULTS.md section 4):

```bash
python experiments/datasets/fetch_xfund.py                                  # 40 train + 15 test forms per language
modal volume put idp-data data/real/xfund_train /xfund_train && modal volume put idp-data data/real/xfund_dev /xfund_dev
IDP_FT_MODEL=Qwen/Qwen2.5-VL-3B-Instruct IDP_FT_TIMEOUT=4000 modal run --detach experiments/training/modal_finetune.py \
    --tag ft3b_xfund --epochs 2 --n-train 250 --per-example-s 8 --sets xfund_train --dev-sets xfund_dev
IDP_MODEL=Qwen/Qwen2.5-VL-3B-Instruct modal run experiments/serving/modal_run.py --suite real --chunk 10 --per-item-s 40 \
    --data data/real/xfund_test --tag real_zs3b_xfund_test
IDP_MODEL=Qwen/Qwen2.5-VL-3B-Instruct modal run experiments/serving/modal_run.py --suite real --chunk 10 --per-item-s 40 \
    --data data/real/xfund_test --tag real_ft3b_xfund_xfund_test --adapter ft3b_xfund
python experiments/evaluation/analyze_xfund.py --out results/raw/xfund_compare.json
```

**Document-type stage** (CPU only, no Modal): fetch RVL-CDIP, embed with DiT, fit and evaluate the probe.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers
python experiments/datasets/fetch_rvlcdip.py                                                # 100 test + 100 train + 60 dev per class (about 0.5 GB saved)
python experiments/evaluation/embed_doctype.py sroie_ft sroie_dev sroie_test cord_train cord_validation cord_test xfund_train xfund_test     rvlcdip_train rvlcdip_dev rvlcdip_test                                        # 21 minutes for the three RVL-CDIP sets on a laptop CPU (440 + 299 + 500 s), a few more for the rest; resumable
python experiments/evaluation/analyze_doctype.py --receipts sroie --out results/raw/doctype_compare_sroie_only.json
python experiments/evaluation/analyze_doctype.py --receipts sroie+cord --write-config --out results/raw/doctype_compare.json   # also writes configs/doctype*.{json,npz}
```

## 6. Keeping strangers off your GPU

The Modal app (`idp-vlm`) has no public endpoint: it can only be called with credentials from your Modal workspace. The risk is the service you put in front of it, because every upload to `/receipt` or `/document` is a GPU call on your account. Three guards are built in:

| Guard | Setting | Behaviour |
|---|---|---|
| API key | `IDP_API_KEY` | every endpoint except `/health` needs `X-API-Key: <key>`; 401 otherwise |
| Local-only default | (no key set) | only a client on the same machine is answered; a request carrying `X-Forwarded-For` is refused even from localhost, so a reverse proxy cannot silently open the service |
| Daily GPU cap | `IDP_MAX_GPU_CALLS_PER_DAY` (default 200) | the 201st call of the UTC day gets 429; per process, resets on restart |
| Container cap | `IDP_MAX_CONTAINERS` (default 3, applied at `modal deploy`) | Modal cannot start more GPU containers than this at once |
| Demo password | `IDP_DEMO_PASSWORD` | the Streamlit live-upload tab asks for it (recorded examples stay open); the same daily cap applies |

Also: bind the Streamlit demo to the local machine unless you mean to share it (`streamlit run app.py --server.address 127.0.0.1`; its default is every interface), keep your Modal token out of the repository (`.env` and `~/.modal.toml` are not tracked), and set a spend limit in the Modal dashboard as the backstop. Not built: per-user accounts, a shared counter across several service processes.

## 7. Model licences

Qwen2.5-VL-3B-Instruct's model card lists the Qwen Research licence ([discussion](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/discussions/12)). For the 7B, a mirror lists Apache-2.0 while a [question on the official repository](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/discussions/31) about whether that is right had no answer when I checked. DiT-base is `microsoft/dit-base-finetuned-rvlcdip`. Read the licence in each official model card before any commercial use.

## 8. Before putting anything real through it

- The measurements are on public receipts and old scanned documents. Build a small labelled set of your own documents (the review queue records corrected values but not the images, so it does not produce one by itself), re-run the analysis on it and **refit the calibration and the document-type probe**: calibration fitted on a different document family auto-accepted 89% of real fields at 89% precision instead of 99%, and a receipt format the probe had not seen was recognised 70% of the time.
- Add authentication and a persistent store behind the API; `ReviewStore` is an append-only file for one process.
- Decide the accept threshold from what a wrong field costs you. The default (`IDP_ACCEPT_THRESHOLD` unset, calibration config 0.80) sends 13% of fields to a person and lets about 1 in 27 accepted fields through wrong; `0.95` is the strict setting (42% to a person, 99.2% precision). The measured trade-off is in RESULTS.md section 2.3.
