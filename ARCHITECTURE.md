# Architecture

```
                      POST /document                                   POST /receipt
                           |                                                |
                  prepare_image (RGB PNG, <= 1.4 MP)  <--------------------+
                           |
            DiTEmbedder (CPU) -> 768-d embedding
                           |
            Probe: 17 classes (16 RVL-CDIP + receipt) -> IsotonicMap -> confidence
                           |
            route(): receipt -> receipt | form, questionnaire -> form | else, or confidence < accept -> human
              |                      |                                         |
     ReceiptExtractor         FormExtractor                          ReviewStore.enqueue("unrouted", type)
   (VLM + calibration)      (VLM, every pair "review")
              |                      |
     fields: value, normalized, probability, auto|review
              |
     ReviewStore (append-only JSONL)   <-- POST /review/{doc}/{field}     DriftMonitor (PSI) --> GET /monitor
```

## Components

| Module | Role |
|---|---|
| `src/doctype/classifier.py` | `DiTEmbedder` (hook on the classifier input for the pooled embedding), `Probe` (softmax over a linear map, calibrated confidence), the class lists. `DIT_LABELS` is DiT's own output order, which differs from the dataset's alphabetical order |
| `src/doctype/router.py` | `ROUTES` (the only document types with an extractor) and `route()`: below the accept threshold or no extractor means a person |
| `src/doctype/service.py` | `DocTypeService`: image bytes to type, confidence and route |
| `src/extract/vlm.py` | prompts, schemas, JSON parsing, and per-field confidence from token log-probabilities (`min_prob`). The prompts are fixed: the adapters were trained on them |
| `src/pipeline/receipts.py` | `ReceiptExtractor`: calls the VLM, calibrates each field, applies the accept threshold, never auto-accepts a missing field |
| `src/pipeline/forms.py` | `FormExtractor`: label/value pairs, no per-pair confidence, so everything is flagged |
| `src/pipeline/calibration.py` | `IsotonicMap` (fit, apply, serialise) |
| `src/api/app.py` | FastAPI service: the endpoints, and one shared extract-queue-monitor path for receipts and routed documents |
| `src/api/backends.py` | where the models run: the Modal class (looked up by app name) for extraction, DiT on CPU for document types; each returns None when not configured |
| `src/api/images.py` | upload validation (suffix, size) and conversion to the RGB PNG the model was fine-tuned on |
| `src/utils/guard.py` | API key, local-only default and the daily GPU-call cap |
| `src/monitor/` | `ReviewStore` (queue plus resolved corrections as labelled data), `DriftMonitor` (PSI of expected errors per receipt against the dev reference) |
| `src/eval/`, `src/utils/normalize.py` | scoring and the normalisation (dates, amounts) used by training data, scorers and the service |
| `benchmarks/modal_finetune.py`, `modal_run.py` | LoRA fine-tuning and evaluation/serving on Modal, behind a cost guard (`modal_common.py`) |

## Decisions and the evidence for each

- **Fine-tune a VLM, not an OCR pipeline.** On SROIE the zero-shot 7B VLM reached 0.825 field accuracy; the fine-tuned 3B reaches 0.892 (+0.057 at the same size, paired interval +0.042 to +0.072). The earlier Tesseract path was removed from this repository: on real receipts it recovered fewer than half the fields.
- **Calibrate on a dev set of the same kind of document, split by shop.** Calibration fitted on a different document family auto-accepted 89% of real fields at 89% precision against a 99% target. Fitted on real dev receipts, the same rule gives 99.2% precision (fine-tuned 7B).
- **Min-token probability, then isotonic map.** The confidence feature is the lowest token probability inside a field's value; an isotonic regression maps it to the probability that the field is right. It halves ECE (0.039 to 0.016 for the fine-tuned 7B).
- **The accept threshold is a product setting, not a fitted one.** The default is 0.80 (87% of fields accepted at 96.3% precision on test); `DOCINTEL_ACCEPT_THRESHOLD=0.95` is the strict setting (58 to 59% at 99.2%). Both are measured and the whole curve is published (RESULTS section 2.3), so the person deploying it can pick the error rate they can afford.
- **7B over 3B for the receipt route.** The accuracy difference is not reliable (+0.009, interval -0.002 to +0.020), but at the strict setting the 7B's probabilities let 58% of fields be auto-accepted at 99.2% precision against 39% for the 3B, which is cheaper overall once review time is priced.
- **A linear probe on DiT embeddings, not a fine-tuned classifier.** It needs no GPU, trains in seconds, and adding a new document family means adding embeddings and refitting. It costs 1.6 points against DiT's own head on RVL-CDIP (0.922 against 0.938). Adding the CORD format to the training data lifted CORD recognition from 70% to 98%.
- **Only measured types are routed.** `ROUTES` lists receipts and forms; invoices are classified but sent to a person because no extractor has been validated on them.
- **Forms are always reviewed.** The form adapter has pair F1 0.44 and no per-pair confidence, so nothing is auto-accepted.
- **Cost guard.** Every Modal script estimates worst-case cost and refuses to start past the budget left in the ledger, after a workspace was disabled mid-run by spend limits.

- **The GPU is protected at the service, not the model.** The Modal class has no public URL; the API requires `X-API-Key` (or a same-machine client when no key is set), caps GPU calls per day and limits Modal containers (`src/utils/guard.py`, DEPLOYMENT section 6).

What is not built: per-user accounts, a database-backed review store, autoscaling, an invoice extractor. See [docs/ROADMAP.md](docs/ROADMAP.md).
