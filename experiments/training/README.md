# training

Fine-tuning Qwen2.5-VL with LoRA (supervised fine-tuning) on a rented GPU.

| File | What it does |
|---|---|
| [modal_finetune.py](modal_finetune.py) | **The training code.** LoRA on the language model, loss on the answer tokens only, AdamW with a cosine schedule, a checkpoint per epoch, behind a cost guard |
| [prep_real_split.py](prep_real_split.py) | makes the company-disjoint fine-tune and dev split of the SROIE receipts |

```bash
python experiments/training/prep_real_split.py
IDP_FT_MODEL=Qwen/Qwen2.5-VL-3B-Instruct modal run --detach experiments/training/modal_finetune.py --tag ft3b --epochs 3
```

The full sequence, including the 7B and the form model, is in [docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md) section 5, and the settings used are in the README's [Fine-tuning](../../README.md#4-fine-tuning-experimentstrainingmodal_finetunepy) section. The trained adapters are saved to the Modal volume `idp-models` and are not in this repository.
