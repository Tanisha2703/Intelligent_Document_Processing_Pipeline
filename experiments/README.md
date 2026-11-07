# experiments

How the numbers in [docs/RESULTS.md](../docs/RESULTS.md) were produced. The product itself is in [src/](../src/); nothing in `src/` imports from here.

| Folder | Contents |
|---|---|
| [training/](training/) | the fine-tuning code (`modal_finetune.py`) and the train/dev split |
| [serving/](serving/) | the GPU model app (`modal_run.py`) that the API and the evaluation runs call |
| [datasets/](datasets/) | scripts that download the public datasets |
| [evaluation/](evaluation/) | one script per result, plus the table and figure builders |

Typical order: `datasets/` to download, `training/` and `serving/` to produce model outputs, then `evaluation/`. The commands are in [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md) section 5.
