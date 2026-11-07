# serving

[modal_run.py](modal_run.py) is the GPU app. It loads Qwen2.5-VL with a LoRA adapter on a Modal GPU and answers extraction requests. Two things use it:

- the HTTP service in [src/api/](../../src/api/) (`IDP_VLM_APP=idp-vlm`), and
- batch evaluation runs that write the model outputs in `results/raw/`.

```bash
modal run experiments/serving/modal_run.py --suite fetch     # cache the base weights on a CPU container
modal deploy experiments/serving/modal_run.py                # on Windows set PYTHONUTF8=1 first
modal run experiments/serving/modal_run.py --suite real --data data/real/sroie_test --tag real_ft7b_sroie_test --adapter ft7b
```

Redeploy after pulling changes: the deployed app is a snapshot of this file. The cost guard it shares with training is [src/utils/gpu_budget.py](../../src/utils/gpu_budget.py).
