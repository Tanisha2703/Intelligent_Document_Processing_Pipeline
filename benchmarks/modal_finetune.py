"""LoRA fine-tune Qwen2.5-VL on real receipts (SROIE) on a rented GPU (Modal), with the shared cost guard and ledger.

  python benchmarks/prep_real_split.py                                  # fine-tune / dev split (company-disjoint)
  modal volume put docintel-data data/real/sroie_ft /sroie_ft
  modal volume put docintel-data data/real/sroie_dev /sroie_dev
  DOCINTEL_FT_MODEL=Qwen/Qwen2.5-VL-3B-Instruct modal run benchmarks/modal_finetune.py --tag ft3b --max-examples 12   # smoke
  DOCINTEL_FT_MODEL=Qwen/Qwen2.5-VL-3B-Instruct modal run benchmarks/modal_finetune.py --tag ft3b

Trains only the language model's attention and MLP projections (LoRA), on the answer tokens only. The adapter is
saved to the `docintel-models` volume under <tag>; benchmarks/modal_run.py --adapter <tag> evaluates it. Output
(loss curve, timings) goes to results/raw/ft_<tag>.json.
"""

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

import modal  # noqa: E402
import modal_common as mc  # noqa: E402

GPU = os.environ.get("DOCINTEL_FT_GPU", "A100-40GB")
MODEL = os.environ.get("DOCINTEL_FT_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
MAX_PIXELS = int(os.environ.get("DOCINTEL_MAX_PIXELS", "1400000"))
CPU, MEM = 4, 16384
TIMEOUT = int(os.environ.get("DOCINTEL_FT_TIMEOUT", "7200"))  # hard cap on one run; bounds the worst-case cost

app = modal.App("docintel-ft")
hf_cache = modal.Volume.from_name("docintel-hf-cache", create_if_missing=True)
models = modal.Volume.from_name("docintel-models", create_if_missing=True)
data_vol = modal.Volume.from_name("docintel-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "torchvision==0.29.1",
        "transformers==5.20.2",
        "accelerate==1.15.0",
        "pillow==11.3.0",
        "numpy==2.5.4",
        "safetensors==0.8.0",
        "huggingface_hub",
        "peft",
    )
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1", "PYTHONPATH": "/root"})
    .add_local_python_source("src")
    .add_local_file(str(ROOT / "benchmarks" / "modal_common.py"), remote_path="/root/modal_common.py")
)
LORA_TARGET = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


@app.function(
    gpu=GPU,
    image=image,
    cpu=CPU,
    memory=MEM,
    timeout=TIMEOUT,
    volumes={"/cache": hf_cache, "/models": models, "/data": data_vol},
)
def finetune(
    model_id: str,
    tag: str,
    epochs: int,
    lr: float,
    rank: int,
    grad_accum: int,
    max_pixels: int,
    max_examples: int,
    sets: str = "sroie_ft",
    dev_sets: str = "sroie_dev:40",
    seed: int = 0,
) -> dict:
    import json as _json
    import math
    import random

    import peft
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.extract.vlm import build_prompt

    t_start = time.time()
    proc = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=LORA_TARGET, task_type="CAUSAL_LM"
        ),
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    load_s = time.time() - t_start

    orders = {
        "scanned_receipt": ("company", "date", "address", "total"),
        "cord_receipt": ("total_price", "subtotal_price", "tax_price", "menu"),
        "xfund_form": ("pairs",),
    }

    def example(folder: str, rec: dict):
        doc_type = rec.get("doc_type", "scanned_receipt")
        prompt, order = build_prompt(doc_type), orders[doc_type]
        img = Image.open(f"/data/{folder}/{rec['file']}").convert("RGB")
        w, h = img.size
        if w * h > max_pixels:
            s = math.sqrt(max_pixels / (w * h))
            img = img.resize((max(int(w * s), 28), max(int(h * s), 28)), Image.LANCZOS)
        answer = _json.dumps(
            {k: rec["entities"][k] for k in order}, ensure_ascii=False, indent=None if doc_type == "xfund_form" else 2
        )
        user = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        full = user + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        enc = proc.apply_chat_template(full, tokenize=True, return_dict=True, return_tensors="pt")
        n_prompt = proc.apply_chat_template(
            user, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :n_prompt] = -100
        enc["labels"] = labels
        return {k: v.to("cuda") for k, v in enc.items()}

    def load_sets(spec: str):
        """ "folder:count,folder" -> [(folder, record)]; count takes a fixed random subset (seeded)."""
        out = []
        for part in spec.split(","):
            folder, _, cnt = part.partition(":")
            recs = _json.loads(pathlib.Path(f"/data/{folder}/labels.json").read_text(encoding="utf-8"))
            if cnt:
                recs = random.Random(seed).sample(recs, min(int(cnt), len(recs)))
            out += [(folder, r) for r in recs]
        return out

    train, dev = load_sets(sets), load_sets(dev_sets)
    if max_examples:
        random.Random(seed).shuffle(train)
        train, dev = train[:max_examples], dev[: max(2, max_examples // 4)]
    steps_total = max(1, math.ceil(len(train) * epochs / grad_accum))
    warm = max(1, int(0.05 * steps_total))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    def lr_at(step):
        if step < warm:
            return lr * (step + 1) / warm
        return lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps_total - warm)))

    def dev_loss():
        model.eval()
        tot = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for folder, rec in dev:
                tot += float(model(**example(folder, rec)).loss)
        model.train()
        return tot / len(dev)

    rng = random.Random(seed)
    history, step, micro, run_loss, n_run = [], 0, 0, 0.0, 0
    model.train()
    t_train = time.time()
    for ep in range(epochs):
        idx = list(range(len(train)))
        rng.shuffle(idx)
        for i in idx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**example(*train[i])).loss
            (loss / grad_accum).backward()
            run_loss += float(loss)
            n_run += 1
            micro += 1
            if micro % grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0:
                    history.append(
                        {
                            "step": step,
                            "epoch": ep + (idx.index(i) + 1) / len(idx),
                            "train_loss": run_loss / n_run,
                            "lr": lr_at(step),
                            "elapsed_s": time.time() - t_train,
                        }
                    )
                    print(history[-1], flush=True)
                    run_loss, n_run = 0.0, 0
        d = dev_loss()
        history.append({"epoch_end": ep + 1, "dev_loss": d, "elapsed_s": time.time() - t_train})
        print(history[-1], flush=True)
        # checkpoint + log to the volume every epoch: a lost client (or a run started with --detach) loses nothing
        model.save_pretrained(f"/models/{tag}")
        with open(f"/models/{tag}/train_log.json", "w", encoding="utf-8") as fh:
            _json.dump({"tag": tag, "model": model_id, "epochs_done": ep + 1, "epochs": epochs, "history": history}, fh)
        models.commit()
    model.save_pretrained(f"/models/{tag}")
    models.commit()
    hf_cache.commit()
    return {
        "tag": tag,
        "model": model_id,
        "gpu": torch.cuda.get_device_name(0),
        "peft": peft.__version__,
        "torch": torch.__version__,
        "trainable_params": n_train,
        "all_params": n_all,
        "load_seconds": load_s,
        "train_seconds": time.time() - t_train,
        "train_examples": len(train),
        "dev_examples": len(dev),
        "epochs": epochs,
        "lr": lr,
        "rank": rank,
        "grad_accum": grad_accum,
        "optimizer_steps": step,
        "max_pixels": max_pixels,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
        "history": history,
    }


@app.local_entrypoint()
def main(
    tag: str = "ft3b",
    epochs: int = 3,
    lr: float = 1e-4,
    rank: int = 16,
    grad_accum: int = 8,
    max_examples: int = 0,
    per_example_s: int = 4,
    sets: str = "sroie_ft",
    dev_sets: str = "sroie_dev:40",
    n_train: int = 506,
):
    n = max_examples or n_train
    mc.guard(f"finetune {tag}", 900 + n * epochs * per_example_s, GPU, CPU, MEM)
    t0 = time.time()
    r = finetune.remote(MODEL, tag, epochs, lr, rank, grad_accum, MAX_PIXELS, max_examples, sets, dev_sets)
    wall = time.time() - t0
    est = (wall + 60) * mc.cost_per_second(GPU, CPU, MEM)
    mc.ledger(
        {
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "suite": f"finetune/{tag}",
            "model": MODEL,
            "gpu": r["gpu"],
            "examples": r["train_examples"],
            "epochs": epochs,
            "wall_seconds": round(wall),
            "est_cost_usd": round(est, 3),
        }
    )
    out = ROOT / "results" / "raw" / f"ft_{tag}{'_smoke' if max_examples else ''}.json"
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(
        f"done in {wall:.0f}s (estimated ${est:.2f}; total spent ${mc.spent():.2f} of ${mc.BUDGET:.2f}); "
        f"peak {r['peak_gpu_gb']:.1f} GB; final dev loss {[h for h in r['history'] if 'dev_loss' in h][-1]['dev_loss']:.4f}"
    )
