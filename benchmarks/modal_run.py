"""Run the vision-language model on a rented GPU (Modal), with a cost guard and a spend ledger.

  modal run benchmarks/modal_run.py --suite fetch                                    # download the base weights on a CPU container first
  modal run benchmarks/modal_run.py --suite real --data data/real/sroie_test --n 2   # a 2-receipt smoke test (cents)
  modal run benchmarks/modal_run.py --suite real --data data/real/sroie_test --tag real_zs7b_sroie_test   # zero-shot, all receipts
  modal run benchmarks/modal_run.py --suite real --data data/real/sroie_test --tag real_ft3b_sroie_test --adapter ft3b
  modal deploy benchmarks/modal_run.py                                               # the app the service and the demo call

Credentials: Modal reads MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from the environment or ~/.modal.toml; this file never prints them.
Settings (all optional):
  DOCINTEL_GPU          GPU type, default A10G
  DOCINTEL_MODEL        Hugging Face model id, default Qwen/Qwen2.5-VL-7B-Instruct
  DOCINTEL_MAX_PIXELS   image pixel budget per page (more pixels = more image tokens = slower, default 1400000)
  DOCINTEL_BUDGET_USD   total budget across ALL runs; a run is refused if its WORST-CASE cost would exceed what is
                        left after the ledger in results/raw/modal_spend.json (default 15)

What runs where: this file runs on your machine, reads a dataset folder written by benchmarks/fetch_*.py, and sends the
images in chunks to a Modal container holding the model (optionally with a LoRA adapter from the `docintel-models`
volume), which returns the raw generation, the parsed fields and per-field token probabilities. Results are appended
to results/raw/vlm_<tag>.jsonl after every chunk, so an interrupted run resumes where it stopped.
"""

import io
import json
import math
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                v = v.split("#")[0].strip().strip('"').strip("'")
                if v:
                    os.environ.setdefault(k.strip(), v)


_load_dotenv()

import modal  # noqa: E402  (after .env so the client sees the token)

sys.path.insert(0, str(ROOT / "benchmarks"))
import modal_common as mc  # noqa: E402  (price table, budget, ledger and cost guard shared with modal_train.py)
from modal_common import BUDGET, CPU_PRICE, MEM_PRICE  # noqa: E402,F401

GPU = os.environ.get("DOCINTEL_GPU", "A10G")
MODEL = os.environ.get("DOCINTEL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
MAX_PIXELS = int(os.environ.get("DOCINTEL_MAX_PIXELS", "1400000"))
CPU_CORES, MEM_MIB = 4, 16384


def cost_per_second(gpu: str) -> float:
    return mc.cost_per_second(gpu, CPU_CORES, MEM_MIB)


app = modal.App("docintel-vlm")
hf_cache = modal.Volume.from_name("docintel-hf-cache", create_if_missing=True)
models_vol = modal.Volume.from_name("docintel-models", create_if_missing=True)
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


# ---------------------------------------------------------------------------------------------- remote side
@app.function(image=image, cpu=2, memory=8192, volumes={"/cache": hf_cache}, timeout=1800)
def fetch(model_id: str) -> dict:
    """Download the weights into the cached volume on a CPU-only container (a GPU container would bill GPU time
    for the download)."""
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
    hf_cache.commit()
    return {"path": path, "seconds": time.time() - t0}


@app.cls(
    gpu=GPU,
    image=image,
    cpu=CPU_CORES,
    memory=MEM_MIB,
    volumes={"/cache": hf_cache, "/models": models_vol},
    timeout=3600,
    scaledown_window=60,
    max_containers=int(os.environ.get("DOCINTEL_MAX_CONTAINERS", "3")),
)  # caps burst spend
class VLM:
    model_id: str = modal.parameter()
    max_pixels: int = modal.parameter(default=MAX_PIXELS)
    adapter: str = modal.parameter(default="")  # LoRA adapter saved by modal_finetune.py under /models/<name>

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        t0 = time.time()
        self.proc = AutoProcessor.from_pretrained(self.model_id, min_pixels=256 * 28 * 28, max_pixels=self.max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
        ).eval()
        if self.adapter:
            from peft import PeftModel

            models_vol.reload()
            self.model = PeftModel.from_pretrained(self.model, f"/models/{self.adapter}").merge_and_unload().eval()
        hf_cache.commit()
        self.load_seconds = time.time() - t0
        self.gpu = torch.cuda.get_device_name(0)
        self.eos = {self.proc.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}

    @modal.method()
    def info(self) -> dict:
        import torch

        return {
            "gpu": self.gpu,
            "load_seconds": self.load_seconds,
            "torch": torch.__version__,
            "model": self.model_id,
            "max_pixels": self.max_pixels,
            "adapter": self.adapter,
        }

    @modal.method()
    def extract(self, items: list[dict]) -> list[dict]:
        import torch
        from PIL import Image

        from src.extract.vlm import build_prompt, field_confidences, parse_json, pieces_and_logprobs

        tok = self.proc.tokenizer
        out = []
        for it in items:
            t_item = time.time()
            img = Image.open(io.BytesIO(it["image"])).convert("RGB")
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": build_prompt(it["doc_type"])},
                    ],
                }
            ]
            enc = self.proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            ).to("cuda")
            n_in = int(enc["input_ids"].shape[1])
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=int(it.get("max_new_tokens", 320)),
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            torch.cuda.synchronize()
            gen_s = time.time() - t0
            ids = gen.sequences[0, n_in:].tolist()
            lps = self.model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)[0].tolist()
            for k, t in enumerate(ids):  # keep the answer, drop the end-of-turn token and anything after
                if t in self.eos:
                    ids, lps = ids[:k], lps[:k]
                    break
            text = tok.decode(ids, skip_special_tokens=True)
            pieces, plps = pieces_and_logprobs(lambda x: tok.decode(x, skip_special_tokens=True), ids, lps)
            out.append(
                {
                    "id": it["id"],
                    "raw": text,
                    "fields": parse_json(text, it["doc_type"]),
                    "confidence": field_confidences("".join(pieces), pieces, plps, it["doc_type"]),
                    "n_input_tokens": n_in,
                    "n_output_tokens": len(ids),
                    "generate_seconds": gen_s,
                    "item_seconds": time.time() - t_item,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
        return out


# ---------------------------------------------------------------------------------------------- local side
_spent, _ledger = mc.spent, mc.ledger


def _guard(label: str, worst_seconds: float) -> None:
    mc.guard(label, worst_seconds, GPU, CPU_CORES, MEM_MIB)


def _shrink(img, max_pixels: int):
    """Resize on this side so only the pixels the model will use are sent (aspect preserved)."""
    from PIL import Image

    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = math.sqrt(max_pixels / (w * h))
    return img.resize((max(int(w * s), 28), max(int(h * s), 28)), Image.LANCZOS)


def _png(img) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG", optimize=False)
    return b.getvalue()


def _plan_real(n: int, data: str = "data/real/sroie_test", files: list | None = None):
    """Items for a dataset folder written by benchmarks/fetch_*.py: (id, args, build), truth from the dataset labels."""
    from types import SimpleNamespace

    from PIL import Image

    folder = ROOT / data
    recs = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    recs = [r for r in recs if r["file"] in set(files)] if files else recs
    recs = recs[:n] if n else recs

    def build(rec):
        doc = SimpleNamespace(doc_type=rec.get("doc_type", "scanned_receipt"), fields=rec["entities"])
        return doc, None, Image.open(folder / rec["file"]).convert("RGB")

    for rec in recs:
        yield f"real|{pathlib.Path(data).name}|{rec['file']}", (rec,), build


@app.local_entrypoint()
def main(
    suite: str = "real",
    n: int = 0,
    chunk: int = 20,
    tag: str = "",
    per_item_s: int = 25,
    data: str = "data/real/sroie_test",
    adapter: str = "",
):
    t_start = time.time()
    if suite == "fetch":
        r = fetch.remote(MODEL)
        cpu_cost = r["seconds"] * (2 * CPU_PRICE + 8 * MEM_PRICE)
        _ledger(
            {
                "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                "suite": "fetch",
                "model": MODEL,
                "gpu": "none (cpu)",
                "wall_seconds": round(r["seconds"]),
                "est_cost_usd": round(cpu_cost, 4),
            }
        )
        print(f"fetched {MODEL} in {r['seconds']:.0f}s (estimated ${cpu_cost:.3f})")
        return
    if suite != "real":
        raise SystemExit(f"unknown suite {suite!r}; use fetch or real")
    split = pathlib.Path(data).name
    tag = tag or f"{suite}_{MODEL.split('/')[-1]}_{split}"
    out_path = ROOT / "results" / "raw" / f"vlm_{tag}.jsonl"
    done = (
        {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines()}
        if out_path.exists()
        else set()
    )

    plan = list(_plan_real(n, data))
    todo = [p for p in plan if p[0] not in done]
    print(f"suite={suite} model={MODEL} split={split}: {len(plan)} pages, {len(plan) - len(todo)} already done")
    if not todo:
        return
    # worst case: container start + model load/download (10 min) + every page hitting the per-page limit
    _guard(f"{suite}", 600 + len(todo) * per_item_s)

    vlm = VLM(model_id=MODEL, max_pixels=MAX_PIXELS, adapter=adapter)
    info = vlm.info.remote()
    print("container:", info)
    n_done, gen_s = 0, 0.0
    with out_path.open("a", encoding="utf-8") as sink:
        for s in range(0, len(todo), chunk):
            batch = []
            for iid, args, build in todo[s : s + chunk]:
                doc, page, img = build(*args)
                batch.append(
                    {
                        "id": iid,
                        "doc_type": doc.doc_type,
                        "image": _png(_shrink(img, MAX_PIXELS)),
                        "max_new_tokens": 2048 if doc.doc_type == "xfund_form" else 320,
                    }
                )
                batch[-1]["_truth"] = {"truth": doc.fields}
            truth = {b["id"]: b.pop("_truth") for b in batch}
            for r in vlm.extract.remote(batch):
                r.update(truth[r["id"]])
                r["split"], r["doc_type"] = split, next(b["doc_type"] for b in batch if b["id"] == r["id"])
                r["gpu"], r["model"], r["max_pixels"], r["adapter"] = info["gpu"], MODEL, MAX_PIXELS, adapter
                sink.write(json.dumps(r, ensure_ascii=False) + "\n")
                gen_s += r["item_seconds"]
            sink.flush()
            n_done += len(batch)
            print(f"{n_done}/{len(todo)} pages, {time.time() - t_start:.0f}s elapsed", flush=True)

    wall = time.time() - t_start
    # billed time ~ container lifetime: the wall time of this run plus the scale-down window
    est = (wall + 60) * cost_per_second(GPU)
    _ledger(
        {
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "suite": suite,
            "split": split,
            "model": MODEL,
            "gpu": info["gpu"],
            "pages": len(todo),
            "wall_seconds": round(wall),
            "model_load_seconds": round(info["load_seconds"]),
            "est_cost_usd": round(est, 3),
            "price_per_second": cost_per_second(GPU),
        }
    )
    print(
        f"done: {len(todo)} pages in {wall:.0f}s (estimated ${est:.2f}; total spent ${_spent():.2f} of ${BUDGET:.2f})"
    )
