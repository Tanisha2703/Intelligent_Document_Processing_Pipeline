"""
Compute DiT embeddings and DiT's own 16-class logits for every document set used by the document-type stage, on CPU.
Resumable: a set whose cache file exists is skipped.

    python benchmarks/embed_doctype.py                 # all sets
    python benchmarks/embed_doctype.py rvlcdip_test    # one set

Writes data/real/doctype_cache/<set>.npz (gitignored: regenerable) with `emb`, `logits`, `files` and `labels`
(class names; "receipt" for SROIE and CORD, "form_xfund" for XFUND, which has no label in RVL-CDIP's class list).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.doctype.classifier import DiTEmbedder  # noqa: E402

DATA = ROOT / "data" / "real"
CACHE = DATA / "doctype_cache"
SETS = [
    "rvlcdip_train",
    "rvlcdip_dev",
    "rvlcdip_test",
    "sroie_ft",
    "sroie_dev",
    "sroie_test",
    "cord_train",
    "cord_validation",
    "cord_test",
    "xfund_train",
    "xfund_test",
]


def labels_of(name: str, recs: list[dict]) -> list[str]:
    if name.startswith("rvlcdip"):
        return [r["class"] for r in recs]
    return ["receipt" if name.startswith(("sroie", "cord")) else "form_xfund"] * len(recs)


def main(names: list[str], batch: int = 16):
    CACHE.mkdir(parents=True, exist_ok=True)
    emb = None
    for name in names:
        out = CACHE / f"{name}.npz"
        if out.exists() or not (DATA / name / "labels.json").exists():
            print("skip", name, "(cached)" if out.exists() else "(no data)")
            continue
        emb = emb or DiTEmbedder()
        recs = json.loads((DATA / name / "labels.json").read_text(encoding="utf-8"))
        E, L, t0 = [], [], time.time()
        for i in range(0, len(recs), batch):
            imgs = []
            for r in recs[i : i + batch]:
                with Image.open(DATA / name / r["file"]) as im:
                    imgs.append(im.convert("RGB"))
            e, lg = emb(imgs)
            E.append(e)
            L.append(lg)
            if (i // batch) % 20 == 0:
                print(f"{name} {i + len(imgs)}/{len(recs)} {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(
            out,
            emb=np.concatenate(E),
            logits=np.concatenate(L),
            files=np.array([r["file"] for r in recs]),
            labels=np.array(labels_of(name, recs)),
        )
        print("saved", name, len(recs), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or SETS)
