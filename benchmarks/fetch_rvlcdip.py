"""Download a class-balanced sample of RVL-CDIP (real scanned documents, 16 classes) for the document-type stage.

    python benchmarks/fetch_rvlcdip.py        # 100 test + (100 train + 60 dev) per class -> data/real/rvlcdip_{test,train,dev}

Source: Hugging Face `chainyo/rvl-cdip`, a copy of RVL-CDIP (Harley et al., 2015; derived from the IIT-CDIP collection of
scanned tobacco-industry documents), read from its auto-converted parquet files with HTTP range requests, one row group
(2,600 documents) at a time, so only the groups holding the sampled documents are downloaded (about 2 GB per split).
The converted `train` split is partial and covers only 2 classes, so the probe's training and dev documents come from the
`val` split (disjoint from each other), and the evaluation documents from `test`.

The documents are old, mostly black-and-white scans: they say nothing about modern phone photos.
"""

import json
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

DATASET = "chainyo/rvl-cdip"
CLASSES = [
    "advertisement",
    "budget",
    "email",
    "file folder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "news article",
    "presentation",
    "questionnaire",
    "resume",
    "scientific publication",
    "scientific report",
    "specification",
]
OUT = Path("data/real")


def shard_urls(split: str) -> list[str]:
    for k in range(6):
        try:
            d = json.load(
                urllib.request.urlopen(f"https://datasets-server.huggingface.co/parquet?dataset={DATASET}", timeout=60)
            )
            return sorted(f["url"] for f in d["parquet_files"] if f["split"] == split)
        except Exception:
            time.sleep(2**k)
    raise RuntimeError("could not list the parquet files")


def index(split: str):
    """[(shard url, row group, row within group, label)] for every document, reading only the label column."""
    rows = []
    for url in shard_urls(split):
        pf = pq.ParquetFile(fsspec.open(url, "rb").open())
        labels = pf.read(columns=["label"]).column("label").to_pylist()
        pos = 0
        for g in range(pf.metadata.num_row_groups):
            for r in range(pf.metadata.row_group(g).num_rows):
                rows.append((url, g, r, labels[pos]))
                pos += 1
    return rows


def pick(rows, per_class: int, seed: int, exclude=frozenset()):
    """Per class, sample from the single row group holding most of its documents (topped up from the next ones)."""
    rng = random.Random(seed)
    chosen = []
    for c in range(len(CLASSES)):
        by_group = defaultdict(list)
        for i, (url, g, _, lab) in enumerate(rows):
            if lab == c and i not in exclude:
                by_group[(url, g)].append(i)
        got = []
        for key in sorted(by_group, key=lambda k: -len(by_group[k])):
            pool = by_group[key][:]
            rng.shuffle(pool)
            got += pool[: per_class - len(got)]
            if len(got) >= per_class:
                break
        chosen.append(got)
    return chosen


def save(rows, groups_of_indices: list[list[int]], name: str, cache: dict):
    out = OUT / name
    if (out / "labels.json").exists():
        print(name, "already downloaded, skipping")  # a finished set is never re-downloaded: reruns resume
        return
    out.mkdir(parents=True, exist_ok=True)
    labels = []
    for c, idxs in enumerate(groups_of_indices):
        for i in idxs:
            url, g, r, lab = rows[i]
            if (url, g) not in cache:
                cache.clear()  # keep one row group in memory at a time
                pf = pq.ParquetFile(fsspec.open(url, "rb").open())
                cache[(url, g)] = pf.read_row_group(g, columns=["image"]).column("image").to_pylist()
            data = cache[(url, g)][r]["bytes"]
            fname = f"{name.split('_')[1]}_{i:05d}" + (".png" if data[:4] == b"\x89PNG" else ".jpg")
            (out / fname).write_bytes(data)
            labels.append({"file": fname, "label": c, "class": CLASSES[c]})
        print(name, CLASSES[c], len(idxs), flush=True)
    (out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    print(name, "saved", len(labels), flush=True)


def main(n_test: int = 100, n_train: int = 100, n_dev: int = 60):
    test = index("test")
    save(test, pick(test, n_test, 0), "rvlcdip_test", {})
    val = index("val")
    train = pick(val, n_train, 1)
    used = {i for g in train for i in g}
    dev = pick(val, n_dev, 2, exclude=used)
    save(val, sorted_by_group(val, train), "rvlcdip_train", {})
    save(val, sorted_by_group(val, dev), "rvlcdip_dev", {})


def sorted_by_group(rows, per_class):
    """Order each class's indices by row group so consecutive documents come from the same downloaded group."""
    return [sorted(idxs, key=lambda i: (rows[i][0], rows[i][1], rows[i][2])) for idxs in per_class]


if __name__ == "__main__":
    main()
