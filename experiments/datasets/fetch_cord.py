"""Download CORD-v2 (real Indonesian shop receipts, CC-BY-4.0) from the Hugging Face datasets server.

    python experiments/datasets/fetch_cord.py            # train (800), validation (100), test (100) -> data/real/cord_<split>

Target per receipt: total_price, subtotal_price, tax_price (strings or null) and menu = [{name, quantity, price}],
taken from CORD's `gt_parse`.
"""

import json
import pathlib
import sys
import time
import urllib.request

BASE = "https://datasets-server.huggingface.co/rows?dataset=naver-clova-ix/cord-v2&config=default&split=%s&offset=%d&length=10"


def get(url: str, tries: int = 8):
    """urlopen with exponential backoff: the datasets server returns the odd 500/429 on large rows."""
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=90)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2**k)


def as_list(x):
    return x if isinstance(x, list) else ([x] if x else [])


def target(gt: dict) -> dict:
    parse = gt["gt_parse"]
    tot, sub = parse.get("total") or {}, parse.get("sub_total") or {}
    menu = []
    for m in as_list(parse.get("menu")):
        if isinstance(m, dict) and (m.get("nm") or m.get("price")):
            menu.append(
                {
                    "name": str(m.get("nm") or ""),
                    "quantity": str(m.get("cnt") or ""),
                    "price": str(m.get("price") or ""),
                }
            )
    pick = lambda d, k: str(d[k]) if d.get(k) not in (None, "") else None  # noqa: E731
    return {
        "total_price": pick(tot, "total_price"),
        "subtotal_price": pick(sub, "subtotal_price"),
        "tax_price": pick(sub, "tax_price"),
        "menu": menu,
    }


PARQUET = "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/refs%2Fconvert%2Fparquet/default/{split}/{i:04d}.parquet"


def from_parquet(split: str, shards: int) -> None:
    """Fallback when the rows API keeps returning 500: read the auto-converted parquet shards directly."""

    import pyarrow.parquet as pq

    out = pathlib.Path(f"data/real/cord_{split}")
    out.mkdir(parents=True, exist_ok=True)
    labels, k = [], 0
    for i in range(shards):
        tmp = pathlib.Path(f"data/real/_cord_{split}_{i}.parquet")
        with get(PARQUET.format(split=split, i=i)) as r, tmp.open("wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        for row in pq.read_table(tmp).to_pylist():
            name = f"{k:04d}.jpg"
            (out / name).write_bytes(row["image"]["bytes"])
            labels.append(
                {"file": name, "doc_type": "cord_receipt", "entities": target(json.loads(row["ground_truth"]))}
            )
            k += 1
        tmp.unlink()
        print(split, "shard", i, k, flush=True)
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    print(split, len(labels), flush=True)


def main():
    if sys.argv[1:2] == ["--parquet"]:
        from_parquet(sys.argv[2], int(sys.argv[3]))
        return
    for split in sys.argv[1:] or ["train", "validation", "test"]:
        out = pathlib.Path(f"data/real/cord_{split}")
        out.mkdir(parents=True, exist_ok=True)
        labels, off = [], 0
        while True:
            d = json.load(get(BASE % (split, off)))
            rows = d["rows"]
            if not rows:
                break
            for r in rows:
                name = f"{r['row_idx']:04d}.jpg"
                (out / name).write_bytes(get(r["row"]["image"]["src"]).read())
                labels.append(
                    {"file": name, "doc_type": "cord_receipt", "entities": target(json.loads(r["row"]["ground_truth"]))}
                )
            off += len(rows)
            if off >= d["num_rows_total"]:
                break
        (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
        print(split, len(labels), flush=True)


if __name__ == "__main__":
    main()
