"""Download SROIE receipts (real scanned receipts, CC-BY-4.0) from the Hugging Face datasets server.

python experiments/datasets/fetch_real.py test     # 361 receipts -> data/real/sroie_test
python experiments/datasets/fetch_real.py train    # 626 receipts -> data/real/sroie_train
"""

import json
import pathlib
import sys
import urllib.request

split = sys.argv[1] if len(sys.argv) > 1 else "test"
out = pathlib.Path(f"data/real/sroie_{split}")
out.mkdir(parents=True, exist_ok=True)
base = (
    "https://datasets-server.huggingface.co/rows?dataset=jsdnrs/ICDAR2019-SROIE&config=default&split="
    + split
    + "&offset=%d&length=50"
)
labels, off = [], 0
while True:
    d = json.load(urllib.request.urlopen(base % off, timeout=60))
    rows = d["rows"]
    if not rows:
        break
    for r in rows:
        row = r["row"]
        name = f"{r['row_idx']:04d}.jpg"
        (out / name).write_bytes(urllib.request.urlopen(row["image"]["src"], timeout=60).read())
        labels.append({"file": name, "key": row["key"], "entities": row["entities"]})
    off += len(rows)
    print(off, flush=True)
    if off >= d["num_rows_total"]:
        break
(out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved", len(labels), "to", out)
