"""Download XFUND (real forms in 7 languages) and turn the question/answer links into key-value pairs.

    python experiments/datasets/fetch_xfund.py                   # 40 train + 15 val forms per language -> data/real/xfund_train, xfund_test

Source: https://github.com/doc-analysis/XFUND (release v1.0; forms in zh, ja, es, fr, it, de, pt; the `val` split is
used as the test set). A pair is (text of a `question` item, text of an `answer` item it is linked to). Forms are sampled
with a fixed seed. The zip files are read member by member into our own file names (nothing is extracted to disk by
path), and deleted after use.
"""

import json
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

LANGS = ["zh", "ja", "es", "fr", "it", "de", "pt"]
BASE = "https://github.com/doc-analysis/XFUND/releases/download/v1.0/{lang}.{split}.{ext}"
OUT = Path("data/real")


def get(url: str, dest: Path, tries: int = 5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            return
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2**k)


def pairs_of(doc: dict) -> list[dict]:
    items = {it["id"]: it for it in doc["document"]}
    out = []
    for it in doc["document"]:
        if it["label"] != "question":
            continue
        for a, b in it["linking"]:
            other = items.get(b if a == it["id"] else a)
            if other and other["label"] == "answer" and it["text"].strip() and other["text"].strip():
                out.append({"key": it["text"].strip(), "value": other["text"].strip()})
    return out


def main(n_train: int = 40, n_test: int = 15, seed: int = 0):
    tmp = OUT / "_xfund_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for split, name, n in (("train", "xfund_train", n_train), ("val", "xfund_test", n_test)):
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        labels = []
        for lang in LANGS:
            js, zp = tmp / f"{lang}.{split}.json", tmp / f"{lang}.{split}.zip"
            get(BASE.format(lang=lang, split=split, ext="json"), js)
            docs = json.loads(js.read_text(encoding="utf-8"))["documents"]
            docs = [d for d in docs if pairs_of(d)]
            random.Random(seed).shuffle(docs)
            chosen = docs[:n]
            get(BASE.format(lang=lang, split=split, ext="zip"), zp)
            with zipfile.ZipFile(zp) as zf:
                members = {Path(m).name: m for m in zf.namelist() if not m.endswith("/")}
                for d in chosen:
                    fname = d["img"]["fname"]
                    (out / fname).write_bytes(zf.read(members[fname]))
                    labels.append(
                        {"file": fname, "doc_type": "xfund_form", "lang": lang, "entities": {"pairs": pairs_of(d)}}
                    )
            zp.unlink()
            js.unlink()
            print(split, lang, len(chosen), flush=True)
        (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
        print(name, len(labels), "forms", sum(len(x["entities"]["pairs"]) for x in labels), "pairs", flush=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
