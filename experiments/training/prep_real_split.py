"""
Split the SROIE train receipts into a fine-tuning set and a company-disjoint dev set, and tag the test receipts
whose company never appears in the fine-tuning set ("unseen shop"). Deterministic.

Why: the same shop prints near-identical receipts, so a model can score well by memorising shop layouts. The dev set
(used to fit calibration and choose thresholds) shares no company with the fine-tuning set, and the test results are
reported overall AND on the unseen-shop subset.

    python experiments/training/prep_real_split.py
"""

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "real"


def norm(c: str) -> str:
    return " ".join(c.upper().split())


def main(dev_target: int = 120, seed: int = 0):
    train = json.loads((DATA / "sroie_train" / "labels.json").read_text(encoding="utf-8"))
    test = json.loads((DATA / "sroie_test" / "labels.json").read_text(encoding="utf-8"))
    by_company: dict[str, list] = {}
    for r in train:
        by_company.setdefault(norm(r["entities"]["company"]), []).append(r)
    companies = sorted(by_company)
    random.Random(seed).shuffle(companies)
    dev, dev_companies = [], set()
    for c in companies:
        if len(dev) >= dev_target:
            break
        if len(by_company[c]) <= 6:  # keep the big chains in fine-tuning
            dev += by_company[c]
            dev_companies.add(c)
    ft = [r for r in train if norm(r["entities"]["company"]) not in dev_companies]
    ft_companies = {norm(r["entities"]["company"]) for r in ft}
    for name, recs in (("sroie_ft", ft), ("sroie_dev", dev)):
        out = DATA / name
        out.mkdir(parents=True, exist_ok=True)
        for r in recs:
            shutil.copy(DATA / "sroie_train" / r["file"], out / r["file"])
        (out / "labels.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    unseen = [r["file"] for r in test if norm(r["entities"]["company"]) not in ft_companies]
    info = {
        "train_receipts": len(train),
        "fine_tune": len(ft),
        "dev": len(dev),
        "dev_companies": len(dev_companies),
        "test_receipts": len(test),
        "test_unseen_shop": len(unseen),
        "test_seen_shop": len(test) - len(unseen),
        "dev_company_overlap_with_ft": len(dev_companies & ft_companies),
        "test_unseen_shop_files": unseen,
    }
    (DATA / "split_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print({k: v for k, v in info.items() if k != "test_unseen_shop_files"})


if __name__ == "__main__":
    main()
