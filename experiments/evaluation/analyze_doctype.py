"""
Document-type stage: train the 17-class linear probe on DiT embeddings, calibrate it on dev, evaluate on everything it
never saw, and write the config the service loads (configs/doctype.json, configs/doctype_probe.npz).

Train : RVL-CDIP (100 docs per class, from the val split) + SROIE fine-tune receipts (+ CORD train unless --receipts sroie)
Dev   : RVL-CDIP dev (60 per class) + SROIE dev receipts: picks the regularisation, fits the confidence calibration
        (isotonic: top probability -> P(prediction is right)) and the accept threshold
Test  : RVL-CDIP test (100 per class), SROIE test receipts, CORD test receipts (a different country and receipt
        format: never trained on), XFUND forms (modern forms: also never trained on)

Needs the caches from experiments/evaluation/embed_doctype.py.

    python experiments/evaluation/analyze_doctype.py --out results/raw/doctype_compare.json
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
from src.doctype.classifier import CLASSES, DIT_LABELS, RVL_CLASSES, softmax  # noqa: E402
from src.doctype.router import ROUTES  # noqa: E402
from src.eval.calibration import choose_accept, ece  # noqa: E402
from src.pipeline.calibration import IsotonicMap  # noqa: E402

CACHE = ROOT / "data" / "real" / "doctype_cache"


def load(name):
    z = np.load(CACHE / f"{name}.npz", allow_pickle=False)
    return z["emb"], z["logits"], z["labels"].tolist()


def stack(names):
    parts = [load(n) for n in names]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), sum((p[2] for p in parts), [])


def boot_acc(ok, n=2000, seed=0):
    rng = random.Random(seed)
    ok = list(map(float, ok))
    m = sorted(sum(rng.choice(ok) for _ in ok) / len(ok) for _ in range(n))
    return [m[int(0.025 * n)], m[int(0.975 * n)]]


def fit(X, y, C, mu, sd):
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit((X - mu) / sd, y)
    W = clf.coef_ / sd  # fold the standardisation into the weights
    b = clf.intercept_ - (W * mu).sum(axis=1)
    return W, b, list(clf.classes_)


def aligned(W, b, classes):
    """Reorder to CLASSES (a class absent from the training data would get a very negative bias)."""
    Wf, bf = np.zeros((len(CLASSES), W.shape[1])), np.full(len(CLASSES), -50.0)
    for i, c in enumerate(classes):
        Wf[CLASSES.index(c)], bf[CLASSES.index(c)] = W[i], b[i]
    return Wf, bf


def truth_route(label: str) -> str:
    return ROUTES.get("form" if label == "form_xfund" else label, "human")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--receipts",
        choices=["sroie", "sroie+cord"],
        default="sroie+cord",
        help="which receipt formats the probe trains on; with 'sroie', CORD is a never-seen format",
    )
    ap.add_argument("--write-config", action="store_true", help="write configs/doctype.json and the probe weights")
    a = ap.parse_args()
    with_cord = a.receipts == "sroie+cord"

    Xtr, _, ytr = stack(["rvlcdip_train", "sroie_ft"] + (["cord_train"] if with_cord else []))
    Xdv, _, ydv = stack(["rvlcdip_dev", "sroie_dev"] + (["cord_validation"] if with_cord else []))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6

    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        W, b, cl = fit(Xtr, ytr, C, mu, sd)
        Wf, bf = aligned(W, b, cl)
        acc = float(np.mean(np.array(CLASSES)[(Xdv @ Wf.T + bf).argmax(1)] == np.array(ydv)))
        if best is None or acc > best[0]:
            best = (acc, C, Wf, bf)
    dev_acc, C, W, b = best
    pdv = softmax(Xdv @ W.T + b)
    top, pred = pdv.max(1), np.array(CLASSES)[pdv.argmax(1)]
    right = (pred == np.array(ydv)).astype(float)
    cal = IsotonicMap.fit(top, right)
    calp = np.array([cal(t) for t in top])
    accept = choose_accept(calp, right, 0.95)

    res = {
        "_run": {
            "probe": "logistic regression on 768-d DiT embeddings, 17 classes",
            "C": C,
            "dev_accuracy": dev_acc,
            "train_docs": len(ytr),
            "dev_docs": len(ydv),
            "receipt_formats_in_training": a.receipts,
            "accept_threshold_for_95pct_dev_precision": accept,
        }
    }

    # --- RVL-CDIP test: native DiT head against the probe
    Xt, Lt, yt = stack(["rvlcdip_test"])
    yt_arr = np.array(yt)
    native = np.array(DIT_LABELS)[Lt.argmax(1)]
    pt = softmax(Xt @ W.T + b)
    probe_pred = np.array(CLASSES)[pt.argmax(1)]
    nat_ok, pr_ok = native == yt_arr, probe_pred == yt_arr
    per_class = {
        c: {"native": float(nat_ok[yt_arr == c].mean()), "probe": float(pr_ok[yt_arr == c].mean())} for c in RVL_CLASSES
    }
    conf = Counter((t, p) for t, p in zip(yt, probe_pred, strict=True) if t != p)
    res["rvlcdip_test"] = {
        "n": len(yt),
        "per_class_n": 100,
        "native_dit_accuracy": float(nat_ok.mean()),
        "native_ci95": boot_acc(nat_ok),
        "probe_accuracy": float(pr_ok.mean()),
        "probe_ci95": boot_acc(pr_ok),
        "probe_predicted_receipt": float((probe_pred == "receipt").mean()),
        "per_class": per_class,
        "top_confusions": [[t, p, n] for (t, p), n in conf.most_common(8)],
    }

    # --- receipts (SROIE test; CORD test and validation: a different format, never trained on)
    receipt_sets = [
        ("sroie_test", "sroie_test_receipts"),
        ("cord_test", "cord_test_receipts" if with_cord else "cord_test_receipts_unseen_format"),
    ]
    if not with_cord:
        receipt_sets.append(("cord_validation", "cord_validation_receipts_unseen_format"))
    for name, key in receipt_sets:
        X, L, _ = load(name)
        p = softmax(X @ W.T + b)
        res[key] = {
            "n": len(X),
            "probe_predicted_receipt": float((p.argmax(1) == CLASSES.index("receipt")).mean()),
            "probe_predictions": dict(Counter(np.array(CLASSES)[p.argmax(1)]).most_common(5)),
            "native_dit_predictions": dict(Counter(np.array(DIT_LABELS)[L.argmax(1)]).most_common(5)),
        }

    # --- XFUND forms: modern forms, never trained on
    X, L, _ = load("xfund_test")
    p = softmax(X @ W.T + b)
    pr = np.array(CLASSES)[p.argmax(1)]
    res["xfund_test_forms"] = {
        "n": len(X),
        "probe_predictions": dict(Counter(pr).most_common(6)),
        "native_dit_predictions": dict(Counter(np.array(DIT_LABELS)[L.argmax(1)]).most_common(6)),
        "routed_to_form_extractor": float(np.mean([truth_route(c) == "form" for c in pr])),
        "wrongly_routed_to_receipt_extractor": float((pr == "receipt").mean()),
    }

    # --- calibration and routing on the combined test set
    sets = ["rvlcdip_test", "sroie_test", "cord_test", "xfund_test"]
    X, _, y = stack(sets)
    p = softmax(X @ W.T + b)
    top, pred = p.max(1), np.array(CLASSES)[p.argmax(1)]
    calt = np.array([cal(t) for t in top])
    known = np.array([label != "form_xfund" for label in y])
    right = (pred == np.array(y)).astype(float)
    res["calibration_on_known_labels"] = {
        "n": int(known.sum()),
        "ece_raw": ece(top[known], right[known]),
        "ece_calibrated": ece(calt[known], right[known]),
    }
    route_pred = np.array(
        [("human" if c < accept else ROUTES.get(pd, "human")) for c, pd in zip(calt, pred, strict=True)]
    )
    route_true = np.array([truth_route(label) for label in y])
    out = {"accept_threshold": accept, "n": len(y)}
    for r in ("receipt", "form", "human"):
        sel = route_pred == r
        out[r] = {
            "routed": int(sel.sum()),
            "precision": float((route_true[sel] == r).mean()) if sel.any() else None,
            "recall": float((route_pred[route_true == r] == r).mean()) if (route_true == r).any() else None,
        }
    out["receipts_sent_to_the_wrong_extractor"] = int(((route_true == "receipt") & np.isin(route_pred, ["form"])).sum())
    out["by_source"] = {}
    idx, i0 = {}, 0
    for s in sets:
        n = len(load(s)[2])
        idx[s] = (i0, i0 + n)
        i0 += n
    for s, (lo, hi) in idx.items():
        out["by_source"][s] = {
            "n": hi - lo,
            "routing_correct": float((route_pred[lo:hi] == route_true[lo:hi]).mean()),
            "to_human": float((route_pred[lo:hi] == "human").mean()),
        }
    res["routing"] = out

    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    cfgdir = ROOT / "configs"
    if a.write_config:
        write_config(cfgdir, W, b, cal, accept)
    print(json.dumps({k: v for k, v in res.items() if k != "rvlcdip_test"}, indent=1)[:3000])
    r = res["rvlcdip_test"]
    print(
        "RVL test: native",
        round(r["native_dit_accuracy"], 3),
        "probe",
        round(r["probe_accuracy"], 3),
        "probe->receipt",
        r["probe_predicted_receipt"],
    )


def write_config(cfgdir, W, b, cal, accept):
    """The config the service loads."""
    np.savez_compressed(cfgdir / "doctype_probe.npz", W=W.astype(np.float32), b=b.astype(np.float32))
    (cfgdir / "doctype.json").write_text(
        json.dumps(
            {
                "model": "microsoft/dit-base-finetuned-rvlcdip",
                "classes": CLASSES,
                "probe_file": "doctype_probe.npz",
                "map": cal.to_dict(),
                "accept": accept,
                "fitted_on": "RVL-CDIP val (100 per class) + SROIE and CORD receipts; calibrated on a disjoint dev set",
                "routes": ROUTES,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
