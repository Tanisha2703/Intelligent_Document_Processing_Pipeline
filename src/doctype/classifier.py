"""
Document-type classification: what kind of document is this?

Stage one of the pipeline. DiT (microsoft/dit-base-finetuned-rvlcdip, MIT) was fine-tuned on RVL-CDIP and knows 16
document classes, but none of them is "receipt". A linear probe on DiT's pooled image embedding adds that class
(and re-fits the others) on real documents: 100 RVL-CDIP documents per class plus receipts from SROIE.

Two layers so most of it runs without torch:
  - `Probe`: numpy only. softmax(W @ embedding + b) over the 17 classes, then an isotonic map turns the top
    probability into a calibrated probability that the predicted class is right.
  - `DiTEmbedder`: loads the DiT model (torch + transformers, CPU is fine) and returns the pooled embedding that
    feeds DiT's own classification head, plus that head's 16 logits.
"""

import json
from pathlib import Path

import numpy as np

from src.pipeline.calibration import IsotonicMap

RVL_CLASSES = [
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
CLASSES = RVL_CLASSES + ["receipt"]
# The DiT checkpoint numbers its 16 outputs in the original RVL-CDIP order, not alphabetically like the dataset copy used here.
DIT_LABELS = [
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific report",
    "scientific publication",
    "specification",
    "file folder",
    "news article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
]
MODEL_ID = "microsoft/dit-base-finetuned-rvlcdip"
CONFIG = Path(__file__).parents[2] / "configs" / "doctype.json"


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Probe:
    """Linear classifier over DiT embeddings with a calibrated confidence."""

    def __init__(
        self, W: np.ndarray, b: np.ndarray, calibration: IsotonicMap, accept: float, classes: list[str] = CLASSES
    ):
        self.W, self.b, self.cal, self.accept, self.classes = (
            np.asarray(W, float),
            np.asarray(b, float),
            calibration,
            accept,
            classes,
        )

    def proba(self, emb: np.ndarray) -> np.ndarray:
        return softmax(np.atleast_2d(emb) @ self.W.T + self.b)

    def predict(self, emb: np.ndarray) -> dict:
        """The most likely class for one embedding: raw probability, calibrated confidence, and all class scores."""
        p = self.proba(emb)[0]
        k = int(p.argmax())
        return {
            "doc_type": self.classes[k],
            "probability": float(p[k]),
            "confidence": float(self.cal(p[k])),
            "scores": {c: float(x) for c, x in zip(self.classes, p, strict=True)},
        }

    @classmethod
    def load(cls, path: str | Path = CONFIG) -> "Probe":
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        w = np.load(path.parent / d["probe_file"])
        return cls(w["W"], w["b"], IsotonicMap.from_dict(d["map"]), d["accept"], d["classes"])


class DiTEmbedder:
    """DiT image classifier on CPU: images in, (pooled embedding, 16 DiT logits) out."""

    def __init__(self, model_id: str = MODEL_ID):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_id).eval()
        self._pooled = {}
        # the input of the classification head is the pooled embedding; capture it with a hook
        self.model.classifier.register_forward_hook(lambda m, inp, out: self._pooled.__setitem__("x", inp[0].detach()))

    def __call__(self, images: list) -> tuple[np.ndarray, np.ndarray]:
        with self.torch.no_grad():
            enc = self.proc(images=[im.convert("RGB") for im in images], return_tensors="pt")
            logits = self.model(**enc).logits
        return self._pooled["x"].numpy().astype(np.float32), logits.numpy().astype(np.float32)
