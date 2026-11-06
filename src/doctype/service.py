"""Image bytes in, routed document type out: the DiT embedder, the probe and the router together."""

import io

from src.doctype.classifier import CONFIG, DiTEmbedder, Probe
from src.doctype.router import route


class DocTypeService:
    """Image bytes -> document type, calibrated confidence and route (DiT embedder + probe + router)."""

    def __init__(self, embedder, probe: Probe):
        self.embedder, self.probe = embedder, probe

    @classmethod
    def load(cls, config=CONFIG) -> "DocTypeService":
        return cls(DiTEmbedder(), Probe.load(config))

    def classify(self, image_bytes: bytes) -> dict:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as im:
            emb, _ = self.embedder([im.convert("RGB")])
        return route(self.probe.predict(emb), self.probe.accept)
