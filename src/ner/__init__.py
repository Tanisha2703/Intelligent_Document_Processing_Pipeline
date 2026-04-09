"""NER sub-package: model inference, post-processing, entity definitions."""
from .model import NERModel
from .entities import ENTITY_TYPES, IOB2_LABELS, ENTITY_COLORS

__all__ = ["NERModel", "ENTITY_TYPES", "IOB2_LABELS", "ENTITY_COLORS"]
