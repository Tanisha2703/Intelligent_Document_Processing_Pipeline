"""Upload handling: validate a file, and turn any supported image into what the model was fine-tuned on."""

import io
import math
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
# The budget the adapters were fine-tuned with; it also sets the image-token count, and so the time per page.
MAX_PIXELS = 1_400_000


def read_upload(file: UploadFile, max_file_mb: float) -> bytes:
    """The uploaded bytes, or an HTTP error: 415 for a non-image suffix, 413 for a file over the size limit."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise HTTPException(415, f"unsupported file type {suffix!r}; send an image (png, jpg, tiff)")
    data = file.file.read()
    if len(data) > max_file_mb * 1024 * 1024:
        raise HTTPException(413, f"file larger than {max_file_mb} MB")
    return data


def prepare_image(data: bytes, max_pixels: int = MAX_PIXELS) -> bytes:
    """Any supported image -> RGB PNG of at most `max_pixels` pixels."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        k = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(int(w * k), 28), max(int(h * k), 28)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
