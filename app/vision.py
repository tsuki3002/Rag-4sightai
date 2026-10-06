"""Read images: local OCR (free, offline) plus an optional vision-LLM description for diagrams and charts."""
import base64
import io
import logging
from functools import lru_cache

import numpy as np
from PIL import Image

from . import llm
from .config import get_settings

log = logging.getLogger(__name__)

DESCRIBE_PROMPT = """This image comes from course material. Describe it for a student who can't see it: \
what kind of figure it is, what it shows, and every label, axis, value or relationship it contains. \
Be factual and concise (under 200 words). Don't speculate beyond what is visible."""


@lru_cache
def _ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr(img: Image.Image) -> str:
    result, _ = _ocr_engine()(np.array(img.convert("RGB")))
    return " ".join(text for _box, text, conf in result or [] if float(conf) > 0.5)


def describe(img: Image.Image) -> str | None:
    model = get_settings().vision_model
    if not model:
        return None
    img = img.convert("RGB")
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    try:
        completion = llm.vision_client().chat.completions.create(
            model=model, temperature=0.1,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": DESCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        )
        return completion.choices[0].message.content
    except Exception as e:  # a failed description shouldn't fail the whole ingest; OCR text still gets indexed
        log.warning("Vision description failed (%s); falling back to OCR only", e)
        return None


def read_image(img: Image.Image, label: str) -> str:
    """Searchable text for an image: description (if a vision model is configured) + OCR'd text."""
    if min(img.size) < 32:  # icons, bullets, spacers
        return ""
    parts = []
    if description := describe(img):
        parts.append(f"Figure description: {description}")
    if text := ocr(img):
        parts.append(f"Text in figure: {text}")
    return f"[Image: {label}]\n" + "\n".join(parts) if parts else ""
