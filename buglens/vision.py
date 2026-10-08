"""Stage 5: prepare the screenshot and ask Gemma what it shows."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from buglens import config
from buglens.errors import UnsupportedImage
from buglens.llm.base import LLMClient
from buglens.llm.json_utils import generate_structured
from buglens.models import ScreenshotAnalysis
from buglens.prompts import render


def prepare_image(source: bytes | str | Path) -> bytes:
    """Validate a screenshot, shrink it to at most MAX_IMAGE_SIDE pixels and return PNG bytes."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise UnsupportedImage(f"Screenshot not found: {path}")
        source = path.read_bytes()
    try:
        image = Image.open(io.BytesIO(source))
        image_format = image.format
        image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise UnsupportedImage("The screenshot could not be read as an image.") from exc
    if image_format not in config.ALLOWED_IMAGE_FORMATS:
        raise UnsupportedImage(f"Unsupported image format '{image_format}'.")

    image = _flatten(image)
    image.thumbnail((config.MAX_IMAGE_SIDE, config.MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _flatten(image: Image.Image) -> Image.Image:
    """Convert to RGB, putting transparent pixels on a white background."""
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def _unique(items: list[str], limit: int) -> list[str]:
    """Drop duplicates (ignoring case) while keeping the order."""
    seen: set[str] = set()
    result = []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result[:limit]


def tidy_analysis(analysis: ScreenshotAnalysis, user_text: str) -> ScreenshotAnalysis:
    """Deduplicate lists, cap their sizes and make sure there is at least one search query."""
    queries = _unique(analysis.search_queries, config.MAX_SEARCH_QUERIES)
    if not queries:
        queries = _unique([analysis.apparent_problem or user_text], 1)
    return analysis.model_copy(
        update={
            "visible_text": _unique(analysis.visible_text, config.MAX_LEXICAL_STRINGS),
            "error_messages": _unique(analysis.error_messages, config.MAX_LEXICAL_STRINGS),
            "ui_components": _unique(analysis.ui_components, 20),
            "framework_hints": _unique(analysis.framework_hints, 10),
            "file_hints": _unique(analysis.file_hints, config.MAX_FILE_HINTS),
            "identifiers": _unique(analysis.identifiers, config.MAX_IDENTIFIERS),
            "search_queries": queries,
        }
    )


def analyze_screenshot(llm: LLMClient, image_png: bytes, user_text: str) -> ScreenshotAnalysis:
    """LLM call 1 of 3: describe the screenshot as structured data."""
    prompt = render("vision", user_text=user_text)
    analysis = generate_structured(llm, prompt, ScreenshotAnalysis, image_png=image_png)
    return tidy_analysis(analysis, user_text)


def text_only_analysis(user_text: str) -> ScreenshotAnalysis:
    """Used with --no-image: no LLM call, the user's sentence is the only evidence."""
    return ScreenshotAnalysis(apparent_problem=user_text, search_queries=[user_text])
