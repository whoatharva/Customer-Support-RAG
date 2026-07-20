"""Shared helpers for parsing LLM text output."""

import json
import re
from typing import Any

from app.helpers.logger import get_logger

logger = get_logger(__name__)


def strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences some models wrap JSON output in."""
    return re.sub(r"```(?:json)?\s*|\s*```", "", text or "").strip()


def parse_json_lenient(raw: str, default: Any = None, *, slice_from: str | None = None) -> Any:
    """Parse JSON from LLM output, tolerating markdown fences and leading chatter.

    slice_from: optional character (e.g. "[" or "{") — if present in the text,
    parsing starts at its first occurrence (drops any preamble the model added).

    Returns `default` (and logs a warning) when the text is not valid JSON.
    """
    text = strip_json_fences(raw)
    if slice_from and slice_from in text:
        text = text[text.find(slice_from):]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("lenient JSON parse failed | raw=%r", (raw or "")[:200])
        return default
