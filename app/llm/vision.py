"""Groq Vision API wrapper for image analysis (Pipeline 4).

Uses the Groq multimodal model to analyze product/package photos.
The same groq_api_key used for lightweight text LLM is reused here.
"""

import base64

from app.config import settings
from app.helpers.logger import get_logger

logger = get_logger(__name__)


def analyze_image(image_bytes: bytes, mime_type: str, prompt: str) -> str:
    """Send an image + prompt to Groq Vision; return the raw text response."""
    if not settings.groq_api_key:
        raise RuntimeError("groq_api_key is not configured — vision analysis unavailable")

    from groq import Groq
    b64 = base64.b64encode(image_bytes).decode()
    client = Groq(api_key=settings.groq_api_key)

    response = client.chat.completions.create(
        model=settings.groq_vision_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            ],
        }],
        max_tokens=512,
        temperature=0.1,
    )
    result = response.choices[0].message.content.strip()
    logger.debug("vision response (%d chars): %s", len(result), result[:120])
    return result
