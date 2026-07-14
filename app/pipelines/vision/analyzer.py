"""Pipeline 4: Vision-based product/package issue classifier.

Flow:
  1. Validate image (size, MIME type)
  2. Call Groq Vision with a structured classification prompt
  3. Parse JSON response → ImageAnalysis
  4. Apply confidence thresholds to decide next action:
     - confidence >= vision_high_confidence  → auto-escalate
     - vision_low_confidence <= conf < high  → ask user to confirm
     - confidence < vision_low_confidence    → request better image
"""

import json
import re
from dataclasses import dataclass

from app.config import settings
from app.llm import vision as vision_llm
from app.llm.prompts import SUPPORT_CONTACT
from app.helpers.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

ISSUE_TYPES = [
    "damaged_packaging",
    "damaged_product",
    "wrong_item",
    "missing_components",
    "tampering",
    "no_issue",
    "invalid_image",
]

_CLASSIFICATION_PROMPT = """\
You are a customer support agent reviewing a photo submitted by a customer about their order.

Carefully examine the image and classify it into exactly one of these issue types:
- damaged_packaging: outer box or packaging is crushed, torn, wet, or visibly damaged
- damaged_product: the product itself is broken, cracked, scratched, or non-functional in appearance
- wrong_item: the item received is clearly different from what was expected (wrong model, color, size)
- missing_components: the product is present but accessories, parts, or items are visibly absent
- tampering: seal is broken, package appears previously opened, or signs of interference
- no_issue: product and packaging appear to be in good condition
- invalid_image: image is too blurry, dark, unrelated, or cannot be used to assess the product

Respond ONLY with a JSON object in this exact format (no markdown, no explanation):
{"issue_type": "<one of the types above>", "confidence": <0.0 to 1.0>, "description": "<one sentence for the customer>", "evidence": "<what specifically you observed in the image>"}
"""


@dataclass
class ImageAnalysis:
    issue_type: str
    confidence: float
    description: str
    evidence: str
    should_escalate: bool


def analyze_product_image(image_bytes: bytes, mime_type: str) -> ImageAnalysis:
    """Run the full vision pipeline and return a structured analysis."""
    _validate_image(image_bytes, mime_type)

    raw = vision_llm.analyze_image(image_bytes, mime_type, _CLASSIFICATION_PROMPT)
    analysis = _parse_response(raw)

    analysis.should_escalate = (
        analysis.confidence >= settings.vision_high_confidence
        and analysis.issue_type not in ("no_issue", "invalid_image")
    )

    logger.info(
        "vision analysis: issue=%s confidence=%.2f escalate=%s",
        analysis.issue_type, analysis.confidence, analysis.should_escalate,
    )
    return analysis


def build_follow_up_message(analysis: ImageAnalysis) -> str:
    """Return the message to show the user based on confidence tier."""
    issue_label = analysis.issue_type.replace("_", " ")

    if analysis.issue_type == "invalid_image":
        return (
            "I wasn't able to assess your image clearly. "
            "Please upload a well-lit, focused photo of the product or packaging."
        )

    if analysis.issue_type == "no_issue":
        return (
            f"Your {issue_label} — the product and packaging appear to be in good condition based on the photo. "
            "If you're still experiencing a problem, please describe it in text so I can help further."
        )

    if analysis.confidence >= settings.vision_high_confidence:
        return (
            f"I can see **{issue_label}** in your photo — {analysis.description}\n\n"
            f"I'm flagging this for our support team. "
            f"You can also reach us directly at {SUPPORT_CONTACT}."
        )

    if analysis.confidence >= settings.vision_low_confidence:
        return (
            f"I notice what might be **{issue_label}** — {analysis.description}\n\n"
            "Could you confirm: is this the issue you're experiencing? "
            "A reply of 'yes' will escalate this to our support team."
        )

    return (
        "I couldn't clearly identify an issue from this image. "
        "Please upload a clearer, well-lit photo, or describe the problem in text."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_image(image_bytes: bytes, mime_type: str) -> None:
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise ValueError(f"Image too large ({len(image_bytes) // (1024*1024)}MB). Max is {settings.max_image_size_mb}MB.")
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(f"Unsupported image type: {mime_type}. Use JPEG, PNG, or WebP.")


def _parse_response(raw: str) -> ImageAnalysis:
    """Extract JSON from model output and convert to ImageAnalysis."""
    # Strip markdown code fences if the model adds them
    json_str = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("vision model returned non-JSON: %s", raw[:200])
        return ImageAnalysis(
            issue_type="invalid_image",
            confidence=0.0,
            description="Could not parse the image analysis result.",
            evidence=raw[:200],
            should_escalate=False,
        )

    issue_type = data.get("issue_type", "invalid_image")
    if issue_type not in ISSUE_TYPES:
        issue_type = "invalid_image"

    return ImageAnalysis(
        issue_type=issue_type,
        confidence=float(data.get("confidence", 0.0)),
        description=str(data.get("description", "")),
        evidence=str(data.get("evidence", "")),
        should_escalate=False,  # set by caller after threshold check
    )
