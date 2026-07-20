"""Refund verification — single combined vision call.

Stage module for Pipeline 4. Sends the customer's photo + their claim to the Groq
vision model in ONE call and returns all signals the orchestrator needs: authenticity,
issue type, whether the visible issue matches the claim, and a confidence score.

Kept deliberately thin and swappable — the orchestrator (analyzer.py) owns the decision
policy; this module only produces evidence.
"""

from dataclasses import dataclass

from app.config import settings
from app.llm import vision as vision_llm
from app.llm.parsing import parse_json_lenient
from app.llm.prompts import REFUND_VERIFICATION_PROMPT
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


@dataclass
class VerificationResult:
    authentic: bool
    authenticity_reason: str
    issue_type: str
    issue_matches_claim: bool
    match_reason: str
    confidence: float
    description: str
    evidence: str
    image_description: str
    contradicts_claim: bool
    claim_verifiable_from_photo: bool


def verify_image(image_bytes: bytes, mime_type: str, claim: str) -> VerificationResult:
    """Run the combined authenticity + issue-match vision check against the claim."""
    _validate_image(image_bytes, mime_type)

    prompt = REFUND_VERIFICATION_PROMPT.format(claim=claim)
    raw = vision_llm.analyze_image(image_bytes, mime_type, prompt)
    return _parse_response(raw)


def _validate_image(image_bytes: bytes, mime_type: str) -> None:
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise ValueError(
            f"Image too large ({len(image_bytes) // (1024 * 1024)}MB). "
            f"Max is {settings.max_image_size_mb}MB."
        )
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(f"Unsupported image type: {mime_type}. Use JPEG, PNG, or WebP.")


def _parse_response(raw: str) -> VerificationResult:
    """Extract JSON from model output. Safe fallback routes to reject/HIL, never crashes."""
    data = parse_json_lenient(raw, slice_from="{")
    if not isinstance(data, dict):
        logger.warning("vision verifier returned non-JSON: %s", raw[:200])
        return VerificationResult(
            authentic=False,
            authenticity_reason="Could not parse the image analysis result.",
            issue_type="invalid_image",
            issue_matches_claim=False,
            match_reason="Analysis unavailable.",
            confidence=0.0,
            description="Could not assess the image.",
            evidence=raw[:200],
            image_description="",
            contradicts_claim=True,
            claim_verifiable_from_photo=False,
        )

    issue_type = data.get("issue_type", "invalid_image")
    if issue_type not in ISSUE_TYPES:
        issue_type = "invalid_image"

    return VerificationResult(
        authentic=bool(data.get("authentic", False)),
        authenticity_reason=str(data.get("authenticity_reason", "")),
        issue_type=issue_type,
        issue_matches_claim=bool(data.get("issue_matches_claim", False)),
        match_reason=str(data.get("match_reason", "")),
        confidence=float(data.get("confidence", 0.0)),
        description=str(data.get("description", "")),
        evidence=str(data.get("evidence", "")),
        image_description=str(data.get("image_description", "")),
        contradicts_claim=bool(data.get("contradicts_claim", False)),
        claim_verifiable_from_photo=bool(data.get("claim_verifiable_from_photo", False)),
    )
