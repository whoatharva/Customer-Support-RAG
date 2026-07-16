"""Pipeline 4: Refund verification orchestrator.

Runs the refund claim through modular stages and applies the decision policy:

  1. Authenticity   — reject if the photo looks fake/manipulated.
  2. Issue match    — reject if the visible issue doesn't match the customer's claim.
  3. Confidence     — score how certain we are the claim is valid.
  4. Decision       — auto-approve (high confidence), auto-reject (low), or Human-in-the-Loop.

Each stage lives in its own module (verifier, hil) so it can be improved or replaced
independently. This orchestrator owns only the decision policy and persistence.
"""

from app.config import settings
from app.pipelines.vision import verifier, hil
from app.pipelines.vision.verifier import VerificationResult
from app.helpers import database
from app.helpers.logger import get_logger

logger = get_logger(__name__)


async def process_refund_claim(
    image_bytes: bytes,
    mime_type: str,
    claim: str,
    user_email: str,
    session_id: str,
) -> dict:
    """Verify a refund claim end-to-end and return the decision.

    Returns: {decision, reason, confidence, issue_type, customer_message}
    where decision is "approved" or "rejected".
    """
    result = verifier.verify_image(image_bytes, mime_type, claim)
    logger.info(
        "refund verify | user=%s authentic=%s issue=%s match=%s contradicts=%s verifiable=%s confidence=%.2f",
        user_email, result.authentic, result.issue_type, result.issue_matches_claim,
        result.contradicts_claim, result.claim_verifiable_from_photo, result.confidence,
    )

    # ── Stage 1: authenticity ────────────────────────────────────────────────
    if not result.authentic:
        return _finalize(
            "rejected", result.authenticity_reason or "The image could not be verified as authentic.",
            result, "auto", user_email, session_id, claim,
        )

    # ── Stage 2: photo contradicts the claim (e.g. item present but claimed missing) ──
    if result.contradicts_claim:
        return _finalize(
            "rejected",
            result.match_reason or "The photo appears to contradict the reported issue.",
            result, "auto", user_email, session_id, claim,
        )

    # ── Stage 3: issue matches the claim ─────────────────────────────────────
    if not result.issue_matches_claim:
        return _finalize(
            "rejected", result.match_reason or "The reported issue was not visible in the image.",
            result, "auto", user_email, session_id, claim,
        )

    # ── Stage 4: claim can't be proven from a photo (absence/missing) → HIL ───
    # A photo can never prove something is missing or was never received; a human must
    # check order/warehouse records rather than auto-approving on the image alone.
    if not result.claim_verifiable_from_photo:
        logger.info("refund verify | user=%s → HIL (claim not verifiable from photo)", user_email)
        decision = await hil.request_human_decision(_hil_summary(user_email, claim, result))
        reason = (
            "Approved by a reviewer after records check." if decision == "approved"
            else "This type of claim can't be confirmed from a photo alone and was declined on review."
        )
        return _finalize(decision, reason, result, "human", user_email, session_id, claim)

    # ── Stage 5 + 6: confidence scoring → decision policy ────────────────────
    if result.confidence >= settings.refund_auto_approve_threshold:
        return _finalize(
            "approved", "Verified with high confidence.",
            result, "auto", user_email, session_id, claim,
        )

    if result.confidence < settings.refund_auto_reject_threshold:
        return _finalize(
            "rejected", "Could not verify the claim with enough confidence.",
            result, "auto", user_email, session_id, claim,
        )

    # ── Borderline → Human-in-the-Loop ───────────────────────────────────────
    logger.info("refund verify | user=%s → HIL (confidence=%.2f)", user_email, result.confidence)
    decision = await hil.request_human_decision(_hil_summary(user_email, claim, result))
    reason = "Approved by a reviewer." if decision == "approved" else "Declined by a reviewer."
    return _finalize(decision, reason, result, "human", user_email, session_id, claim)


def _hil_summary(user_email: str, claim: str, result: VerificationResult) -> dict:
    """Build the review payload shown to the terminal reviewer."""
    return {
        "user_email": user_email,
        "claim": claim,
        "issue_type": result.issue_type,
        "confidence": result.confidence,
        "match_reason": result.match_reason,
        "description": result.description,
        "evidence": result.evidence,
        "image_description": result.image_description,
        "claim_verifiable_from_photo": result.claim_verifiable_from_photo,
    }


def _finalize(
    decision: str,
    reason: str,
    result: VerificationResult,
    decided_by: str,
    user_email: str,
    session_id: str,
    claim: str,
) -> dict:
    """Persist the outcome and build the response payload."""
    try:
        database.create_refund_request(
            user_email=user_email,
            session_id=session_id,
            claim=claim,
            verification=result,
            decision=decision,
            decided_by=decided_by,
        )
    except Exception:
        logger.error("failed to persist refund request | user=%s", user_email, exc_info=True)

    customer_message = _customer_message(decision, reason, result)

    # Persist the refund turn to chat_messages so a follow-up question (e.g. "when will
    # it get refunded") has the approved/rejected decision as conversation context.
    try:
        database.ensure_chat_session(session_id, user_email)
        database.save_chat_message(session_id, "user", f"[Refund claim via photo] {claim}")
        database.save_chat_message(session_id, "assistant", customer_message)
    except Exception:
        logger.error("failed to persist refund turn to chat history | user=%s", user_email, exc_info=True)

    logger.info("refund decision=%s by=%s | user=%s", decision, decided_by, user_email)
    return {
        "decision": decision,
        "reason": reason,
        "confidence": result.confidence,
        "issue_type": result.issue_type,
        "customer_message": customer_message,
    }


def _customer_message(decision: str, reason: str, result: VerificationResult) -> str:
    """User-facing wording for the decision."""
    if decision == "approved":
        return (
            f"Good news — your refund claim has been **approved**. {result.description} "
            "Our team will process your refund shortly."
        )
    return (
        f"We're sorry, but we couldn't approve this refund claim. {reason} "
        "If you believe this is a mistake, please reply with more details or contact support."
    )
