"""Order cancellation — pure deterministic Python business logic.

No LLM, no external API calls. All decisions are made from the order's current
status and the user's ownership of the order.

Public API:
    CANCELLABLE_STATUSES               frozenset[str]
    NON_CANCELLABLE_STATUSES           frozenset[str]
    CancellationResult                 dataclass
    matches_cancel_keywords(query)     -> bool          # cheap gate before detect
    detect_cancel_intent(query)        -> dict           # {order_id?, product_name?}  regex only
    find_order_to_cancel(info, email)  -> dict | None   # resolve order from DB
    cancel_order(order_id, email)      -> CancellationResult

Called by: app/services/chat_service.py
"""

import re
from dataclasses import dataclass

from app.helpers import database, order_lookup
from app.helpers.logger import get_logger

logger = get_logger(__name__)


# ── Policy ────────────────────────────────────────────────────────────────────

CANCELLABLE_STATUSES: frozenset[str] = frozenset({
    "Pending",
    "Processing",
    "Confirmed",
    "Packed",
    "Ready to Ship",
})

NON_CANCELLABLE_STATUSES: frozenset[str] = frozenset({
    "In Transit",
    "Out for Delivery",
    "Delivered",
    "Cancelled",
})

_REJECTION_REASONS: dict[str, str] = {
    "In Transit":       "Your order is already with the courier and cannot be cancelled.",
    "Out for Delivery": "Your order is out for delivery and cannot be cancelled.",
    "Delivered":        "This order has already been delivered and cannot be cancelled.",
    "Cancelled":        "This order is already cancelled.",
}


# ── Intent Detection (regex-only, no LLM) ────────────────────────────────────

# Keywords that signal the user wants to cancel an order.
_CANCEL_KEYWORDS = (
    "cancel", "cancellation", "cancel my order", "cancel the order",
    "call off", "revoke order", "stop my order",
)

# Regex for common order ID formats: ORD-123, ORD123, #12345, #ORD123, etc.
_ORDER_ID_RE = re.compile(
    r"(?:ORD[-_]?\d+|#(?:ORD[-_]?)?\d{4,})",
    re.IGNORECASE,
)


def matches_cancel_keywords(query: str) -> bool:
    """Cheap substring gate — True if the query plausibly requests a cancellation."""
    q = query.lower()
    return any(kw in q for kw in _CANCEL_KEYWORDS)


def detect_cancel_intent(query: str) -> dict:
    """Extract order ID or product name from a cancel request without any LLM call.

    Returns:
        {"order_id": str | None, "product_name": str | None}
    """
    order_id_match = _ORDER_ID_RE.search(query)
    order_id = order_id_match.group(0).upper().replace("_", "-") if order_id_match else None

    # Strip cancel-related verbs and filler to leave the product keyword.
    product_name: str | None = None
    if not order_id:
        clean = re.sub(
            r"\b(cancel|cancellation|my|the|order|please|i want to|can you)\b",
            "",
            query,
            flags=re.IGNORECASE,
        ).strip()
        clean = re.sub(r"\s+", " ", clean).strip(" ,.")
        if clean:
            product_name = clean

    logger.debug("cancel intent detected | order_id=%s product_name=%s", order_id, product_name)
    return {"order_id": order_id, "product_name": product_name}


# ── Order Resolution ──────────────────────────────────────────────────────────

def find_order_to_cancel(intent: dict, user_email: str) -> dict | None:
    """Resolve the order to cancel from the intent dict.

    Tries by explicit order_id first; falls back to matching an item name in the
    user's most recent cancellable order containing a product with that keyword.

    Returns the order dict or None if no match found.
    """
    order_id = intent.get("order_id")
    if order_id:
        return order_lookup.get_order_by_id(order_id, user_email)

    product_name = intent.get("product_name")
    if not product_name:
        return None

    # Search the user's orders for one whose items include the keyword.
    keyword = product_name.lower()
    orders = order_lookup.get_orders_for_user(user_email)
    for order in orders:
        for item in order.get("items", []):
            if keyword in (item.get("name") or "").lower():
                logger.debug(
                    "matched order %s via product keyword '%s'",
                    order.get("order_id"), keyword,
                )
                return order

    return None


# ── Cancellation Result ───────────────────────────────────────────────────────

@dataclass
class CancellationResult:
    success: bool
    order_id: str
    previous_status: str
    message: str


# ── Core Logic ────────────────────────────────────────────────────────────────

def cancel_order(order_id: str, user_email: str) -> CancellationResult:
    """Apply the cancellation policy and update the DB if the order is cancellable.

    Steps:
      1. Fetch the order; fail fast if not found or not owned by this user.
      2. Check the status against the policy.
      3. Update DB to 'Cancelled' if allowed.

    No LLM calls are made at any step.
    """
    order = order_lookup.get_order_by_id(order_id, user_email)

    if order is None:
        logger.info("cancel attempt: order not found | order_id=%s user=%s", order_id, user_email)
        return CancellationResult(
            success=False,
            order_id=order_id,
            previous_status="",
            message="I couldn't find that order on your account.",
        )

    current_status = order.get("status", "")
    logger.info(
        "cancel attempt | order_id=%s user=%s status=%s",
        order_id, user_email, current_status,
    )

    if current_status in CANCELLABLE_STATUSES:
        database.cancel_order(order_id)
        logger.info("order cancelled successfully | order_id=%s", order_id)
        return CancellationResult(
            success=True,
            order_id=order_id,
            previous_status=current_status,
            message=(
                f"Your order **{order_id}** has been successfully cancelled. "
                "Any payment will be refunded within 5–7 business days."
            ),
        )

    rejection_reason = _REJECTION_REASONS.get(
        current_status,
        f"This order cannot be cancelled (current status: {current_status}).",
    )
    logger.info(
        "cancel rejected | order_id=%s status=%s reason=%s",
        order_id, current_status, rejection_reason,
    )
    return CancellationResult(
        success=False,
        order_id=order_id,
        previous_status=current_status,
        message=rejection_reason,
    )
