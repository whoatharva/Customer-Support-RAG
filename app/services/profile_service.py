"""Self-service profile: detect intent, validate, render, and apply updates.

Isolated from the RAG path. The chat service calls `detect()` only when a
message trips the cheap keyword pre-filter, so ordinary questions never pay for
the extra LLM call.

Public API:
    matches_keywords(query)          -> bool   # cheap gate before detect()
    detect(query)                    -> dict   # {"action", "field", "values"}
    validate(field, values)          -> (ok: bool, error: str)
    render_profile(email)            -> str    # markdown summary (read-only view)
    apply(email, field, values)      -> str    # perform the write, return confirmation
"""

import json

from app.helpers import database, order_lookup
from app.llm import lightweight
from app.llm.prompts import DETECT_PROMPT
from app.helpers.logger import get_logger

logger = get_logger(__name__)

# Cheap substring gate — if none of these appear, skip the LLM entirely.
PROFILE_KEYWORDS = (
    "phone", "mobile", "contact number",
    "address", "pincode", "pin code",
    "my profile", "my details", "my account", "my info",
)


def matches_keywords(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in PROFILE_KEYWORDS)


def detect(query: str) -> dict:
    """LLM classify into {action, field, values}. Safe default on any failure."""
    raw = lightweight.call(DETECT_PROMPT.format(query=query), max_tokens=200)
    try:
        parsed = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, TypeError):
        logger.warning("profile detect parse failed | raw=%r", raw)
        return {"action": "none", "field": None, "values": {}}

    action = parsed.get("action", "none")
    field = parsed.get("field")
    # Drop null/empty values so validation/apply only see real input.
    values = {k: v for k, v in (parsed.get("values") or {}).items() if v}
    logger.info("profile detect | action=%s field=%s values=%s", action, field, values)
    return {"action": action, "field": field, "values": values}


def validate(field: str, values: dict) -> tuple[bool, str]:
    if field == "phone":
        phone = str(values.get("phone", "")).strip()
        if not (phone.isdigit() and len(phone) == 10):
            return False, "Phone must be a 10-digit number."
        return True, ""

    if field == "address":
        if not any(values.get(k) for k in database.ADDRESS_EDITABLE_FIELDS):
            return False, "Please provide at least one address field to change."
        pincode = str(values.get("pincode", "")).strip()
        if pincode and not (pincode.isdigit() and len(pincode) == 6):
            return False, "Pincode must be a 6-digit number."
        return True, ""

    return False, "I can only update your phone or address."


def render_profile(email: str) -> str:
    """Markdown summary of phone, default address, and recent orders."""
    customer = database.get_customer_full(email)
    if not customer:
        return "I couldn't find a profile associated with your account."

    lines = ["**Your profile**", f"- **Phone:** {customer.get('phone') or '—'}"]

    addresses = customer.get("addresses") or []
    if addresses:
        a = addresses[0]
        parts = [a.get("line1"), a.get("line2"), a.get("city"), a.get("state"), a.get("pincode")]
        pretty = ", ".join(p for p in parts if p)
        label = a.get("label") or "Default"
        lines.append(f"- **Address ({label}):** {pretty}")
    else:
        lines.append("- **Address:** —")

    orders = order_lookup.get_orders_for_user(email)
    if orders:
        lines.append("\n**Recent orders:**")
        for o in orders[:5]:
            items = ", ".join(i.get("name", "") for i in o.get("items", []))
            lines.append(f"- `{o.get('order_id')}` — {o.get('status', 'n/a')}"
                         + (f" ({items})" if items else ""))

    return "\n".join(lines)


def apply(email: str, field: str, values: dict) -> str:
    """Validate then persist the change. Returns a user-facing confirmation."""
    ok, error = validate(field, values)
    if not ok:
        return error

    if field == "phone":
        database.update_customer_phone(email, values["phone"])
        return f"Done — your phone number is now **{values['phone']}**."

    if field == "address":
        database.update_customer_default_address(email, values)
        changed = ", ".join(f"{k}: {v}" for k, v in values.items()
                            if k in database.ADDRESS_EDITABLE_FIELDS)
        return f"Done — your address has been updated ({changed})."

    return "I can only update your phone or address."


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences some models wrap JSON in."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()
