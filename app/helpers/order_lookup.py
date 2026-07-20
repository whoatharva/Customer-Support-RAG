"""Business data access — queries customers, orders, products, and invoices from Supabase.

Public API:
    get_orders_for_user(email)            -> list[dict]
    get_order_by_id(order_id, user_email) -> dict | None
    get_product_by_name(name)            -> dict | None   # case-insensitive substring
    get_invoices_for_user(email)          -> list[dict]
    format_order_items(order)             -> str           # "Laptop, Mouse" display join

Called by: app/pipelines/retrieval/retriever.py  (_load_live_data)
"""

from app.helpers.database import get_db, first_row
from app.helpers.logger import get_logger

logger = get_logger(__name__)


# ── Orders ────────────────────────────────────────────────────────────────────

def get_orders_for_user(email: str) -> list[dict]:
    result = (
        get_db().table("orders")
        .select("*")
        .ilike("user_email", email)
        .execute()
    )
    orders = result.data or []
    logger.debug("orders for %s: %d found", email, len(orders))
    return orders


def get_order_by_id(order_id: str, user_email: str = "") -> dict | None:
    query = get_db().table("orders").select("*").eq("order_id", order_id)
    if user_email:
        query = query.eq("user_email", user_email)
    result = query.limit(1).execute()
    return first_row(result)


def format_order_items(order: dict) -> str:
    """Comma-joined item names for user-facing order summaries."""
    return ", ".join(i.get("name", "") for i in order.get("items", []))


# ── Products ──────────────────────────────────────────────────────────────────

def get_product_by_name(name: str) -> dict | None:
    result = (
        get_db().table("products")
        .select("*")
        .ilike("name", f"%{name}%")
        .limit(1)
        .execute()
    )
    return first_row(result)


# ── Invoices ──────────────────────────────────────────────────────────────────

def get_invoices_for_user(email: str) -> list[dict]:
    result = (
        get_db().table("invoices")
        .select("invoice_id, order_id, content")
        .eq("user_email", email)
        .execute()
    )
    return result.data or []
