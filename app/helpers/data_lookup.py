"""Business data access — queries customers, orders, products, and invoices from Supabase.

Public API:
    get_orders_for_user(email)            -> list[dict]
    get_order_by_id(order_id, user_email) -> dict | None
    get_product_by_name(name)             -> dict | None   # case-insensitive substring
    get_invoices_for_user(email)          -> list[dict]

Called by: app/pipelines/retrieval/engine.py  (_load_live_data)
"""

from app.helpers.database import get_db, get_invoices_for_user  # noqa: F401 — re-exported
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
    return result.data[0] if result.data else None


# ── Products ──────────────────────────────────────────────────────────────────

def get_product_by_name(name: str) -> dict | None:
    result = (
        get_db().table("products")
        .select("*")
        .ilike("name", f"%{name}%")
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None
