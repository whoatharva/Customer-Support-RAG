"""Business data access — queries customers, orders, products, and invoices from Supabase.

Public API:
    get_customer_by_email(email)          -> dict | None
    get_orders_for_user(email)            -> list[dict]
    get_order_by_id(order_id)             -> dict | None
    get_product_by_name(name)             -> dict | None   # case-insensitive substring
    get_product_by_id(product_id)         -> dict | None
    get_invoices_for_user(email)          -> list[dict]
    get_invoice_by_id(invoice_id, email)  -> dict | None

Called by: app/pipelines/retrieval/engine.py  (_load_live_data)
"""

from app.helpers.database import get_db, get_invoices_for_user, get_invoice_by_id  # noqa: F401 — re-exported
from app.helpers.logger import get_logger

logger = get_logger(__name__)


# ── Users / Customers ─────────────────────────────────────────────────────────

def get_customer_by_email(email: str) -> dict | None:
    result = (
        get_db().table("customers")
        .select("*")
        .ilike("email", email)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


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


def get_order_by_id(order_id: str) -> dict | None:
    result = (
        get_db().table("orders")
        .select("*")
        .eq("order_id", order_id)
        .limit(1)
        .execute()
    )
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


def get_product_by_id(product_id: str) -> dict | None:
    result = (
        get_db().table("products")
        .select("*")
        .eq("product_id", product_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None
