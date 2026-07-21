"""Business data access — queries customers, orders, products, and invoices from Supabase.

Public API:
    get_orders_for_user(email)            -> list[dict]
    get_order_by_id(order_id, user_email) -> dict | None
    get_product_by_name(name)            -> dict | None   # case-insensitive substring
    get_invoices_for_user(email)          -> list[dict]
    format_order_items(order)             -> str           # "Laptop, Mouse" display join
    load_live_data(email, entities)       -> list[str]     # assembled order/product context blocks

Called by: app/pipelines/retrieval/retriever.py
"""

import json as _json

from app.helpers.database import get_db, first_row
from app.helpers import date_facts
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


# ── Assembled live-data context ─────────────────────────────────────────────────

def load_live_data(user_email: str, entities: dict) -> list[str]:
    """Load relevant user/order/product data from Supabase based on context."""
    parts: list[str] = []
    if not user_email:
        return parts

    user_orders = get_orders_for_user(user_email)
    if user_orders:
        order_id = entities.get("order_id")
        if order_id:
            order = get_order_by_id(order_id, user_email)
            if order:
                order_block = f"[ORDER: {order_id}]\n{_json.dumps(order, indent=2)}"
                order_timing = date_facts.order_date_facts(order)
                if order_timing:
                    order_block += "\n" + "\n".join(order_timing)
                parts.append(order_block)
                logger.debug("loaded specific order: %s", order_id)
        else:
            summary = [
                {
                    "order_id": o["order_id"],
                    "status": o.get("status"),
                    "placed_at": o.get("placed_at"),
                    "delivery_date": o.get("delivery_date"),
                    "items": [i.get("name") for i in o.get("items", [])],
                }
                for o in user_orders
            ]
            orders_block = f"[USER ORDERS]\n{_json.dumps(summary, indent=2)}"
            # Add precomputed date facts for each order
            all_timing = []
            for o in user_orders:
                timing = date_facts.order_date_facts(o)
                all_timing.extend(timing)
            if all_timing:
                orders_block += "\n[ORDER TIMING FACTS]\n" + "\n".join(all_timing)
            parts.append(orders_block)
            logger.debug("loaded %d order summaries for user", len(user_orders))

    product_name = entities.get("product_name")
    if product_name:
        product = get_product_by_name(product_name)
        if product:
            parts.append(f"[PRODUCT: {product_name}]\n{_json.dumps(product, indent=2)}")
            logger.debug("loaded product: %s", product_name)

    return parts
