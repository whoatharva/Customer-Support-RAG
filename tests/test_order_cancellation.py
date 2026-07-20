"""Order cancellation — unit tests.

All DB and order_lookup calls are patched so tests are hermetic: no network,
no Supabase, no actual data. Every supported order state is exercised across
both the service layer and the intent detection helpers.
"""

from unittest.mock import MagicMock

import pytest

from app.services.order_cancellation import (
    CANCELLABLE_STATUSES,
    NON_CANCELLABLE_STATUSES,
    cancel_order,
    detect_cancel_intent,
    find_order_to_cancel,
    matches_cancel_keywords,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_order(order_id: str, status: str, items: list[str] | None = None) -> dict:
    return {
        "order_id": order_id,
        "user_email": "user@test.com",
        "status": status,
        "items": [{"name": n} for n in (items or [])],
    }


# ── matches_cancel_keywords ───────────────────────────────────────────────────

def test_matches_cancel_keywords_positive():
    assert matches_cancel_keywords("I want to cancel my order") is True
    assert matches_cancel_keywords("please cancel ORD-001") is True
    assert matches_cancel_keywords("cancel my laptop") is True


def test_matches_cancel_keywords_negative():
    assert matches_cancel_keywords("where is my order?") is False
    assert matches_cancel_keywords("I'd like a refund") is False
    assert matches_cancel_keywords("track my delivery") is False


# ── detect_cancel_intent ──────────────────────────────────────────────────────

def test_detect_cancel_intent_by_order_id():
    result = detect_cancel_intent("cancel order ORD-12345")
    assert result["order_id"] == "ORD-12345"
    assert result["product_name"] is None


def test_detect_cancel_intent_by_order_id_hash_format():
    result = detect_cancel_intent("cancel #12345")
    assert result["order_id"] is not None
    assert result["product_name"] is None


def test_detect_cancel_intent_by_product_name():
    result = detect_cancel_intent("cancel my laptop")
    assert result["order_id"] is None
    assert result["product_name"] is not None
    assert "laptop" in result["product_name"].lower()


def test_detect_cancel_intent_no_id_or_product():
    result = detect_cancel_intent("cancel")
    # product_name may be empty string or None — either is acceptable
    assert result["order_id"] is None


# ── find_order_to_cancel ──────────────────────────────────────────────────────

def test_find_order_by_id(monkeypatch):
    fake_order = _make_order("ORD-001", "Pending", ["Laptop"])
    monkeypatch.setattr(
        "app.services.order_cancellation.order_lookup.get_order_by_id",
        lambda oid, email: fake_order if oid == "ORD-001" else None,
    )
    result = find_order_to_cancel({"order_id": "ORD-001", "product_name": None}, "user@test.com")
    assert result is not None
    assert result["order_id"] == "ORD-001"


def test_find_order_by_product_name(monkeypatch):
    orders = [
        _make_order("ORD-001", "Pending", ["Laptop Pro"]),
        _make_order("ORD-002", "Delivered", ["Headphones"]),
    ]
    monkeypatch.setattr(
        "app.services.order_cancellation.order_lookup.get_orders_for_user",
        lambda email: orders,
    )
    result = find_order_to_cancel({"order_id": None, "product_name": "laptop"}, "user@test.com")
    assert result is not None
    assert result["order_id"] == "ORD-001"


def test_find_order_no_match(monkeypatch):
    monkeypatch.setattr(
        "app.services.order_cancellation.order_lookup.get_orders_for_user",
        lambda email: [],
    )
    result = find_order_to_cancel({"order_id": None, "product_name": "laptop"}, "user@test.com")
    assert result is None


def test_find_order_no_intent_returns_none():
    result = find_order_to_cancel({"order_id": None, "product_name": None}, "user@test.com")
    assert result is None


# ── cancel_order — cancellable / non-cancellable statuses ────────────────────

# Keyword expected in the rejection message for each non-cancellable status.
_REJECTION_KEYWORDS = {
    "In Transit": "courier",
    "Out for Delivery": "delivery",
    "Delivered": "delivered",
    "Cancelled": "already cancelled",
}


def _patch_lookup(monkeypatch, order: dict) -> MagicMock:
    """Patch order lookup to return `order` and return a spy on the DB write."""
    db_spy = MagicMock()
    monkeypatch.setattr(
        "app.services.order_cancellation.order_lookup.get_order_by_id",
        lambda oid, email: order if oid == order["order_id"] else None,
    )
    monkeypatch.setattr("app.services.order_cancellation.database.cancel_order", db_spy)
    return db_spy


@pytest.mark.parametrize("status", sorted(CANCELLABLE_STATUSES))
def test_cancel_cancellable_status(monkeypatch, status):
    order = _make_order("ORD-001", status)
    db_spy = _patch_lookup(monkeypatch, order)
    result = cancel_order("ORD-001", "user@test.com")
    assert result.success is True
    assert result.previous_status == status
    assert "cancelled" in result.message.lower()
    db_spy.assert_called_once_with("ORD-001")


@pytest.mark.parametrize("status", sorted(NON_CANCELLABLE_STATUSES))
def test_cancel_non_cancellable_status_rejected(monkeypatch, status):
    order = _make_order("ORD-006", status)
    db_spy = _patch_lookup(monkeypatch, order)
    result = cancel_order("ORD-006", "user@test.com")
    assert result.success is False
    assert _REJECTION_KEYWORDS[status] in result.message.lower()
    db_spy.assert_not_called()


# ── cancel_order — edge cases ─────────────────────────────────────────────────

def test_cancel_order_not_found(monkeypatch):
    monkeypatch.setattr(
        "app.services.order_cancellation.order_lookup.get_order_by_id",
        lambda oid, email: None,
    )
    result = cancel_order("ORD-GHOST", "user@test.com")
    assert result.success is False
    assert "couldn't find" in result.message.lower()


# ── Status set completeness ───────────────────────────────────────────────────

def test_cancellable_and_noncancellable_are_disjoint():
    assert CANCELLABLE_STATUSES.isdisjoint(NON_CANCELLABLE_STATUSES)
