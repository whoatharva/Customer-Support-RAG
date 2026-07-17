"""Order cancellation — unit tests.

All DB and order_lookup calls are patched so tests are hermetic: no network,
no Supabase, no actual data. Every supported order state is exercised across
both the service layer and the intent detection helpers.
"""

from unittest.mock import MagicMock, patch

from app.services.order_cancellation import (
    CANCELLABLE_STATUSES,
    NON_CANCELLABLE_STATUSES,
    CancellationResult,
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


def test_find_order_no_intent_returns_none(monkeypatch):
    result = find_order_to_cancel({"order_id": None, "product_name": None}, "user@test.com")
    assert result is None


# ── cancel_order — cancellable statuses ──────────────────────────────────────

def _patch_lookup_and_db(monkeypatch, order: dict):
    monkeypatch.setattr(
        "app.services.order_cancellation.order_lookup.get_order_by_id",
        lambda oid, email: order if oid == order["order_id"] else None,
    )
    monkeypatch.setattr(
        "app.services.order_cancellation.database.cancel_order",
        MagicMock(),
    )


def test_cancel_pending_order(monkeypatch):
    order = _make_order("ORD-001", "Pending")
    _patch_lookup_and_db(monkeypatch, order)
    result = cancel_order("ORD-001", "user@test.com")
    assert result.success is True
    assert result.previous_status == "Pending"
    assert "cancelled" in result.message.lower()


def test_cancel_processing_order(monkeypatch):
    order = _make_order("ORD-002", "Processing")
    _patch_lookup_and_db(monkeypatch, order)
    result = cancel_order("ORD-002", "user@test.com")
    assert result.success is True
    assert result.previous_status == "Processing"


def test_cancel_confirmed_order(monkeypatch):
    order = _make_order("ORD-003", "Confirmed")
    _patch_lookup_and_db(monkeypatch, order)
    result = cancel_order("ORD-003", "user@test.com")
    assert result.success is True
    assert result.previous_status == "Confirmed"


def test_cancel_packed_order(monkeypatch):
    order = _make_order("ORD-004", "Packed")
    _patch_lookup_and_db(monkeypatch, order)
    result = cancel_order("ORD-004", "user@test.com")
    assert result.success is True
    assert result.previous_status == "Packed"


def test_cancel_ready_to_ship_order(monkeypatch):
    order = _make_order("ORD-005", "Ready to Ship")
    _patch_lookup_and_db(monkeypatch, order)
    result = cancel_order("ORD-005", "user@test.com")
    assert result.success is True
    assert result.previous_status == "Ready to Ship"


# ── cancel_order — non-cancellable statuses ───────────────────────────────────

def _patch_lookup_no_db(monkeypatch, order: dict):
    db_spy = MagicMock()
    monkeypatch.setattr(
        "app.services.order_cancellation.order_lookup.get_order_by_id",
        lambda oid, email: order if oid == order["order_id"] else None,
    )
    monkeypatch.setattr("app.services.order_cancellation.database.cancel_order", db_spy)
    return db_spy


def test_cancel_in_transit_rejected(monkeypatch):
    order = _make_order("ORD-006", "In Transit")
    db_spy = _patch_lookup_no_db(monkeypatch, order)
    result = cancel_order("ORD-006", "user@test.com")
    assert result.success is False
    assert "courier" in result.message.lower()
    db_spy.assert_not_called()


def test_cancel_out_for_delivery_rejected(monkeypatch):
    order = _make_order("ORD-007", "Out for Delivery")
    db_spy = _patch_lookup_no_db(monkeypatch, order)
    result = cancel_order("ORD-007", "user@test.com")
    assert result.success is False
    assert "delivery" in result.message.lower()
    db_spy.assert_not_called()


def test_cancel_delivered_rejected(monkeypatch):
    order = _make_order("ORD-008", "Delivered")
    db_spy = _patch_lookup_no_db(monkeypatch, order)
    result = cancel_order("ORD-008", "user@test.com")
    assert result.success is False
    assert "delivered" in result.message.lower()
    db_spy.assert_not_called()


def test_cancel_already_cancelled_rejected(monkeypatch):
    order = _make_order("ORD-009", "Cancelled")
    db_spy = _patch_lookup_no_db(monkeypatch, order)
    result = cancel_order("ORD-009", "user@test.com")
    assert result.success is False
    assert "already cancelled" in result.message.lower()
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


def test_cancel_db_write_called_once(monkeypatch):
    order = _make_order("ORD-010", "Pending")
    db_spy = MagicMock()
    monkeypatch.setattr(
        "app.services.order_cancellation.order_lookup.get_order_by_id",
        lambda oid, email: order,
    )
    monkeypatch.setattr("app.services.order_cancellation.database.cancel_order", db_spy)
    cancel_order("ORD-010", "user@test.com")
    db_spy.assert_called_once_with("ORD-010")


# ── Status set completeness ───────────────────────────────────────────────────

def test_cancellable_and_noncancellable_are_disjoint():
    assert CANCELLABLE_STATUSES.isdisjoint(NON_CANCELLABLE_STATUSES)
