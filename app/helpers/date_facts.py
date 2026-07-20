"""Precomputed date facts for accurate temporal reasoning.

Computes and formats date-based information so the LLM never has to do date math.
Uses stdlib datetime (ISO 8601 parsing) and calendar (weekday/month names).
"""

import calendar
from datetime import datetime, date as date_type


def parse_dt(value: str | None) -> datetime | None:
	"""Parse an ISO-8601 datetime string (with or without timezone) safely.

	Returns None if value is None, empty, or unparseable.
	"""
	if not value or not isinstance(value, str):
		return None
	try:
		return datetime.fromisoformat(value)
	except (ValueError, TypeError):
		return None


def today_str() -> str:
	"""Today formatted for prompts, e.g. '20 July 2026'."""
	today = date_type.today()
	return today.strftime("%d %B %Y")


def today_facts() -> str:
	"""Return a [DATE FACTS] block with today's date, weekday, and ISO format."""
	today = date_type.today()
	weekday = calendar.day_name[today.weekday()]
	month = today.strftime("%B")
	iso = today.isoformat()
	return f"[DATE FACTS]\nCurrent date: {weekday}, {today.day} {month} {today.year} ({iso})"


def _days_ago_phrase(verb: str, d: date_type, today: date_type) -> str:
	"""Format a past-event fact, e.g. 'Order placed 3 days ago (2026-07-17)'."""
	days_ago = (today - d).days
	iso = d.isoformat()
	if days_ago == 0:
		return f"Order {verb} today ({iso})"
	if days_ago == 1:
		return f"Order {verb} 1 day ago ({iso})"
	return f"Order {verb} {days_ago} days ago ({iso})"


def order_date_facts(order: dict) -> list[str]:
	"""Compute and format precomputed date facts for a single order.

	Returns a list of formatted strings (one per computed field).
	Only includes fields that are present and parseable in the order dict.
	"""
	if not isinstance(order, dict):
		return []

	facts = []
	today = date_type.today()

	# placed_at: compute "placed N days ago"
	placed = parse_dt(order.get("placed_at"))
	if placed:
		facts.append(_days_ago_phrase("placed", placed.date(), today))

	# delivery_date: compute "delivered N days ago" or "not yet delivered"
	delivery = parse_dt(order.get("delivery_date"))
	if delivery:
		facts.append(_days_ago_phrase("delivered", delivery.date(), today))
	elif "delivery_date" in order and order["delivery_date"] is None:
		facts.append("Order has not been delivered yet (no delivery date recorded)")

	# estimated_delivery: compute "expected in N days" / "was expected N days ago (overdue)"
	estimated_dt = parse_dt(order.get("estimated_delivery"))
	if estimated_dt:
		estimated = estimated_dt.date()
		days_delta = (estimated - today).days
		if days_delta > 0:
			if days_delta == 1:
				fact_str = f"Estimated delivery: tomorrow ({estimated.isoformat()})"
			else:
				fact_str = f"Estimated delivery: in {days_delta} days ({estimated.isoformat()})"
		elif days_delta == 0:
			fact_str = f"Estimated delivery: today ({estimated.isoformat()})"
		else:
			days_overdue = abs(days_delta)
			if days_overdue == 1:
				fact_str = f"Expected delivery was 1 day ago (OVERDUE, {estimated.isoformat()})"
			else:
				fact_str = f"Expected delivery was {days_overdue} days ago (OVERDUE, {estimated.isoformat()})"
		facts.append(fact_str)

	return facts
