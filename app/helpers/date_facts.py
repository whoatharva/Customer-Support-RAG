"""Precomputed date facts for accurate temporal reasoning.

Computes and formats date-based information so the LLM never has to do date math.
Uses stdlib datetime (ISO 8601 parsing) and calendar (weekday/month names).

AVAILABLE TOOLS:
  • get_orders_for_user(email)           → all orders for a user
  • get_order_by_id(order_id)            → one order
  • get_product_by_name(name)            → product search
  • get_product_by_id(product_id)        → product by id
  • get_invoices_for_user(email)         → all invoices for user
  • get_invoice_by_id(id, email)         → one invoice
  • profile view/update                  → self-service account changes
  • date_facts.today_facts()             → current date block
  • date_facts.order_date_facts(order)   → precomputed order timing
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


def today_facts() -> str:
	"""Return a [DATE FACTS] block with today's date, weekday, and ISO format."""
	today = date_type.today()
	weekday = calendar.day_name[today.weekday()]
	month = today.strftime("%B")
	iso = today.isoformat()
	return f"[DATE FACTS]\nCurrent date: {weekday}, {today.day} {month} {today.year} ({iso})"


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
		placed_date = placed.date()
		days_ago = (today - placed_date).days
		if days_ago == 0:
			fact_str = f"Order placed today ({placed_date.isoformat()})"
		elif days_ago == 1:
			fact_str = f"Order placed 1 day ago ({placed_date.isoformat()})"
		else:
			fact_str = f"Order placed {days_ago} days ago ({placed_date.isoformat()})"
		facts.append(fact_str)

	# delivery_date: compute "delivered N days ago" or "not yet delivered"
	delivery = parse_dt(order.get("delivery_date"))
	if delivery:
		delivery_date = delivery.date()
		days_ago = (today - delivery_date).days
		if days_ago == 0:
			fact_str = f"Order delivered today ({delivery_date.isoformat()})"
		elif days_ago == 1:
			fact_str = f"Order delivered 1 day ago ({delivery_date.isoformat()})"
		else:
			fact_str = f"Order delivered {days_ago} days ago ({delivery_date.isoformat()})"
		facts.append(fact_str)
	elif "delivery_date" in order and order["delivery_date"] is None:
		facts.append("Order has not been delivered yet (no delivery date recorded)")

	# estimated_delivery: compute "expected in N days" / "was expected N days ago (overdue)"
	estimated_str = order.get("estimated_delivery")
	if estimated_str:
		try:
			estimated = datetime.fromisoformat(estimated_str).date()
		except (ValueError, TypeError):
			try:
				estimated = datetime.strptime(estimated_str, "%Y-%m-%d").date()
			except (ValueError, TypeError):
				estimated = None

		if estimated:
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
