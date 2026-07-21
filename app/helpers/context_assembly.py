"""Assemble the LLM context string from date facts, invoices, live data, and FAQ chunks.

Public API:
  assemble_context(db_invoices, live_data_parts, results) -> str
"""

from app.helpers import date_facts


def assemble_context(db_invoices: list, live_data_parts: list[str], results: list) -> str:
    """Combine date facts, invoices, live order/product data, and FAQ chunks."""
    context_parts: list[str] = [date_facts.today_facts()]
    for inv in db_invoices:
        context_parts.append(f"[INVOICE: {inv['invoice_id']}]\n{inv['content']}")
    context_parts.extend(live_data_parts)
    for r in results:
        p = r.payload
        context_parts.append(f"[{p['doc_filename']} / {p.get('section', '')}]\n{p['text']}")
    return "\n\n---\n\n".join(context_parts)
