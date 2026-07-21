"""Citation builders — turn invoices and vector-search hits into Citation objects.

Public API:
  build_invoice_citations(db_invoices) -> list[Citation]
  build_faq_citations(results)         -> list[Citation]
"""

from app.schemas import Citation


def build_invoice_citations(db_invoices: list) -> list[Citation]:
    return [
        Citation(
            chunk_id=f"invoice_{inv['invoice_id']}",
            source_document=f"INVOICE: {inv['invoice_id']}",
            section="Invoice",
            text=inv["content"][:200],
            score=1.0,
        )
        for inv in db_invoices
    ]


def build_faq_citations(results: list) -> list[Citation]:
    return [
        Citation(
            chunk_id=r.payload.get("chunk_id", ""),
            source_document=r.payload.get("doc_filename", ""),
            section=r.payload.get("section", ""),
            text=r.payload.get("text", "")[:200],
            score=round(r.score, 4),
        )
        for r in results
    ]
