"""Pipeline 3: Retrieval & Response Generation

PIPELINE FLOW (easy to explain):

  [User query]
       │
       ▼
  ① HyDE Lite         — generate a hypothetical answer, blend its embedding with
                         the query embedding for better semantic search
       │
       ▼
  ② Invoice load       — if user uploaded an invoice, load it as direct context
  ② Live data load     — auto-load user's orders + product info from JSON files
       │
       ▼
  ③ Vector search      — search Qdrant knowledge base with the HyDE-enhanced vector
       │
       ▼
  ④ GATE 1             — Retrieval relevancy gate:
                         if top score < retrieval_min_score AND no direct data → escalate immediately
       │
       ▼
  ⑤ Context assembly   — combine: invoice + orders + FAQ chunks into one string
       │
       ▼
  ⑥ LLM generation     — call Azure OpenAI gpt-4.1 with assembled context
       │
       ▼
  ⑦ GATE 2             — Answer relevancy check (Gemini/Groq):
                         does the answer actually address the question?
       │
       ▼
  ⑧ Confidence score   — composite: invoice/data boosts base Qdrant score
       │
       ▼
  [ChatResponse]
"""

import json as _json
from pathlib import Path

from app.config import settings
from app.llm import client as llm
from app.llm.prompts import RESPONSE_TEMPLATE, SUPPORT_CONTACT
from app.vectorstore import store as vector_store
from app.schemas import Citation
from app.helpers.langfuse import get_langfuse
from app.helpers import data_lookup
from app.pipelines.query import hyde
from app.pipelines.retrieval import gates
from app.helpers.logger import get_logger

logger = get_logger(__name__)

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

_ESCALATION_ANSWER = (
    f"I'm sorry, I wasn't able to find relevant information to answer your question. "
    f"Please contact our support team at {SUPPORT_CONTACT} for further assistance."
)


def generate_response(query: str, context: dict) -> dict:
    """
    Run the full retrieval + generation pipeline with quality gates.

    Args:
        query:   Rewritten, self-contained query from Pipeline 2.
        context: {invoice_id, session_id, intent, entities, user_email}

    Returns:
        {answer, citations, confidence, should_escalate}
    """
    session_id  = context.get("session_id", "")
    invoice_ids = context.get("invoice_ids") or []
    user_email  = context.get("user_email", "")
    entities    = context.get("entities") or {}

    logger.info("P3 start | session=%s invoices=%s user=%s", session_id, invoice_ids, user_email)

    langfuse = get_langfuse()
    trace = langfuse.trace(name="p3_generate_response", session_id=session_id, input=query)

    # ── ① HyDE Lite: blend query vector with hypothetical-answer vector ───────
    span = trace.span(name="p3_hyde", input=query)
    search_vector = hyde.get_blended_vector(query)
    span.end(output=f"blended vector dim={len(search_vector)}")

    # ── ② Load direct data: invoice(s) + user orders/products ───────────────────
    invoice_text   = _load_invoices(invoice_ids, user_email)
    live_data_parts = _load_live_data(user_email, entities)
    has_direct_data = bool(invoice_text or live_data_parts)

    # ── ③ Vector search (with HyDE-blended vector) ───────────────────────────
    span = trace.span(name="p3_retrieve", input=f"query={query[:60]}")
    results = vector_store.search(search_vector, top_k=5)
    top_score = results[0].score if results else 0.0
    span.end(output=f"{len(results)} chunks | top_score={top_score:.3f}")
    logger.info("P3 retrieved %d chunks | top_score=%.3f", len(results), top_score)

    # ── ④ GATE 1: retrieval relevancy gate ───────────────────────────────────
    if not gates.retrieval_gate(results, has_direct_data=has_direct_data):
        logger.info("P3 GATE1 blocked — escalating without LLM call")
        trace.score(name="confidence", value=0.0)
        return _escalate(citations=[])

    # ── ⑤ Assemble context: invoice(s) → live data → FAQ chunks ─────────────────
    context_parts: list[str] = []
    if invoice_text:
        context_parts.append(invoice_text)
    context_parts.extend(live_data_parts)
    for r in results:
        p = r.payload
        context_parts.append(f"[{p['doc_filename']} / {p.get('section', '')}]\n{p['text']}")
    full_context = "\n\n---\n\n".join(context_parts)

    # ── ⑥ LLM generation (Azure OpenAI gpt-4.1) ──────────────────────────────
    prompt = RESPONSE_TEMPLATE.format(context=full_context, query=query)
    generation = trace.generation(
        name="p3_generate",
        model=settings.azure_openai_deployment_name,
        input=prompt[:500],
    )
    response = llm.chat([{"role": "user", "content": prompt}])
    answer = response["choices"][0]["message"]["content"].strip()
    _usage = response.get("usage") or {}
    generation.end(
        output=answer[:300],
        usage={
            "input":  _usage.get("prompt_tokens", 0),
            "output": _usage.get("completion_tokens", 0),
            "total":  _usage.get("total_tokens", 0),
        },
    )
    logger.info(
        "P3 generated answer: %d chars | tokens in=%s out=%s",
        len(answer), _usage.get("prompt_tokens"), _usage.get("completion_tokens"),
    )

    # ── ⑦ GATE 2: answer relevancy check (Gemini/Groq) ───────────────────────
    span = trace.span(name="p3_answer_gate", input=f"query={query[:60]}")
    answer_is_relevant = gates.answer_relevancy_gate(query, answer)
    span.end(output="relevant" if answer_is_relevant else "off-topic")

    if not answer_is_relevant:
        logger.info("P3 GATE2 blocked — answer deemed off-topic, escalating")
        trace.score(name="confidence", value=0.0)
        return _escalate(citations=_build_faq_citations(results))

    # ── ⑧ Citations: invoice(s) first, then scored FAQ chunks ────────────────────
    citations: list[Citation] = []
    for inv_id in invoice_ids:
        inv_text = _load_single_invoice(inv_id, user_email)
        if inv_text:
            citations.append(Citation(
                chunk_id=f"invoice_{inv_id}",
                source_document=f"INVOICE: {inv_id}",
                section="Invoice",
                text=inv_text[:200],
                score=1.0,
            ))
    citations.extend(_build_faq_citations(results))

    # ── ⑧ Composite confidence score ─────────────────────────────────────────
    # Direct data (invoice/orders) boosts confidence — we have authoritative info.
    # Without direct data, confidence is the raw Qdrant similarity score.
    if has_direct_data:
        confidence = round(min(0.5 * top_score + 0.5, 1.0), 2)
    else:
        confidence = round(min(top_score, 1.0), 2)

    should_escalate = confidence < settings.confidence_threshold
    trace.score(name="confidence", value=confidence)
    logger.info(
        "P3 done | confidence=%.2f (direct_data=%s top_score=%.3f) escalate=%s",
        confidence, has_direct_data, top_score, should_escalate,
    )

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "should_escalate": should_escalate,
        "total_tokens": _usage.get("total_tokens"),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_single_invoice(invoice_id: str, user_email: str) -> str:
    """Load one invoice file from the user's own subfolder (security-gated)."""
    path = _DATA_ROOT / "invoices" / user_email / f"{invoice_id}.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        logger.debug("invoice loaded: %s/%s (%d chars)", user_email, invoice_id, len(text))
        return text
    logger.warning("invoice not found (or not owned by user): %s/%s", user_email, invoice_id)
    return ""


def _load_invoices(invoice_ids: list[str], user_email: str = "") -> str:
    """Load one or more invoices and return them as a single labelled context block."""
    if not invoice_ids:
        return ""
    if not user_email:
        logger.warning("invoice load skipped: no user_email provided (security check)")
        return ""
    parts: list[str] = []
    for inv_id in invoice_ids:
        text = _load_single_invoice(inv_id, user_email)
        if text:
            parts.append(f"[INVOICE: {inv_id}]\n{text}")
    return "\n\n---\n\n".join(parts)


def _load_live_data(user_email: str, entities: dict) -> list[str]:
    """Load relevant user/order/product data from JSON files based on context."""
    parts: list[str] = []
    if not user_email:
        return parts

    user_orders = data_lookup.get_orders_for_user(user_email)
    if user_orders:
        order_id = entities.get("order_id")
        if order_id:
            order = data_lookup.get_order_by_id(order_id)
            if order:
                parts.append(f"[ORDER: {order_id}]\n{_json.dumps(order, indent=2)}")
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
            parts.append(f"[USER ORDERS]\n{_json.dumps(summary, indent=2)}")
            logger.debug("loaded %d order summaries for user", len(user_orders))

    product_name = entities.get("product_name")
    if product_name:
        product = data_lookup.get_product_by_name(product_name)
        if product:
            parts.append(f"[PRODUCT: {product_name}]\n{_json.dumps(product, indent=2)}")
            logger.debug("loaded product: %s", product_name)

    return parts


def _build_faq_citations(results: list) -> list[Citation]:
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


def _escalate(citations: list[Citation]) -> dict:
    return {
        "answer": _ESCALATION_ANSWER,
        "citations": citations,
        "confidence": 0.0,
        "should_escalate": True,
    }
