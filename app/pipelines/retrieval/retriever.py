"""Pipeline 3: Retrieval & Response Generation

PIPELINE FLOW:

  [User query]  (already scope-gated in chat_service before this pipeline runs)
       │
       ▼
  ① HyDE Lite         — generate a hypothetical answer, blend its embedding with
                         the query embedding for better semantic search
       │
       ▼
  ② Live data load     — auto-load user's invoices + orders + products from Supabase
       │
       ▼
  ③ Vector search      — search Qdrant knowledge base with the HyDE-enhanced vector
       │
       ▼
  ④ Context assembly   — combine: invoices + orders + FAQ chunks into one string
       │
       ▼
  ⑤ LLM generation     — call Azure OpenAI gpt-4.1 with assembled context
       │
       ▼
  ⑥ Confidence score   — composite: direct DB data boosts base Qdrant score
       │
       ▼
  [ChatResponse]
"""

from app.config import settings
from app.vectorstore import store as vector_store
from app.helpers.langfuse import get_langfuse
from app.helpers import order_lookup
from app.helpers.citations import build_invoice_citations, build_faq_citations
from app.helpers.confidence import compute_confidence
from app.helpers.context_assembly import assemble_context
from app.helpers.generation import generate_answer
from app.pipelines.query import hyde
from app.helpers.logger import get_logger

logger = get_logger(__name__)


def generate_response(query: str, context: dict) -> dict:
    """
    Run the full retrieval + generation pipeline with quality gates.

    Args:
        query:   Rewritten, self-contained query from Pipeline 2.
        context: {session_id, intent, entities, user_email}

    Returns:
        {answer, citations, confidence, should_escalate}
    """
    session_id = context.get("session_id", "")
    user_email = context.get("user_email", "")
    entities   = context.get("entities") or {}

    logger.info("P3 start | session=%s user=%s", session_id, user_email)

    langfuse = get_langfuse()
    trace = langfuse.trace(name="p3_generate_response", session_id=session_id, input=query)

    # ── ① HyDE Lite: blend query vector with hypothetical-answer vector ───────
    span = trace.span(name="p3_hyde", input=query)
    search_vector = hyde.get_blended_vector(query)
    span.end(output=f"blended vector dim={len(search_vector)}")

    # ── ② Load direct data from Supabase: invoices + orders + products ────────
    db_invoices     = order_lookup.get_invoices_for_user(user_email) if user_email else []
    live_data_parts = order_lookup.load_live_data(user_email, entities)
    has_direct_data = bool(db_invoices or live_data_parts)
    logger.debug("P3 direct data: %d invoices, %d live parts", len(db_invoices), len(live_data_parts))

    # ── ③ Vector search (with HyDE-blended vector) ───────────────────────────
    span = trace.span(name="p3_retrieve", input=f"query={query[:60]}")
    results = vector_store.search(search_vector, top_k=5)
    top_score = results[0].score if results else 0.0
    span.end(output=f"{len(results)} chunks | top_score={top_score:.3f}")
    logger.info("P3 retrieved %d chunks | top_score=%.3f", len(results), top_score)

    # ── ④ Assemble context: date facts → invoices → live data → FAQ chunks ────
    full_context = assemble_context(db_invoices, live_data_parts, results)

    # ── ⑤ LLM generation (Azure OpenAI gpt-4.1) ──────────────────────────────
    answer, _usage = generate_answer(trace, full_context, query)

    # ── ⑥ Citations: invoices first, then scored FAQ chunks ──────────────────
    citations = build_invoice_citations(db_invoices)
    citations.extend(build_faq_citations(results))

    # ── ⑥ Composite confidence score ─────────────────────────────────────────
    confidence = compute_confidence(top_score, has_direct_data)
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

