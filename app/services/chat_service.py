"""Orchestrates a single chat turn: Pipeline 2 → Pipeline 3 → persist.

Flow:
  1. Call P2 (process_query) → intent, entities, rewritten query.
  2. Call P3 (generate_response) → answer, citations, confidence.
  3. Persist turn to in-memory session (for multi-turn context in P2).
  4. Persist turn to Supabase (for history endpoint + durability).
  5. Update Langfuse trace with final output + metadata.
"""

from app.schemas import ChatRequest, ChatResponse
from app.pipelines.query.processor import process_query
from app.pipelines.retrieval.engine import generate_response
from app.helpers import session as mem
from app.helpers import database
from app.helpers.langfuse import get_langfuse
from app.helpers.logger import get_logger

logger = get_logger(__name__)


def handle_chat(request: ChatRequest, user_email: str = "") -> ChatResponse:
    logger.info(
        "chat turn | session=%s user=%s invoice=%s query_len=%d",
        request.session_id, user_email, request.invoice_id, len(request.query),
    )

    langfuse = get_langfuse()
    trace = langfuse.trace(
        name="chat_turn",
        session_id=request.session_id,
        user_id=user_email,
        input=request.query,
        metadata={"invoice_id": request.invoice_id},
    )

    # ── Pipeline 2: process query ─────────────────────────────────────────────
    processed = process_query(request.query, request.session_id)

    # ── Pipeline 3: retrieve + generate ──────────────────────────────────────
    context = {
        "invoice_id": request.invoice_id,
        "session_id": request.session_id,
        "intent": processed["intent"],
        "entities": processed["entities"],
        "user_email": user_email,
    }
    result = generate_response(processed["rewritten_query"], context)

    answer = result["answer"]
    citations = result["citations"]
    confidence = result["confidence"]
    should_escalate = result["should_escalate"]
    total_tokens = result.get("total_tokens")  # set by engine when LLM is called

    # ── Persist: in-memory session (used by P2 for next-turn rewrite) ─────────
    mem.append(request.session_id, "user", request.query)
    mem.append(request.session_id, "assistant", answer)

    # ── Persist: Supabase (durable history) ───────────────────────────────────
    try:
        database.ensure_chat_session(request.session_id, user_email)
        database.save_chat_message(request.session_id, "user", request.query)
        database.save_chat_message(request.session_id, "assistant", answer)
    except Exception:
        logger.error("failed to persist chat to Supabase | session=%s", request.session_id, exc_info=True)

    # ── Langfuse: update top-level trace with outcome ─────────────────────────
    trace.update(
        output=answer,
        metadata={
            "intent": processed["intent"],
            "confidence": confidence,
            "should_escalate": should_escalate,
            "citations_count": len(citations),
            "total_tokens": total_tokens,
        },
    )

    logger.info(
        "chat turn done | session=%s confidence=%.2f escalate=%s tokens=%s",
        request.session_id, confidence, should_escalate, total_tokens,
    )
    response = ChatResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        should_escalate=should_escalate,
    )
    # carry token count for the wrapper's cost score log (non-API field)
    response._total_tokens = total_tokens
    return response
