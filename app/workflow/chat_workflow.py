"""End-to-end chat workflow.

Wraps the chat service with a fallback so the API never returns a 500.
The escalation flag is set by the retriever when confidence is below the threshold.
Any unexpected exception returns a safe fallback response.

Cost tracking:
  Each LLM call logs a Langfuse `generation` with model name + token counts.
  Langfuse calculates dollar cost automatically from those.
  This wrapper logs a per-turn `cost_summary` score (total tokens) on the
  parent trace so you can filter expensive turns in the Langfuse UI.
"""

from app.helpers.langfuse import get_langfuse
from app.llm.prompts import ESCALATION_ANSWER
from app.schemas import ChatRequest, ChatResponse
from app.services.chat_service import handle_chat
from app.helpers.logger import get_logger

logger = get_logger(__name__)


def run(request: ChatRequest, user_email: str = "") -> ChatResponse:
    """Run the full chat pipeline; returns a safe ChatResponse in all cases."""
    try:
        response = handle_chat(request, user_email=user_email)

        # Log total token count as a score so Langfuse can surface expensive turns.
        # Per-generation costs (USD) are tracked automatically via trace.generation()
        # in retriever.py — this score is for quick filtering by token volume.
        _log_cost_score(request.session_id, response)

        return response

    except Exception:
        logger.error(
            "chat workflow failed | session=%s query=%r",
            request.session_id, request.query[:80], exc_info=True,
        )
        return ChatResponse(
            answer=ESCALATION_ANSWER,
            citations=[],
            confidence=0.0,
            should_escalate=True,
        )


def _log_cost_score(session_id: str, response: ChatResponse) -> None:
    """Add a token-count score to the Langfuse trace for this session turn."""
    try:
        total_tokens = response.total_tokens
        if total_tokens is None:
            return
        get_langfuse().score(
            trace_id=session_id,
            name="total_tokens",
            value=float(total_tokens),
            comment="input + output tokens for this chat turn",
        )
        logger.debug("cost score logged: session=%s tokens=%d", session_id, total_tokens)
    except Exception:
        logger.debug("cost score logging skipped (non-critical)")

