"""Orchestrates a single chat turn: Pipeline 2 → Pipeline 3 → persist.

Flow:
  1. Order cancellation check (deterministic, no LLM) — intercepts before RAG.
  2. Call P2 (process_query) → intent, entities, rewritten query.
  3. Call P3 (generate_response) → answer, citations, confidence.
  4. Persist turn to Supabase (durable history for P2 rewrite + UI).
  5. Update Langfuse trace with final output + metadata.
"""

from app.schemas import ChatRequest, ChatResponse
from app.pipelines.query.processor import process_query
from app.pipelines.retrieval.retriever import generate_response
from app.services import profile_service
from app.services import order_cancellation
from app.helpers import database
from app.helpers import order_lookup
from app.helpers.text import contains_any
from app.llm import lightweight
from app.llm.parsing import parse_json_lenient
from app.llm.prompts import SPLIT_PROMPT
from app.helpers.langfuse import get_langfuse
from app.helpers.logger import get_logger

logger = get_logger(__name__)


def _persist_turn(session_id: str, user_email: str, query: str, answer: str):
    """Store a user/assistant exchange in Supabase (best effort)."""
    database.persist_chat_turn(session_id, user_email, query, answer)


def handle_chat(request: ChatRequest, user_email: str = "") -> ChatResponse:
    logger.info(
        "chat turn | session=%s user=%s query_len=%d",
        request.session_id, user_email, len(request.query),
    )

    # ── Order cancellation branch (deterministic, no LLM) ──────────────────
    if user_email and order_cancellation.matches_cancel_keywords(request.query):
        return _handle_cancel(request, user_email)

    # ── Multi-intent branch (gated) ──────────────────────────────────────────
    # Cheap heuristic first; only if it fires do we pay for the LLM splitter.
    if _looks_multi_intent(request.query):
        subs = _split_intents(request.query)
        if len(subs) > 1:
            return _handle_multi(request, user_email, subs)

    # ── Self-service profile branch (skips RAG) ──────────────────────────────
    if user_email and profile_service.matches_keywords(request.query):
        handled = _handle_profile(request, user_email)
        if handled is not None:
            return handled
        # action == "none" → fall through to RAG

    result = _run_rag(request, user_email)
    _persist_turn(request.session_id, user_email, request.query, result["answer"])
    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"],
        confidence=result["confidence"],
        should_escalate=result["should_escalate"],
        total_tokens=result.get("total_tokens"),
    )


def _handle_cancel(request: ChatRequest, user_email: str) -> ChatResponse:
    """Handle an order cancellation intent entirely in Python — no LLM, no RAG.

    Detects whether the user named an order ID or a product, resolves the order,
    checks cancellability, then returns either:
      - A Confirm/Cancel prompt (action="order_cancel") if the order can be cancelled.
      - A direct rejection message if it cannot.
    The actual DB write happens only when the user clicks Confirm in the UI.
    """
    intent = order_cancellation.detect_cancel_intent(request.query)
    order = order_cancellation.find_order_to_cancel(intent, user_email)

    if order is None:
        answer = (
            "I couldn't find an order matching that description on your account. "
            "Please check your order ID or item name and try again."
        )
        _persist_turn(request.session_id, user_email, request.query, answer)
        return ChatResponse(answer=answer, citations=[], confidence=1.0)

    order_id = order.get("order_id", "")
    current_status = order.get("status", "")
    items = order_lookup.format_order_items(order)

    if current_status in order_cancellation.CANCELLABLE_STATUSES:
        answer = (
            f"I found your order **{order_id}**"
            + (f" ({items})" if items else "")
            + f", currently **{current_status}**. "
            "Do you want to cancel it?"
        )
        _persist_turn(request.session_id, user_email, request.query, answer)
        return ChatResponse(
            answer=answer,
            citations=[],
            confidence=1.0,
            action="order_cancel",
            action_payload={"order_id": order_id, "previous_status": current_status},
        )

    # Non-cancellable — look up the rejection reason from the service.
    result = order_cancellation.cancel_order(order_id, user_email)
    _persist_turn(request.session_id, user_email, request.query, result.message)
    return ChatResponse(answer=result.message, citations=[], confidence=1.0)


def _handle_profile(request: ChatRequest, user_email: str) -> ChatResponse | None:
    """Handle a self-service profile view/update. Returns None to fall through to RAG."""
    det = profile_service.detect(request.query)
    if det["action"] == "view":
        answer = profile_service.render_profile(user_email)
        _persist_turn(request.session_id, user_email, request.query, answer)
        return ChatResponse(answer=answer, citations=[], confidence=1.0)

    if det["action"] == "update" and det["field"] in ("phone", "address"):
        ok, error = profile_service.validate(det["field"], det["values"])
        if not ok:
            _persist_turn(request.session_id, user_email, request.query, error)
            return ChatResponse(answer=error, citations=[], confidence=1.0)

        confirm = profile_service.confirm_prompt(det)
        payload = {"field": det["field"], "values": det["values"]}
        # Compound query (e.g. "status of my order AND update my phone"):
        # answer the question part via RAG first, then append the confirm
        # prompt so the user gets both in one turn without re-asking.
        if _has_secondary_question(request.query):
            rag = _run_rag(request, user_email)
            answer = f"{rag['answer']}\n\n---\n\nAlso, about your profile update: {confirm}"
            _persist_turn(request.session_id, user_email, request.query, answer)
            return ChatResponse(
                answer=answer,
                citations=rag["citations"],
                confidence=rag["confidence"],
                should_escalate=rag["should_escalate"],
                total_tokens=rag.get("total_tokens"),
                action="profile_update",
                action_payload=payload,
            )
        _persist_turn(request.session_id, user_email, request.query, confirm)
        return ChatResponse(
            answer=confirm, citations=[], confidence=1.0,
            action="profile_update", action_payload=payload,
        )

    # action == "none" → caller falls through to RAG
    return None


def _classify_sub(sub: str, user_email: str) -> dict:
    """Classify one sub-query into a profile action (if any) without emitting a
    response. Returns {"kind": "profile_update"|"profile_view"|"rag", "det"?: dict}."""
    if user_email and profile_service.matches_keywords(sub):
        det = profile_service.detect(sub)
        if det["action"] == "view":
            return {"kind": "profile_view"}
        if det["action"] == "update" and det["field"] in ("phone", "address"):
            ok, _ = profile_service.validate(det["field"], det["values"])
            if ok:
                return {"kind": "profile_update", "det": det}
    return {"kind": "rag"}


def _handle_multi(request: ChatRequest, user_email: str, subs: list[str]) -> ChatResponse:
    """Process each sub-query in order and combine into one response.

    Question/order sub-queries are answered inline; the first valid profile update is
    deferred and appended last with Confirm/Cancel buttons (only one update per turn).
    """
    sections: list[str] = []
    citations: list = []
    should_escalate = False
    total_tokens = 0
    pending_update: dict | None = None
    skipped_update = False

    for i, sub in enumerate(subs, 1):
        cls = _classify_sub(sub, user_email)
        if cls["kind"] == "profile_update":
            if pending_update is None:
                pending_update = cls["det"]
            else:
                skipped_update = True  # only one update actioned per turn
            continue
        if cls["kind"] == "profile_view":
            sections.append(f"**{i}. Your profile**\n\n{profile_service.render_profile(user_email)}")
            continue
        rag = _run_rag(request, user_email, query=sub)
        sections.append(f"**{i}. {sub}**\n\n{rag['answer']}")
        citations.extend(rag["citations"])
        should_escalate = should_escalate or rag["should_escalate"]
        total_tokens += rag.get("total_tokens") or 0

    answer = "\n\n---\n\n".join(sections) if sections else ""

    action = None
    action_payload = None
    if pending_update is not None:
        confirm = profile_service.confirm_prompt(pending_update)
        prefix = f"{answer}\n\n---\n\n" if answer else ""
        answer = f"{prefix}Also, about your profile update: {confirm}"
        if skipped_update:
            answer += (
                "\n\n_(You asked to change more than one profile field — please confirm "
                "this one, then ask for the other.)_"
            )
        action = "profile_update"
        action_payload = {"field": pending_update["field"], "values": pending_update["values"]}

    _persist_turn(request.session_id, user_email, request.query, answer)
    return ChatResponse(
        answer=answer,
        citations=citations,
        confidence=0.9,
        should_escalate=should_escalate,
        total_tokens=total_tokens or None,
        action=action,
        action_payload=action_payload,
    )


def _run_rag(request: ChatRequest, user_email: str, query: str | None = None) -> dict:
    """Run the full RAG path (P2 → P3) and return the raw result dict.

    `query` overrides `request.query` when handling a single sub-query of a
    decomposed multi-intent message; session_id/user_email are unchanged.
    """
    q = query or request.query
    langfuse = get_langfuse()
    trace = langfuse.trace(
        id=request.session_id,
        name="chat_turn",
        session_id=request.session_id,
        user_id=user_email,
        input=q,
    )

    # ── Pipeline 2: process query ─────────────────────────────────────────────
    processed = process_query(q, request.session_id)

    # ── Pipeline 3: retrieve + generate ──────────────────────────────────────
    context = {
        "session_id": request.session_id,
        "intent": processed["intent"],
        "entities": processed["entities"],
        "user_email": user_email,
    }
    result = generate_response(processed["rewritten_query"], context)

    confidence = result["confidence"]
    should_escalate = result["should_escalate"]
    total_tokens = result.get("total_tokens")  # set by engine when LLM is called

    # ── Langfuse: update top-level trace with outcome ─────────────────────────
    trace.update(
        output=result["answer"],
        metadata={
            "intent": processed["intent"],
            "confidence": confidence,
            "should_escalate": should_escalate,
            "citations_count": len(result["citations"]),
            "total_tokens": total_tokens,
        },
    )

    logger.info(
        "chat turn done | session=%s confidence=%.2f escalate=%s tokens=%s",
        request.session_id, confidence, should_escalate, total_tokens,
    )
    return result


# Words that signal a question beyond the profile update itself. A profile-update
# query legitimately contains phrases like "update"/"change"/"my phone", so we look
# for order/status/tracking-type asks instead of a generic "?".
_SECONDARY_QUESTION_TERMS = (
    "order", "status", "track", "tracking", "delivery", "deliver", "shipped",
    "shipping", "arrive", "invoice", "refund", "return", "warranty", "where is",
)


def _has_secondary_question(query: str) -> bool:
    """True if a profile-update query ALSO carries an order/support question, so the
    turn should answer that via RAG before showing the update confirm."""
    return contains_any(query, _SECONDARY_QUESTION_TERMS)


# Cheap connector signals that a message may bundle more than one request. Used only
# to gate the (LLM) splitter — conservative on purpose: a miss just falls back to
# today's single-shot path, a false positive costs one lightweight-LLM call.
_MULTI_INTENT_CONNECTORS = (" and ", " also ", " plus ", ";", " as well as ")


def _looks_multi_intent(query: str) -> bool:
    """Cheap heuristic: does this message plausibly hold more than one request?"""
    q = query.lower().strip()
    if len(q) < 25:
        return False
    if q.count("?") >= 2:
        return True
    if not contains_any(q, _MULTI_INTENT_CONNECTORS):
        return False
    # A connector plus either a profile ask or an order/support term is a strong
    # signal of two distinct intents (question + question, or question + update).
    return profile_service.matches_keywords(q) or _has_secondary_question(q)


def _split_intents(query: str) -> list[str]:
    """Decompose a multi-intent message into sub-queries via the lightweight LLM.

    Degrades to `[query]` (single-shot) on any parse/LLM failure or a 1-item result.
    """
    try:
        raw = lightweight.call(SPLIT_PROMPT.format(query=query), max_tokens=300)
        parsed = parse_json_lenient(raw, slice_from="[")
        subs = [s.strip() for s in (parsed or []) if isinstance(s, str) and s.strip()]
    except Exception:
        logger.warning("intent split failed, treating as single query", exc_info=True)
        return [query]
    if len(subs) <= 1:
        return [query]
    logger.info("split into %d sub-queries: %s", len(subs), subs)
    return subs
