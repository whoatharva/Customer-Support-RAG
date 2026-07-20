"""Pipeline 2: Query Input Processing

Steps per turn:
  1. Load recent conversation history from Supabase.
  2. One combined LLM call that rewrites the query into a self-contained question
     AND detects intent + extracts entities (order_id, product_name) as JSON.
  3. Return a dict that Pipeline 3 and the chat service consume.
"""

from app.helpers import database
from app.llm import client as llm
from app.llm.parsing import parse_json_lenient
from app.llm.prompts import PROCESS_QUERY_PROMPT
from app.helpers.langfuse import get_langfuse
from app.helpers.logger import get_logger

logger = get_logger(__name__)

# Only the most recent messages are useful for pronoun/ellipsis resolution.
HISTORY_WINDOW = 6


def process_query(query: str, session_id: str) -> dict:
    """
    Returns:
        {
            "original_query": str,
            "rewritten_query": str,
            "intent": str,
            "entities": dict
        }
    """
    logger.info("P2 start | session=%s query_len=%d", session_id, len(query))
    langfuse = get_langfuse()
    trace = langfuse.trace(name="p2_process_query", session_id=session_id, input=query)

    # ── Step 1: conversation history (from Supabase, durable across workers) ────
    history = database.get_chat_history_db(session_id, limit=HISTORY_WINDOW)
    logger.debug("P2 history: %d message(s)", len(history))
    history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)

    # ── Step 2: combined rewrite + intent + entity extraction (one LLM call) ────
    prompt = PROCESS_QUERY_PROMPT.format(history=history_text or "(none)", query=query)
    span = trace.span(name="p2_process", input=prompt)
    response = llm.chat([{"role": "user", "content": prompt}])
    raw_text = response["choices"][0]["message"]["content"].strip()
    span.end(output=raw_text)

    rewritten_query = query
    intent = "other"
    entities: dict = {}
    parsed = parse_json_lenient(raw_text, slice_from="{")
    if isinstance(parsed, dict):
        rewritten_query = (parsed.get("rewritten_query") or query).strip() or query
        intent = parsed.get("intent", "other")
        entities = parsed.get("entities", {})
    else:
        logger.warning("P2 parse failed, defaulting | raw=%r", raw_text)

    if rewritten_query != query:
        logger.debug("P2 rewrite: %r → %r", query, rewritten_query)
    logger.info("P2 done | intent=%s entities=%s", intent, entities)
    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "intent": intent,
        "entities": entities,
    }
