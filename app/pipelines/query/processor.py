"""Pipeline 2: Query Input Processing

Steps per turn:
  1. Load conversation history from in-memory session store.
  2. Rewrite the raw query into a self-contained question (only when history exists).
  3. Detect intent and extract key entities (order_id, invoice_id, product_name) via
     a single LLM call that returns JSON.
  4. Return a dict that Pipeline 3 and the chat service consume.
"""

import json
from app.helpers import session as memory
from app.llm import client as llm
from app.llm.prompts import QUERY_REWRITE_TEMPLATE
from app.helpers.langfuse import get_langfuse
from app.helpers.logger import get_logger

logger = get_logger(__name__)

_INTENT_PROMPT = """
Analyse the customer support query below and respond with ONLY valid JSON — no extra text.

Query: {query}

JSON format:
{{
  "intent": "<one of: return_request | shipping_inquiry | warranty_inquiry | payment_inquiry | account_inquiry | general_policy | other>",
  "entities": {{
    "order_id": "<order id string or null>",
    "invoice_id": "<invoice id string or null>",
    "product_name": "<product name or null>"
  }}
}}
""".strip()


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

    # ── Step 1: conversation history ─────────────────────────────────────────
    history = memory.get_history(session_id)
    logger.debug("P2 history: %d message(s)", len(history))

    # ── Step 2: query rewrite (only when there is prior context) ─────────────
    if history:
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history[-6:])
        prompt = QUERY_REWRITE_TEMPLATE.format(history=history_text, query=query)
        span = trace.span(name="p2_query_rewrite", input=prompt)
        response = llm.chat([{"role": "user", "content": prompt}])
        rewritten_query = response["choices"][0]["message"]["content"].strip()
        span.end(output=rewritten_query)
        logger.debug("P2 rewrite: %r → %r", query, rewritten_query)
    else:
        rewritten_query = query
        logger.debug("P2 rewrite: skipped (no history)")

    # ── Step 3: intent + entity extraction ───────────────────────────────────
    intent_prompt = _INTENT_PROMPT.format(query=rewritten_query)
    span = trace.span(name="p2_intent_extract", input=intent_prompt)
    raw = llm.chat([{"role": "user", "content": intent_prompt}])
    raw_text = raw["choices"][0]["message"]["content"].strip()
    span.end(output=raw_text)

    try:
        parsed = json.loads(raw_text)
        intent = parsed.get("intent", "other")
        entities = parsed.get("entities", {})
    except json.JSONDecodeError:
        logger.warning("P2 intent parse failed, defaulting | raw=%r", raw_text)
        intent = "other"
        entities = {}

    logger.info("P2 done | intent=%s entities=%s", intent, entities)
    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "intent": intent,
        "entities": entities,
    }
