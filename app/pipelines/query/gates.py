"""Input scope gate — a single quality check that runs BEFORE retrieval.

WHY A SCOPE GATE?
  The chatbot only helps with this store's orders, products, invoices, shipping,
  returns, warranty, payments, and account/profile. Without a scope check, a user
  could ask it to write code, answer trivia, or otherwise act outside customer
  support — and we'd pay for retrieval + the main LLM to (badly) attempt it.

  The gate asks a cheap lightweight LLM: "Is this question in scope?" If not, we
  escalate immediately and never touch Qdrant or the main LLM.

  Uses Gemini (primary) / Groq (fallback). If both are unavailable it defaults to
  True (pass) so the pipeline degrades open rather than blocking all traffic.
"""

from app.llm import lightweight
from app.llm.prompts import SCOPE_CHECK_PROMPT
from app.helpers.logger import get_logger

logger = get_logger(__name__)


def scope_gate(query: str) -> bool:
    """Ask a lightweight LLM whether the question is within customer-support scope.

    Args:
        query: The user's question.

    Returns:
        True  → pass (in scope, proceed to retrieval)
        False → block (off-topic, escalate before spending retrieval/LLM cost)
    """
    prompt = SCOPE_CHECK_PROMPT.format(query=query)
    verdict = lightweight.call(prompt, max_tokens=5).lower()

    if not verdict:
        # Lightweight LLM unavailable — let the query through (degrade open).
        logger.warning("SCOPE GATE SKIP: lightweight LLM unavailable — defaulting to pass")
        return True

    passed = verdict.startswith("yes")
    logger.info("SCOPE GATE %s: verdict=%r", "PASS" if passed else "BLOCK", verdict)
    return passed
