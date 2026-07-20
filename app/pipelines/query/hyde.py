"""HyDE Lite — Hypothetical Document Embeddings (lightweight version).

WHAT IS HyDE?
  Normal RAG embeds the user's question and searches for similar chunks.
  Problem: a short question like "warranty on my laptop?" has a very different
  embedding than the FAQ answer it's looking for.

  HyDE fixes this by:
    1. Asking a lightweight LLM "what would a good answer to this question look like?"
    2. Embedding THAT hypothetical answer instead of (or alongside) the question
    3. The hypothetical answer embedding lives in the same semantic space as real answers

WHY "LITE"?
  Full HyDE generates a whole paragraph. We generate only 2 sentences using a cheap
  model (Gemini / Groq) — fast enough to add < 300 ms to the pipeline.

HOW WE BLEND:
  final_vector = 0.7 * query_vector + 0.3 * hyde_vector  (then normalise)
  Keeps the original query intent dominant while shifting toward answer-space.
"""

from app.llm import client as llm
from app.llm import lightweight
from app.llm.prompts import HYDE_PROMPT
from app.config import settings
from app.helpers.logger import get_logger

logger = get_logger(__name__)


def get_blended_vector(query: str) -> list[float]:
    """
    Returns a blended embedding vector:
      70% from the original query
      30% from a hypothetical answer to that query

    Falls back to plain query embedding if HyDE generation fails.
    """
    # Step 1: generate a hypothetical answer via lightweight LLM
    prompt = HYDE_PROMPT.format(query=query)
    hypothetical_answer = lightweight.call(prompt, max_tokens=120)

    if not hypothetical_answer:
        logger.warning("HyDE: hypothetical answer generation failed — using plain query vector")
        return llm.embed([query])[0]

    logger.debug("HyDE: hypothetical answer = %r", hypothetical_answer[:80])

    # Step 2: embed query + hypothetical answer in ONE batched API call
    query_vector, hyde_vector = llm.embed([query, hypothetical_answer])

    # Step 3: blend and normalise
    blended = [settings.hyde_query_weight * q + settings.hyde_vector_weight * h for q, h in zip(query_vector, hyde_vector)]
    magnitude = sum(x * x for x in blended) ** 0.5
    if magnitude == 0:
        return query_vector
    normalised = [x / magnitude for x in blended]

    logger.debug("HyDE: blended vector produced (%.0f%% query + %.0f%% hypothetical)", settings.hyde_query_weight * 100, settings.hyde_vector_weight * 100)
    return normalised
