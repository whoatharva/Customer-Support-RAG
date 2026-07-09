"""Lightweight LLM client for cheap, fast operations (HyDE generation, relevancy checks).

Strategy:
  Primary  — Google Gemini (gemini-1.5-flash): fast and cheap
  Fallback — Groq (llama-3.1-8b-instant): free tier, very fast inference

These are used only for:
  - HyDE: generating a short hypothetical answer to improve query embedding
  - Relevancy gate: checking if retrieved context actually matches the query
  - Answer check: verifying the LLM's answer is on-topic

Heavy answer generation still uses Azure OpenAI (gpt-4.1) via app/llm/client.py.
"""

from app.config import settings
from app.helpers.logger import get_logger

logger = get_logger(__name__)


def call(prompt: str, max_tokens: int = 300) -> str:
    """
    Run a short prompt through Gemini first; fall back to Groq if Gemini fails.
    Returns the model's text response, or empty string if both fail.
    """
    if settings.gemini_api_key:
        try:
            return _gemini(prompt, max_tokens)
        except Exception as e:
            logger.warning("Gemini failed, trying Groq: %s", e)

    if settings.groq_api_key:
        try:
            return _groq(prompt, max_tokens)
        except Exception as e:
            logger.warning("Groq also failed: %s", e)

    logger.error("Both Gemini and Groq unavailable — lightweight LLM call skipped")
    return ""


# ── Providers ─────────────────────────────────────────────────────────────────

def _gemini(prompt: str, max_tokens: int) -> str:
    from google import genai
    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={"max_output_tokens": max_tokens, "temperature": 0.2},
    )
    return response.text.strip()


def _groq(prompt: str, max_tokens: int) -> str:
    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
