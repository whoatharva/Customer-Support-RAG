"""Main-LLM answer generation with Langfuse tracing.

Public API:
  generate_answer(trace, full_context, query) -> (answer, usage)
"""

from app.config import settings
from app.llm import client as llm
from app.llm.prompts import SYSTEM_PROMPT, USER_PROMPT
from app.helpers import date_facts
from app.helpers.logger import get_logger

logger = get_logger(__name__)


def generate_answer(trace, full_context: str, query: str) -> tuple[str, dict]:
    """Call the LLM with assembled context; return (answer, usage) and log the generation."""
    system_msg = SYSTEM_PROMPT.format(today=date_facts.today_str())
    user_msg = USER_PROMPT.format(context=full_context, query=query)
    generation = trace.generation(
        name="p3_generate",
        model=settings.azure_openai_deployment_name,
        input=user_msg[:500],
    )
    response = llm.chat([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])
    answer = response["choices"][0]["message"]["content"].strip()
    usage = response.get("usage") or {}
    generation.end(
        output=answer[:300],
        usage={
            "input":  usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
            "total":  usage.get("total_tokens", 0),
        },
    )
    logger.info(
        "P3 generated answer: %d chars | tokens in=%s out=%s",
        len(answer), usage.get("prompt_tokens"), usage.get("completion_tokens"),
    )
    return answer, usage
