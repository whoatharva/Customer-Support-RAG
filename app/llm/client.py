from openai import AzureOpenAI
from app.config import settings
from app.helpers.logger import get_logger

logger = get_logger(__name__)
_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version="2024-08-01-preview",
        )
        logger.info("AzureOpenAI client initialised")
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input string."""
    logger.debug("embedding %d text(s)", len(texts))
    response = _get_client().embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    logger.debug("embedding done: %d vector(s), dim=%d", len(vectors), len(vectors[0]) if vectors else 0)
    return vectors


def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Call the chat completion model. Returns the full response as a dict."""
    logger.debug("chat call: %d message(s), tools=%s", len(messages), bool(tools))
    kwargs = dict(model=settings.azure_openai_deployment_name, messages=messages)
    if tools:
        kwargs["tools"] = tools
    response = _get_client().chat.completions.create(**kwargs)
    result = response.model_dump()
    usage = result.get("usage", {})
    logger.debug(
        "chat done: prompt_tokens=%s, completion_tokens=%s",
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
    )
    return result
