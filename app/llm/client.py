# Azure OpenAI wrapper used by all pipelines.
# embed(texts) → list of float vectors (Pipeline 1 + 3)
# chat(messages, tools) → LLM response (Pipeline 3)

def embed(texts: list[str]) -> list[list[float]]:
    raise NotImplementedError


def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    raise NotImplementedError
