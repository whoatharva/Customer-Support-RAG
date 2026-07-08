# Pure text utilities — no LLM, no DB dependencies.


def truncate(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def clean(text: str) -> str:
    return " ".join(text.split())
