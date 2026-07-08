# In-memory conversation history keyed by session_id.
# Each session stores a list of {"role": "user"|"assistant", "content": str} dicts.
# Pipeline 2 reads history for query rewriting; Pipeline 3 appends after each turn.

_store: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    return _store.get(session_id, [])


def append(session_id: str, role: str, content: str):
    _store.setdefault(session_id, []).append({"role": role, "content": content})


def clear(session_id: str):
    _store.pop(session_id, None)
