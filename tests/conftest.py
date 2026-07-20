"""Shared pytest fixtures + hermetic setup for the unit test suite.

Importing any `app.*` module instantiates `app.config.Settings()`, which reads
required fields from the environment. We inject dummy values here — BEFORE any
`app` import — with `setdefault`, so the suite runs with zero real credentials
and never touches Azure / Qdrant / Supabase / Langfuse. All network calls are
mocked in the tests themselves.
"""

import os

# ── Hermetic env: set required settings to dummies (only if not already set) ──
_DUMMY_ENV = {
    "AZURE_OPENAI_API_KEY": "test-key",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-4.1",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_API_KEY": "test-qdrant",
    "SUPABASE_URL": "http://localhost:54321",
    "SUPABASE_KEY": "test-supabase",
    "JWT_SECRET": "test-secret",
    "ADMIN_USERNAME": "admin",
    "ADMIN_PASSWORD": "admin",
    "LANGFUSE_SECRET_KEY": "test-lf-secret",
    "LANGFUSE_PUBLIC_KEY": "test-lf-public",
}
for _k, _v in _DUMMY_ENV.items():
    os.environ.setdefault(_k, _v)

import pytest


@pytest.fixture
def make_scored_point():
    """Factory building a Qdrant-style scored point with `.score` and `.payload`.

    A lightweight stub is used rather than the real `ScoredPoint` so tests don't
    depend on qdrant_client's required constructor fields (id/version).
    """
    class _StubPoint:
        def __init__(self, score: float, payload: dict):
            self.score = score
            self.payload = payload

    def _make(score: float, **payload) -> _StubPoint:
        payload.setdefault("chunk_id", "chunk_0")
        payload.setdefault("doc_filename", "shipping.md")
        payload.setdefault("section", "General")
        payload.setdefault("text", "Some retrieved FAQ text.")
        return _StubPoint(score, payload)

    return _make


@pytest.fixture
def chat_response():
    """Factory building the dict shape `app.llm.client.chat` returns."""
    def _make(content: str, total_tokens: int = 42) -> dict:
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": total_tokens // 2,
                "completion_tokens": total_tokens - total_tokens // 2,
                "total_tokens": total_tokens,
            },
        }

    return _make
