"""Pipeline 3 — retrieval & response generation tests.

`retriever.py` references external modules via alias, so we patch within the retriever
module's namespace: `retriever.vector_store.search`, `retriever.hyde.get_blended_vector`,
`retriever.order_lookup.*`. The main LLM call now lives in `app.helpers.generation`, so
`generation.llm.chat` and `date_facts.today_facts` are patched at their source modules.
Langfuse is stubbed to a MagicMock.

The input scope gate now lives in `app.pipelines.query.gates.scope_gate` and is
enforced by the chat service before retrieval, so it is unit-tested directly here.
"""

from unittest.mock import MagicMock

from app.pipelines.retrieval import retriever
from app.pipelines.query import gates
from app.helpers import date_facts, generation


def _patch_common(monkeypatch):
    """Neutralise everything the retriever touches except what a test overrides."""
    monkeypatch.setattr(retriever, "get_langfuse", lambda: MagicMock())
    monkeypatch.setattr(retriever.hyde, "get_blended_vector", lambda q: [0.0] * 8)
    monkeypatch.setattr(retriever.order_lookup, "get_invoices_for_user", lambda e: [])
    monkeypatch.setattr(retriever.order_lookup, "get_orders_for_user", lambda e: [])
    monkeypatch.setattr(date_facts, "today_facts", lambda: "Today is a test day.")


def test_generate_response_returns_answer(monkeypatch, make_scored_point, chat_response):
    _patch_common(monkeypatch)
    points = [
        make_scored_point(0.85, chunk_id="c1", text="Standard shipping is 4-7 days."),
        make_scored_point(0.72, chunk_id="c2", text="Express shipping is 1-3 days."),
    ]
    monkeypatch.setattr(retriever.vector_store, "search", lambda vec, top_k=5: points)
    monkeypatch.setattr(generation.llm, "chat", lambda msgs: chat_response("Here are the shipping options.", total_tokens=120))

    result = retriever.generate_response("shipping options?", {"session_id": "s", "user_email": ""})

    assert result["answer"] == "Here are the shipping options."
    assert len(result["citations"]) == 2
    assert result["confidence"] == 0.85  # no direct data → raw top score
    assert result["should_escalate"] is False
    assert result["total_tokens"] == 120


def test_low_confidence_sets_escalate_flag(monkeypatch, make_scored_point, chat_response):
    _patch_common(monkeypatch)
    points = [make_scored_point(0.2, chunk_id="c1", text="Barely relevant text.")]
    monkeypatch.setattr(retriever.vector_store, "search", lambda vec, top_k=5: points)
    monkeypatch.setattr(generation.llm, "chat", lambda msgs: chat_response("A weak answer."))

    result = retriever.generate_response("obscure question?", {"session_id": "s", "user_email": ""})

    assert result["confidence"] == 0.2
    assert result["should_escalate"] is True  # 0.2 < confidence_threshold (0.5)


# ── Focused scope-gate unit tests ────────────────────────────────────────────

def test_scope_gate_in_scope_passes(monkeypatch):
    monkeypatch.setattr(gates.lightweight, "call", lambda *a, **k: "yes")
    assert gates.scope_gate("where is my order?") is True


def test_scope_gate_off_topic_blocks(monkeypatch):
    monkeypatch.setattr(gates.lightweight, "call", lambda *a, **k: "no")
    assert gates.scope_gate("write me python code to sort a list") is False


def test_scope_gate_empty_defaults_true(monkeypatch):
    # Lightweight LLM unavailable → degrade open (pass).
    monkeypatch.setattr(gates.lightweight, "call", lambda *a, **k: "")
    assert gates.scope_gate("anything") is True
