"""Pipeline 3 — retrieval & response generation tests.

`retriever.py` references external modules via alias, so we patch within the retriever
module's namespace: `retriever.vector_store.search`, `retriever.gates.*`,
`retriever.llm.chat`, `retriever.hyde.get_blended_vector`, `retriever.order_lookup.*`,
`retriever.date_facts.today_facts`. Langfuse is stubbed to a MagicMock.

Gate unit tests patch `app.llm.lightweight.call` (the only external call in
`gates.answer_relevancy_gate`).
"""

from unittest.mock import MagicMock

from app.pipelines.retrieval import retriever
from app.pipelines.retrieval import gates


def _patch_common(monkeypatch):
    """Neutralise everything the retriever touches except what a test overrides."""
    monkeypatch.setattr(retriever, "get_langfuse", lambda: MagicMock())
    monkeypatch.setattr(retriever.hyde, "get_blended_vector", lambda q: [0.0] * 8)
    monkeypatch.setattr(retriever.order_lookup, "get_invoices_for_user", lambda e: [])
    monkeypatch.setattr(retriever.order_lookup, "get_orders_for_user", lambda e: [])
    monkeypatch.setattr(retriever.date_facts, "today_facts", lambda: "Today is a test day.")


def test_generate_response_returns_answer(monkeypatch, make_scored_point, chat_response):
    _patch_common(monkeypatch)
    points = [
        make_scored_point(0.85, chunk_id="c1", text="Standard shipping is 4-7 days."),
        make_scored_point(0.72, chunk_id="c2", text="Express shipping is 1-3 days."),
    ]
    monkeypatch.setattr(retriever.vector_store, "search", lambda vec, top_k=5: points)
    monkeypatch.setattr(retriever.gates, "retrieval_gate", lambda results, has_direct_data=False: True)
    monkeypatch.setattr(retriever.gates, "answer_relevancy_gate", lambda q, a: True)
    monkeypatch.setattr(retriever.llm, "chat", lambda msgs: chat_response("Here are the shipping options.", total_tokens=120))

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
    # Gate 1 lets it through (so we exercise the confidence branch, not the block).
    monkeypatch.setattr(retriever.gates, "retrieval_gate", lambda results, has_direct_data=False: True)
    monkeypatch.setattr(retriever.gates, "answer_relevancy_gate", lambda q, a: True)
    monkeypatch.setattr(retriever.llm, "chat", lambda msgs: chat_response("A weak answer."))

    result = retriever.generate_response("obscure question?", {"session_id": "s", "user_email": ""})

    assert result["confidence"] == 0.2
    assert result["should_escalate"] is True  # 0.2 < confidence_threshold (0.5)


def test_gate1_blocks_before_llm(monkeypatch, make_scored_point):
    _patch_common(monkeypatch)
    points = [make_scored_point(0.1, chunk_id="c1")]
    monkeypatch.setattr(retriever.vector_store, "search", lambda vec, top_k=5: points)
    monkeypatch.setattr(retriever.gates, "retrieval_gate", lambda results, has_direct_data=False: False)
    llm_chat = MagicMock()
    monkeypatch.setattr(retriever.llm, "chat", llm_chat)

    result = retriever.generate_response("nothing relevant?", {"session_id": "s", "user_email": ""})

    llm_chat.assert_not_called()
    assert result["confidence"] == 0.0
    assert result["should_escalate"] is True


# ── Focused gate unit tests ──────────────────────────────────────────────────

def test_answer_relevancy_gate_yes(monkeypatch):
    monkeypatch.setattr(gates.lightweight, "call", lambda *a, **k: "yes, it does")
    assert gates.answer_relevancy_gate("q", "a") is True


def test_answer_relevancy_gate_no(monkeypatch):
    monkeypatch.setattr(gates.lightweight, "call", lambda *a, **k: "no, off topic")
    assert gates.answer_relevancy_gate("q", "a") is False


def test_answer_relevancy_gate_empty_defaults_true(monkeypatch):
    monkeypatch.setattr(gates.lightweight, "call", lambda *a, **k: "")
    assert gates.answer_relevancy_gate("q", "a") is True


def test_retrieval_gate_direct_data_always_passes(make_scored_point):
    # Low score, but direct data present → passes regardless.
    points = [make_scored_point(0.01)]
    assert gates.retrieval_gate(points, has_direct_data=True) is True


def test_retrieval_gate_below_threshold_blocks(make_scored_point):
    points = [make_scored_point(0.1)]  # below default retrieval_min_score (0.35)
    assert gates.retrieval_gate(points, has_direct_data=False) is False


def test_retrieval_gate_no_results_blocks():
    assert gates.retrieval_gate([], has_direct_data=False) is False
