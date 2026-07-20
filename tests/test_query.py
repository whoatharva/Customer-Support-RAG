"""Pipeline 2 — query processing tests.

`processor.py` references `database` and `llm` via module alias, so we patch at
the source: `app.helpers.database.get_chat_history_db` and `app.llm.client.chat`.
Langfuse is a no-op shim in tests (real client requires network), so we patch
`get_langfuse` to a MagicMock to keep tracing calls inert.
"""

import json
from unittest.mock import MagicMock

from app.pipelines.query import processor


def _patch_langfuse(monkeypatch):
    monkeypatch.setattr(processor, "get_langfuse", lambda: MagicMock())


def test_process_query_returns_dict(monkeypatch, chat_response):
    _patch_langfuse(monkeypatch)
    monkeypatch.setattr(processor.database, "get_chat_history_db", lambda *a, **k: [])
    combined_json = json.dumps({
        "rewritten_query": "What are shipping options?",
        "intent": "shipping_inquiry",
        "entities": {"order_id": None, "product_name": None},
    })
    monkeypatch.setattr(processor.llm, "chat", lambda *a, **k: chat_response(combined_json))

    result = processor.process_query("What are shipping options?", "sess1")

    assert set(result) == {"original_query", "rewritten_query", "intent", "entities"}
    assert result["rewritten_query"] == result["original_query"] == "What are shipping options?"
    assert result["intent"] == "shipping_inquiry"
    assert result["entities"] == {"order_id": None, "product_name": None}


def test_process_query_rewrites_pronoun(monkeypatch, chat_response):
    _patch_langfuse(monkeypatch)
    history = [
        {"role": "user", "content": "I ordered a laptop last week."},
        {"role": "assistant", "content": "Your laptop order is confirmed."},
    ]
    monkeypatch.setattr(processor.database, "get_chat_history_db", lambda *a, **k: history)

    combined = chat_response(json.dumps({
        "rewritten_query": "Where is my laptop shipment?",
        "intent": "shipping_inquiry",
        "entities": {},
    }))
    chat = MagicMock(return_value=combined)
    monkeypatch.setattr(processor.llm, "chat", chat)

    result = processor.process_query("Where is it?", "sess2")

    assert result["original_query"] == "Where is it?"
    assert result["rewritten_query"] == "Where is my laptop shipment?"
    assert "laptop" in result["rewritten_query"]
    assert chat.call_count == 1  # rewrite + intent combined into one call


def test_process_query_bad_json_defaults(monkeypatch, chat_response):
    _patch_langfuse(monkeypatch)
    monkeypatch.setattr(processor.database, "get_chat_history_db", lambda *a, **k: [])
    monkeypatch.setattr(processor.llm, "chat", lambda *a, **k: chat_response("not valid json at all"))

    result = processor.process_query("Random query", "sess3")

    assert result["rewritten_query"] == "Random query"
    assert result["intent"] == "other"
    assert result["entities"] == {}
