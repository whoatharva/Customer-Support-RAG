"""Pipeline 1 — ingestion tests (loader, chunker, run_ingestion).

Fully mocked: no filesystem beyond pytest's tmp_path, no Qdrant, no Supabase,
no embedding calls.
"""

from unittest.mock import MagicMock

from app.pipelines.ingestion.loader import RawDocument, load_documents
from app.pipelines.ingestion.chunker import Chunk, chunk_document
from app.pipelines.ingestion import ingester
from app.config import settings


def test_load_documents(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nAlpha document body with enough text.")
    (tmp_path / "b.md").write_text("# B\n\nBravo document body with enough text.")

    docs, errors = load_documents(str(tmp_path))

    assert errors == []
    assert len(docs) == 2
    assert all(isinstance(d, RawDocument) for d in docs)
    assert all(d.doc_type == "md" for d in docs)
    assert all(d.content_hash for d in docs)
    assert {d.filename for d in docs} == {"a.md", "b.md"}


def test_chunk_document(sample_raw_doc):
    # Force multiple chunks from the sample doc.
    body = "\n\n".join(f"## Section {i}\n\n" + ("word " * 120) for i in range(4))
    doc = RawDocument(
        filepath="/tmp/big.md",
        filename="big.md",
        doc_type="md",
        content="# Big Doc\n\n" + body,
        content_hash="abc123",
    )

    chunks = chunk_document(doc)

    assert chunks, "expected at least one chunk"
    assert all(isinstance(c, Chunk) for c in chunks)
    # No chunk shorter than the configured minimum.
    assert all(len(c.text) >= settings.min_chunk_length for c in chunks)
    # chunk_index is contiguous from 0.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Required metadata carried through from the source doc.
    assert all(c.doc_filename == "big.md" and c.content_hash == "abc123" for c in chunks)


def _fake_doc(filename="shipping.md", content_hash="hash1") -> RawDocument:
    return RawDocument(
        filepath=f"/tmp/{filename}",
        filename=filename,
        doc_type="md",
        content="# T\n\nSome document content here.",
        content_hash=content_hash,
    )


def test_run_ingestion_skips_unchanged(monkeypatch):
    monkeypatch.setattr(ingester.vector_store, "ensure_collection", MagicMock())
    upsert_points = MagicMock()
    monkeypatch.setattr(ingester.vector_store, "upsert_points", upsert_points)
    monkeypatch.setattr(ingester.database, "is_document_changed", lambda *a, **k: False)
    monkeypatch.setattr(ingester.database, "upsert_document", MagicMock())
    monkeypatch.setattr(ingester.database, "save_ingestion_log", MagicMock())
    monkeypatch.setattr(ingester, "load_documents", lambda p: ([_fake_doc()], []))
    embed_chunks = MagicMock()
    monkeypatch.setattr(ingester, "embed_chunks", embed_chunks)

    result = ingester.run_ingestion("x")

    assert result["files_skipped"] == 1
    assert result["files_processed"] == 0
    assert result["chunks_created"] == 0
    embed_chunks.assert_not_called()
    upsert_points.assert_not_called()


def test_run_ingestion_processes_changed(monkeypatch):
    monkeypatch.setattr(ingester.vector_store, "ensure_collection", MagicMock())
    upsert_points = MagicMock()
    monkeypatch.setattr(ingester.vector_store, "upsert_points", upsert_points)
    monkeypatch.setattr(ingester.database, "is_document_changed", lambda *a, **k: True)
    monkeypatch.setattr(ingester.database, "upsert_document", MagicMock())
    monkeypatch.setattr(ingester.database, "save_ingestion_log", MagicMock())
    monkeypatch.setattr(ingester, "load_documents", lambda p: ([_fake_doc()], []))

    fake_chunk = Chunk(
        chunk_id="shipping.md_0", doc_filename="shipping.md", doc_type="md",
        content_hash="hash1", section="General", chunk_index=0, text="chunk text",
    )
    monkeypatch.setattr(ingester, "chunk_document", lambda doc: [fake_chunk])
    monkeypatch.setattr(ingester, "embed_chunks", lambda chunks: [(fake_chunk, [0.1] * 8)])

    result = ingester.run_ingestion("x")

    assert result["files_processed"] == 1
    assert result["files_skipped"] == 0
    assert result["chunks_created"] == 1
    upsert_points.assert_called_once()
