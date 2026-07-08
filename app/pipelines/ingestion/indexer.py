import uuid
import hashlib
from datetime import datetime, timezone

from qdrant_client.models import PointStruct

from app import database
from app.vector_store import store as vector_store
from app.pipelines.ingestion.loader import load_documents
from app.pipelines.ingestion.chunker import chunk_document
from app.pipelines.ingestion.embedder import embed_chunks
from app.logger import get_logger

logger = get_logger(__name__)


def _make_point_id(chunk_id: str) -> str:
    return str(uuid.UUID(hashlib.md5(chunk_id.encode()).hexdigest()))


def run_ingestion(folder_path: str) -> dict:
    started_at = datetime.now(timezone.utc)
    run_id = str(uuid.uuid4())
    logger.info("ingestion run started: run_id=%s folder=%s", run_id, folder_path)

    vector_store.ensure_collection()

    raw_docs, errors = load_documents(folder_path)
    processed = skipped = total_chunks = 0

    for doc in raw_docs:
        if not database.is_document_changed(doc.filename, doc.content_hash):
            logger.debug("skipped (unchanged): %s", doc.filename)
            skipped += 1
            continue

        logger.debug("processing: %s", doc.filename)
        try:
            chunks = chunk_document(doc)
            logger.debug("chunked: %s → %d chunks", doc.filename, len(chunks))

            embedded = embed_chunks(chunks)

            points = [
                PointStruct(
                    id=_make_point_id(chunk.chunk_id),
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "doc_filename": chunk.doc_filename,
                        "doc_type": chunk.doc_type,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                )
                for chunk, embedding in embedded
            ]

            vector_store.upsert_points(points)
            database.upsert_document(doc.filename, doc.filepath, doc.content_hash, doc.doc_type, len(chunks), "indexed")
            processed += 1
            total_chunks += len(chunks)

        except Exception as e:
            logger.error("failed to process: %s — %s", doc.filename, e, exc_info=True)
            errors.append({"file": doc.filename, "reason": str(e)})

    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    logger.info(
        "ingestion run completed: run_id=%s processed=%d skipped=%d chunks=%d errors=%d elapsed=%.1fs",
        run_id, processed, skipped, total_chunks, len(errors), elapsed,
    )

    database.save_ingestion_log(run_id, processed, skipped, total_chunks, errors, started_at, datetime.now(timezone.utc))

    return {"run_id": run_id, "files_processed": processed, "files_skipped": skipped, "chunks_created": total_chunks, "errors": errors}
