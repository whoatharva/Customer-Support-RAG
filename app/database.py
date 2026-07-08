import httpx
from datetime import datetime
from supabase import create_client, Client, ClientOptions
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)
_client: Client | None = None


def get_db() -> Client:
    global _client
    if _client is None:
        _client = create_client(
            settings.supabase_url,
            settings.supabase_key,
            options=ClientOptions(httpx_client=httpx.Client(verify=False)),
        )
        logger.info("Supabase client initialized")
    return _client


# ── Documents ─────────────────────────────────────────────────────────────────

def is_document_changed(filename: str, content_hash: str) -> bool:
    result = get_db().table("documents").select("content_hash").eq("filename", filename).execute()
    if not result.data:
        logger.debug("document not found in DB, treating as new: %s", filename)
        return True
    changed = result.data[0]["content_hash"] != content_hash
    logger.debug("hash check for %s: %s", filename, "changed" if changed else "unchanged")
    return changed


def upsert_document(filename: str, filepath: str, content_hash: str, doc_type: str, chunk_count: int, status: str):
    try:
        get_db().table("documents").upsert({
            "filename": filename,
            "filepath": filepath,
            "content_hash": content_hash,
            "doc_type": doc_type,
            "chunk_count": chunk_count,
            "status": status,
            "updated_at": datetime.utcnow().isoformat(),
        }, on_conflict="filename").execute()
    except Exception:
        logger.error("failed to upsert document: %s", filename, exc_info=True)
        raise


# ── Ingestion logs ────────────────────────────────────────────────────────────

def save_ingestion_log(run_id: str, processed: int, skipped: int, chunks: int, errors: list, started_at: datetime, completed_at: datetime):
    try:
        get_db().table("ingestion_logs").insert({
            "run_id": run_id,
            "files_processed": processed,
            "files_skipped": skipped,
            "chunks_created": chunks,
            "errors": errors,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
        }).execute()
    except Exception:
        logger.error("failed to save ingestion log: run_id=%s", run_id, exc_info=True)
        raise


def get_ingestion_logs(limit: int = 20) -> list:
    return get_db().table("ingestion_logs").select("*").order("started_at", desc=True).limit(limit).execute().data
