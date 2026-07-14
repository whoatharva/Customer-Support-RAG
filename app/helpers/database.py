"""Supabase client and all database operations for the Customer Support RAG system.

Tables managed here:
  users            — auth credentials (email, password_hash, is_active)
  chat_sessions    — one row per conversation (session_id → user_email)
  chat_messages    — full chat history (role, content, created_at)
  documents        — ingestion tracking (filename, content_hash, chunk_count)
  ingestion_logs   — pipeline run history (processed, skipped, errors)
  customers        — customer profiles (loyalty, subscription, addresses)
  products         — product catalog (warranty, pricing, stock)
  orders           — order history with items/payment/logistics as JSONB

"""

import os
from datetime import datetime, timezone
from supabase import create_client, Client
from app.config import settings
from app.helpers.logger import get_logger

# certifi's bundle is missing the CA that signs Supabase's cert on this machine;
# the system bundle has it, so point all httpx/ssl calls there.
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")

logger = get_logger(__name__)
_client: Client | None = None


def get_db() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_key)
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
            "updated_at": datetime.now(timezone.utc).isoformat(),
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


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(name: str, email: str, password_hash: str) -> dict:
    result = get_db().table("users").insert({
        "name": name,
        "email": email,
        "password_hash": password_hash,
    }).execute()
    logger.info("user created: %s", email)
    return result.data[0]


def get_user_by_email(email: str) -> dict | None:
    result = get_db().table("users").select("*").eq("email", email).eq("is_active", True).execute()
    return result.data[0] if result.data else None


# ── Chat history ──────────────────────────────────────────────────────────────

def ensure_chat_session(session_id: str, user_email: str):
    """Create session row if it doesn't exist; update updated_at if it does."""
    get_db().table("chat_sessions").upsert(
        {"id": session_id, "user_email": user_email, "updated_at": datetime.now(timezone.utc).isoformat()},
        on_conflict="id",
    ).execute()
    logger.debug("chat session ensured: %s", session_id)


def save_chat_message(session_id: str, role: str, content: str):
    get_db().table("chat_messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
    }).execute()
    logger.debug("message saved: session=%s role=%s", session_id, role)


def get_chat_history_db(session_id: str, limit: int = 50) -> list[dict]:
    result = (
        get_db()
        .table("chat_messages")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data


def delete_chat_session(session_id: str):
    # chat_messages rows cascade-delete via FK
    get_db().table("chat_sessions").delete().eq("id", session_id).execute()
    logger.info("chat session deleted: %s", session_id)
