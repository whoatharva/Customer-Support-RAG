from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 1800


# ── Ingestion (Pipeline 1) ────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    path: str


class IngestError(BaseModel):
    file: str
    reason: str


class IngestResponse(BaseModel):
    run_id: str
    files_processed: int
    files_skipped: int
    chunks_created: int
    errors: list[IngestError]


class IngestionStatusResponse(BaseModel):
    run_id: str
    files_processed: int
    files_skipped: int
    chunks_created: int
    errors: list[dict]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


# ── Chat (Pipeline 2 + 3) ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    query: str
    invoice_id: Optional[str] = None


class Citation(BaseModel):
    chunk_id: str
    source_document: str
    section: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    should_escalate: bool = False
