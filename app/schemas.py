from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from app.config import settings


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = settings.jwt_expiry_minutes * 60


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


class Citation(BaseModel):
    chunk_id: str
    source_document: str
    section: str
    text: str
    score: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    should_escalate: bool = False
    total_tokens: int | None = None
    # Pending client-side action (e.g. a profile update awaiting Confirm/Cancel).
    action: str | None = None
    action_payload: dict | None = None


# ── Profile (self-service) ────────────────────────────────────────────────────

class UpdateProfileRequest(BaseModel):
    field: str            # "phone" | "address"
    values: dict          # {"phone": ...} or {"line1": ..., "city": ..., ...}


class ProfileResponse(BaseModel):
    phone: Optional[str] = None
    address: Optional[dict] = None
    orders: list[dict] = []


# ── Auth (extended) ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


# ── Chat history ──────────────────────────────────────────────────────────────

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]


# ── Vision (Pipeline 4) ───────────────────────────────────────────────────────

class ImageAnalysisResponse(BaseModel):
    issue_type: str
    confidence: float
    description: str
    evidence: str
    should_escalate: bool
    follow_up_message: str
