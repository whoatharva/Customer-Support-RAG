from fastapi import APIRouter, Depends
from app.routes.deps import verify_jwt
from app.schemas import ChatRequest, ChatResponse, ChatHistoryResponse
from app.workflow.chat_workflow import run as workflow_run
from app.helpers import database
from app.helpers import session as mem
from app.helpers.logger import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


@router.post("/query", response_model=ChatResponse)
def chat(body: ChatRequest, user_email: str = Depends(verify_jwt)):
    """Run the full RAG pipeline (P2 + P3) and return an answer with citations."""
    logger.info("POST /chat/query | session=%s user=%s", body.session_id, user_email)
    return workflow_run(body, user_email=user_email)


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
def get_history(session_id: str, _: str = Depends(verify_jwt)):
    """Return the full message history for a session from Supabase."""
    messages = database.get_chat_history_db(session_id)
    return ChatHistoryResponse(session_id=session_id, messages=messages)


@router.delete("/session/{session_id}")
def clear_session(session_id: str, _: str = Depends(verify_jwt)):
    """Delete all messages for a session from Supabase and in-memory store."""
    database.delete_chat_session(session_id)
    mem.clear(session_id)
    logger.info("session cleared: %s", session_id)
    return {"status": "cleared", "session_id": session_id}
