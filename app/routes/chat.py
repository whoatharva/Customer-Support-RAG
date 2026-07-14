from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from app.routes.deps import verify_jwt
from app.schemas import ChatRequest, ChatResponse, ChatHistoryResponse, ImageAnalysisResponse
from app.workflow.chat_workflow import run as workflow_run
from app.helpers import database
from app.helpers import session as mem
from app.helpers.logger import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)

_SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/query", response_model=ChatResponse)
def chat(body: ChatRequest, user_email: str = Depends(verify_jwt)):
    """Run the full RAG pipeline (P2 + P3) and return an answer with citations."""
    logger.info("POST /chat/query | session=%s user=%s", body.session_id, user_email)
    return workflow_run(body, user_email=user_email)


@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user_email: str = Depends(verify_jwt),
):
    """Run Pipeline 4: classify a product/package photo for support issues."""
    from app.pipelines.vision.analyzer import analyze_product_image, build_follow_up_message

    mime_type = file.content_type or "image/jpeg"
    if mime_type not in _SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {mime_type}. Use JPEG, PNG, or WebP.",
        )

    logger.info("POST /chat/analyze-image | session=%s user=%s file=%s", session_id, user_email, file.filename)
    image_bytes = await file.read()

    try:
        analysis = analyze_product_image(image_bytes, mime_type)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    follow_up = build_follow_up_message(analysis)
    return ImageAnalysisResponse(
        issue_type=analysis.issue_type,
        confidence=analysis.confidence,
        description=analysis.description,
        evidence=analysis.evidence,
        should_escalate=analysis.should_escalate,
        follow_up_message=follow_up,
    )


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
