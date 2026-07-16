from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from app.routes.deps import verify_jwt
from app.schemas import ChatRequest, ChatResponse, RefundDecisionResponse
from app.workflow.chat_workflow import run as workflow_run
from app.helpers.logger import get_logger

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)

_SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/query", response_model=ChatResponse)
def chat(body: ChatRequest, user_email: str = Depends(verify_jwt)):
    """Run the full RAG pipeline (P2 + P3) and return an answer with citations."""
    logger.info("POST /chat/query | session=%s user=%s", body.session_id, user_email)
    return workflow_run(body, user_email=user_email)


@router.post("/analyze-image", response_model=RefundDecisionResponse)
async def analyze_image(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    claim: str = Form(...),
    user_email: str = Depends(verify_jwt),
):
    """Run Pipeline 4: verify a refund claim from a product/package photo."""
    from app.pipelines.vision.analyzer import process_refund_claim

    mime_type = file.content_type or "image/jpeg"
    if mime_type not in _SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {mime_type}. Use JPEG, PNG, or WebP.",
        )

    logger.info("POST /chat/analyze-image | session=%s user=%s file=%s", session_id, user_email, file.filename)
    image_bytes = await file.read()

    try:
        result = await process_refund_claim(image_bytes, mime_type, claim, user_email, session_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return RefundDecisionResponse(**result)
