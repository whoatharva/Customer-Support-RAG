from fastapi import APIRouter, Depends
from app.api.dependencies import verify_jwt
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
def chat(body: ChatRequest, _: str = Depends(verify_jwt)):
    # TODO: call pipeline 2 (process_query) then pipeline 3 (generate_response)
    raise NotImplementedError
