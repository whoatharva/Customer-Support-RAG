# Orchestrates a single chat turn: P2 (process) → P3 (retrieve + respond).
# Called by the /chat/query route; returns a ChatResponse.

from app.schemas import ChatRequest, ChatResponse


def handle_chat(request: ChatRequest) -> ChatResponse:
    # TODO: call pipelines.query.processor.process_query
    # TODO: call pipelines.retrieval.engine.generate_response
    # TODO: persist turn to memory.session
    raise NotImplementedError
