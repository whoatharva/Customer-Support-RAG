# End-to-end chat workflow: process → retrieve → respond → validate → fallback.
# Wraps services.chat_service with retry/fallback logic for low-confidence answers.

from app.schemas import ChatRequest, ChatResponse


def run(request: ChatRequest) -> ChatResponse:
    # Step 1: process query (intent, entities, rewrite)
    # Step 2: retrieve relevant chunks
    # Step 3: generate response with citations
    # Step 4: check confidence threshold → escalate if below 0.6
    # Step 5: fallback to "I don't know" + escalation flag
    raise NotImplementedError
