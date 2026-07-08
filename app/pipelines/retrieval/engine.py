# Pipeline 3: Retrieval & Response Generation
#
# Responsibilities:
#   - Embed the rewritten query
#   - Hybrid search (semantic + keyword) over Qdrant
#   - Rerank top-K results with a cross-encoder
#   - Build prompt (invoice facts + retrieved policy chunks + instructions)
#   - Call LLM, extract inline citations
#   - Score confidence, run groundedness check
#   - Return answer + citations + confidence + escalation flag


def generate_response(query: str, context: dict) -> dict:
    # TODO: implement retrieval, reranking, prompt building, LLM call
    raise NotImplementedError
