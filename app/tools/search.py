# LLM function-calling tool definition for knowledge base search.
# The LLM in Pipeline 3 may call this tool to fetch relevant chunks.

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the customer support knowledge base for relevant policy or FAQ chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query derived from the customer question."},
                "top_k": {"type": "integer", "description": "Number of chunks to return.", "default": 5},
            },
            "required": ["query"],
        },
    },
}


def search_knowledge_base(query: str, top_k: int = 5) -> list[dict]:
    # TODO: embed query via llm.client.embed, then call vector_store.store.search
    raise NotImplementedError
