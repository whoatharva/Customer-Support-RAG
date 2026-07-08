# Pipeline 2: Query Input Processing
#
# Responsibilities:
#   - Detect intent from raw query (return_request, warranty_inquiry, etc.)
#   - Extract entities (product names, order numbers, dates)
#   - Rewrite query using conversation history (resolve pronouns, ellipsis)
#   - Validate and sanitize input (prompt injection guard)


def process_query(query: str, session_id: str) -> dict:
    # TODO: implement intent detection, entity extraction, query rewriting
    raise NotImplementedError
