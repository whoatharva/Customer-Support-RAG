# Prompt templates for Pipeline 2 (query rewrite) and Pipeline 3 (response generation).

SUPPORT_CONTACT = "support@ourstore.com or use the live chat (Mon–Sat, 9am–8pm IST)"

QUERY_REWRITE_TEMPLATE = """
Given the conversation history and the latest user message, rewrite the user's question
as a fully self-contained query suitable for semantic search. Resolve pronouns and ellipsis.

Conversation history:
{history}

User message:
{query}

Rewritten query:
""".strip()

RESPONSE_TEMPLATE = """
You are a helpful customer support assistant. Use only the provided context to answer
the customer's question. Cite the source document and section for each fact you use.
If you cannot answer from the context, say so and suggest escalation.

Context:
{context}

Customer question:
{query}

Answer:
""".strip()
