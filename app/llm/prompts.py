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

SYSTEM_PROMPT = """
You are a helpful customer support assistant. Today's date is {today}.
When documents in the context use the word "today", they refer to the date that document was written — not the current date. Always treat {today} as the current date when answering questions about timing, delivery expectations, or whether deadlines have passed.
Use only the provided context to answer the customer's question. Cite the source document for each fact you use. If you cannot answer from the context, say so and suggest escalation.
""".strip()

USER_PROMPT = """
Context:
{context}

Customer question:
{query}

Answer:
""".strip()

# Keep for backwards-compat with anything that still imports RESPONSE_TEMPLATE
RESPONSE_TEMPLATE = SYSTEM_PROMPT + "\n\n" + USER_PROMPT
