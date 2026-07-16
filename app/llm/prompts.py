# Prompt templates for all pipelines: query rewrite (P2), response generation (P3),
# HyDE query expansion, intent extraction, relevancy gating, multi-intent splitting,
# profile intent detection, and refund verification (P4 vision).

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

A [DATE FACTS] block is provided in the context with precomputed date information. Use ONLY those precomputed values for anything involving timing, delivery dates, days ago/from now, return windows, or deadlines. NEVER calculate dates or day-counts yourself.

NEGATIVE CONSTRAINTS for date reasoning:
  • NEVER perform date arithmetic yourself (e.g., "today is July 15, order placed July 1, so 14 days ago"). Use the [DATE FACTS] block.
  • NEVER guess a weekday or month name. If a user asks "What day was it?" use the precomputed facts.
  • If a timing question is asked and the needed date value is NOT in [DATE FACTS], say "I don't have that date information" and suggest escalation. Do NOT estimate or fabricate.

Use only the provided context to answer the customer's question. Cite the source document for each fact you use. If you cannot answer from the context, say so and suggest escalation.
""".strip()

USER_PROMPT = """
Context:
{context}

Customer question:
{query}

Answer:
""".strip()

HYDE_PROMPT = """You are a customer support assistant. Write a SHORT 2-sentence answer
to the following question, as if you had access to all order and policy information.
Be specific and factual. Do NOT say "I don't know".

Question: {query}

Answer (2 sentences max):""".strip()

INTENT_PROMPT = """
Analyse the customer support query below and respond with ONLY valid JSON — no extra text.

Query: {query}

JSON format:
{{
  "intent": "<one of: return_request | shipping_inquiry | warranty_inquiry | payment_inquiry | account_inquiry | general_policy | other>",
  "entities": {{
    "order_id": "<order id string or null>",
    "invoice_id": "<invoice id string or null>",
    "product_name": "<product name or null>"
  }}
}}
""".strip()

RELEVANCY_CHECK_PROMPT = """You are a strict relevancy judge for a customer support chatbot.

Customer question: {query}

Answer given: {answer}

Does the answer directly address the customer's question?
Reply with ONLY one word: "yes" or "no".
""".strip()

SPLIT_PROMPT = """
Split the customer message below into independent, self-contained requests.
Respond with ONLY a JSON array of strings — no extra text.
Rewrite each part so it stands alone (resolve "it"/"that" if possible).
If the message is really a single request, return a one-element array.

Message: {query}
""".strip()

DETECT_PROMPT = """
You classify a customer message about their own account profile.
Respond with ONLY valid JSON — no extra text.

Message: {query}

JSON format:
{{
  "action": "<view | update | none>",
  "field": "<phone | address | null>",
  "values": {{
    "phone": "<10-digit number, if updating phone, else null>",
    "line1": "<street/flat, if updating address, else null>",
    "line2": "<area/landmark, else null>",
    "city": "<city, else null>",
    "state": "<state, else null>",
    "pincode": "<6-digit pincode, else null>"
  }}
}}

Rules:
- "action": "view" when the user wants to SEE their profile/details/orders.
- "action": "update" when they want to CHANGE phone or address.
- "action": "none" when the message is unrelated to their profile.
- Only fill "values" keys that the user actually provided; use null otherwise.
""".strip()

REFUND_VERIFICATION_PROMPT = """\
You are a strict refund verification agent reviewing a photo a customer submitted to
support a refund claim. Base every judgement on the IMAGE, never on the claim text alone.

STEP 1 — DESCRIBE FIRST (claim-blind). Before reading the claim, describe ONLY what is
objectively visible in the photo (the item, its condition, packaging, whether the product
is present). Do NOT restate the customer's claim as your description.

The customer claims: "{claim}"

STEP 2 — AUTHENTICITY. Decide if the photo is a genuine, unedited photograph of a real
physical item. Mark it NOT authentic if it looks AI-generated, morphed, digitally
manipulated, a screenshot, a stock/marketing image, or a photo of a screen.

STEP 3 — ISSUE TYPE. Classify the visible condition into exactly one of:
   - damaged_packaging: outer box or packaging is crushed, torn, wet, or visibly damaged
   - damaged_product: the product itself is broken, cracked, scratched, or non-functional in appearance
   - wrong_item: the item is clearly different from what was expected (wrong model, color, size)
   - missing_components: the product is present but accessories, parts, or items are visibly absent
   - tampering: seal is broken, package appears previously opened, or signs of interference
   - no_issue: product and packaging appear to be in good condition
   - invalid_image: image is too blurry, dark, unrelated, or cannot be used to assess the product

STEP 4 — CLAIM CHECK. Judge the claim strictly against what you saw in STEP 1:
   - issue_matches_claim: true ONLY if the visible evidence directly supports the claim.
   - contradicts_claim: true if the photo shows the OPPOSITE of the claim. Example: the
     customer claims "laptop is missing, only box received" but the photo clearly shows a
     laptop present inside the box — that is a contradiction.
   - claim_verifiable_from_photo: false when the claim asserts something a single photo
     CANNOT prove — e.g. an item is missing/absent, the box was empty, the item was never
     received, or a quantity that is out of frame. A photo that shows the item present can
     NEVER prove it is "missing" or "not received".

CONFIDENCE RULE: Give confidence > 0.5 ONLY if the photo VISIBLY and DIRECTLY shows the
exact problem claimed. If the photo contradicts the claim, set confidence to 0.0. If the
claim cannot be proven from a photo, set confidence to 0.0.

Respond ONLY with a JSON object in this exact format (no markdown, no explanation):
{{"image_description": "<what is objectively visible, claim-blind>", "authentic": <true|false>, "authenticity_reason": "<short reason>", "issue_type": "<one of the types above>", "issue_matches_claim": <true|false>, "contradicts_claim": <true|false>, "claim_verifiable_from_photo": <true|false>, "match_reason": "<short reason>", "confidence": <0.0 to 1.0>, "description": "<one sentence for the customer>", "evidence": "<what specifically you observed in the image>"}}
"""
