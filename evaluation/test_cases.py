"""Golden test set for RAG evaluation.

Each case is a question a real customer might ask, paired with a `ground_truth`
answer derived from the FAQ documents in `data/documents/`. The eval runs each
`question` through the live pipeline (see `ragas_eval.py`) to obtain the actual
answer + retrieved contexts, then scores them against `ground_truth`.

Fields:
  id            unique slug (also used as the per-case session_id, so each case
                gets a clean conversation with no history bleed)
  question      the user query sent to the pipeline
  ground_truth  reference answer, written from the source FAQ
  category      source doc / topic (account | payment | returns | shipping |
                warranties | out_of_scope) — usable with --categories

`out_of_scope` cases have no supporting document. A well-behaved RAG system
should decline or escalate rather than fabricate an answer; these cases exist to
check that faithfulness/answer_correctness stay LOW (i.e. the system isn't
hallucinating confident answers to things it can't know).

To grow the set: copy a dict, keep `id` unique, and base `ground_truth` on the
actual FAQ text so context_recall / answer_correctness stay meaningful.
"""

TEST_CASES = [
    # ── Shipping ──────────────────────────────────────────────────────────────
    {
        "id": "shipping_options",
        "question": "What shipping options are available and when do I get free shipping?",
        "ground_truth": (
            "Two main options: Standard Shipping (4–7 business days, ₹49–₹99, free on "
            "orders above ₹499) and Express Shipping (1–3 business days, ₹99–₹199, "
            "available in metros and Tier-1 cities). Free shipping applies automatically "
            "on standard orders over ₹499."
        ),
        "category": "shipping",
    },
    {
        "id": "shipping_delivered_not_received",
        "question": "My tracking says delivered but I never got my package. What should I do?",
        "ground_truth": (
            "Check your surroundings and with neighbours, family, or building reception, "
            "since agents sometimes leave packages at the door or lobby. Wait up to 24 "
            "hours, then contact the courier using your tracking number. If still "
            "unresolved after 24–48 hours, contact our support with your order number; "
            "we review the proof of delivery, raise a dispute with the courier, and issue "
            "a replacement or refund if the package can't be located within 5 business days."
        ),
        "category": "shipping",
    },
    {
        "id": "shipping_change_address",
        "question": "Can I change my delivery address after I've placed an order?",
        "ground_truth": (
            "You can change the delivery address only before the order is dispatched/shipped. "
            "Update it from your order details or by contacting support quickly; once the "
            "package is handed to the courier the address usually cannot be changed."
        ),
        "category": "shipping",
    },

    # ── Returns ───────────────────────────────────────────────────────────────
    {
        "id": "returns_policy",
        "question": "What is your return policy?",
        "ground_truth": (
            "Most items can be returned within a 30-day window from the date of delivery. "
            "The item must be unused, unworn, and in original condition with all original "
            "packaging, tags, labels, and accessories, must not be in a non-returnable "
            "category, and needs a valid order number or proof of purchase. Accepted "
            "reasons include defective/damaged items, wrong item, item not matching the "
            "description, size/fit issues, or change of mind subject to condition. A refund "
            "or exchange is processed after inspection and approval."
        ),
        "category": "returns",
    },
    {
        "id": "returns_non_returnable",
        "question": "Which items cannot be returned?",
        "ground_truth": (
            "Non-returnable items include perishable goods, customized or personalized "
            "items, activated digital products, intimate wear and swimwear, hazardous "
            "materials, opened consumables like cosmetics and supplements, opened "
            "mattresses and bedding, and anything marked Final Sale or Non-Returnable. "
            "If such an item arrives damaged or defective, contact support with photos "
            "within 48 hours of delivery for a possible exception."
        ),
        "category": "returns",
    },
    {
        "id": "returns_initiate",
        "question": "How do I start a return for something I ordered?",
        "ground_truth": (
            "Initiate the return from your account: go to your orders, select the item, "
            "choose a return reason, and submit the request. Ensure the item is within the "
            "return window and meets the condition requirements, then follow the pickup or "
            "drop-off instructions provided. A refund or exchange follows inspection."
        ),
        "category": "returns",
    },

    # ── Payment ───────────────────────────────────────────────────────────────
    {
        "id": "payment_money_deducted_failed",
        "question": "My payment failed but money was deducted from my account. What happens now?",
        "ground_truth": (
            "This is usually a network/timeout issue where the bank processed the deduction "
            "but our system didn't get confirmation. Within 24–48 hours the payment gateway "
            "reconciles and either confirms the order or marks it failed and refunds you. If "
            "the order isn't confirmed within 48 hours, the amount is auto-refunded to your "
            "original payment method within 5–7 business days. You can also contact support "
            "with your order ID. Do not pay again immediately, or you may be double-charged."
        ),
        "category": "payment",
    },
    {
        "id": "payment_no_cost_emi",
        "question": "What is No-Cost EMI? Is there really no extra charge?",
        "ground_truth": (
            "No-Cost EMI means you pay the product price spread across installments with no "
            "interest on top — the interest is discounted from the price upfront so the total "
            "equals the original price. Your card statement may still show a bank interest "
            "charge, but it's offset by the upfront discount. Some banks charge a one-time "
            "processing fee (around ₹99–₹299), and availability varies by bank, card, and product."
        ),
        "category": "payment",
    },
    {
        "id": "payment_methods",
        "question": "What payment methods do you accept?",
        "ground_truth": (
            "We accept credit and debit cards (Visa, Mastercard, American Express, Rupay, "
            "and prepaid cards), UPI apps such as Google Pay, PhonePe, Paytm, and BHIM, "
            "net banking, EMI, wallets, and Cash on Delivery where available. "
            "Cryptocurrency is not accepted."
        ),
        "category": "payment",
    },

    # ── Warranties ────────────────────────────────────────────────────────────
    {
        "id": "warranty_accidental_damage",
        "question": "Does the warranty cover accidental damage like drops or spills?",
        "ground_truth": (
            "Standard manufacturer warranties do NOT cover accidental damage such as drops, "
            "spills, or cracks. For that coverage you need our Extended Warranty with "
            "Accidental Damage Protection (bought at purchase or within 30 days), a "
            "manufacturer's own protection plan (e.g. AppleCare+), or third-party gadget/home "
            "insurance. ADP is available at checkout for eligible categories."
        ),
        "category": "warranties",
    },
    {
        "id": "warranty_lost_card",
        "question": "I lost my warranty card. Can I still claim warranty?",
        "ground_truth": (
            "Yes. A physical warranty card is usually not mandatory — your proof of purchase "
            "(invoice or order details) generally serves as warranty proof. Keep your invoice "
            "and product/serial details, and you can typically still file a warranty claim "
            "through the manufacturer's authorized service center."
        ),
        "category": "warranties",
    },
    {
        "id": "warranty_platform_vs_manufacturer",
        "question": "Does your platform provide its own warranty or only the manufacturer's?",
        "ground_truth": (
            "Products are covered primarily by the manufacturer's warranty, which is a "
            "contract between you and the maker and serviced through their authorized service "
            "centers. The platform itself doesn't provide a separate default warranty, but it "
            "offers optional paid Extended Warranty / Accidental Damage Protection plans you "
            "can buy for eligible products."
        ),
        "category": "warranties",
    },

    # ── Account ───────────────────────────────────────────────────────────────
    {
        "id": "account_forgot_password",
        "question": "I forgot my password. How do I reset it?",
        "ground_truth": (
            "On the Login page click 'Forgot Password?', enter your registered email, and "
            "click 'Send Reset Link'. Open the reset email (check spam) and use the link, "
            "which is valid for 60 minutes, to set and confirm a new password, then log in. "
            "If you've lost access to your registered email, contact support with proof of "
            "identity to regain access."
        ),
        "category": "account",
    },
    {
        "id": "account_password_requirements",
        "question": "What are the password requirements when creating an account?",
        "ground_truth": (
            "A password must be at least 8 characters and include at least one uppercase "
            "letter, one lowercase letter, one number, and one special character, and it "
            "cannot match any of your last 3 passwords. Using a passphrase or password "
            "manager is recommended."
        ),
        "category": "account",
    },

    # ── Out of scope (should decline / escalate, NOT hallucinate) ─────────────
    {
        "id": "oos_weather",
        "question": "What's the weather forecast in Mumbai this weekend?",
        "ground_truth": (
            "This is outside the scope of customer support for this store. The assistant "
            "should say it cannot help with weather forecasts rather than inventing an answer."
        ),
        "category": "out_of_scope",
    },
    {
        "id": "oos_competitor_price",
        "question": "Is the same TV cheaper on a competitor's website right now?",
        "ground_truth": (
            "The assistant has no knowledge of competitor pricing and should decline or "
            "suggest checking the competitor directly, rather than fabricating a price "
            "comparison."
        ),
        "category": "out_of_scope",
    },
]
