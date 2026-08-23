"""
Part 3 Task 1: Flipkart-style policy knowledge base.

14 short (2-4 sentence) documents covering, at minimum: return windows by
category, COD refund timelines, delivery SLAs, and reverse-pickup eligibility.

Each document has a stable `doc_id` -- build_index.py chunks these documents
SENTENCE-WISE (each sentence becomes its own chunk), but every chunk keeps a
`parent_doc_id` pointer back here, since Task 10's Precision@3/Recall@3
scoring is done at the DOCUMENT level, not the chunk level.
"""

POLICY_DOCS = [
    {
        "doc_id": "return_apparel_footwear",
        "category": "returns",
        "text": (
            "Apparel and footwear items can be returned within 30 days of delivery. "
            "The item must be unworn, unwashed, and have all original tags attached. "
            "Footwear must include the original shoebox in undamaged condition."
        ),
    },
    {
        "doc_id": "return_electronics",
        "category": "returns",
        "text": (
            "Electronics items have a 10-day return window from the date of delivery. "
            "The product must be in its original packaging with all accessories, manuals, "
            "and the invoice included. Physical damage or missing accessories void return eligibility."
        ),
    },
    {
        "doc_id": "return_home",
        "category": "returns",
        "text": (
            "Home and furniture items can be returned within 15 days of delivery. "
            "Large furniture returns require the item to be repackaged in its original crate or box. "
            "A reverse-pickup fee may apply for items over 20 kg in weight."
        ),
    },
    {
        "doc_id": "return_beauty",
        "category": "returns",
        "text": (
            "Beauty and personal care products are non-returnable once the seal is broken, "
            "for hygiene reasons. Unopened, sealed beauty products can be returned within 7 days of delivery. "
            "Damaged-on-arrival items are always eligible regardless of seal status."
        ),
    },
    {
        "doc_id": "cod_refund_timeline",
        "category": "refunds",
        "text": (
            "For Cash on Delivery orders, refunds are issued to the customer's bank account via NEFT, "
            "since no original payment method exists to refund back to. COD refunds typically take "
            "7-10 business days to reflect after the returned item passes quality inspection."
        ),
    },
    {
        "doc_id": "prepaid_refund_timeline",
        "category": "refunds",
        "text": (
            "For prepaid orders (card, UPI, or wallet), refunds are credited back to the original "
            "payment method within 3-5 business days of the return being approved. "
            "Wallet refunds are typically the fastest, often reflecting within 24 hours."
        ),
    },
    {
        "doc_id": "refund_inspection_process",
        "category": "refunds",
        "text": (
            "All returned items undergo a quality inspection at the returns warehouse before a refund "
            "is initiated. Items that fail inspection (e.g. signs of use, missing parts) are shipped "
            "back to the customer instead of being refunded, with an explanation email sent."
        ),
    },
    {
        "doc_id": "delivery_sla_metro",
        "category": "delivery",
        "text": (
            "For metro cities, standard delivery SLA is 2-4 business days from order confirmation. "
            "Express delivery, where available, guarantees next-day delivery for an additional fee. "
            "Delivery delays beyond the SLA automatically trigger a customer notification with a revised ETA."
        ),
    },
    {
        "doc_id": "delivery_sla_non_metro",
        "category": "delivery",
        "text": (
            "For non-metro and rural pin codes, standard delivery SLA is 5-9 business days. "
            "Express delivery is not available for most non-metro pin codes due to logistics network limits. "
            "Customers can track real-time delivery status via the order tracking page."
        ),
    },
    {
        "doc_id": "delivery_delay_compensation",
        "category": "delivery",
        "text": (
            "If a delivery is delayed by more than 5 days beyond the promised SLA, the customer becomes "
            "eligible for a delivery-delay coupon worth 10% of the order value. "
            "This coupon is auto-applied to the customer's account and does not require a support request."
        ),
    },
    {
        "doc_id": "reverse_pickup_eligibility",
        "category": "reverse_pickup",
        "text": (
            "Reverse pickup (a courier collecting the returned item from the customer's address) is "
            "available in all serviceable pin codes for items above 500 grams. "
            "For lightweight items or non-serviceable pin codes, customers must self-ship the return "
            "and are reimbursed shipping costs up to 100 rupees upon approval."
        ),
    },
    {
        "doc_id": "reverse_pickup_scheduling",
        "category": "reverse_pickup",
        "text": (
            "Once a return is approved, reverse pickup is scheduled within 2 business days. "
            "Customers receive an SMS and app notification with the pickup date and a 4-hour time window. "
            "A pickup can be rescheduled up to twice before the return request is automatically cancelled."
        ),
    },
    {
        "doc_id": "cancellation_policy",
        "category": "cancellation",
        "text": (
            "Orders can be cancelled free of charge any time before they are shipped. "
            "Once an order has shipped, it cannot be cancelled but can be refused at the doorstep or "
            "returned through the standard return process after delivery."
        ),
    },
    {
        "doc_id": "exchange_policy",
        "category": "exchange",
        "text": (
            "Size and color exchanges for apparel and footwear are allowed within the same 30-day "
            "return window, subject to stock availability of the requested size or color. "
            "Exchanges do not require a refund-and-repurchase; the new item ships once the original is picked up."
        ),
    },
]

# --- Task 1: retrieval-evaluation answer key ---
# For each test query, the document(s) considered "relevant" (document-level,
# not chunk-level). Used by retrieval_eval.py for Task 10's Precision@3/Recall@3.
QUERY_ANSWER_KEY = [
    {
        "query": "How many days do I have to return a pair of shoes?",
        "relevant_doc_ids": ["return_apparel_footwear"],
    },
    {
        "query": "When will I get my refund if I paid cash on delivery?",
        "relevant_doc_ids": ["cod_refund_timeline"],
    },
    {
        "query": "How long does delivery take outside major cities?",
        "relevant_doc_ids": ["delivery_sla_non_metro"],
    },
    {
        "query": "Can someone come pick up my return from my house?",
        "relevant_doc_ids": ["reverse_pickup_eligibility", "reverse_pickup_scheduling"],
    },
    {
        "query": "What's the return policy for a laptop I bought?",
        "relevant_doc_ids": ["return_electronics"],
    },
    {
        "query": "Can I cancel my order after it's already shipped?",
        "relevant_doc_ids": ["cancellation_policy"],
    },
    {
        "query": "I want a different size of the shirt I ordered, is that possible?",
        "relevant_doc_ids": ["exchange_policy"],
    },
]
