"""
Part 3 Task 5 (intent node logic): deterministic, rule-based intent
classification that mirrors the few-shot examples in prompts.py. This is
what MOCK_LLM mode actually calls (zero network, zero API keys) -- the same
routing decision a live LLM prompted with prompts.SYSTEM_PROMPT would make.
"""

RETURN_RISK_KEYWORDS = [
    "return risk", "likely to be returned", "will it be returned",
    "order #", "order id", "order number", "risk score", "return probability",
    "risk bucket", "that order",
]
IMAGE_KEYWORDS = [
    "image", "photo", "picture", ".png", ".jpg", "this product photo",
    "classify this", "what category is this",
]


def classify_intent(user_input: str) -> str:
    """Returns one of: 'policy', 'return_risk', 'image_classification'."""
    lowered = user_input.lower()

    for kw in RETURN_RISK_KEYWORDS:
        if kw in lowered:
            return "return_risk"

    for kw in IMAGE_KEYWORDS:
        if kw in lowered:
            return "image_classification"

    return "policy"  # default: policy question, routed to RAG