"""
Part 3 Task 8: Guardrails.

- Input-side: block/flag prompt-injection attempts before intent classification.
- Output-side: refuse to answer a policy question if no retrieved chunk clears
  a minimum similarity threshold, instead of letting the mock generator
  fabricate a policy that isn't in the knowledge base.
"""

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all rules",
    "ignore all previous",
    "disregard your instructions",
    "disregard previous instructions",
    "pretend you are",
    "pretend to be",
    "you are now",
    "forget your instructions",
    "act as if you have no rules",
    "system prompt",
    "reveal your prompt",
]

GROUNDEDNESS_SIMILARITY_THRESHOLD = 0.35


def check_input_injection(user_input: str) -> dict:
    """Returns {"blocked": bool, "matched_pattern": str|None}."""
    lowered = user_input.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lowered:
            return {"blocked": True, "matched_pattern": pattern}
    return {"blocked": False, "matched_pattern": None}


def check_groundedness(retrieved_chunks: list) -> dict:
    """
    retrieved_chunks: list of {chunk_id, parent_doc_id, text, score}
    Returns {"grounded": bool, "top_score": float, "threshold": float}
    """
    if not retrieved_chunks:
        return {"grounded": False, "top_score": 0.0, "threshold": GROUNDEDNESS_SIMILARITY_THRESHOLD}

    top_score = max(c["score"] for c in retrieved_chunks)
    grounded = top_score >= GROUNDEDNESS_SIMILARITY_THRESHOLD
    return {"grounded": grounded, "top_score": round(top_score, 4), "threshold": GROUNDEDNESS_SIMILARITY_THRESHOLD}
