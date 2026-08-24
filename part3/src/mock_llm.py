"""
Part 3 Task 7: MOCK_LLM deterministic mode.

Rule-based/template function that composes the final structured JSON answer
from retrieved KB chunk(s) and/or tool output -- zero network calls, zero API
keys. This is the DEFAULT mode and what every graded transcript must run
against (per the brief, USE_LIVE_LLM is optional and never scored).
"""


def generate_blocked_response(matched_pattern: str) -> dict:
    return {
        "answer": (
            "I can't comply with that request -- it looks like an attempt to override "
            "my instructions, which I'm not able to do. I'm happy to help with a genuine "
            "policy, return-risk, or product-category question instead."
        ),
        "source": "policy_kb",
        "confidence": 0.0,
    }


def generate_ungrounded_refusal(top_score: float, threshold: float) -> dict:
    return {
        "answer": (
            f"I don't have a policy document confident enough to answer that "
            f"(best match similarity {top_score:.4f} is below my {threshold:.2f} "
            f"grounding threshold), so rather than guess, I'd recommend checking "
            f"with a human support agent for this specific question."
        ),
        "source": "policy_kb",
        "confidence": round(top_score, 4),
    }


def generate_policy_response(retrieved_chunks: list) -> dict:
    """Paraphrase-composes an answer from the top retrieved chunk(s)."""
    top = sorted(retrieved_chunks, key=lambda c: c["score"], reverse=True)[:2]
    combined_text = " ".join(c["text"] for c in top)
    answer = f"Based on our policy documentation: {combined_text}"
    avg_conf = sum(c["score"] for c in top) / len(top)
    return {
        "answer": answer,
        "source": "policy_kb",
        "confidence": round(avg_conf, 4),
    }


def generate_return_risk_response(tool_output: dict) -> dict:
    prob = tool_output["return_probability"]
    bucket = tool_output["risk_bucket"]
    answer = (
        f"This order has an estimated return probability of {prob:.2%}, "
        f"placing it in the '{bucket}' risk bucket (cut points: {tool_output['cut_points']})."
    )
    return {
        "answer": answer,
        "source": "return_risk_tool",
        "confidence": round(prob, 4),
    }

def generate_image_classification_response(tool_output: dict) -> dict:
    category = tool_output["predicted_category"]
    confidence = tool_output["confidence"]
    answer = f"This product image is classified as '{category}' with {confidence:.2%} confidence."
    return {
        "answer": answer,
        "source": "image_classifier_tool",
        "confidence": round(confidence, 4),
    }

def generate_missing_context_response(kind: str) -> dict:
    """kind: 'order' or 'image' -- used when session has no prior context to reuse."""
    if kind == "order":
        answer = ("I don't have an order to reference yet in this conversation -- "
                   "could you share the order ID or its details so I can check the return risk?")
    else:
        answer = ("I don't have a product image to reference yet in this conversation -- "
                   "could you point me to the image file you'd like classified?")
    return {"answer": answer, "source": "policy_kb", "confidence": 0.0}
    category = tool_output["predicted_category"]
    confidence = tool_output["confidence"]
    answer = f"This product image is classified as '{category}' with {confidence:.2%} confidence."
    return {
        "answer": answer,
        "source": "image_classifier_tool",
        "confidence": round(confidence, 4),
    }