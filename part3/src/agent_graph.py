"""
Part 3 Task 5: LangGraph agent graph.

4 nodes: intent -> (rag_retrieval | tool_calling) -> response_generation
1 conditional edge: routes by intent (and by the input guardrail's block flag).

Short-term conversational state (Task 5 multi-turn requirement) is carried in
state["session"], a dict that the CALLER threads between invoke() calls
within one conversation. A fresh conversation starts with session={}, so
"last_order_features" / "last_image_path" are correctly absent until the
first turn that provides them -- see run_agent.py for both the multi-turn
transcript and the fresh-conversation transcript.
"""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from guardrails import check_input_injection, check_groundedness
from intent_classifier import classify_intent
from retriever import retrieve
from check_return_risk_tool import check_return_risk
from classify_product_image_tool import classify_product_image
from mock_llm import (
    generate_blocked_response,
    generate_ungrounded_refusal,
    generate_policy_response,
    generate_return_risk_response,
    generate_image_classification_response,
    generate_missing_context_response,
)


class AgentState(TypedDict):
    user_input: str
    intent: Optional[str]
    blocked: bool
    block_reason: Optional[str]
    retrieved_chunks: List[Dict]
    groundedness: Optional[Dict]
    tool_output: Optional[Dict]
    missing_context: Optional[str]  # 'order' | 'image' | None
    order_features: Optional[Dict]  # explicit per-turn input, if provided
    image_path: Optional[str]       # explicit per-turn input, if provided
    final_answer: Optional[Dict]
    session: Dict[str, Any]         # persisted short-term state across turns


# --- Node 1: intent (also runs the input-side guardrail) ---
def intent_node(state: AgentState) -> dict:
    injection_check = check_input_injection(state["user_input"])
    if injection_check["blocked"]:
        return {"blocked": True, "block_reason": injection_check["matched_pattern"]}

    intent = classify_intent(state["user_input"])
    return {"blocked": False, "block_reason": None, "intent": intent}


# --- Node 2: RAG retrieval (policy intent only) ---
def rag_retrieval_node(state: AgentState) -> dict:
    chunks = retrieve(state["user_input"], k=3)
    groundedness = check_groundedness(chunks)
    return {"retrieved_chunks": chunks, "groundedness": groundedness}


# --- Node 3: tool calling (return_risk / image_classification intent) ---
def tool_calling_node(state: AgentState) -> dict:
    session = state.get("session", {})

    if state["intent"] == "return_risk":
        # Use this turn's explicit order_features if given, else fall back to
        # the last order mentioned earlier in this conversation (multi-turn state).
        order_features = state.get("order_features") or session.get("last_order_features")
        if order_features is None:
            return {"tool_output": None, "missing_context": "order", "session": session}
        result = check_return_risk(order_features)
        session["last_order_features"] = order_features  # persist for follow-ups
        return {"tool_output": result, "missing_context": None, "session": session}

    elif state["intent"] == "image_classification":
        image_path = state.get("image_path") or session.get("last_image_path")
        if image_path is None:
            return {"tool_output": None, "missing_context": "image", "session": session}
        result = classify_product_image(image_path)
        session["last_image_path"] = image_path
        return {"tool_output": result, "missing_context": None, "session": session}

    return {}


# --- Node 4: response generation ---
def response_generation_node(state: AgentState) -> dict:
    if state.get("blocked"):
        return {"final_answer": generate_blocked_response(state["block_reason"])}

    if state["intent"] == "policy":
        if not state["groundedness"]["grounded"]:
            return {"final_answer": generate_ungrounded_refusal(
                state["groundedness"]["top_score"], state["groundedness"]["threshold"]
            )}
        return {"final_answer": generate_policy_response(state["retrieved_chunks"])}

    elif state["intent"] == "return_risk":
        if state.get("missing_context") == "order":
            return {"final_answer": generate_missing_context_response("order")}
        return {"final_answer": generate_return_risk_response(state["tool_output"])}

    elif state["intent"] == "image_classification":
        if state.get("missing_context") == "image":
            return {"final_answer": generate_missing_context_response("image")}
        return {"final_answer": generate_image_classification_response(state["tool_output"])}

    return {"final_answer": {"answer": "I couldn't determine how to help with that.",
                              "source": "policy_kb", "confidence": 0.0}}


# --- Conditional edge routing ---
def route_after_intent(state: AgentState) -> str:
    if state.get("blocked"):
        return "response_generation"
    if state["intent"] == "policy":
        return "rag_retrieval"
    return "tool_calling"  # return_risk or image_classification


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("intent", intent_node)
    graph.add_node("rag_retrieval", rag_retrieval_node)
    graph.add_node("tool_calling", tool_calling_node)
    graph.add_node("response_generation", response_generation_node)

    graph.set_entry_point("intent")

    graph.add_conditional_edges("intent", route_after_intent, {
        "rag_retrieval": "rag_retrieval",
        "tool_calling": "tool_calling",
        "response_generation": "response_generation",
    })

    graph.add_edge("rag_retrieval", "response_generation")
    graph.add_edge("tool_calling", "response_generation")
    graph.add_edge("response_generation", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "user_input": "What is the return window for electronics?",
        "session": {},
    })
    print(result["final_answer"])