"""
Retriever: loads the FAISS index built by build_index.py and returns the
top-k most similar chunks (with similarity scores) for a query.
Used by both agent_graph.py (live retrieval) and retrieval_eval.py (Task 10).
"""
import json
import os
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "../index"
INDEX_PATH = os.path.join(INDEX_DIR, "policy_index.faiss")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")

_cache = {}


def _load():
    if "model" not in _cache:
        _cache["model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _cache["index"] = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH) as f:
            _cache["chunks"] = json.load(f)
    return _cache["model"], _cache["index"], _cache["chunks"]


def retrieve(query: str, k: int = 3):
    """Returns a list of up to k dicts: {chunk_id, parent_doc_id, text, score}."""
    model, index, chunks = _load()
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "parent_doc_id": chunk["parent_doc_id"],
            "text": chunk["text"],
            "score": float(score),
        })
    return results


def retrieve_documents(query: str, k: int = 3):
    """Task 10 helper: dedupe retrieved chunks down to unique parent document ids,
    preserving the highest score seen for each document."""
    chunk_results = retrieve(query, k=k)
    doc_scores = {}
    for r in chunk_results:
        doc_id = r["parent_doc_id"]
        if doc_id not in doc_scores or r["score"] > doc_scores[doc_id]:
            doc_scores[doc_id] = r["score"]
    return doc_scores  # {doc_id: best_score}
