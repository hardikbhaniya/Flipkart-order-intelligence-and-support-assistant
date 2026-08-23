"""
Part 3 Task 2: Chunk every policy document SENTENCE-WISE, embed each chunk
with a free local sentence-transformer, and build a FAISS index over them.

Sentence-wise chunking (over fixed-size/overlapping windows) is used because
each policy document is short and each sentence already carries one complete
policy fact -- splitting by sentence keeps each retrievable unit maximally
specific without fragmenting a single fact across two chunks.
"""
import json
import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from knowledge_base import POLICY_DOCS

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "../index"
INDEX_PATH = os.path.join(INDEX_DIR, "policy_index.faiss")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")


def split_sentences(text: str):
    """Simple sentence splitter: splits on '. ' boundaries, keeps trailing period."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def build_chunks():
    chunks = []
    for doc in POLICY_DOCS:
        sentences = split_sentences(doc["text"])
        for i, sentence in enumerate(sentences):
            chunks.append({
                "chunk_id": f"{doc['doc_id']}__{i}",
                "parent_doc_id": doc["doc_id"],
                "category": doc["category"],
                "text": sentence,
            })
    return chunks


def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (free, local, no API key)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunks = build_chunks()
    print(f"Built {len(chunks)} sentence-wise chunks from {len(POLICY_DOCS)} documents.")

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine similarity
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved FAISS index ({index.ntotal} vectors, dim={dim}) to {INDEX_PATH}")
    print(f"Saved chunk metadata to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
