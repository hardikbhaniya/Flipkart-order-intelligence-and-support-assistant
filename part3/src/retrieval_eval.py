"""
Part 3 Task 10: Evaluate retrieval using the query/relevant-document pairs
from knowledge_base.QUERY_ANSWER_KEY.

Computes Precision@3 and Recall@3 PER QUERY at the DOCUMENT level (each
retrieved chunk mapped back to its parent document and deduplicated before
scoring), shows the per-query arithmetic, then reports the average of each
across all queries.
"""
from knowledge_base import QUERY_ANSWER_KEY
from retriever import retrieve_documents


def evaluate_query(query: str, relevant_doc_ids: list, k: int = 3):
    retrieved_doc_scores = retrieve_documents(query, k=k)
    retrieved_doc_ids = list(retrieved_doc_scores.keys())

    relevant_set = set(relevant_doc_ids)
    retrieved_set = set(retrieved_doc_ids)

    true_positives = relevant_set & retrieved_set
    precision = len(true_positives) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(true_positives) / len(relevant_set) if relevant_set else 0.0

    return {
        "query": query,
        "relevant_doc_ids": relevant_doc_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "true_positives": list(true_positives),
        "precision_at_3": round(precision, 4),
        "recall_at_3": round(recall, 4),
    }


def main():
    print("=" * 70)
    print("TASK 10: RETRIEVAL EVALUATION (Precision@3 / Recall@3, document-level)")
    print("=" * 70)

    results = []
    for item in QUERY_ANSWER_KEY:
        r = evaluate_query(item["query"], item["relevant_doc_ids"])
        results.append(r)

        print(f"\nQuery: \"{r['query']}\"")
        print(f"  Relevant docs (answer key):  {r['relevant_doc_ids']}")
        print(f"  Retrieved docs (top-3, deduped): {r['retrieved_doc_ids']}")
        print(f"  True positives: {r['true_positives']}")
        print(f"  Precision@3 = {len(r['true_positives'])}/{len(r['retrieved_doc_ids'])} = {r['precision_at_3']}")
        print(f"  Recall@3    = {len(r['true_positives'])}/{len(r['relevant_doc_ids'])} = {r['recall_at_3']}")

    avg_precision = sum(r["precision_at_3"] for r in results) / len(results)
    avg_recall = sum(r["recall_at_3"] for r in results) / len(results)

    print("\n" + "=" * 70)
    print(f"AVERAGE Precision@3 across {len(results)} queries: {avg_precision:.4f}")
    print(f"AVERAGE Recall@3 across {len(results)} queries: {avg_recall:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
