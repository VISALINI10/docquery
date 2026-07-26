"""
evaluate.py
Evaluation harness for the DocQuery retrieval pipeline.

Rather than just eyeballing a few queries, this tests retrieval against
three categories designed to expose specific failure modes:

1. EXACT-MATCH  -- query shares vocabulary with the source chunk.
   Any working retriever should pass these. A baseline sanity check,
   not a meaningful signal of quality on its own.

2. PARAPHRASED  -- same underlying question, deliberately reworded to
   avoid shared vocabulary with the source text. A lexical method
   (TF-IDF, keyword search) predictably fails here because it has no
   concept of meaning -- only token overlap. A real sentence-transformer
   should do meaningfully better because it embeds semantic similarity.
   This category is the whole point of using dense embeddings over
   keyword search, so it's the most important one to report.

3. OUT-OF-SCOPE -- questions the corpus cannot answer. A good system
   should retrieve low-confidence results (score below a threshold) so
   the generation step can correctly say "not found in context" instead
   of confidently retrieving something irrelevant and letting the LLM
   hallucinate an answer grounded in the wrong passage.

Metric: Recall@k -- for each labeled query, did the correct doc_id appear
in the top-k retrieved results? For out-of-scope queries, "correct"
means the top score falls below a chosen confidence threshold.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from retrieve import retrieve_local_test  # noqa: E402


EXACT_MATCH_QUERIES = [
    {"query": "What is the Fast Gradient Sign Method?", "expected_doc": "doc1_adversarial_robustness"},
    {"query": "What is FAISS used for?", "expected_doc": "doc3_faiss_vector_search"},
    {"query": "What is prompt injection?", "expected_doc": "doc4_prompt_injection"},
]

PARAPHRASED_QUERIES = [
    # Same question as above, reworded to avoid shared terms
    {"query": "How can you find tiny input changes that fool a neural network?", "expected_doc": "doc1_adversarial_robustness"},
    {"query": "What tool would you use to quickly search millions of embedding vectors?", "expected_doc": "doc3_faiss_vector_search"},
    {"query": "How might someone hide malicious commands inside a document a chatbot reads?", "expected_doc": "doc4_prompt_injection"},
]

OUT_OF_SCOPE_QUERIES = [
    {"query": "What is the capital of France?"},
    {"query": "How do I bake a chocolate cake?"},
    {"query": "What was the score of yesterday's football match?"},
]

OUT_OF_SCOPE_CONFIDENCE_THRESHOLD = 0.15  # tuned empirically for this TF-IDF local test


def evaluate_recall_at_k(labeled_queries, k=3, label=""):
    hits, total = 0, len(labeled_queries)
    print(f"\n=== {label} (Recall@{k}) ===")
    for item in labeled_queries:
        results = retrieve_local_test(item["query"], top_k=k)
        retrieved_docs = [r["doc_id"] for r in results]
        hit = item["expected_doc"] in retrieved_docs
        hits += int(hit)
        status = "PASS" if hit else "FAIL"
        top_score = results[0]["score"] if results else 0.0
        print(f"  [{status}] '{item['query']}'")
        print(f"         expected={item['expected_doc']}  retrieved_top={retrieved_docs[0] if retrieved_docs else None}"
              f"  top_score={top_score:.3f}")
    recall = hits / total if total else 0.0
    print(f"  --> Recall@{k}: {hits}/{total} = {recall:.2f}")
    return recall


def evaluate_out_of_scope(queries, threshold=OUT_OF_SCOPE_CONFIDENCE_THRESHOLD):
    correct, total = 0, len(queries)
    print(f"\n=== OUT-OF-SCOPE (should score below {threshold}) ===")
    for item in queries:
        results = retrieve_local_test(item["query"], top_k=1)
        top_score = results[0]["score"] if results else 0.0
        correctly_low_confidence = top_score < threshold
        correct += int(correctly_low_confidence)
        status = "PASS" if correctly_low_confidence else "FAIL"
        print(f"  [{status}] '{item['query']}'  top_score={top_score:.3f}"
              f"  (would retrieve: {results[0]['doc_id'] if results else None})")
    rate = correct / total if total else 0.0
    print(f"  --> Correctly flagged as out-of-scope: {correct}/{total} = {rate:.2f}")
    return rate


if __name__ == "__main__":
    exact_recall = evaluate_recall_at_k(EXACT_MATCH_QUERIES, k=3, label="EXACT-MATCH")
    paraphrase_recall = evaluate_recall_at_k(PARAPHRASED_QUERIES, k=3, label="PARAPHRASED")
    oos_rate = evaluate_out_of_scope(OUT_OF_SCOPE_QUERIES)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Exact-match Recall@3:   {exact_recall:.2f}")
    print(f"Paraphrased Recall@3:   {paraphrase_recall:.2f}")
    print(f"Out-of-scope detection: {oos_rate:.2f}")
    print("\nNote: this run uses TF-IDF (local-test stand-in, no HF access).")
    print("Expect the paraphrase gap to shrink significantly once run against")
    print("the real sentence-transformer embeddings in Colab -- that gap IS")
    print("the empirical case for using dense embeddings over keyword search.")
