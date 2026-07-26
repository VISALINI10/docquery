"""
retrieve.py
Given a natural-language query, embed it and search the FAISS index for
the most relevant document chunks.

This module is written against the REAL pipeline (sentence-transformers +
docquery.index). Use retrieve_local_test() below when running in the
sandbox without Hugging Face access.
"""

import json
from pathlib import Path
from typing import List, Dict

import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = Path("/home/claude/docquery/index")


def load_index_and_metadata(index_dir: Path = INDEX_DIR):
    index = faiss.read_index(str(index_dir / "docquery.index"))
    with open(index_dir / "metadata.json") as f:
        metadata = json.load(f)
    return index, metadata


def retrieve(query: str, top_k: int = 3, index_dir: Path = INDEX_DIR) -> List[Dict]:
    model = SentenceTransformer(MODEL_NAME)
    index, metadata = load_index_and_metadata(index_dir)

    query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata[idx].copy()
        entry["score"] = float(score)
        results.append(entry)
    return results


# ---- local-test variant (TF-IDF, no internet required) ----

def retrieve_local_test(query: str, top_k: int = 3, index_dir: Path = INDEX_DIR) -> List[Dict]:
    import pickle
    from sklearn.preprocessing import normalize

    index = faiss.read_index(str(index_dir / "docquery_localtest.index"))
    with open(index_dir / "metadata_localtest.json") as f:
        metadata = json.load(f)
    with open(index_dir / "vectorizer_localtest.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    query_vec = vectorizer.transform([query]).toarray().astype("float32")
    query_vec = normalize(query_vec, axis=1)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata[idx].copy()
        entry["score"] = float(score)
        results.append(entry)
    return results


if __name__ == "__main__":
    test_queries = [
        "How does self-attention work in transformers?",
        "What is the FGSM adversarial attack?",
        "How can retrieved documents be used to attack a language model?",
        "What index type does FAISS use for exact nearest neighbor search?",
    ]

    for q in test_queries:
        print(f"\nQUERY: {q}")
        results = retrieve_local_test(q, top_k=2)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['doc_id']} (chunk {r['chunk_id']}): {r['text'][:120]}...")
