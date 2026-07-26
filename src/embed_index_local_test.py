"""
embed_index_local_test.py

NOT part of the DocQuery pipeline you'll submit/present.

This exists purely to validate the chunking -> indexing -> retrieval logic
in an environment without internet access to Hugging Face. It swaps
all-MiniLM-L6-v2 for a TF-IDF vectorizer, which needs no download, so we
can prove the FAISS indexing and retrieval code is correct before running
the real embed_index.py (with the real transformer model) in Colab.

TF-IDF is a much weaker retriever than a sentence transformer -- it matches
on shared vocabulary, not semantic meaning -- so don't expect Colab-quality
results here. We're testing plumbing, not retrieval quality.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ingest import ingest_directory

INDEX_DIR = Path("/home/claude/docquery/index")


def build_local_test_index(data_dir: str, index_dir: Path = INDEX_DIR) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    print("Ingesting and chunking documents...")
    chunks = ingest_directory(data_dir)
    if not chunks:
        raise RuntimeError("No chunks produced")

    texts = [c.text for c in chunks]

    print(f"Fitting TF-IDF vectorizer on {len(texts)} chunks (local-test stand-in for embeddings)...")
    vectorizer = TfidfVectorizer(max_features=2000)
    tfidf_matrix = vectorizer.fit_transform(texts).toarray().astype("float32")

    # normalize so inner product == cosine similarity, matching the real pipeline
    tfidf_matrix = normalize(tfidf_matrix, axis=1)

    dim = tfidf_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(tfidf_matrix)

    faiss.write_index(index, str(index_dir / "docquery_localtest.index"))

    metadata = [
        {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text, "source_path": c.source_path}
        for c in chunks
    ]
    with open(index_dir / "metadata_localtest.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # save vectorizer so retrieve step can embed queries the same way
    import pickle
    with open(index_dir / "vectorizer_localtest.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Local-test index built: {index.ntotal} vectors, dim={dim}")


if __name__ == "__main__":
    build_local_test_index("/home/claude/docquery/data")
