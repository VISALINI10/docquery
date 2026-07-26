"""
embed_index.py
Embeds document chunks with a sentence-transformer model and builds a
FAISS index for similarity search.

Model choice: all-MiniLM-L6-v2
- 384-dimensional embeddings, ~80MB, fast on CPU.
- Good general-purpose semantic similarity performance for its size, which
  matters here since this needs to run without a GPU in Colab's free tier.
- Tradeoff worth knowing: a larger model (e.g. all-mpnet-base-v2) gives
  better retrieval quality but is slower and heavier — a real production
  system would benchmark this tradeoff rather than assume MiniLM is enough.

Index choice: FAISS IndexFlatIP (inner product) over L2-normalized vectors
- With normalized vectors, inner product is mathematically equivalent to
  cosine similarity, which is the standard metric for sentence embeddings.
- IndexFlatIP does exact search (no approximation), which is appropriate
  at this corpus scale (tens to low thousands of chunks). At production
  scale (millions of chunks) you'd switch to an approximate index like
  IVF or HNSW to trade a small amount of recall for much faster search.
"""

import json
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import Chunk, ingest_directory

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = Path("/home/claude/docquery/index")


def build_index(data_dir: str, index_dir: Path = INDEX_DIR) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Ingesting and chunking documents...")
    chunks: List[Chunk] = ingest_directory(data_dir)
    if not chunks:
        raise RuntimeError("No chunks produced — check that data_dir has .txt/.pdf files")

    print(f"Embedding {len(chunks)} chunks...")
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize so inner product == cosine similarity
    )
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP with dimension {dim}...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "docquery.index"))

    metadata = [
        {
            "doc_id": c.doc_id,
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source_path": c.source_path,
        }
        for c in chunks
    ]
    with open(index_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nIndex built: {index.ntotal} vectors, dim={dim}")
    print(f"Saved to: {index_dir}")


if __name__ == "__main__":
    build_index("/home/claude/docquery/data")
