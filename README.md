# DocQuery — Retrieval-Augmented Generation Pipeline

A RAG system built from scratch: document ingestion → sentence-aware chunking →
dense embedding → FAISS indexing → retrieval → grounded generation → evaluation.

## Why this exists

Most RAG tutorials stop at "it retrieved something and the LLM answered." This
project treats retrieval as something to be *measured*, not assumed — the
evaluation harness specifically tests paraphrased queries (no shared vocabulary
with the source text) and out-of-scope queries (unanswerable from the corpus),
because those are the two failure modes that matter most in a real deployment
and the two that a naive "does it retrieve *something*" test won't catch.

## Architecture

```
Documents (.txt/.pdf)
    │
    ▼
[ingest.py]        sentence-aware chunking, ~500 chars/chunk, 100-char overlap
    │
    ▼
[embed_index.py]   all-MiniLM-L6-v2 embeddings → FAISS IndexFlatIP (cosine sim)
    │
    ▼
[retrieve.py]       query embedding → top-k nearest chunks
    │
    ▼
[generate.py]       retrieved chunks + query → grounded LLM answer
                     (explicitly instructed to say "not found" rather than guess)
    │
    ▼
[evaluate.py]        Recall@k across exact-match / paraphrased / out-of-scope queries
```

## Key design decisions

- **Chunking**: sentence-boundary-aware with overlap, not a hard character cut —
  prevents an answer from being split away from its supporting context.
- **Embedding model**: `all-MiniLM-L6-v2` — small enough to run on CPU, good
  enough for general semantic similarity. A larger model (e.g. `all-mpnet-base-v2`)
  trades speed for quality; worth A/B testing at production scale.
- **Index**: `IndexFlatIP` (exact search) over normalized vectors — appropriate
  at this corpus scale. At millions of chunks, switch to IVF or HNSW for
  approximate search.
- **Generation prompt**: explicitly instructed to say "not found in context"
  rather than answer from outside knowledge — this is what prevents silent
  hallucination when retrieval fails.

## Evaluation results

Measured locally with a TF-IDF stand-in (no internet access to Hugging Face
in the dev sandbox used to build this):

| Category | Recall@3 |
|---|---|
| Exact-match | 1.00 |
| Paraphrased | 0.67 |
| Out-of-scope correctly flagged | 0.00 |

**Finding**: TF-IDF (lexical matching) fails on paraphrased queries with no
shared vocabulary, and its similarity scores don't have enough dynamic range
to reliably separate relevant from irrelevant queries by threshold alone.
This is the empirical case for using dense embeddings instead of keyword
search, and it's also why production RAG systems typically add a dedicated
groundedness/relevance check after retrieval rather than trusting a raw
similarity score.

`colab/DocQuery_Colab.ipynb` reruns the identical evaluation using the real
`all-MiniLM-L6-v2` embeddings (requires internet access) — run it and compare
the numbers to see whether semantic embeddings close these gaps.

## Running it

**Local (TF-IDF stand-in, no internet needed):**
```bash
pip install faiss-cpu scikit-learn pypdf
cd src && python embed_index_local_test.py && python retrieve.py
cd .. && python evaluate.py
```

**Real pipeline (Colab or any environment with internet access):**
Open `colab/DocQuery_Colab.ipynb` in Google Colab and run all cells. Add your
own API key in Colab's Secrets manager to enable the generation step.

## What I'd build next

- Reranking step (cross-encoder) after initial retrieval to improve precision
- A learned or better-calibrated out-of-scope threshold, tuned against a real
  score-distribution histogram rather than a guessed cutoff
- Swap in a larger embedding model and re-run the same evaluation harness to
  quantify the quality/speed tradeoff directly, rather than assuming it
