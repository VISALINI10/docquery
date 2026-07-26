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

Tested across three query categories designed to expose specific retrieval
failure modes: exact-match (baseline sanity check), paraphrased (semantic vs.
lexical matching), and out-of-scope (does the system know what it doesn't know).

| Category | TF-IDF baseline | Real embeddings (all-MiniLM-L6-v2) |
|---|---|---|
| Exact-match Recall@3 | 1.00 | 1.00 |
| Paraphrased Recall@3 | 0.67 | **1.00** |
| Out-of-scope correctly flagged | 0.00 | **1.00** |

**Finding**: TF-IDF (lexical matching) fails on paraphrased queries with no
shared vocabulary — e.g. "How might someone hide malicious commands inside a
document a chatbot reads?" has almost no words in common with the source
text's "prompt injection... redirect the model's behavior," so a keyword
matcher has no signal to work with. Switching to dense sentence embeddings
closed this gap entirely, because semantic similarity captures meaning rather
than surface wordforms.

TF-IDF's out-of-scope detection also failed completely (0.00) because its
similarity scores don't have enough dynamic range to separate "irrelevant"
from "vaguely related" by a fixed threshold. Real embeddings gave a much
cleaner score separation (irrelevant queries scored 0.07–0.12, relevant
queries scored 0.44–0.69), which is what let a simple threshold work here.

**Caveat**: each category was tested with only 3 queries. A perfect score on
9 total queries demonstrates the evaluation methodology and the lexical-vs-
semantic gap clearly, but is too small a sample to claim strong generalization
— a natural next step is expanding each category with harder, more borderline
examples to find where the system actually starts to fail.

`colab/DocQuery_Colab.ipynb` contains this full run with real sentence-transformer
embeddings, including the exact query-by-query output.

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
