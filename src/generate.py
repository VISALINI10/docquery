"""
generate.py
Takes retrieved chunks + the original query and produces a grounded answer.

Two modes:
1. LLM mode (real RAG): sends retrieved chunks + query to a language model
   with an instruction to answer ONLY from the provided context, and to say
   so explicitly if the context doesn't contain the answer. This is the
   mode you'd use in Colab / production.
2. Extractive fallback (no API key required): returns the single highest-
   scoring chunk verbatim, clearly labeled as unsynthesized. This exists so
   the pipeline is testable end-to-end without any API access -- it is
   NOT a substitute for real generation and should not be presented as such.

The prompt template below is a deliberate design choice, not boilerplate:
- Instructing the model to say "not found in context" instead of guessing
  is what prevents a RAG system from silently hallucinating when retrieval
  fails -- this is one of the most common real-world RAG failure modes.
"""

import os
from typing import List, Dict


SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer the user's "
    "question using ONLY the context provided below. Do not use outside "
    "knowledge. If the context does not contain enough information to "
    "answer the question, say exactly: 'The retrieved context does not "
    "contain enough information to answer this question.' Cite which "
    "chunk(s) you used by their doc_id."
)


def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(f"[{c['doc_id']} | chunk {c['chunk_id']}]\n{c['text']}")
    context = "\n\n".join(context_blocks)

    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above:"
    )


def generate_answer_llm(query: str, retrieved_chunks: List[Dict], client=None) -> str:
    """
    Real generation using an LLM. `client` should be an already-configured
    API client (e.g. anthropic.Anthropic()). Left generic so you can swap
    providers without changing the pipeline logic.
    """
    if client is None:
        raise ValueError("No LLM client provided. Use generate_answer_extractive() "
                          "for testing without API access.")

    prompt = build_prompt(query, retrieved_chunks)

    # Example for the Anthropic API -- adapt if using a different provider.
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_answer_extractive(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    No-API fallback: returns the top retrieved chunk verbatim with a clear
    label. Use this to validate the pipeline end-to-end without a key --
    do not present this as the RAG system's real generation behavior.
    """
    if not retrieved_chunks:
        return "[EXTRACTIVE FALLBACK] No relevant chunks retrieved."

    top = retrieved_chunks[0]
    return (
        f"[EXTRACTIVE FALLBACK -- NOT LLM-GENERATED]\n"
        f"Most relevant passage (score={top['score']:.3f}, source={top['doc_id']} "
        f"chunk {top['chunk_id']}):\n\n{top['text']}"
    )


def answer_query(query: str, retrieved_chunks: List[Dict], use_llm: bool = False, client=None) -> str:
    if use_llm:
        return generate_answer_llm(query, retrieved_chunks, client=client)
    return generate_answer_extractive(query, retrieved_chunks)


if __name__ == "__main__":
    # Local smoke test using the TF-IDF retrieval pipeline
    from retrieve import retrieve_local_test

    query = "How does self-attention work in transformers?"
    chunks = retrieve_local_test(query, top_k=2)
    answer = answer_query(query, chunks, use_llm=False)
    print(f"QUERY: {query}\n")
    print(answer)
