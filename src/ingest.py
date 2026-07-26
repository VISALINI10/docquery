"""
ingest.py
Loads raw documents (.txt, .pdf) and splits them into overlapping chunks
suitable for embedding.

Design choices (worth understanding, not just running):
- Chunk size ~500 characters: small enough that each chunk stays topically
  coherent for embedding, large enough to retain context for the generator.
- Overlap ~100 characters: prevents information at a chunk boundary from
  being split away from its context (e.g. a sentence cut in half).
- Splitting prefers sentence boundaries over hard character cuts, so chunks
  don't start/end mid-sentence when avoidable.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    source_path: str


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt" or suffix == ".md":
        return load_text_file(path)
    elif suffix == ".pdf":
        return load_pdf_file(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def split_into_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter; avoids pulling in a full NLP toolkit
    # for something this simple. Handles ., !, ? followed by whitespace+capital.
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    sentences = split_into_sentences(text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying over the tail of the previous chunk
            # for overlap so context isn't lost at the boundary
            overlap_text = current[-overlap:] if current else ""
            current = f"{overlap_text} {sentence}".strip()

    if current:
        chunks.append(current)

    return chunks


def ingest_directory(data_dir: str) -> List[Chunk]:
    data_path = Path(data_dir)
    all_chunks: List[Chunk] = []

    files = sorted(
        [p for p in data_path.iterdir() if p.suffix.lower() in (".txt", ".md", ".pdf")]
    )

    for file_path in files:
        doc_id = file_path.stem
        try:
            raw_text = load_document(file_path)
        except Exception as e:
            print(f"[WARN] Failed to load {file_path.name}: {e}")
            continue

        text_chunks = chunk_text(raw_text)
        for i, chunk_str in enumerate(text_chunks):
            all_chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=i,
                    text=chunk_str,
                    source_path=str(file_path),
                )
            )
        print(f"[OK] {file_path.name}: {len(text_chunks)} chunks")

    return all_chunks


if __name__ == "__main__":
    chunks = ingest_directory("/home/claude/docquery/data")
    print(f"\nTotal chunks across all documents: {len(chunks)}")
    if chunks:
        print("\n--- Sample chunk ---")
        print(f"doc_id: {chunks[0].doc_id}, chunk_id: {chunks[0].chunk_id}")
        print(chunks[0].text[:300])
