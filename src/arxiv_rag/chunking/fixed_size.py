### Baseline : Naive chunking (not-section aware), splits a document into overlapping windows
### Disadvantages : can cut mid sentencen destroying meaning, text covering many topics can be averaged out, resulting to noisy representations. 

from __future__ import annotations
from src.arxiv_rag.chunking.models import Chunk

def chunk_fixed_size(text, paper_id, *, chunk_size=300, overlap=40):
    words = text.split()
    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    for i, start in enumerate(range(0, len(words), stride)):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(Chunk(
            chunk_id=f"{paper_id}_fixed_{i}",
            paper_id=paper_id,
            chunk_index=i,
            text=" ".join(window),
            method="fixed",
        ))
        if start + chunk_size >= len(words):
            break
    return chunks