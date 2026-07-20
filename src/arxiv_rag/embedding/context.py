from __future__ import annotations
from arxiv_rag.chunking.models import Chunk

def build_embedding_text(chunk: Chunk, *, use_context: bool = True) -> str:
    if use_context and chunk.section_title:
        return f"[{chunk.section_title}] {chunk.text}"
    return chunk.text

