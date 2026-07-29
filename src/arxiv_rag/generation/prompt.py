from __future__ import annotations
from arxiv_rag.chunking.models import Chunk

_SYSTEM_INSTRUCTIONS = (
    "You are a research assistant answering questions about a corpus of "
    "arXiv papers.\n"
    "Answer ONLY using the context provided below. If the answer is not "
    "in the context, say so explicitly -- do not guess or use outside "
    "knowledge.\n"
    "When you use information from a chunk, cite it inline using its "
    "paper id, e.g. [2401.00001]."
)

def build_prompt(query: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(
        f"[{chunk.paper_id}] (section: {chunk.section_title or 'unknown'})\n{chunk.text}"
        for chunk in chunks
    )
    return f"{_SYSTEM_INSTRUCTIONS}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"