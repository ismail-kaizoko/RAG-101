from __future__ import annotations
from rank_bm25 import BM25Okapi
from arxiv_rag.chunking.models import Chunk

def _tokenize(text: str) -> list[str]:
    return text.lower().split()

class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunk_ids = [c.chunk_id for c in chunks]
        corpus_tokens = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]