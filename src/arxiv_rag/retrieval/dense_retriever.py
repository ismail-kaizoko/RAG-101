from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arxiv_rag.embedding.embedder import Embedder
    from arxiv_rag.embedding.index import VectorIndex

class DenseRetriever:
    def __init__(self, index: "VectorIndex", embedder: "Embedder"):
        self._index = index
        self._embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_vector = self._embedder.embed_query(query)
        return self._index.search(query_vector, top_k=top_k)