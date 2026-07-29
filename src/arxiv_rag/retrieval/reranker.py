# src/arxiv_rag/retrieval/reranker.py
from __future__ import annotations
from src.arxiv_rag.chunking.models import Chunk

DEFAULT_MODEL = "BAAI/bge-reranker-base"

class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)
        return self._sort_by_score(chunks, scores, top_k)

    @staticmethod
    def _sort_by_score(chunks, scores, top_k):
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [chunk for chunk, _score in ranked[:top_k]]