from __future__ import annotations
from collections import defaultdict

from arxiv_rag.retrieval.bm25_retriever import BM25Retriever
from arxiv_rag.retrieval.dense_retriever import DenseRetriever

def reciprocal_rank_fusion(ranked_lists, *, k=60, top_k=5):
    fused_scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (chunk_id, _original_score) in enumerate(ranked, start=1):
            fused_scores[chunk_id] += 1.0 / (k + rank)
    fused = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return fused[:top_k]

class HybridRetriever:
    def __init__(self, dense, bm25, *, k=60):
        self._dense = dense
        self._bm25 = bm25
        self._k = k

    def search(self, query, top_k=5, candidate_k=20):
        dense_results = self._dense.search(query, top_k=candidate_k)
        bm25_results = self._bm25.search(query, top_k=candidate_k)
        return reciprocal_rank_fusion([dense_results, bm25_results], k=self._k, top_k=top_k)