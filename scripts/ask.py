"""CLI: retrieve relevant chunks for a query, then generate a grounded answer.

Usage:
    python scripts/ask.py "what noise schedule does DDPM use" --backend ollama
    python scripts/ask.py "what noise schedule does DDPM use" --backend local
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.arxiv_rag.chunking.models import Chunk
from src.arxiv_rag.embedding.embedder import Embedder
from src.arxiv_rag.embedding.index import VectorIndex
from src.arxiv_rag.generation.prompt import build_prompt
from src.arxiv_rag.retrieval.bm25_retriever import BM25Retriever
from src.arxiv_rag.retrieval.dense_retriever import DenseRetriever
from src.arxiv_rag.retrieval.hybrid_retriever import HybridRetriever


def load_chunks(path: Path) -> list[Chunk]:
    with path.open("r", encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--method", choices=["fixed", "structured"], default="structured")
    parser.add_argument("--backend", choices=["ollama", "local","api"], default="api")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    chunks = load_chunks(Path(f"data/processed/chunks_{args.method}.jsonl"))
    embedder = Embedder()
    index = VectorIndex.load(Path(f"data/index/{args.method}"))
    dense = DenseRetriever(index, embedder)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25)

    chunks_by_id = {c.chunk_id: c for c in chunks}
    results = hybrid.search(args.query, top_k=args.top_k)
    retrieved_chunks = [chunks_by_id[chunk_id] for chunk_id, _score in results]

    prompt = build_prompt(args.query, retrieved_chunks)

    if args.backend == "ollama":
        from arxiv_rag.generation.ollama_generator import OllamaGenerator
        generator = OllamaGenerator()
    else:
        from arxiv_rag.generation.local_hf_generator import LocalHFGenerator
        generator = LocalHFGenerator()

    answer = generator.generate(prompt)
    print(answer)


if __name__ == "__main__":
    main()