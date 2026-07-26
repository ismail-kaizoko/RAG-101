from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.arxiv_rag.chunking.models import Chunk
from src.arxiv_rag.embedding.embedder import Embedder
from src.arxiv_rag.embedding.index import VectorIndex

def load_chunks(path):
    with path.open("r", encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--input_dir")
    # parser.add_argument("--output_dir")
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()

    chunks_path = Path(f"data/examples/chunks/chunks_structured.jsonl")
    index_dir = Path(f"data/examples/indexes/indexes_structured.jsonl")

    chunks = load_chunks(chunks_path)
    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks, use_context=not args.no_context)

    index = VectorIndex(dim=embeddings.shape[1])
    index.add(embeddings, [c.chunk_id for c in chunks])
    index.save(index_dir)
    print(f"Saved index to {index_dir}")

if __name__ == "__main__":
    main()