from __future__ import annotations
from pathlib import Path
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.arxiv_rag.chunking.fixed_size import chunk_fixed_size
from src.arxiv_rag.chunking.structure_aware import chunk_by_structure
from src.arxiv_rag.ingestion.storage import load_papers

INPUT_PATH = Path("data/examples/raw/example_papers.jsonl")
FIXED_OUTPUT = Path("data/examples/chunks/chunks_fixed.jsonl")
STRUCTURED_OUTPUT = Path("data/examples/chunks/chunks_structured.jsonl")

def save_chunks(path, chunks):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

def main():
    papers = load_papers(INPUT_PATH)
    fixed_chunks, structured_chunks = [], []
    for paper in papers:
        fixed_chunks.extend(chunk_fixed_size(paper.full_text, paper.arxiv_id))
        structured_chunks.extend(chunk_by_structure(paper.full_text, paper.arxiv_id))

    save_chunks(FIXED_OUTPUT, fixed_chunks)
    save_chunks(STRUCTURED_OUTPUT, structured_chunks)
    print(f"Fixed-size: {len(fixed_chunks)} chunks -> {FIXED_OUTPUT}")
    print(f"Structured: {len(structured_chunks)} chunks -> {STRUCTURED_OUTPUT}")

if __name__ == "__main__":
    main()