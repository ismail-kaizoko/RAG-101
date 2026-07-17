"""CLI: search arXiv, download PDFs, extract text, save to data/raw/papers.jsonl.

Usage:
    python scripts/ingest.py "cat:cs.LG AND abs:diffusion" --max-results 75
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


from src.arxiv_rag.ingestion.client import ArxivClient
from src.arxiv_rag.ingestion.pdf_parser import fetch_and_extract
from src.arxiv_rag.ingestion.storage import save_papers


DEFAULT_OUTPUT = Path("data/raw/papers.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="arXiv search query, e.g. 'cat:cs.LG AND abs:diffusion'")
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    client = ArxivClient()
    print(f"Searching arXiv for: {args.query!r}")
    papers = client.search(args.query, max_results=args.max_results)
    print(f"Found {len(papers)} papers. Downloading and extracting text...")

    fetched = []
    for i, paper in enumerate(papers, start=1):
        try:
            fetched.append(fetch_and_extract(paper))
            print(f"  [{i}/{len(papers)}] {paper.arxiv_id} — {paper.title[:60]}")
        except Exception as e:
            print(f"  [{i}/{len(papers)}] FAILED {paper.arxiv_id}: {e}")

    save_papers(args.output, fetched)
    print(f"Saved {len(fetched)} papers to {args.output}")


if __name__ == "__main__":
    main()
