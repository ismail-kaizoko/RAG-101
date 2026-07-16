"""Persist and load papers as JSONL.

JSONL (one JSON object per line) rather than a single JSON array: papers can
be appended one at a time as they're fetched, so a crash mid-ingestion
doesn't lose already-downloaded papers, and the file stays readable with
plain `grep`/`jq`.
"""

from __future__ import annotations

from pathlib import Path

from arxiv_rag.ingestion.models import Paper


def save_papers(path: Path, papers: list[Paper]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(paper.model_dump_json() + "\n")


def load_papers(path: Path) -> list[Paper]:
    with path.open("r", encoding="utf-8") as f:
        return [Paper.model_validate_json(line) for line in f if line.strip()]
