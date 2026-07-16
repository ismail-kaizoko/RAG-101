"""Data contracts for ingested papers.

Using pydantic here (rather than a plain dataclass) buys us validation and
free JSON serialization — every downstream stage (chunking, indexing) reads
and writes these as JSONL, so a stable, validated schema matters more than
it would in a throwaway script.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A single arXiv paper: metadata + full extracted text."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: str  # ISO date string; kept as str for simple JSONL round-tripping
    pdf_url: str
    full_text: str = Field(default="", repr=False)  # populated after PDF parsing
