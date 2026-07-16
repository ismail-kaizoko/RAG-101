"""Download a paper's PDF and extract plain text from it.

pymupdf (imported as `fitz`) over pypdf: faster, and handles multi-column
academic PDF layouts (arXiv papers are almost all two-column) noticeably
better — pypdf tends to interleave columns as if they were one, which
scrambles chunk boundaries downstream.
"""

from __future__ import annotations

import httpx
import pymupdf

from arxiv_rag.ingestion.models import Paper


def fetch_and_extract(paper: Paper, *, timeout: float = 30.0) -> Paper:
    """Download `paper.pdf_url` and return a copy of `paper` with full_text filled."""
    response = httpx.get(paper.pdf_url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()

    with pymupdf.open(stream=response.content, filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)

    return paper.model_copy(update={"full_text": text})
