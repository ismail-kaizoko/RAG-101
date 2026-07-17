"""Thin wrapper around the `arxiv` package for querying arXiv's API.

We keep this as a thin wrapper (not a re-export) so the rest of the codebase
depends on our `Paper` model, not on the third-party library's result type.
If `arxiv` changes its interface or we swap it out later, only this file
needs to change.
"""

from __future__ import annotations

import arxiv
 
from src.arxiv_rag.ingestion.models import Paper


class ArxivClient:
    def __init__(self, page_size: int = 50, delay_seconds: float = 3.0):
        # delay_seconds respects arXiv's rate-limit guidance for the API
        self._client = arxiv.Client(page_size=page_size, delay_seconds=delay_seconds)

    def search(self, query: str, max_results: int = 100) -> list[Paper]:
        """Search arXiv and return papers with metadata only (no full_text yet).

        `query` follows arXiv's search syntax, e.g. 'cat:cs.LG AND abs:diffusion'.
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        return [self._to_paper(result) for result in self._client.results(search)]

    @staticmethod
    def _to_paper(result: arxiv.Result) -> Paper:
        return Paper(
            arxiv_id= result.get_short_id(),
            title= result.title.strip(),
            authors= [a.name for a in result.authors],
            abstract= result.summary.strip(),
            categories= result.categories,
            published= result.published.isoformat(),
            pdf_url= result.pdf_url,
        )
