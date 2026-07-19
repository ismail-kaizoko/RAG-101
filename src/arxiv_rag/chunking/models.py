from __future__ import annotations
from pydantic import BaseModel

class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    chunk_index: int
    text: str
    method: str
    section_title: str | None = None