from __future__ import annotations
from typing import Protocol

class Generator(Protocol):
    def generate(self, prompt: str, *, max_tokens: int = 512) -> str: ...