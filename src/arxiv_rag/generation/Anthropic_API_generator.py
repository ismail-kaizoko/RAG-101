from __future__ import annotations
import os
import anthropic

DEFAULT_MODEL = "claude-sonnet-5"

class ClaudeGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text