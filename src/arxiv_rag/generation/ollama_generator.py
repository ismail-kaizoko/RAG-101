from __future__ import annotations
import httpx


DEFAULT_MODEL = "phi4-mini"

class OllamaGenerator:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = "http://localhost:11434"):
        self._model = model
        self._host = host

    def generate(self, prompt: str, *, max_tokens: int = 512) -> str:
        response = httpx.post(
            f"{self._host}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["response"]