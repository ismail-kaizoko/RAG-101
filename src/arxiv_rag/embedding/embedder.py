from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

from src.arxiv_rag.chunking.models import Chunk
from src.arxiv_rag.embedding.context import build_embedding_text

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: list[Chunk], *, use_context: bool = True) -> np.ndarray:
        texts = [build_embedding_text(c, use_context=use_context) for c in chunks]
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_query(self, query: str) -> np.ndarray:
        return self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]