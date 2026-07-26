from __future__ import annotations
import json
from pathlib import Path
import faiss
import numpy as np

class VectorIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.chunk_ids: list[str] = []

    def add(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError("embeddings and chunk_ids must have the same length")
        self.index.add(embeddings.astype("float32"))
        self.chunk_ids.extend(chunk_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        query = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query, top_k)
        return [
            (self.chunk_ids[i], float(score))
            for score, i in zip(scores[0], indices[0])
            if i != -1
        ]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "index.faiss"))
        (dir_path / "chunk_ids.json").write_text(json.dumps(self.chunk_ids))

    @classmethod
    def load(cls, dir_path: Path) -> "VectorIndex":
        index = faiss.read_index(str(dir_path / "index.faiss"))
        chunk_ids = json.loads((dir_path / "chunk_ids.json").read_text())
        instance = cls(dim=index.d)
        instance.index = index
        instance.chunk_ids = chunk_ids
        return instance