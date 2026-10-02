from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .chunking import Chunk
from .embeddings import Embedder, TfidfEmbedder

@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float

class VectorIndex:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or TfidfEmbedder()
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build an index with no chunks.")
        self._chunks = chunks
        self._matrix = self.embedder.fit_transform([c.text for c in chunks])

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        if self._matrix is None or not self._chunks:
            raise RuntimeError("Index has not been built.")
        q = self.embedder.transform([query])
        scores = cosine_similarity(q, self._matrix)[0]
        order = np.argsort(scores)[::-1][:top_k]
        return [SearchHit(self._chunks[i], float(scores[i])) for i in order]
