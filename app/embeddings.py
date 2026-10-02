from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class Embedder(ABC):
    @abstractmethod
    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

class TfidfEmbedder(Embedder):
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(texts).toarray()

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()

class SentenceTransformerEmbedder(Embedder):
    """Optional deep-learning embedding backend using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True))
