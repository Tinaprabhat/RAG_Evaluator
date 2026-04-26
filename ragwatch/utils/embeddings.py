"""
Embedding cache — compute MiniLM embeddings once, reuse aggressively.

Same context chunks appear across many queries during batch evaluation.
Caching saves the bulk of CPU time.
"""

from __future__ import annotations
import hashlib
import os
from typing import Any

import numpy as np


class EmbeddingCache:
    """Lazy-loaded sentence-transformer with in-memory hash cache."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any = None
        self._cache: dict[str, np.ndarray] = {}

    def _load_model(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self.model_name)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string (cached)."""
        k = self._key(text)
        if k in self._cache:
            return self._cache[k]
        self._load_model()
        vec = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        self._cache[k] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings — uses cache where available."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        keys = [self._key(t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]

        if missing_idx:
            self._load_model()
            missing_texts = [texts[i] for i in missing_idx]
            new_vecs = self._model.encode(
                missing_texts, convert_to_numpy=True, show_progress_bar=False
            )
            for j, i in enumerate(missing_idx):
                self._cache[keys[i]] = new_vecs[j]

        return np.vstack([self._cache[k] for k in keys])

    def clear(self) -> None:
        self._cache.clear()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine between rows of A and rows of B."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_n @ B_n.T
