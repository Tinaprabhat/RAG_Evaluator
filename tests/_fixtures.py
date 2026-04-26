"""
Test fixtures module — pure helpers, no pytest dependency.

`conftest.py` imports from here and wraps these in pytest fixtures.
This separation lets the `DummyEmbedder` be reused outside the test runner
(e.g. in interactive smoke tests).
"""

import hashlib

import numpy as np

from ragwatch.utils.embeddings import EmbeddingCache


class DummyEmbedder(EmbeddingCache):
    """Deterministic, model-less embedder for unit tests.

    Hashes text into a fixed 384-dim vector so cosine similarity is reproducible
    but related strings still cluster (we mix a hash signal with a token-overlap
    signal so semantically similar strings get higher cosine).
    """

    def __init__(self, dim: int = 384):
        self.model_name = "dummy"
        self._cache = {}
        self.dim = dim

    def _load_model(self):  # noqa: D401
        pass

    def embed(self, text: str) -> np.ndarray:
        k = self._key(text)
        if k in self._cache:
            return self._cache[k]
        # weak deterministic random base (low variance, won't dominate)
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        base = rng.normal(0, 0.1, self.dim).astype(np.float32)

        # strong token-overlap signal so similar strings → similar vectors
        for tok in text.lower().split():
            tseed = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % self.dim
            base[tseed] += 5.0

        self._cache[k] = base
        return base

    def embed_batch(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


def make_good_input():
    """A clean (q, c, a, gt) where everything is well-supported."""
    from ragwatch.core.schemas import EvalInput
    return EvalInput(
        query="Who developed the theory of general relativity, and in what year?",
        context=[
            "Albert Einstein developed the theory of general relativity in 1915.",
            "The theory describes gravity as a curvature of spacetime.",
        ],
        answer="The theory of general relativity was developed by Albert Einstein in 1915.",
        ground_truth="Albert Einstein developed general relativity in 1915.",
    )


def make_hallucinated_input():
    """An answer that contradicts the context."""
    from ragwatch.core.schemas import EvalInput
    return EvalInput(
        query="How many planets are in our solar system?",
        context=[
            "Our solar system contains eight planets.",
            "Pluto was reclassified as a dwarf planet in 2006.",
        ],
        answer="There are nine planets in our solar system, including Pluto.",
        ground_truth="Eight planets.",
    )
