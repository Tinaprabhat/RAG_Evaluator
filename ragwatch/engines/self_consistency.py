"""
Self-Consistency Engine — internal claim agreement check.

Inspired by SelfCheckGPT (Manakul et al., EMNLP 2023). We don't have access to
the original LLM that produced the answer, so we adapt the idea: compute
pairwise semantic agreement between claims in the answer. Highly self-contradictory
answers signal lower reliability.

This is a pure-embedding measure — no extra model calls.
"""

from __future__ import annotations
import numpy as np

from ragwatch.core.schemas import ScoreWithCI
from ragwatch.utils.embeddings import EmbeddingCache, cosine_matrix


class SelfConsistencyEngine:
    """Internal-coherence scoring across claims of an answer."""

    def __init__(self, embedder: EmbeddingCache):
        self.embedder = embedder

    def score(self, claims: list[str]) -> ScoreWithCI:
        """
        Mean pairwise cosine between claim embeddings.
        Returns higher = more internally consistent.
        Single claim → consistency = 1 by definition.
        """
        n = len(claims)
        if n == 0:
            return ScoreWithCI(0.0, 0.0, 0, "no claims")
        if n == 1:
            return ScoreWithCI(1.0, 0.0, 1, "single claim — trivially consistent")

        E = self.embedder.embed_batch(claims)
        sim = cosine_matrix(E, E)
        iu = np.triu_indices(n, k=1)
        pair_sims = np.clip(sim[iu], 0.0, 1.0)
        mean = float(pair_sims.mean())
        std = float(pair_sims.std(ddof=1)) if pair_sims.size > 1 else 0.0
        return ScoreWithCI(
            score=mean,
            std=std,
            n_samples=int(pair_sims.size),
            notes="mean pairwise cosine across claims",
        )
