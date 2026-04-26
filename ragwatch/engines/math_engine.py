"""
MathEngine — pure deterministic scoring.

No models, no LLMs. Just numpy + cosine similarity.
Handles:
    - context_relevance (query ↔ chunks)
    - context_precision (which chunks the answer actually used)
    - context_redundancy (chunk-pair overlap)
    - answer_relevance (answer ↔ query)
    - correctness (answer ↔ ground_truth, if provided)
"""

from __future__ import annotations
import numpy as np

from ragwatch.core.schemas import EvalInput, ScoreWithCI
from ragwatch.utils.embeddings import EmbeddingCache, cosine_matrix


class MathEngine:
    """Deterministic similarity-based scoring engine."""

    def __init__(self, embedder: EmbeddingCache, redundancy_threshold: float = 0.85):
        self.embedder = embedder
        self.redundancy_threshold = redundancy_threshold

    # ---------- public scoring methods ----------

    def context_relevance(self, inp: EvalInput) -> ScoreWithCI:
        """Mean cosine(query, chunk_i) across all retrieved chunks."""
        if not inp.context:
            return ScoreWithCI(0.0, 0.0, 0, "no context")
        q = self.embedder.embed_batch([inp.query])
        C = self.embedder.embed_batch(inp.context)
        sims = cosine_matrix(q, C)[0]
        sims = np.clip(sims, 0.0, 1.0)
        return _agg(sims.tolist(), notes="cosine(query, chunk)")

    def context_precision(self, inp: EvalInput) -> ScoreWithCI:
        """
        Approximation of context-precision:
        How much of each retrieved chunk is actually reflected in the answer?
        cosine(answer, chunk_i) — averaged.
        """
        if not inp.context:
            return ScoreWithCI(0.0, 0.0, 0, "no context")
        a = self.embedder.embed_batch([inp.answer])
        C = self.embedder.embed_batch(inp.context)
        sims = cosine_matrix(a, C)[0]
        sims = np.clip(sims, 0.0, 1.0)
        return _agg(sims.tolist(), notes="cosine(answer, chunk)")

    def context_redundancy(self, inp: EvalInput) -> ScoreWithCI:
        """
        Fraction of chunk-pairs whose pairwise similarity exceeds threshold.
        Lower = better diversity. Returns 1 - redundancy so higher = better.
        """
        n = len(inp.context)
        if n < 2:
            return ScoreWithCI(1.0, 0.0, 0, "single or no chunk; no redundancy possible")

        C = self.embedder.embed_batch(inp.context)
        sim = cosine_matrix(C, C)
        # take strict upper triangle
        iu = np.triu_indices(n, k=1)
        pair_sims = sim[iu]

        n_redundant = int(np.sum(pair_sims > self.redundancy_threshold))
        n_pairs = len(pair_sims)
        redundancy = n_redundant / n_pairs if n_pairs > 0 else 0.0
        # invert so 1 = no redundancy, 0 = fully redundant
        diversity = 1.0 - redundancy

        # std across pair similarities gives a CI
        std = float(np.std(pair_sims)) if n_pairs > 1 else 0.0
        return ScoreWithCI(
            score=float(diversity),
            std=std,
            n_samples=n_pairs,
            notes=f"{n_redundant}/{n_pairs} pairs above {self.redundancy_threshold} similarity",
        )

    def answer_relevance(self, inp: EvalInput) -> ScoreWithCI:
        """Cosine similarity between answer and query."""
        q = self.embedder.embed_batch([inp.query])[0]
        a = self.embedder.embed_batch([inp.answer])[0]
        from ragwatch.utils.embeddings import cosine
        sim = max(0.0, cosine(q, a))
        return ScoreWithCI(score=float(sim), std=0.0, n_samples=1, notes="cosine(query, answer)")

    def correctness(self, inp: EvalInput) -> ScoreWithCI | None:
        """Cosine(answer, ground_truth). Returns None if no ground truth."""
        if not inp.ground_truth:
            return None
        a = self.embedder.embed_batch([inp.answer])[0]
        g = self.embedder.embed_batch([inp.ground_truth])[0]
        from ragwatch.utils.embeddings import cosine
        sim = max(0.0, cosine(a, g))
        return ScoreWithCI(score=float(sim), std=0.0, n_samples=1, notes="cosine(answer, ground_truth)")


# ---------- internals ----------

def _agg(values: list[float], notes: str = "") -> ScoreWithCI:
    """Local aggregator (kept here to avoid circular import)."""
    if not values:
        return ScoreWithCI(0.0, 0.0, 0, notes or "empty")
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return ScoreWithCI(score=mean, std=std, n_samples=int(arr.size), notes=notes)
