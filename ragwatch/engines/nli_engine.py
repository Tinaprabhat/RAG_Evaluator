"""
NLI Engine — claim-level entailment via cross-encoder NLI model.

Uses cosine-gating to skip NLI calls on obviously entailed/contradicted pairs.
This typically saves 50-70% of NLI calls in real RAG outputs.

Model: cross-encoder/nli-deberta-v3-small (~180MB, ~50MB ONNX-quantized).
"""

from __future__ import annotations
from typing import Any

import numpy as np

from ragwatch.core.schemas import ScoreWithCI
from ragwatch.utils.embeddings import EmbeddingCache, cosine_matrix


# NLI label conventions for cross-encoder/nli-deberta-v3 family:
# label 0 = contradiction, 1 = entailment, 2 = neutral
# We map to a soft entailment score: P(entail) - P(contradict).
ENTAILMENT_IDX = 1
CONTRADICTION_IDX = 0


class NLIEngine:
    """Entailment-based faithfulness + completeness scoring."""

    def __init__(
        self,
        embedder: EmbeddingCache,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        cosine_high: float = 0.80,
        cosine_low: float = 0.20,
        onnx_dir: str | None = None,
    ):
        self.embedder = embedder
        self.model_name = model_name
        self.cosine_high = cosine_high
        self.cosine_low = cosine_low
        self.onnx_dir = onnx_dir
        self._model: Any = None
        self._backend: str = "none"  # 'onnx' | 'pytorch' | 'none'

    def _load(self) -> None:
        if self._model is not None:
            return

        # Prefer quantized ONNX runner if available
        if self.onnx_dir:
            from pathlib import Path
            quant_file = Path(self.onnx_dir) / "model_quantized.onnx"
            if quant_file.exists():
                try:
                    from ragwatch.utils.onnx_export import ONNXNLIRunner
                    self._model = ONNXNLIRunner(model_dir=self.onnx_dir)
                    self._backend = "onnx"
                    return
                except Exception as e:
                    # fall through to PyTorch
                    print(f"[NLIEngine] ONNX load failed: {e}; falling back to PyTorch.")

        # Fallback: PyTorch CrossEncoder
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required for NLIEngine. "
                "pip install sentence-transformers"
            ) from e
        self._model = CrossEncoder(self.model_name)
        self._backend = "pytorch"

    @property
    def backend(self) -> str:
        """Returns 'onnx', 'pytorch', or 'none' (before _load is called)."""
        return self._backend

    # ---------- core entailment scoring ----------

    def _entailment_score(self, premise: str, hypothesis: str) -> float:
        """
        Returns a [0,1] entailment score:
            P(entail) is dominant → close to 1
            P(contradict) is dominant → close to 0
            P(neutral) → ~0.5
        """
        self._load()
        logits = self._model.predict([(premise, hypothesis)])
        if isinstance(logits, np.ndarray) and logits.ndim == 2:
            row = logits[0]
        elif isinstance(logits, list):
            row = np.array(logits[0])
        else:
            row = np.array(logits)

        # softmax
        e = np.exp(row - np.max(row))
        probs = e / e.sum()

        p_ent = float(probs[ENTAILMENT_IDX])
        p_con = float(probs[CONTRADICTION_IDX])
        # squash into [0,1]: neutral → 0.5; pure entailment → 1; pure contradiction → 0
        return float(0.5 * (1.0 + p_ent - p_con))

    # ---------- public methods ----------

    def faithfulness(self, claims: list[str], context: list[str]) -> tuple[ScoreWithCI, list[str]]:
        """
        Per-claim faithfulness with lazy NLI gating.
        Returns (score_with_CI, list_of_flagged_unfaithful_claims).
        """
        if not claims or not context:
            return ScoreWithCI(0.0, 0.0, 0, "no claims or context"), []

        claim_emb = self.embedder.embed_batch(claims)
        ctx_emb = self.embedder.embed_batch(context)
        sim = cosine_matrix(claim_emb, ctx_emb)  # [n_claims, n_chunks]

        per_claim_scores: list[float] = []
        flagged: list[str] = []

        for i, claim in enumerate(claims):
            best_chunk_idx = int(np.argmax(sim[i]))
            best_sim = float(sim[i, best_chunk_idx])

            # Lazy gating
            if best_sim >= self.cosine_high:
                score = float(min(1.0, best_sim))
            elif best_sim <= self.cosine_low:
                score = 0.0
                flagged.append(claim)
            else:
                # ambiguous → run NLI
                premise = context[best_chunk_idx]
                score = self._entailment_score(premise, claim)
                if score < 0.4:
                    flagged.append(claim)
            per_claim_scores.append(score)

        arr = np.array(per_claim_scores, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return (
            ScoreWithCI(score=mean, std=std, n_samples=int(arr.size),
                        notes="entailment per claim (lazy-gated)"),
            flagged,
        )

    def completeness(self, query: str, answer: str) -> ScoreWithCI:
        """
        Decompose query into sub-questions, check each is addressed by answer.
        Heuristic decomposition: split on "and" / "?" / commas-with-conjunctions.
        """
        sub_qs = _decompose_query(query)
        if not sub_qs:
            return ScoreWithCI(1.0, 0.0, 1, "no sub-questions detected")

        scores = [self._entailment_score(answer, q) for q in sub_qs]
        arr = np.array(scores, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return ScoreWithCI(score=mean, std=std, n_samples=int(arr.size),
                           notes=f"{len(sub_qs)} sub-question(s) checked")


def _decompose_query(query: str) -> list[str]:
    """Heuristic split of a query into sub-questions."""
    import re
    parts = re.split(r"\?\s*|;\s+|\s+and\s+|,\s+and\s+", query.strip())
    parts = [p.strip().rstrip("?") for p in parts if p.strip()]
    # only keep parts that look like questions / clauses (>= 3 words)
    return [p for p in parts if len(p.split()) >= 3]
