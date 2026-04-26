"""
Schemas — strict input/output contracts.

Pipeline-agnostic interface. The evaluator only needs:
    - query:        what the user asked
    - context:      list of retrieved chunks
    - answer:       what the LLM generated
    - ground_truth: (optional) reference answer
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class EvalInput:
    """A single (query, context, answer, ground_truth?) tuple."""
    query: str
    context: list[str]
    answer: str
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(self.context, list) or not all(isinstance(c, str) for c in self.context):
            raise ValueError("context must be a list of strings")
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValueError("answer must be a non-empty string")


@dataclass
class ScoreWithCI:
    """A score with its confidence interval (mean ± std)."""
    score: float
    std: float = 0.0
    n_samples: int = 1
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.score:.3f} ± {self.std:.3f}"


@dataclass
class EvalResult:
    """Full evaluation output for one input."""
    # Retrieval-layer scores
    context_relevance: ScoreWithCI | None = None
    context_precision: ScoreWithCI | None = None
    context_redundancy: ScoreWithCI | None = None

    # Generation-layer scores
    faithfulness: ScoreWithCI | None = None
    answer_relevance: ScoreWithCI | None = None
    completeness: ScoreWithCI | None = None
    hallucination_score: ScoreWithCI | None = None

    # End-to-end
    correctness: ScoreWithCI | None = None  # only if ground_truth given
    self_consistency: ScoreWithCI | None = None
    composite: ScoreWithCI | None = None

    # Diagnostics
    n_claims: int = 0
    n_atomic_claims: int = 0
    flagged_claims: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    engines_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, val in asdict(self).items():
            if isinstance(val, dict) and "score" in val:
                out[key] = val
            else:
                out[key] = val
        return out
