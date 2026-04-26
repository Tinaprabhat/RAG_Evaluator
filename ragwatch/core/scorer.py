"""
Scorer — aggregates engine outputs into final per-parameter and composite scores.

Confidence intervals are derived from variance across evidence units (claims, chunks).
"""

from __future__ import annotations
import math
from ragwatch.core.schemas import ScoreWithCI
from ragwatch.core.config import Config


def aggregate_score(values: list[float], notes: str = "") -> ScoreWithCI:
    """
    Aggregate a list of per-unit scores into mean ± std.

    For example:
        - Faithfulness: per-claim entailment scores → mean = faithfulness, std = uncertainty
        - Context relevance: per-chunk similarity → mean = relevance
    """
    if not values:
        return ScoreWithCI(score=0.0, std=0.0, n_samples=0, notes=notes or "no evidence")

    n = len(values)
    mean = sum(values) / n
    if n == 1:
        std = 0.0
    else:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)

    return ScoreWithCI(
        score=float(mean),
        std=float(std),
        n_samples=n,
        notes=notes,
    )


def composite_score(
    scores: dict[str, ScoreWithCI | None],
    cfg: Config,
) -> ScoreWithCI:
    """
    Weighted composite of per-parameter scores.
    Skips parameters that are None (engine disabled or not applicable).
    Confidence: propagated as sqrt(sum(w^2 * std^2)).
    """
    total_weight = 0.0
    weighted_sum = 0.0
    var_sum = 0.0
    used = []

    for key, weight in cfg.weights.items():
        sc = scores.get(key)
        if sc is None:
            continue
        weighted_sum += weight * sc.score
        var_sum += (weight ** 2) * (sc.std ** 2)
        total_weight += weight
        used.append(key)

    if total_weight == 0:
        return ScoreWithCI(0.0, 0.0, 0, "no scores available")

    mean = weighted_sum / total_weight
    std = math.sqrt(var_sum) / total_weight if total_weight > 0 else 0.0

    return ScoreWithCI(
        score=float(mean),
        std=float(std),
        n_samples=len(used),
        notes=f"composed from: {', '.join(used)}",
    )


def clamp01(x: float) -> float:
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, float(x)))
