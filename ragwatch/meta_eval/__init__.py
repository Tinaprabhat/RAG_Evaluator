"""
Meta-evaluation — answers the question "Who evaluates the evaluator?".

Components:
    - synthetic_cases: hand-crafted RAG cases with known ground-truth scores
    - meta_evaluator:  runs RAGWatch on the cases, compares outputs to truth
    - correlation:     Pearson, Spearman, AUROC, calibration error
    - report:          generates a self-trust report
"""

from ragwatch.meta_eval.synthetic_cases import (
    LabeledCase,
    get_synthetic_cases,
    get_known_good_cases,
    get_known_bad_cases,
)
from ragwatch.meta_eval.correlation import (
    pearson,
    spearman,
    auroc,
    expected_calibration_error,
)
from ragwatch.meta_eval.meta_evaluator import (
    MetaEvaluator,
    MetaResult,
)

__all__ = [
    "LabeledCase",
    "get_synthetic_cases",
    "get_known_good_cases",
    "get_known_bad_cases",
    "pearson",
    "spearman",
    "auroc",
    "expected_calibration_error",
    "MetaEvaluator",
    "MetaResult",
]
