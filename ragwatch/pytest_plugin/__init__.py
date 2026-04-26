"""
RAGWatch pytest plugin — RAG evaluation as unit tests.

Public API:
    evaluate_rag(cases, threshold=0.7, mode='cpu_safe') -> EvalReport
    evaluate_one(case) -> EvalResult
    EvalReport — assertion-friendly result container

Drop one file in your project's tests/ folder:

    # tests/test_rag_eval.py
    from ragwatch.pytest_plugin import evaluate_rag

    RAG_CASES = [
        {"query": "...", "context": [...], "answer": "...", "ground_truth": "..."},
    ]

    def test_rag_quality():
        report = evaluate_rag(RAG_CASES, threshold=0.7)
        assert report.passed, report.summary()

That's it. RAG quality is now a unit test.
"""

from ragwatch.pytest_plugin.api import (
    evaluate_one,
    evaluate_rag,
    EvalReport,
    Thresholds,
    case_from_dict,
)

__all__ = [
    "evaluate_one",
    "evaluate_rag",
    "EvalReport",
    "Thresholds",
    "case_from_dict",
]
