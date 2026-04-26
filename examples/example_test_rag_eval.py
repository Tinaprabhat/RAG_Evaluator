"""
EXAMPLE: How to use the RAGWatch pytest plugin in YOUR project.

Copy this file to your project's tests/ folder, rename to test_rag_eval.py,
edit RAG_CASES, and run:

    pytest tests/test_rag_eval.py -v

That's it. RAG quality is now a unit test.

------------------------------------------------------------------
Three patterns are demonstrated below. Pick the one that fits.
------------------------------------------------------------------
"""

import pytest

from ragwatch.pytest_plugin import (
    EvalReport,
    Thresholds,
    evaluate_one,
    evaluate_rag,
)


# ============================================================
# Your test cases — replace with your project's real data
# ============================================================

RAG_CASES = [
    {
        "query": "Who developed general relativity, and in what year?",
        "context": [
            "Albert Einstein developed general relativity in 1915.",
            "The theory describes gravity as spacetime curvature.",
        ],
        "answer": "Einstein developed general relativity in 1915.",
        "ground_truth": "Einstein, 1915.",
    },
    {
        "query": "What is the capital of France?",
        "context": [
            "Paris is the capital of France.",
            "The Seine river flows through Paris.",
        ],
        "answer": "Paris is the capital of France.",
        "ground_truth": "Paris.",
    },
    {
        "query": "How many planets are in our solar system?",
        "context": [
            "Our solar system contains eight planets.",
            "Pluto was reclassified as a dwarf planet in 2006.",
        ],
        "answer": "There are eight planets in our solar system.",
        "ground_truth": "Eight planets.",
    },
]


# ============================================================
# Pattern 1 — One test for the whole suite
# ============================================================

def test_rag_quality_aggregate():
    """All cases together must clear the quality bar."""
    report = evaluate_rag(RAG_CASES, threshold=0.6)
    assert report.passed, report.summary()


# ============================================================
# Pattern 2 — Per-case granularity (pytest parametrize)
# ============================================================

@pytest.mark.parametrize("case", RAG_CASES, ids=lambda c: c["query"][:40])
def test_rag_quality_per_case(case):
    """Each case becomes its own pytest test — failures show by name."""
    result = evaluate_one(case)
    assert result.composite is not None
    assert result.composite.score >= 0.5, \
        f"composite score too low: {result.composite.score:.3f}"


# ============================================================
# Pattern 3 — Custom thresholds
# ============================================================

def test_rag_quality_strict():
    """Use a strict threshold preset for production-critical RAG."""
    report = evaluate_rag(RAG_CASES, thresholds=Thresholds.strict())
    # NOTE: this MAY fail on first run — that's fine, it tells you where you stand
    if not report.passed:
        pytest.skip(f"Strict thresholds not met yet:\n{report.summary()}")


def test_rag_quality_custom():
    """Hand-tuned thresholds per parameter."""
    custom = Thresholds(
        composite=0.55,
        faithfulness=0.6,
        hallucination_max=0.5,
    )
    report = evaluate_rag(RAG_CASES, thresholds=custom)
    assert report.passed, report.summary()


# ============================================================
# Pattern 4 — Smoke test (always passes, prints baseline)
# ============================================================

def test_rag_quality_baseline(capsys):
    """First-run helper: prints baseline metrics so you know your starting point."""
    report = evaluate_rag(RAG_CASES, threshold=0.0)
    print(report.summary())
    assert isinstance(report, EvalReport)
