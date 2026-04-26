"""RAGWatch — Framework-agnostic RAG evaluation, CPU-friendly, no LLM-as-judge.

v0.3 — Pytest-Native + Self-Trust + Quantized.
"""

__version__ = "0.3.0"
__author__ = "Tina"

from ragwatch.core.evaluator import Evaluator
from ragwatch.core.schemas import EvalInput, EvalResult

__all__ = ["Evaluator", "EvalInput", "EvalResult"]
