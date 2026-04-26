"""Tests for input/output schemas."""

import pytest

from ragwatch.core.schemas import EvalInput, EvalResult, ScoreWithCI


class TestEvalInput:
    def test_valid_construction(self):
        inp = EvalInput(query="q", context=["c1"], answer="a")
        assert inp.query == "q"
        assert inp.context == ["c1"]
        assert inp.answer == "a"
        assert inp.ground_truth is None
        assert inp.metadata == {}

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError, match="query"):
            EvalInput(query="", context=["c"], answer="a")
        with pytest.raises(ValueError, match="query"):
            EvalInput(query="   ", context=["c"], answer="a")

    def test_non_list_context_rejected(self):
        with pytest.raises(ValueError, match="context"):
            EvalInput(query="q", context="not a list", answer="a")  # type: ignore[arg-type]

    def test_non_string_chunks_rejected(self):
        with pytest.raises(ValueError, match="context"):
            EvalInput(query="q", context=[1, 2, 3], answer="a")  # type: ignore[list-item]

    def test_empty_answer_rejected(self):
        with pytest.raises(ValueError, match="answer"):
            EvalInput(query="q", context=["c"], answer="")

    def test_empty_context_allowed(self):
        # legitimate edge case: pipeline returned no chunks
        inp = EvalInput(query="q", context=[], answer="a")
        assert inp.context == []


class TestScoreWithCI:
    def test_str_format(self):
        s = ScoreWithCI(score=0.8, std=0.1, n_samples=5)
        assert "0.800" in str(s)
        assert "0.100" in str(s)

    def test_to_dict_roundtrip(self):
        s = ScoreWithCI(score=0.5, std=0.05, n_samples=10, notes="hi")
        d = s.to_dict()
        assert d == {"score": 0.5, "std": 0.05, "n_samples": 10, "notes": "hi"}


class TestEvalResult:
    def test_default_empty(self):
        r = EvalResult()
        assert r.faithfulness is None
        assert r.composite is None
        assert r.flagged_claims == []
        assert r.engines_used == []
