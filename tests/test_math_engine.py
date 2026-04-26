"""Tests for MathEngine — deterministic similarity-based scoring."""

import pytest

from ragwatch.engines.math_engine import MathEngine


@pytest.fixture
def math_engine(dummy_embedder):
    return MathEngine(dummy_embedder, redundancy_threshold=0.85)


class TestContextRelevance:
    def test_empty_context(self, math_engine):
        from ragwatch.core.schemas import EvalInput
        inp = EvalInput(query="q", context=[], answer="a")
        s = math_engine.context_relevance(inp)
        assert s.score == 0.0
        assert s.n_samples == 0

    def test_returns_in_range(self, math_engine, good_input):
        s = math_engine.context_relevance(good_input)
        assert 0.0 <= s.score <= 1.0

    def test_n_samples_equals_chunks(self, math_engine, good_input):
        s = math_engine.context_relevance(good_input)
        assert s.n_samples == len(good_input.context)


class TestContextPrecision:
    def test_in_range(self, math_engine, good_input):
        s = math_engine.context_precision(good_input)
        assert 0.0 <= s.score <= 1.0

    def test_empty_context_handled(self, math_engine):
        from ragwatch.core.schemas import EvalInput
        inp = EvalInput(query="q", context=[], answer="a")
        s = math_engine.context_precision(inp)
        assert s.score == 0.0


class TestContextRedundancy:
    def test_single_chunk_perfect_diversity(self, math_engine):
        from ragwatch.core.schemas import EvalInput
        inp = EvalInput(query="q", context=["only one"], answer="a")
        s = math_engine.context_redundancy(inp)
        assert s.score == 1.0  # no redundancy possible

    def test_diverse_chunks(self, math_engine):
        from ragwatch.core.schemas import EvalInput
        inp = EvalInput(
            query="q",
            context=[
                "Apples grow on trees in orchards",
                "Quantum mechanics describes subatomic particles",
                "The Eiffel Tower is in Paris France",
            ],
            answer="a",
        )
        s = math_engine.context_redundancy(inp)
        # very different topics → low redundancy → high diversity score
        assert s.score >= 0.5

    def test_duplicate_chunks_low_diversity(self, math_engine):
        from ragwatch.core.schemas import EvalInput
        inp = EvalInput(
            query="q",
            context=[
                "Einstein developed relativity in 1915",
                "Einstein developed relativity in 1915",
                "Einstein developed relativity in 1915",
            ],
            answer="a",
        )
        s = math_engine.context_redundancy(inp)
        # identical chunks → high redundancy → low diversity
        assert s.score <= 0.5


class TestAnswerRelevance:
    def test_in_range(self, math_engine, good_input):
        s = math_engine.answer_relevance(good_input)
        assert 0.0 <= s.score <= 1.0

    def test_relevant_higher_than_irrelevant(self, math_engine, dummy_embedder):
        from ragwatch.core.schemas import EvalInput
        relevant = EvalInput(
            query="What is the capital of France?",
            context=["Paris is the capital."],
            answer="Paris is the capital of France.",
        )
        irrelevant = EvalInput(
            query="What is the capital of France?",
            context=["Paris is the capital."],
            answer="Bananas grow on trees in tropical climates.",
        )
        s_rel = math_engine.answer_relevance(relevant)
        s_irr = math_engine.answer_relevance(irrelevant)
        assert s_rel.score > s_irr.score


class TestCorrectness:
    def test_none_when_no_ground_truth(self, math_engine):
        from ragwatch.core.schemas import EvalInput
        inp = EvalInput(query="q", context=["c"], answer="a")
        assert math_engine.correctness(inp) is None

    def test_returns_score_when_gt_present(self, math_engine, good_input):
        s = math_engine.correctness(good_input)
        assert s is not None
        assert 0.0 <= s.score <= 1.0
