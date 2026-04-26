"""
End-to-end integration tests for the Evaluator.

We monkey-patch:
  - the EmbeddingCache class to use the DummyEmbedder
  - the NLI _entailment_score method
…so the full pipeline runs offline with no model downloads.
A separate `test_integration_real_models.py` (optional) can be added by users
who want to verify against the real models — kept out of the default suite.
"""

import pytest

from ragwatch.core.config import Config
from ragwatch.core.evaluator import Evaluator
from ragwatch.core.schemas import EvalInput, EvalResult


def _patch_evaluator(evaluator):
    """Replace heavy components with deterministic stand-ins."""
    from tests._fixtures import DummyEmbedder

    evaluator.embedder = DummyEmbedder()
    if evaluator.math is not None:
        evaluator.math.embedder = evaluator.embedder
    if evaluator.nli is not None:
        evaluator.nli.embedder = evaluator.embedder

        def fake_entail(premise, hypothesis):
            p = set(premise.lower().split())
            h = set(hypothesis.lower().split())
            if not h:
                return 0.5
            return float(min(1.0, len(p & h) / len(h)))

        evaluator.nli._entailment_score = fake_entail  # type: ignore[method-assign]
    if evaluator.ann is not None:
        evaluator.ann.embedder = evaluator.embedder
    if evaluator.consistency is not None:
        evaluator.consistency.embedder = evaluator.embedder
    return evaluator


@pytest.fixture
def evaluator():
    cfg = Config.cpu_safe()
    cfg.use_ann_validator = False  # ANN needs trained weights for fair eval
    cfg.use_logit_uncertainty = False
    cfg.use_self_consistency = True
    return _patch_evaluator(Evaluator(cfg))


class TestEvaluatorEndToEnd:
    def test_returns_eval_result(self, evaluator, good_input):
        r = evaluator.evaluate(good_input)
        assert isinstance(r, EvalResult)

    def test_engines_recorded(self, evaluator, good_input):
        r = evaluator.evaluate(good_input)
        assert "math" in r.engines_used
        assert "nli" in r.engines_used
        assert "self_consistency" in r.engines_used

    def test_composite_present(self, evaluator, good_input):
        r = evaluator.evaluate(good_input)
        assert r.composite is not None
        assert 0.0 <= r.composite.score <= 1.0

    def test_latency_recorded(self, evaluator, good_input):
        r = evaluator.evaluate(good_input)
        assert r.latency_seconds > 0

    def test_n_claims_counted(self, evaluator, good_input):
        r = evaluator.evaluate(good_input)
        assert r.n_claims >= 1

    def test_correctness_present_with_ground_truth(self, evaluator, good_input):
        r = evaluator.evaluate(good_input)
        assert r.correctness is not None

    def test_correctness_absent_without_ground_truth(self, evaluator):
        inp = EvalInput(
            query="What is X?",
            context=["X is an example."],
            answer="X is an example.",
        )
        r = _patch_evaluator(Evaluator(Config.cpu_safe())).evaluate(inp)
        assert r.correctness is None

    def test_hallucinated_input_lower_faithfulness(self, evaluator, good_input, hallucinated_input):
        r_good = evaluator.evaluate(good_input)
        r_bad = evaluator.evaluate(hallucinated_input)
        # the hallucinated answer (nine planets w/ Pluto) should score lower on faithfulness
        # than the well-supported answer
        assert r_good.faithfulness is not None and r_bad.faithfulness is not None
        assert r_good.faithfulness.score >= r_bad.faithfulness.score

    def test_batch_evaluation(self, evaluator, good_input, hallucinated_input):
        results = evaluator.evaluate_batch([good_input, hallucinated_input])
        assert len(results) == 2
        assert all(isinstance(r, EvalResult) for r in results)


class TestEdgeCases:
    def test_empty_context(self, evaluator):
        inp = EvalInput(query="Q", context=[], answer="An answer.")
        r = evaluator.evaluate(inp)
        assert r is not None
        # context-relevance n_samples should be 0
        assert r.context_relevance is not None and r.context_relevance.n_samples == 0

    def test_single_word_answer(self, evaluator):
        inp = EvalInput(
            query="Capital of France?",
            context=["Paris is the capital of France."],
            answer="Paris",
        )
        r = evaluator.evaluate(inp)
        # should not crash; should still produce a composite
        assert r.composite is not None


class TestReports:
    def test_console_report_format(self, evaluator, good_input):
        from ragwatch.utils.reports import to_console
        r = evaluator.evaluate(good_input)
        out = to_console(r)
        assert "RAGWatch Evaluation Result" in out
        assert "COMPOSITE" in out

    def test_json_report_writeable(self, evaluator, good_input, tmp_path):
        from ragwatch.utils.reports import to_json
        r = evaluator.evaluate(good_input)
        out = tmp_path / "report.json"
        to_json([r], str(out))
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "composite" in content

    def test_html_report_writeable(self, evaluator, good_input, tmp_path):
        from ragwatch.utils.reports import to_html
        r = evaluator.evaluate(good_input)
        out = tmp_path / "report.html"
        to_html([r], str(out), queries=[good_input.query])
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "RAGWatch" in content
        assert "<table>" in content
