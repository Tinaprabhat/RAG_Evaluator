"""
Tests for the pytest_plugin module — the v0.3 headline feature.

Uses a patched evaluator (DummyEmbedder + token-overlap fake NLI) so the entire
plugin flow runs offline without model downloads.
"""

import pytest

from tests._fixtures import DummyEmbedder
from ragwatch.core.config import Config
from ragwatch.core.evaluator import Evaluator
from ragwatch.core.schemas import EvalInput
from ragwatch.pytest_plugin import (
    EvalReport,
    Thresholds,
    case_from_dict,
    evaluate_one,
    evaluate_rag,
)
from ragwatch.pytest_plugin.api import reset_evaluator_cache, _get_evaluator


# ------------------------------------------------------------
# Test fixtures: patched evaluator (offline)
# ------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_plugin_evaluator(monkeypatch):
    """Replace the plugin's cached evaluator with a patched dummy-embedder one."""
    reset_evaluator_cache()

    cfg = Config.cpu_safe()
    cfg.use_ann_validator = False
    cfg.use_self_consistency = False

    ev = Evaluator(cfg)
    ev.embedder = DummyEmbedder()
    if ev.math is not None:
        ev.math.embedder = ev.embedder
    if ev.nli is not None:
        ev.nli.embedder = ev.embedder

        def fake_entail(p, h):
            ps = set(p.lower().split())
            hs = set(h.lower().split())
            if not hs:
                return 0.5
            return float(min(1.0, len(ps & hs) / len(hs)))

        ev.nli._entailment_score = fake_entail  # type: ignore[method-assign]

    # inject our patched evaluator into the plugin's cache
    from ragwatch.pytest_plugin import api
    api._EVALUATOR_CACHE["cpu_safe"] = ev
    yield
    reset_evaluator_cache()


# ------------------------------------------------------------
# Sample data
# ------------------------------------------------------------

GOOD_CASE = {
    "query": "Who developed general relativity?",
    "context": ["Albert Einstein developed general relativity in 1915."],
    "answer": "Albert Einstein developed general relativity in 1915.",
    "ground_truth": "Einstein, 1915.",
}

BAD_CASE = {
    "query": "Who developed general relativity?",
    "context": ["Albert Einstein developed general relativity in 1915."],
    "answer": "Marie Curie developed general relativity in 1900.",
    "ground_truth": "Einstein, 1915.",
}


# ------------------------------------------------------------
# case_from_dict
# ------------------------------------------------------------

class TestCaseFromDict:
    def test_basic_dict(self):
        inp = case_from_dict(GOOD_CASE)
        assert isinstance(inp, EvalInput)
        assert inp.query.startswith("Who developed")
        assert inp.context == ["Albert Einstein developed general relativity in 1915."]

    def test_alternative_field_names(self):
        d = {
            "question": "Who?",
            "chunks": ["A chunk."],
            "response": "An answer here.",
            "reference": "The truth.",
        }
        inp = case_from_dict(d)
        assert inp.query == "Who?"
        assert inp.answer == "An answer here."
        assert inp.ground_truth == "The truth."

    def test_string_context_promoted_to_list(self):
        d = {"query": "Q", "context": "single string", "answer": "A"}
        inp = case_from_dict(d)
        assert inp.context == ["single string"]

    def test_metadata_preserved(self):
        d = {**GOOD_CASE, "metadata": {"source": "test"}}
        inp = case_from_dict(d)
        assert inp.metadata == {"source": "test"}

    def test_rejects_non_mapping(self):
        with pytest.raises(TypeError):
            case_from_dict("not a dict")  # type: ignore[arg-type]


# ------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------

class TestThresholds:
    def test_default_thresholds_set(self):
        th = Thresholds()
        assert th.composite == 0.7
        assert th.faithfulness == 0.6
        assert th.hallucination_max == 0.4

    def test_strict_preset(self):
        th = Thresholds.strict()
        assert th.composite == 0.80
        assert th.faithfulness == 0.75
        assert th.hallucination_max == 0.25

    def test_permissive_preset(self):
        th = Thresholds.permissive()
        assert th.composite == 0.5


# ------------------------------------------------------------
# evaluate_one
# ------------------------------------------------------------

class TestEvaluateOne:
    def test_accepts_dict(self):
        r = evaluate_one(GOOD_CASE)
        assert r.composite is not None
        assert 0.0 <= r.composite.score <= 1.0

    def test_accepts_eval_input(self):
        r = evaluate_one(case_from_dict(GOOD_CASE))
        assert r.composite is not None

    def test_distinguishes_good_from_bad(self):
        good_r = evaluate_one(GOOD_CASE)
        bad_r = evaluate_one(BAD_CASE)
        # good should at least match bad on faithfulness
        assert good_r.faithfulness.score >= bad_r.faithfulness.score


# ------------------------------------------------------------
# evaluate_rag (the headline feature)
# ------------------------------------------------------------

class TestEvaluateRag:
    def test_returns_report(self):
        report = evaluate_rag([GOOD_CASE])
        assert isinstance(report, EvalReport)
        assert report.n_cases == 1

    def test_empty_cases_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            evaluate_rag([])

    def test_non_list_rejected(self):
        with pytest.raises(TypeError):
            evaluate_rag(GOOD_CASE)  # type: ignore[arg-type]

    def test_passed_field_set(self):
        report = evaluate_rag([GOOD_CASE], threshold=0.0)  # any score >= 0 passes
        assert report.passed is True

    def test_failures_populated_on_low_threshold(self):
        report = evaluate_rag([GOOD_CASE, BAD_CASE], threshold=0.99)
        # threshold=0.99 is unreachable → both should fail
        assert report.passed is False
        assert len(report.failures) >= 1

    def test_summary_contains_status(self):
        report = evaluate_rag([GOOD_CASE], threshold=0.0)
        s = report.summary()
        assert "PASSED" in s

    def test_summary_contains_failures(self):
        report = evaluate_rag([GOOD_CASE], threshold=0.99)
        s = report.summary()
        assert "FAILED" in s

    def test_aggregate_metrics_present(self):
        report = evaluate_rag([GOOD_CASE, BAD_CASE], threshold=0.0)
        assert 0.0 <= report.composite_mean <= 1.0
        assert 0.0 <= report.faithfulness_mean <= 1.0
        assert 0.0 <= report.hallucination_mean <= 1.0
        # std present when n>=2
        assert report.composite_std >= 0.0

    def test_per_case_results_preserved(self):
        cases = [GOOD_CASE, BAD_CASE, GOOD_CASE]
        report = evaluate_rag(cases, threshold=0.0)
        assert len(report.per_case) == 3
        assert len(report.case_labels) == 3

    def test_custom_thresholds(self):
        th = Thresholds(composite=None, faithfulness=0.99, hallucination_max=None)
        report = evaluate_rag([BAD_CASE], thresholds=th)
        # bad case faithfulness will be < 0.99 → should fail
        assert report.passed is False

    def test_strict_thresholds_preset(self):
        report = evaluate_rag([BAD_CASE], thresholds=Thresholds.strict())
        # bad case should not survive strict thresholds
        assert report.passed is False

    def test_bool_dunder(self):
        passing = evaluate_rag([GOOD_CASE], threshold=0.0)
        assert bool(passing) is True
        failing = evaluate_rag([GOOD_CASE], threshold=0.99)
        assert bool(failing) is False


# ------------------------------------------------------------
# Real pytest workflow simulation
# ------------------------------------------------------------

class TestPytestWorkflow:
    """Simulates how a user would actually call the plugin in a test."""

    def test_simple_assertion_pattern(self):
        """The canonical: `assert report.passed, report.summary()` pattern."""
        cases = [GOOD_CASE]
        report = evaluate_rag(cases, threshold=0.0)
        # this is what the user writes:
        assert report.passed, report.summary()

    @pytest.mark.parametrize("case", [GOOD_CASE], ids=lambda c: c["query"][:20])
    def test_parametrize_pattern(self, case):
        """The per-case granularity: each case = its own pytest test."""
        result = evaluate_one(case)
        assert result.composite is not None
        assert 0.0 <= result.composite.score <= 1.0


# ------------------------------------------------------------
# Cache behavior
# ------------------------------------------------------------

class TestEvaluatorCache:
    def test_reset_cache(self):
        from ragwatch.pytest_plugin import api
        # cache has at least our patched entry from autouse fixture
        assert "cpu_safe" in api._EVALUATOR_CACHE
        reset_evaluator_cache()
        assert "cpu_safe" not in api._EVALUATOR_CACHE

    def test_unknown_mode_rejected(self):
        from ragwatch.pytest_plugin import api
        api._EVALUATOR_CACHE.pop("bogus", None)
        with pytest.raises(ValueError, match="unknown mode"):
            _get_evaluator(mode="bogus")
