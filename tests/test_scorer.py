"""Tests for the scorer (aggregate + composite + clamp01)."""

import math

from ragwatch.core.config import Config
from ragwatch.core.schemas import ScoreWithCI
from ragwatch.core.scorer import aggregate_score, clamp01, composite_score


class TestAggregateScore:
    def test_empty(self):
        s = aggregate_score([])
        assert s.score == 0.0
        assert s.std == 0.0
        assert s.n_samples == 0

    def test_single(self):
        s = aggregate_score([0.7])
        assert math.isclose(s.score, 0.7)
        assert s.std == 0.0
        assert s.n_samples == 1

    def test_multiple_mean_std(self):
        vals = [0.2, 0.4, 0.6, 0.8]
        s = aggregate_score(vals)
        assert math.isclose(s.score, 0.5, abs_tol=1e-9)
        # sample std (ddof=1) of [0.2,0.4,0.6,0.8] is sqrt(0.0666...)
        assert math.isclose(s.std, math.sqrt(0.0666666666), abs_tol=1e-3)
        assert s.n_samples == 4


class TestClamp:
    def test_within(self):
        assert clamp01(0.5) == 0.5

    def test_below(self):
        assert clamp01(-0.3) == 0.0

    def test_above(self):
        assert clamp01(1.7) == 1.0


class TestCompositeScore:
    def test_all_none(self):
        cfg = Config()
        c = composite_score({k: None for k in cfg.weights}, cfg)
        assert c.score == 0.0
        assert c.n_samples == 0

    def test_partial_scores(self):
        cfg = Config()
        scores = {
            "context_relevance": ScoreWithCI(0.8, 0.0, 1),
            "context_precision": None,
            "faithfulness": ScoreWithCI(0.9, 0.0, 1),
            "answer_relevance": None,
            "completeness": None,
            "correctness": None,
        }
        c = composite_score(scores, cfg)
        # weights: context_relevance=0.15, faithfulness=0.30 → total=0.45
        # weighted = 0.15*0.8 + 0.30*0.9 = 0.12 + 0.27 = 0.39
        # mean = 0.39 / 0.45 ≈ 0.866...
        assert math.isclose(c.score, 0.39 / 0.45, abs_tol=1e-6)
        assert c.n_samples == 2

    def test_uncertainty_propagates(self):
        cfg = Config()
        scores = {
            "context_relevance": ScoreWithCI(0.8, 0.1, 5),
            "context_precision": None,
            "faithfulness": ScoreWithCI(0.9, 0.2, 5),
            "answer_relevance": None,
            "completeness": None,
            "correctness": None,
        }
        c = composite_score(scores, cfg)
        # std = sqrt((0.15^2 * 0.1^2) + (0.30^2 * 0.2^2)) / (0.15+0.30)
        # = sqrt(0.000225 + 0.0036) / 0.45
        expected = math.sqrt(0.15**2 * 0.1**2 + 0.30**2 * 0.2**2) / 0.45
        assert math.isclose(c.std, expected, abs_tol=1e-6)
