"""Tests for correlation, AUROC, calibration metrics."""

import math

from ragwatch.meta_eval.correlation import (
    auroc,
    expected_calibration_error,
    f1_at_threshold,
    pearson,
    spearman,
)


class TestPearson:
    def test_perfect_positive(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert math.isclose(pearson(x, y), 1.0, abs_tol=1e-9)

    def test_perfect_negative(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        assert math.isclose(pearson(x, y), -1.0, abs_tol=1e-9)

    def test_zero_variance(self):
        x = [1, 1, 1, 1]
        y = [2, 4, 6, 8]
        assert pearson(x, y) == 0.0

    def test_short_input(self):
        assert pearson([], []) == 0.0
        assert pearson([1.0], [1.0]) == 0.0


class TestSpearman:
    def test_monotonic_nonlinear(self):
        # exponential relationship → pearson < spearman
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 9, 16, 25]
        # both monotonic increasing → spearman should be 1.0
        assert math.isclose(spearman(x, y), 1.0, abs_tol=1e-9)

    def test_handles_ties(self):
        x = [1, 2, 2, 3]
        y = [1, 2, 2, 3]
        assert math.isclose(spearman(x, y), 1.0, abs_tol=1e-9)


class TestAUROC:
    def test_perfect_separation(self):
        scores = [0.9, 0.8, 0.7, 0.2, 0.1]
        labels = [1, 1, 1, 0, 0]
        assert math.isclose(auroc(scores, labels), 1.0, abs_tol=1e-9)

    def test_random(self):
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [1, 0, 1, 0]
        # all-equal scores with mixed labels → AUROC = 0.5
        assert math.isclose(auroc(scores, labels), 0.5, abs_tol=1e-9)

    def test_inverted(self):
        # lower score correlates with positive class → AUROC < 0.5
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [1, 1, 0, 0]
        assert auroc(scores, labels) < 0.5

    def test_no_positives(self):
        assert auroc([0.1, 0.2, 0.3], [0, 0, 0]) == 0.5

    def test_no_negatives(self):
        assert auroc([0.1, 0.2, 0.3], [1, 1, 1]) == 0.5


class TestECE:
    def test_perfect_calibration(self):
        # predictions exactly equal truths → ECE = 0
        probs = [0.1, 0.3, 0.5, 0.7, 0.9]
        truths = [0.1, 0.3, 0.5, 0.7, 0.9]
        d = expected_calibration_error(probs, truths, n_bins=5)
        assert d["ece"] < 1e-9

    def test_systematic_overconfidence(self):
        # predict 0.9 but truth is 0.5 → ECE = 0.4
        probs = [0.9, 0.9, 0.9, 0.9]
        truths = [0.5, 0.5, 0.5, 0.5]
        d = expected_calibration_error(probs, truths, n_bins=5)
        assert math.isclose(d["ece"], 0.4, abs_tol=1e-6)

    def test_returns_per_bin_data(self):
        d = expected_calibration_error([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8], n_bins=5)
        assert len(d["bin_means_pred"]) == 5
        assert len(d["bin_counts"]) == 5


class TestF1:
    def test_perfect(self):
        scores = [0.9, 0.8, 0.1, 0.2]
        labels = [1, 1, 0, 0]
        assert math.isclose(f1_at_threshold(scores, labels, 0.5), 1.0, abs_tol=1e-9)

    def test_no_predictions_above_threshold(self):
        scores = [0.1, 0.2, 0.3]
        labels = [1, 1, 1]
        # threshold=0.5 → no positives predicted → f1 = 0
        assert f1_at_threshold(scores, labels, 0.5) == 0.0
