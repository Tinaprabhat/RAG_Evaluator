"""
Statistical correlation and calibration metrics — pure numpy.

Used by the meta-evaluator to measure RAGWatch's agreement with ground truth.

Functions:
    pearson(x, y)  → linear correlation
    spearman(x, y) → rank correlation (robust to monotonic but non-linear)
    auroc(scores, labels) → area under ROC curve (separation power)
    expected_calibration_error(probs, labels, n_bins) → calibration quality
"""

from __future__ import annotations
import math

import numpy as np


def pearson(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """
    Pearson linear correlation coefficient in [-1, 1].
    Returns 0.0 if either array has zero variance (undefined).
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return 0.0
    sa = a.std()
    sb = b.std()
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average rank, ties resolved by mean — same convention as scipy.stats.rankdata."""
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-indexed average
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Spearman rank correlation in [-1, 1]."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return 0.0
    return pearson(_rankdata(a), _rankdata(b))


def auroc(scores: list[float] | np.ndarray, labels: list[int] | np.ndarray) -> float:
    """
    Area Under the ROC Curve.
    `labels` should be 0/1 (negative/positive). `scores` are predicted scores
    where higher = more positive.

    Returns 0.5 if there are no positives or no negatives.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0 or s.size != y.size:
        return 0.5

    # Mann-Whitney U formulation
    ranks = _rankdata(s)
    sum_pos_ranks = float(ranks[y == 1].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def expected_calibration_error(
    probs: list[float] | np.ndarray,
    truths: list[float] | np.ndarray,
    n_bins: int = 10,
) -> dict[str, float | list]:
    """
    ECE for continuous targets:
    Bin predictions into `n_bins` equal-width bins on [0, 1].
    For each bin, compare mean prediction to mean truth in that bin.
    Lower ECE = better calibrated.

    Returns:
        {
          "ece":             float,  # weighted average gap across bins
          "max_gap":         float,  # max bin gap
          "bin_means_pred":  [...],  # per-bin mean predicted score
          "bin_means_true":  [...],  # per-bin mean true score
          "bin_counts":      [...],  # per-bin count
        }
    """
    p = np.asarray(probs, dtype=np.float64)
    t = np.asarray(truths, dtype=np.float64)
    n = p.size
    if n == 0 or n != t.size:
        return {"ece": 0.0, "max_gap": 0.0, "bin_means_pred": [], "bin_means_true": [], "bin_counts": []}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    max_gap = 0.0
    bin_means_pred: list[float] = []
    bin_means_true: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # include upper edge in the last bin
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        bin_counts.append(count)
        if count == 0:
            bin_means_pred.append(0.0)
            bin_means_true.append(0.0)
            continue
        mean_pred = float(p[mask].mean())
        mean_true = float(t[mask].mean())
        bin_means_pred.append(mean_pred)
        bin_means_true.append(mean_true)
        gap = abs(mean_pred - mean_true)
        ece += (count / n) * gap
        max_gap = max(max_gap, gap)

    return {
        "ece": float(ece),
        "max_gap": float(max_gap),
        "bin_means_pred": bin_means_pred,
        "bin_means_true": bin_means_true,
        "bin_counts": bin_counts,
    }


def f1_at_threshold(
    scores: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    threshold: float = 0.5,
) -> float:
    """F1 score for predictions at a fixed threshold."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if s.size != y.size or s.size == 0:
        return 0.0
    pred = (s >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
