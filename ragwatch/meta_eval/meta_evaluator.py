"""
MetaEvaluator — runs RAGWatch on labeled cases and computes agreement metrics.

This is the answer to "Who evaluates the evaluator?".
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from ragwatch.core.config import Config
from ragwatch.core.evaluator import Evaluator
from ragwatch.core.schemas import EvalResult
from ragwatch.meta_eval.correlation import (
    auroc,
    expected_calibration_error,
    f1_at_threshold,
    pearson,
    spearman,
)
from ragwatch.meta_eval.synthetic_cases import LabeledCase, get_synthetic_cases


@dataclass
class MetaResult:
    """Output of meta-evaluation: how well RAGWatch agrees with ground truth."""

    n_cases: int
    n_engines_used: list[str]

    # ---- Faithfulness agreement ----
    faithfulness_pearson: float
    faithfulness_spearman: float
    faithfulness_auroc: float
    faithfulness_f1: float
    faithfulness_ece: float

    # ---- Composite agreement ----
    composite_pearson: float
    composite_spearman: float
    composite_auroc: float
    composite_ece: float

    # ---- Per-tag means (good/hallucinated/off_topic/partial) ----
    per_tag_composite: dict[str, float] = field(default_factory=dict)
    per_tag_faithfulness: dict[str, float] = field(default_factory=dict)

    # ---- Trust grade ----
    trust_score: float = 0.0
    trust_label: str = ""

    # ---- Raw data for debugging / further analysis ----
    raw_scores: dict[str, list[float]] = field(default_factory=dict)
    raw_truths: dict[str, list[float]] = field(default_factory=dict)
    raw_labels: list[int] = field(default_factory=list)
    case_tags: list[str] = field(default_factory=list)


class MetaEvaluator:
    """Runs RAGWatch on a set of labeled cases and grades it."""

    def __init__(self, evaluator: Evaluator | None = None):
        self.evaluator = evaluator or Evaluator(Config.cpu_safe())

    # ---------- core run ----------

    def run(self, cases: list[LabeledCase] | None = None) -> MetaResult:
        cases = cases if cases is not None else get_synthetic_cases()
        if not cases:
            raise ValueError("no cases provided to MetaEvaluator.run")

        results: list[EvalResult] = []
        for c in cases:
            r = self.evaluator.evaluate(c.eval_input)
            results.append(r)

        # collect engines used (from first result that ran)
        engines_used: list[str] = []
        for r in results:
            if r.engines_used:
                engines_used = r.engines_used
                break

        # ---- pull predicted scores ----
        pred_faith: list[float] = []
        pred_comp:  list[float] = []
        true_faith: list[float] = []
        true_relevance: list[float] = []
        labels: list[int] = []
        tags: list[str] = []

        for c, r in zip(cases, results):
            pf = r.faithfulness.score if r.faithfulness is not None else 0.5
            pc = r.composite.score    if r.composite    is not None else 0.5
            pred_faith.append(pf)
            pred_comp.append(pc)
            true_faith.append(c.true_faithfulness)
            true_relevance.append(c.true_relevance)
            labels.append(1 if c.is_good else 0)
            tags.append(c.tag)

        # ---- agreement metrics ----
        faith_pear  = pearson(pred_faith, true_faith)
        faith_spear = spearman(pred_faith, true_faith)
        faith_auc   = auroc(pred_faith, labels)
        faith_f1    = f1_at_threshold(pred_faith, labels, threshold=0.5)
        faith_ece_d = expected_calibration_error(pred_faith, true_faith, n_bins=5)

        # for composite truth-target, blend faithfulness + relevance (50/50)
        comp_truth = [(tf + tr) / 2.0 for tf, tr in zip(true_faith, true_relevance)]
        comp_pear  = pearson(pred_comp, comp_truth)
        comp_spear = spearman(pred_comp, comp_truth)
        comp_auc   = auroc(pred_comp, labels)
        comp_ece_d = expected_calibration_error(pred_comp, comp_truth, n_bins=5)

        # ---- per-tag means ----
        per_tag_comp: dict[str, list[float]] = {}
        per_tag_faith: dict[str, list[float]] = {}
        for tag, pc, pf in zip(tags, pred_comp, pred_faith):
            per_tag_comp.setdefault(tag, []).append(pc)
            per_tag_faith.setdefault(tag, []).append(pf)
        per_tag_comp_avg = {t: float(sum(v) / len(v)) for t, v in per_tag_comp.items()}
        per_tag_faith_avg = {t: float(sum(v) / len(v)) for t, v in per_tag_faith.items()}

        # ---- trust score ----
        # Composite of: faithfulness AUROC (most important), composite AUROC,
        # spearman correlation, and (1 - ECE).
        trust_components = [
            0.35 * _clamp(faith_auc),
            0.25 * _clamp(comp_auc),
            0.20 * _clamp((faith_spear + 1) / 2),  # map [-1,1] -> [0,1]
            0.20 * _clamp(1.0 - faith_ece_d["ece"]),
        ]
        trust = sum(trust_components)
        if trust >= 0.85:
            label = "TRUSTED"
        elif trust >= 0.70:
            label = "ACCEPTABLE"
        elif trust >= 0.55:
            label = "MARGINAL"
        else:
            label = "UNRELIABLE"

        return MetaResult(
            n_cases=len(cases),
            n_engines_used=engines_used,
            faithfulness_pearson=faith_pear,
            faithfulness_spearman=faith_spear,
            faithfulness_auroc=faith_auc,
            faithfulness_f1=faith_f1,
            faithfulness_ece=faith_ece_d["ece"],
            composite_pearson=comp_pear,
            composite_spearman=comp_spear,
            composite_auroc=comp_auc,
            composite_ece=comp_ece_d["ece"],
            per_tag_composite=per_tag_comp_avg,
            per_tag_faithfulness=per_tag_faith_avg,
            trust_score=float(trust),
            trust_label=label,
            raw_scores={"faithfulness": pred_faith, "composite": pred_comp},
            raw_truths={"faithfulness": true_faith, "composite": comp_truth},
            raw_labels=labels,
            case_tags=tags,
        )


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
