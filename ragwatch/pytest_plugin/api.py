"""
Pytest plugin API — the v0.3 headline feature.

Designed to feel like a normal pytest assertion:
    assert report.passed, report.summary()

Three layers of granularity:
  1. evaluate_rag(cases) → one report for all cases (test_rag_quality)
  2. evaluate_one(case)  → per-case granularity (parametrize)
  3. EvalReport          → programmatic access for custom assertions
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping

from ragwatch.core.config import Config
from ragwatch.core.evaluator import Evaluator
from ragwatch.core.schemas import EvalInput, EvalResult


# ---------------------------------------------------------------
# Thresholds — what 'passed' means
# ---------------------------------------------------------------

@dataclass
class Thresholds:
    """Per-parameter pass thresholds. None = not enforced."""
    composite: float | None = 0.7              # overall quality bar
    faithfulness: float | None = 0.6           # don't accept hallucination
    hallucination_max: float | None = 0.4      # hallucination_score must be BELOW this
    answer_relevance: float | None = None
    completeness: float | None = None
    correctness: float | None = None

    @classmethod
    def strict(cls) -> "Thresholds":
        return cls(composite=0.80, faithfulness=0.75, hallucination_max=0.25,
                   answer_relevance=0.65)

    @classmethod
    def permissive(cls) -> "Thresholds":
        return cls(composite=0.5, faithfulness=0.5, hallucination_max=0.5)


# ---------------------------------------------------------------
# EvalReport — the assertion-friendly container
# ---------------------------------------------------------------

@dataclass
class EvalReport:
    """Aggregate report across all cases. Use `.passed` and `.summary()` in asserts."""

    per_case: list[EvalResult] = field(default_factory=list)
    case_labels: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    thresholds: Thresholds = field(default_factory=Thresholds)
    n_cases: int = 0

    # aggregate metrics across cases
    composite_mean: float = 0.0
    composite_std: float = 0.0
    faithfulness_mean: float = 0.0
    hallucination_mean: float = 0.0

    # final pass/fail
    passed: bool = False

    # ------------- assertion-friendly interface -------------

    def summary(self) -> str:
        """Human-readable failure message — embed in `assert ..., report.summary()`."""
        lines = []
        lines.append("")
        lines.append(f"RAGWatch quality report — {self.n_cases} case(s)")
        lines.append("-" * 56)
        lines.append(f"  composite     : {self.composite_mean:.3f} ± {self.composite_std:.3f}")
        lines.append(f"  faithfulness  : {self.faithfulness_mean:.3f}")
        lines.append(f"  hallucination : {self.hallucination_mean:.3f}")
        lines.append(f"  status        : {'PASSED ✓' if self.passed else 'FAILED ✗'}")
        if self.failures:
            lines.append("")
            lines.append(f"  {len(self.failures)} failure(s):")
            for f in self.failures[:10]:
                lines.append(f"    ✗ {f}")
            if len(self.failures) > 10:
                lines.append(f"    ... and {len(self.failures) - 10} more")
        lines.append("")
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------
# Case input helpers
# ---------------------------------------------------------------

def case_from_dict(d: Mapping[str, Any]) -> EvalInput:
    """Convert a dict to EvalInput; tolerant about field names and types."""
    if not isinstance(d, Mapping):
        raise TypeError(f"case must be a dict-like mapping, got {type(d).__name__}")

    query = d.get("query") or d.get("question")
    context = d.get("context") or d.get("contexts") or d.get("chunks") or []
    answer = d.get("answer") or d.get("response") or d.get("output")
    ground_truth = d.get("ground_truth") or d.get("gt") or d.get("reference")

    if isinstance(context, str):
        context = [context]

    return EvalInput(
        query=str(query) if query is not None else "",
        context=[str(c) for c in context],
        answer=str(answer) if answer is not None else "",
        ground_truth=str(ground_truth) if ground_truth is not None else None,
        metadata=dict(d.get("metadata", {})),
    )


# ---------------------------------------------------------------
# Evaluator selection
# ---------------------------------------------------------------

_EVALUATOR_CACHE: dict[str, Evaluator] = {}


def _get_evaluator(mode: str = "cpu_safe", config: Config | None = None) -> Evaluator:
    """Cached evaluator — building it has model-load cost; reuse across cases."""
    if config is not None:
        # custom config — don't cache
        return Evaluator(config)
    if mode not in _EVALUATOR_CACHE:
        if mode == "cpu_safe":
            cfg = Config.cpu_safe()
        elif mode == "full":
            cfg = Config.full()
        else:
            raise ValueError(f"unknown mode: {mode!r}. use 'cpu_safe' or 'full'.")
        # ANN by default off (needs trained weights); user can enable explicitly
        cfg.use_ann_validator = False
        _EVALUATOR_CACHE[mode] = Evaluator(cfg)
    return _EVALUATOR_CACHE[mode]


def reset_evaluator_cache() -> None:
    """Drop cached evaluators — call between test sessions if you change config."""
    _EVALUATOR_CACHE.clear()


# ---------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------

def evaluate_one(
    case: EvalInput | Mapping[str, Any],
    mode: str = "cpu_safe",
    config: Config | None = None,
) -> EvalResult:
    """
    Evaluate a single case. Use with pytest.mark.parametrize for per-case granularity:

        @pytest.mark.parametrize("case", RAG_CASES, ids=lambda c: c["query"][:30])
        def test_individual_case(case):
            result = evaluate_one(case)
            assert result.faithfulness.score > 0.6
    """
    inp = case if isinstance(case, EvalInput) else case_from_dict(case)
    evaluator = _get_evaluator(mode=mode, config=config)
    return evaluator.evaluate(inp)


def evaluate_rag(
    cases: list[EvalInput | Mapping[str, Any]],
    threshold: float | None = 0.7,
    mode: str = "cpu_safe",
    thresholds: Thresholds | None = None,
    config: Config | None = None,
) -> EvalReport:
    """
    Evaluate a list of cases. Returns an EvalReport.

    Args:
        cases: list of EvalInput or dicts. Dicts are auto-converted.
        threshold: shorthand for Thresholds.composite. Ignored if `thresholds` is given.
        mode: 'cpu_safe' (no logit/self-consistency) or 'full' (all engines).
        thresholds: full Thresholds object for fine-grained control.
        config: explicit Config overriding mode preset.

    Use:
        report = evaluate_rag(cases, threshold=0.75)
        assert report.passed, report.summary()
    """
    if not isinstance(cases, list):
        raise TypeError(f"cases must be a list; got {type(cases).__name__}")
    if not cases:
        raise ValueError("evaluate_rag requires at least one case")

    # If shorthand `threshold=` is used, enforce ONLY composite. If full
    # `thresholds=` object is given, use it as-is.
    if thresholds is not None:
        th = thresholds
    else:
        th = Thresholds(
            composite=threshold,
            faithfulness=None,
            hallucination_max=None,
            answer_relevance=None,
            completeness=None,
            correctness=None,
        )

    # convert all to EvalInput
    inputs: list[EvalInput] = []
    labels: list[str] = []
    for i, c in enumerate(cases):
        inp = c if isinstance(c, EvalInput) else case_from_dict(c)
        inputs.append(inp)
        # short label for failure messages
        q = inp.query.strip()
        labels.append(q[:60] + ("…" if len(q) > 60 else "") or f"case_{i}")

    evaluator = _get_evaluator(mode=mode, config=config)
    results = evaluator.evaluate_batch(inputs)

    # per-case failure detection
    failures: list[str] = []
    composites: list[float] = []
    faiths: list[float] = []
    hallus: list[float] = []

    for label, r in zip(labels, results):
        if r.composite is not None:
            composites.append(r.composite.score)
        if r.faithfulness is not None:
            faiths.append(r.faithfulness.score)
        if r.hallucination_score is not None:
            hallus.append(r.hallucination_score.score)

        for msg in _check_thresholds(label, r, th):
            failures.append(msg)

    # aggregate
    def _avg(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _std(xs: list[float], m: float) -> float:
        if len(xs) < 2:
            return 0.0
        return float((sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5)

    comp_mean = _avg(composites)
    comp_std = _std(composites, comp_mean)
    faith_mean = _avg(faiths)
    hallu_mean = _avg(hallus)

    return EvalReport(
        per_case=results,
        case_labels=labels,
        failures=failures,
        thresholds=th,
        n_cases=len(results),
        composite_mean=comp_mean,
        composite_std=comp_std,
        faithfulness_mean=faith_mean,
        hallucination_mean=hallu_mean,
        passed=(len(failures) == 0),
    )


# ---------------------------------------------------------------
# Internals
# ---------------------------------------------------------------

def _check_thresholds(label: str, r: EvalResult, th: Thresholds) -> list[str]:
    """Return human-readable failure messages for any threshold breach."""
    msgs: list[str] = []
    if th.composite is not None and r.composite is not None:
        if r.composite.score < th.composite:
            msgs.append(f"[{label}] composite {r.composite.score:.3f} < {th.composite}")
    if th.faithfulness is not None and r.faithfulness is not None:
        if r.faithfulness.score < th.faithfulness:
            msgs.append(f"[{label}] faithfulness {r.faithfulness.score:.3f} < {th.faithfulness}")
    if th.hallucination_max is not None and r.hallucination_score is not None:
        if r.hallucination_score.score > th.hallucination_max:
            msgs.append(f"[{label}] hallucination {r.hallucination_score.score:.3f} > {th.hallucination_max}")
    if th.answer_relevance is not None and r.answer_relevance is not None:
        if r.answer_relevance.score < th.answer_relevance:
            msgs.append(f"[{label}] answer_relevance {r.answer_relevance.score:.3f} < {th.answer_relevance}")
    if th.completeness is not None and r.completeness is not None:
        if r.completeness.score < th.completeness:
            msgs.append(f"[{label}] completeness {r.completeness.score:.3f} < {th.completeness}")
    if th.correctness is not None and r.correctness is not None:
        if r.correctness.score < th.correctness:
            msgs.append(f"[{label}] correctness {r.correctness.score:.3f} < {th.correctness}")
    return msgs
