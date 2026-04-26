"""
Self-trust report generation for meta-evaluation.

Outputs:
    - Console: detailed text summary
    - JSON:    structured result for downstream analysis
"""

from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path

from ragwatch.meta_eval.meta_evaluator import MetaResult


def _bar(value: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(value * width))))
    return "█" * filled + "░" * (width - filled)


def _grade(value: float, good_th: float = 0.80, ok_th: float = 0.65) -> str:
    if value >= good_th:
        return "✓"
    if value >= ok_th:
        return "~"
    return "✗"


def to_console(meta: MetaResult) -> str:
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("RAGWatch — Self-Trust Report")
    lines.append("=" * 64)
    lines.append(f"  Cases evaluated  : {meta.n_cases}")
    lines.append(f"  Engines used     : {', '.join(meta.n_engines_used) or '—'}")
    lines.append("")

    lines.append("--- Faithfulness Agreement ---")
    lines.append(f"  Pearson r        : {meta.faithfulness_pearson:+.3f}  {_grade(abs(meta.faithfulness_pearson))}")
    lines.append(f"  Spearman ρ       : {meta.faithfulness_spearman:+.3f}  {_grade(abs(meta.faithfulness_spearman))}")
    lines.append(f"  AUROC            : {meta.faithfulness_auroc:.3f}  {_bar(meta.faithfulness_auroc)} {_grade(meta.faithfulness_auroc)}")
    lines.append(f"  F1 @ 0.5         : {meta.faithfulness_f1:.3f}  {_bar(meta.faithfulness_f1)} {_grade(meta.faithfulness_f1)}")
    lines.append(f"  ECE              : {meta.faithfulness_ece:.3f}  (lower is better)")
    lines.append("")

    lines.append("--- Composite Agreement ---")
    lines.append(f"  Pearson r        : {meta.composite_pearson:+.3f}  {_grade(abs(meta.composite_pearson))}")
    lines.append(f"  Spearman ρ       : {meta.composite_spearman:+.3f}  {_grade(abs(meta.composite_spearman))}")
    lines.append(f"  AUROC            : {meta.composite_auroc:.3f}  {_bar(meta.composite_auroc)} {_grade(meta.composite_auroc)}")
    lines.append(f"  ECE              : {meta.composite_ece:.3f}  (lower is better)")
    lines.append("")

    lines.append("--- Mean RAGWatch Composite Score by Tag ---")
    for tag, val in sorted(meta.per_tag_composite.items()):
        lines.append(f"  {tag:<14} : {val:.3f}  {_bar(val)}")
    lines.append("")

    lines.append("--- Mean RAGWatch Faithfulness by Tag ---")
    for tag, val in sorted(meta.per_tag_faithfulness.items()):
        lines.append(f"  {tag:<14} : {val:.3f}  {_bar(val)}")
    lines.append("")

    lines.append("=" * 64)
    label_color = {
        "TRUSTED": "★", "ACCEPTABLE": "◆", "MARGINAL": "▲", "UNRELIABLE": "✗",
    }.get(meta.trust_label, "?")
    lines.append(f"  TRUST SCORE   : {meta.trust_score:.3f}   {_bar(meta.trust_score)}")
    lines.append(f"  TRUST RATING  : {label_color} {meta.trust_label}")
    lines.append("=" * 64)
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - faithfulness AUROC ≥ 0.80 means RAGWatch reliably distinguishes")
    lines.append("    faithful answers from hallucinated ones.")
    lines.append("  - composite AUROC reflects end-to-end quality discrimination.")
    lines.append("  - low ECE means RAGWatch's scores are well-calibrated to truth.")
    lines.append("  - sanity check: per-tag composite should rank")
    lines.append("    `faithful` > `partial` > `hallucinated`/`off_topic`.")
    return "\n".join(lines)


def to_json(meta: MetaResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(meta), indent=2, default=str), encoding="utf-8")
