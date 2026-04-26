"""
Reports — JSON, HTML, and console output.

Generates human-readable reports from EvalResult lists.
"""

from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ragwatch.core.schemas import EvalResult, ScoreWithCI


def _score_to_dict(s: ScoreWithCI | None) -> dict | None:
    if s is None:
        return None
    return s.to_dict()


def to_json(results: list[EvalResult], path: str | Path) -> None:
    """Write a list of EvalResult to JSON."""
    out = [asdict(r) for r in results]
    Path(path).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


def to_console(result: EvalResult) -> str:
    """Pretty single-result console output."""
    lines = []
    lines.append("=" * 60)
    lines.append("RAGWatch Evaluation Result")
    lines.append("=" * 60)

    parameters = [
        ("Context Relevance",   result.context_relevance),
        ("Context Precision",   result.context_precision),
        ("Context Redundancy",  result.context_redundancy),
        ("Faithfulness",        result.faithfulness),
        ("Answer Relevance",    result.answer_relevance),
        ("Completeness",        result.completeness),
        ("Hallucination Score", result.hallucination_score),
        ("Correctness",         result.correctness),
        ("Self Consistency",    result.self_consistency),
    ]

    for name, sc in parameters:
        if sc is None:
            lines.append(f"  {name:<22} : —")
        else:
            lines.append(f"  {name:<22} : {sc.score:.3f} ± {sc.std:.3f}  (n={sc.n_samples})")

    lines.append("-" * 60)
    if result.composite is not None:
        lines.append(f"  COMPOSITE              : {result.composite.score:.3f} ± {result.composite.std:.3f}")
    lines.append(f"  Claims analyzed        : {result.n_claims}")
    lines.append(f"  Atomic claims (post-ANN): {result.n_atomic_claims}")
    if result.flagged_claims:
        lines.append(f"  Flagged claims         : {len(result.flagged_claims)}")
        for c in result.flagged_claims[:3]:
            lines.append(f"     ⚠ {c[:80]}")
    lines.append(f"  Latency                : {result.latency_seconds:.3f}s")
    lines.append(f"  Engines used           : {', '.join(result.engines_used)}")
    lines.append("=" * 60)
    return "\n".join(lines)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>RAGWatch Report</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
  .case {{ border: 1px solid #ccc; border-radius: 8px; padding: 1em; margin: 1em 0; background: #fafafa; }}
  .case h3 {{ margin-top: 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  td, th {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f0f0f0; }}
  .good {{ color: #1a7f37; font-weight: 600; }}
  .mid  {{ color: #b07d00; font-weight: 600; }}
  .bad  {{ color: #c0392b; font-weight: 600; }}
  .composite {{ font-size: 1.4em; padding: 0.5em; background: #eef; border-radius: 6px; }}
  .meta {{ color: #666; font-size: 0.9em; }}
  pre {{ background: #f4f4f4; padding: 8px; border-radius: 4px; overflow-x: auto; }}
</style></head><body>
<h1>RAGWatch Evaluation Report</h1>
<p class="meta">Generated: {generated_at} &middot; {n_cases} case(s)</p>
{cases_html}
</body></html>
"""


def _color_class(score: float) -> str:
    if score >= 0.75:
        return "good"
    if score >= 0.5:
        return "mid"
    return "bad"


def _row(name: str, sc: ScoreWithCI | None) -> str:
    if sc is None:
        return f"<tr><td>{name}</td><td>—</td><td>—</td></tr>"
    cls = _color_class(sc.score)
    return (
        f"<tr><td>{name}</td>"
        f"<td class='{cls}'>{sc.score:.3f} ± {sc.std:.3f}</td>"
        f"<td>n={sc.n_samples}</td></tr>"
    )


def to_html(results: list[EvalResult], path: str | Path, queries: list[str] | None = None) -> None:
    """Write a list of EvalResult to HTML."""
    cases_html = []
    for i, r in enumerate(results):
        title = f"Case {i+1}"
        if queries and i < len(queries):
            title += f": <code>{queries[i][:120]}</code>"

        rows = [
            _row("Context Relevance",   r.context_relevance),
            _row("Context Precision",   r.context_precision),
            _row("Context Redundancy",  r.context_redundancy),
            _row("Faithfulness",        r.faithfulness),
            _row("Answer Relevance",    r.answer_relevance),
            _row("Completeness",        r.completeness),
            _row("Hallucination Score", r.hallucination_score),
            _row("Correctness",         r.correctness),
            _row("Self Consistency",    r.self_consistency),
        ]
        composite_html = ""
        if r.composite is not None:
            cls = _color_class(r.composite.score)
            composite_html = (
                f"<p class='composite'>Composite Score: "
                f"<span class='{cls}'>{r.composite.score:.3f} ± {r.composite.std:.3f}</span></p>"
            )
        flagged = ""
        if r.flagged_claims:
            items = "".join(f"<li>{c[:200]}</li>" for c in r.flagged_claims[:5])
            flagged = f"<details><summary>{len(r.flagged_claims)} flagged claim(s)</summary><ul>{items}</ul></details>"

        cases_html.append(f"""
        <div class="case">
          <h3>{title}</h3>
          {composite_html}
          <table>
            <tr><th>Parameter</th><th>Score</th><th>Evidence</th></tr>
            {''.join(rows)}
          </table>
          {flagged}
          <p class="meta">Engines: {', '.join(r.engines_used)} &middot; Latency: {r.latency_seconds:.3f}s</p>
        </div>
        """)

    html = _HTML_TEMPLATE.format(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_cases=len(results),
        cases_html="\n".join(cases_html),
    )
    Path(path).write_text(html, encoding="utf-8")
