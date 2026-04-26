"""
RAGWatch CLI — one-command pipeline evaluation.

Usage:
    python -m ragwatch.cli.main run --input data.json --report html --out report.html
    python -m ragwatch.cli.main init --train-corpus corpus.txt
    python -m ragwatch.cli.main demo
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from ragwatch.core.config import Config
from ragwatch.core.evaluator import Evaluator
from ragwatch.core.schemas import EvalInput
from ragwatch.engines.ann_validator import ANNValidator
from ragwatch.utils.embeddings import EmbeddingCache
from ragwatch.utils.reports import to_console, to_html, to_json


def _load_inputs(path: str) -> list[EvalInput]:
    """Read the input JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("input JSON must be a list of objects or a single object")
    return [
        EvalInput(
            query=r["query"],
            context=r["context"] if isinstance(r["context"], list) else [r["context"]],
            answer=r["answer"],
            ground_truth=r.get("ground_truth"),
            metadata=r.get("metadata", {}),
        )
        for r in raw
    ]


# ---------- subcommands ----------

def cmd_run(args: argparse.Namespace) -> int:
    """Run evaluation on an input file."""
    inputs = _load_inputs(args.input)

    cfg = Config.full() if args.full else Config.cpu_safe()
    if args.no_ann:
        cfg.use_ann_validator = False
    if args.no_nli:
        cfg.use_nli_engine = False

    evaluator = Evaluator(cfg)
    print(f"[ragwatch] evaluating {len(inputs)} case(s)...", file=sys.stderr)
    results = evaluator.evaluate_batch(inputs)

    # console output (always)
    for r in results:
        print(to_console(r))

    # file output
    out_path = args.out or ("report." + args.report)
    if args.report == "json":
        to_json(results, out_path)
    elif args.report == "html":
        to_html(results, out_path, queries=[i.query for i in inputs])
    else:
        raise ValueError(f"unknown report format: {args.report}")
    print(f"[ragwatch] report written to {out_path}", file=sys.stderr)
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Train the ANN claim validator on a clean text corpus."""
    if not args.train_corpus or not Path(args.train_corpus).exists():
        print("error: --train-corpus path does not exist", file=sys.stderr)
        return 2

    text = Path(args.train_corpus).read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        print("error: no paragraphs found in corpus", file=sys.stderr)
        return 2

    cfg = Config.cpu_safe()
    embedder = EmbeddingCache(cfg.embedding_model)
    ann = ANNValidator(embedder)
    print(f"[ragwatch] training ANN on {len(paragraphs)} paragraph(s)...", file=sys.stderr)
    losses = ann.train_synthetic(paragraphs, epochs=args.epochs)
    for i, loss in enumerate(losses):
        print(f"  epoch {i+1}: loss = {loss:.4f}")
    ann.save(cfg.ann_weights_path)
    print(f"[ragwatch] saved ANN weights → {cfg.ann_weights_path}", file=sys.stderr)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a single canned example (for quick smoke testing)."""
    sample = EvalInput(
        query="Who developed the theory of general relativity, and in what year?",
        context=[
            "Albert Einstein developed the theory of general relativity in 1915.",
            "The theory describes gravity as a curvature of spacetime.",
            "Einstein won the Nobel Prize in Physics in 1921 for the photoelectric effect.",
        ],
        answer="The theory of general relativity was developed by Albert Einstein in 1915.",
        ground_truth="Albert Einstein developed general relativity in 1915.",
    )
    cfg = Config.cpu_safe()
    cfg.use_ann_validator = False  # ANN needs trained weights
    evaluator = Evaluator(cfg)
    result = evaluator.evaluate(sample)
    print(to_console(result))
    return 0


def cmd_meta(args: argparse.Namespace) -> int:
    """Run meta-evaluation: who evaluates the evaluator?"""
    from ragwatch.meta_eval.meta_evaluator import MetaEvaluator
    from ragwatch.meta_eval.report import to_console as meta_console, to_json as meta_json
    from ragwatch.meta_eval.synthetic_cases import case_count_summary

    cfg = Config.cpu_safe()
    if args.no_ann:
        cfg.use_ann_validator = False
    evaluator = Evaluator(cfg)

    print("[ragwatch] case counts:", case_count_summary(), file=sys.stderr)
    print("[ragwatch] running meta-evaluation...", file=sys.stderr)

    meta = MetaEvaluator(evaluator).run()
    print(meta_console(meta))

    if args.out:
        meta_json(meta, args.out)
        print(f"[ragwatch] meta-result JSON written to {args.out}", file=sys.stderr)
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    """Export and quantize the NLI model to ONNX (Option B)."""
    try:
        from ragwatch.utils.onnx_export import export_and_quantize
    except Exception as e:
        print(f"error: cannot import ONNX export module: {e}", file=sys.stderr)
        return 2

    try:
        out = export_and_quantize(
            model_name=args.model,
            out_dir=args.out_dir,
            force=args.force,
        )
    except ImportError as e:
        print(f"error: missing dependencies for ONNX: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error during quantization: {e}", file=sys.stderr)
        return 3
    print(f"[ragwatch] ONNX model ready at: {out}", file=sys.stderr)
    return 0


# ---------- entry point ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragwatch", description="RAGWatch — pipeline-agnostic RAG evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run evaluation on an input JSON file.")
    run.add_argument("--input", required=True, help="Path to input JSON.")
    run.add_argument("--report", choices=["json", "html"], default="json")
    run.add_argument("--out", default=None, help="Output report path.")
    run.add_argument("--full", action="store_true", help="Enable all engines (incl. ollama).")
    run.add_argument("--no-ann", action="store_true", help="Disable ANN claim validator.")
    run.add_argument("--no-nli", action="store_true", help="Disable NLI engine.")
    run.set_defaults(func=cmd_run)

    init = sub.add_parser("init", help="Train ANN validator on a corpus.")
    init.add_argument("--train-corpus", required=True, help="Plain-text corpus file.")
    init.add_argument("--epochs", type=int, default=5)
    init.set_defaults(func=cmd_init)

    demo = sub.add_parser("demo", help="Run a single canned demo case.")
    demo.set_defaults(func=cmd_demo)

    meta = sub.add_parser("meta", help="Run meta-evaluation (Option A — answers 'who evaluates the evaluator?').")
    meta.add_argument("--out", default=None, help="Optional path to write meta-result JSON.")
    meta.add_argument("--no-ann", action="store_true", help="Disable ANN claim validator during meta-eval.")
    meta.set_defaults(func=cmd_meta)

    quant = sub.add_parser("quantize", help="Export and INT8-quantize the NLI model to ONNX (Option B).")
    quant.add_argument("--model", default="cross-encoder/nli-deberta-v3-small")
    quant.add_argument("--out-dir", default=".ragwatch_cache/nli_onnx")
    quant.add_argument("--force", action="store_true", help="Re-export even if quantized model exists.")
    quant.set_defaults(func=cmd_quantize)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
