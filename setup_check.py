"""
setup_check.py — Pre-flight environment verification.

Run this FIRST before doing anything else with RAGWatch.
It confirms your Python version, OS, dependencies, and disk space
are sufficient to run RAGWatch on a CPU-only Lenovo ThinkBook.

Usage:
    python setup_check.py
"""

from __future__ import annotations
import importlib
import os
import platform
import shutil
import sys
from typing import Any


# ---------- ANSI helpers ----------

def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


_USE_COLOR = _supports_color()


def _c(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def ok(msg: str) -> None:
    print(_c("  ✓ ", "32") + msg)


def warn(msg: str) -> None:
    print(_c("  ⚠ ", "33") + msg)


def fail(msg: str) -> None:
    print(_c("  ✗ ", "31") + msg)


def section(title: str) -> None:
    print()
    print(_c("=== " + title + " ===", "1;36"))


# ---------- checks ----------

REQUIRED_PYTHON = (3, 9)

REQUIRED_PACKAGES = {
    "numpy":   "numpy>=1.24",
    "scipy":   "scipy>=1.10",
}

OPTIONAL_PACKAGES = {
    "sentence_transformers": "sentence-transformers (real-model engines)",
    "transformers":          "transformers (NLI model)",
    "torch":                 "torch (CPU build is fine)",
    "spacy":                 "spacy + en_core_web_sm (better sentence splitter)",
    "pytest":                "pytest (test runner)",
    "onnxruntime":           "onnxruntime (Option B — fast CPU inference)",
    "onnx":                  "onnx (Option B — model export)",
    "ollama":                "ollama (optional logit-uncertainty engine)",
}


def check_python_version() -> bool:
    section("Python")
    print(f"  Version : {sys.version.split()[0]}")
    print(f"  Path    : {sys.executable}")
    print(f"  Platform: {platform.platform()}")

    if sys.version_info < REQUIRED_PYTHON:
        fail(f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required.")
        return False
    ok(f"Python {sys.version_info.major}.{sys.version_info.minor} OK.")
    return True


def check_required_packages() -> bool:
    section("Required packages")
    all_ok = True
    for mod, hint in REQUIRED_PACKAGES.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{mod:<10} {ver}")
        except ImportError:
            fail(f"{mod} missing — install with `pip install {hint}`")
            all_ok = False
    return all_ok


def check_optional_packages() -> dict[str, bool]:
    section("Optional packages")
    status: dict[str, bool] = {}
    for mod, desc in OPTIONAL_PACKAGES.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{mod:<22} {ver}  — {desc}")
            status[mod] = True
        except ImportError:
            warn(f"{mod:<22} missing — {desc}")
            status[mod] = False
    return status


def check_spacy_model() -> None:
    section("spaCy model (en_core_web_sm)")
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            ok("en_core_web_sm available — high-quality sentence splitting enabled.")
        except OSError:
            warn("en_core_web_sm not downloaded.")
            warn("  → run: python -m spacy download en_core_web_sm")
            warn("  → fallback: regex splitter will be used (still works).")
    except ImportError:
        warn("spaCy not installed; regex splitter will be used.")


def check_disk_space(path: str = ".", min_gb: float = 1.5) -> None:
    section("Disk space")
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    print(f"  Free space at '{path}': {free_gb:.2f} GB")
    if free_gb < min_gb:
        fail(f"Less than {min_gb} GB free — model downloads may fail.")
    else:
        ok(f"At least {min_gb} GB free.")


def check_ram() -> None:
    section("RAM")
    try:
        import psutil  # type: ignore
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        avail_gb = mem.available / (1024 ** 3)
        print(f"  Total RAM    : {total_gb:.2f} GB")
        print(f"  Available RAM: {avail_gb:.2f} GB")
        if avail_gb < 1.0:
            warn("Less than 1 GB free RAM — close some apps before running real-model tests.")
        else:
            ok("RAM looks fine for CPU inference.")
    except ImportError:
        warn("psutil not installed; skipping RAM check (install with `pip install psutil`).")


def check_ragwatch_imports() -> bool:
    section("RAGWatch package imports")
    try:
        from ragwatch import Evaluator, EvalInput, EvalResult  # noqa: F401
        from ragwatch.core.config import Config  # noqa: F401
        from ragwatch.engines.math_engine import MathEngine  # noqa: F401
        from ragwatch.engines.ann_validator import ANNValidator  # noqa: F401
        from ragwatch.engines.nli_engine import NLIEngine  # noqa: F401
        from ragwatch.engines.self_consistency import SelfConsistencyEngine  # noqa: F401
        from ragwatch.utils.embeddings import EmbeddingCache  # noqa: F401
        from ragwatch.utils.preprocessor import split_sentences  # noqa: F401
        from ragwatch.meta_eval import MetaEvaluator  # noqa: F401
        from ragwatch.pytest_plugin import evaluate_rag, evaluate_one, EvalReport  # noqa: F401
        ok("All RAGWatch modules import correctly (incl. pytest_plugin).")
        return True
    except Exception as e:
        fail(f"Import error: {e}")
        return False


def check_ollama_optional() -> None:
    section("Ollama (optional, for logit uncertainty engine)")
    try:
        import ollama  # type: ignore
        try:
            client = ollama.Client(host="http://localhost:11434")
            models = client.list()
            ok(f"Ollama daemon reachable. Models: {[m.get('name', '?') for m in models.get('models', [])][:5]}")
        except Exception as e:
            warn(f"Ollama installed but daemon unreachable: {e}")
            warn("  → start with: `ollama serve`")
    except ImportError:
        warn("ollama not installed (optional). Skip if you don't plan to use logit-uncertainty.")


def summary(all_required_ok: bool, optional: dict[str, bool]) -> int:
    section("Summary")
    if not all_required_ok:
        fail("REQUIRED packages missing. Install them before running RAGWatch.")
        print()
        print("  pip install -r requirements.txt")
        return 1

    needed_for_real_models = ["sentence_transformers", "transformers", "torch"]
    if not all(optional.get(p, False) for p in needed_for_real_models):
        warn("Real-model engines not fully installed.")
        print("    Unit tests will run, but `demo` and `run` will fail.")
        print("    Install: pip install -r requirements.txt")
    else:
        ok("Real-model engines installed.")

    if optional.get("onnxruntime") and optional.get("onnx"):
        ok("ONNX stack ready — Option B (quantized NLI) available.")
    else:
        warn("ONNX not installed. Option B (quantized NLI) unavailable.")
        print("    Install: pip install onnx onnxruntime")

    if optional.get("pytest"):
        ok("pytest available — run with: pytest -v tests/")
    else:
        warn("pytest missing — install with: pip install pytest")

    print()
    print("Next steps:")
    print("  1. pytest -v tests/                                 # run RAGWatch's own tests")
    print("  2. python -m ragwatch.cli.main demo                 # smoke run with real models")
    print("  3. python -m ragwatch.cli.main meta                 # meta-evaluation (self-trust)")
    print("  4. python -m ragwatch.cli.main quantize             # build ONNX model (faster CPU)")
    print()
    print("To wire RAGWatch into YOUR project:")
    print("  → see docs/connecting-to-projects.md")
    print("  → copy examples/example_test_rag_eval.py to your project's tests/ folder")
    return 0


def main() -> int:
    print(_c("RAGWatch — environment check", "1;35"))
    print(_c("=" * 60, "1;35"))

    py_ok = check_python_version()
    req_ok = check_required_packages()
    optional = check_optional_packages()
    check_spacy_model()
    check_disk_space()
    check_ram()
    rw_ok = check_ragwatch_imports() if (py_ok and req_ok) else False
    if optional.get("ollama"):
        check_ollama_optional()

    return summary(py_ok and req_ok and rw_ok, optional)


if __name__ == "__main__":
    raise SystemExit(main())
