"""
Tests for the ONNX export utility (Option B).

These tests verify the module's contract without actually running a heavy
model export. The full export-and-quantize roundtrip is in
`test_integration_real_models.py` which is opt-in (see test_cases.md).
"""

from __future__ import annotations
import importlib

import pytest


def _has(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


class TestImports:
    def test_module_imports(self):
        from ragwatch.utils.onnx_export import (  # noqa: F401
            DEFAULT_MODEL,
            DEFAULT_OUT_DIR,
            ONNXNLIRunner,
            export_and_quantize,
        )

    def test_default_paths(self):
        from ragwatch.utils.onnx_export import DEFAULT_MODEL, DEFAULT_OUT_DIR
        assert "deberta" in DEFAULT_MODEL.lower()
        assert "nli_onnx" in DEFAULT_OUT_DIR


class TestDependencyChecks:
    def test_export_raises_clear_error_when_deps_missing(self, monkeypatch):
        """If onnx/onnxruntime are missing, _ensure_deps must raise ImportError."""
        from ragwatch.utils import onnx_export

        def fake_import(name, *args, **kwargs):
            if name in ("onnx", "onnxruntime"):
                raise ImportError(f"mocked missing: {name}")
            return importlib.__import__(name, *args, **kwargs)

        monkeypatch.setattr(onnx_export, "__import__", fake_import, raising=False)
        # _ensure_deps uses builtin __import__; patch via builtins
        import builtins
        real_import = builtins.__import__

        def import_blocker(name, *args, **kwargs):
            if name in ("onnx", "onnxruntime"):
                raise ImportError(f"mocked missing: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_blocker)
        with pytest.raises(ImportError, match="onnx"):
            onnx_export._ensure_deps()


class TestONNXRunnerLazyLoad:
    def test_runner_raises_clear_error_when_no_model_file(self, tmp_path):
        from ragwatch.utils.onnx_export import ONNXNLIRunner
        if not (_has("onnx") and _has("onnxruntime") and _has("transformers")):
            pytest.skip("onnx/onnxruntime/transformers not installed")
        with pytest.raises(FileNotFoundError, match="quantize"):
            ONNXNLIRunner(model_dir=str(tmp_path))


class TestNLIEngineWithOnnxFlag:
    def test_engine_constructs_with_onnx_dir(self, dummy_embedder, tmp_path):
        """NLIEngine should accept onnx_dir gracefully even if path doesn't exist."""
        from ragwatch.engines.nli_engine import NLIEngine
        eng = NLIEngine(dummy_embedder, onnx_dir=str(tmp_path / "nonexistent"))
        # before _load: backend == "none"
        assert eng.backend == "none"

    def test_engine_falls_back_to_pytorch_when_onnx_missing(self, dummy_embedder, tmp_path):
        """If onnx_dir given but no quantized file, engine should fall through gracefully.

        We don't actually load PyTorch here — we only verify the *path* through _load
        doesn't raise on the ONNX-check step. We monkey-patch sentence-transformers
        to a stub so we don't need it installed.
        """
        from ragwatch.engines import nli_engine

        class StubCrossEncoder:
            def __init__(self, name): self.name = name
            def predict(self, pairs): return [[0.0, 1.0, 0.0]] * len(pairs)

        # build a fake module structure for sentence_transformers.CrossEncoder
        import sys
        import types
        if "sentence_transformers" not in sys.modules:
            mod = types.ModuleType("sentence_transformers")
            mod.CrossEncoder = StubCrossEncoder
            sys.modules["sentence_transformers"] = mod
        else:
            # don't disturb real module
            pass

        eng = nli_engine.NLIEngine(dummy_embedder, onnx_dir=str(tmp_path / "missing"))
        # call _load; should not raise even though ONNX path is missing
        try:
            eng._load()
            # after _load, backend should be 'pytorch' (real or stubbed)
            assert eng.backend in ("pytorch", "onnx")
        except ImportError:
            # acceptable: real sentence-transformers truly missing
            pytest.skip("sentence-transformers not available")
