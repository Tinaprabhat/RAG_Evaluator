"""
ONNX export and INT8 quantization for the NLI model.

Why this exists
---------------
The PyTorch CrossEncoder for `cross-encoder/nli-deberta-v3-small`:
    ~180 MB on disk, ~250 ms per claim on CPU.

Quantized ONNX:
    ~50 MB on disk, ~60 ms per claim on CPU.
    Accuracy drop typically <1% on NLI tasks.

This module:
1. Loads the HuggingFace model.
2. Traces it into ONNX format (FP32).
3. Quantizes weights to INT8.
4. Saves alongside the tokenizer for offline use.

Run once via:
    python -m ragwatch.cli.main quantize
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_OUT_DIR = ".ragwatch_cache/nli_onnx"


def _ensure_deps() -> None:
    missing: list[str] = []
    for mod in ("onnx", "onnxruntime", "transformers", "torch"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(
            f"Missing dependencies for ONNX export: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )


def export_and_quantize(
    model_name: str = DEFAULT_MODEL,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    force: bool = False,
) -> Path:
    """
    Export an HF cross-encoder NLI model to quantized ONNX.

    Returns the path to the directory containing:
        - model_quantized.onnx
        - tokenizer files
        - metadata.json
    """
    _ensure_deps()

    out_path = Path(out_dir)
    quant_file = out_path / "model_quantized.onnx"
    if quant_file.exists() and not force:
        print(f"[onnx_export] already exists: {quant_file}")
        return out_path

    out_path.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"[onnx_export] loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Create a dummy input matching the model's expected shapes
    dummy = tokenizer(
        "premise sentence here.",
        "hypothesis sentence here.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    fp32_path = out_path / "model_fp32.onnx"
    print(f"[onnx_export] tracing FP32 ONNX → {fp32_path}")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits":         {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"[onnx_export] quantizing → {quant_file}")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(quant_file),
        weight_type=QuantType.QInt8,
    )

    # save tokenizer alongside
    tokenizer.save_pretrained(str(out_path))

    # write small metadata file
    import json
    (out_path / "metadata.json").write_text(json.dumps({
        "source_model": model_name,
        "fp32_size_bytes": fp32_path.stat().st_size,
        "quantized_size_bytes": quant_file.stat().st_size,
        "max_length": 128,
        "num_labels": int(getattr(model.config, "num_labels", 3)),
        "id2label": getattr(model.config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"}),
    }, indent=2, default=str))

    # FP32 file is no longer needed
    try:
        os.remove(fp32_path)
    except OSError:
        pass

    fp32_mb = "—"
    quant_mb = quant_file.stat().st_size / (1024 * 1024)
    print(f"[onnx_export] done. quantized size: {quant_mb:.1f} MB")
    return out_path


# ---------- Inference wrapper ----------

class ONNXNLIRunner:
    """Drop-in replacement for sentence-transformers CrossEncoder.predict."""

    def __init__(self, model_dir: str | Path = DEFAULT_OUT_DIR, max_length: int = 128):
        _ensure_deps()
        self.model_dir = Path(model_dir)
        self.max_length = max_length

        quant_file = self.model_dir / "model_quantized.onnx"
        if not quant_file.exists():
            raise FileNotFoundError(
                f"No quantized ONNX model at {quant_file}. "
                f"Run: python -m ragwatch.cli.main quantize"
            )

        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 1)
        self.session = ort.InferenceSession(
            str(quant_file),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, pairs: list[tuple[str, str]]) -> Any:
        """Return raw logits with shape [batch, num_labels] — same contract as CrossEncoder."""
        import numpy as np

        if not pairs:
            return np.zeros((0, 3), dtype=np.float32)

        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        enc = self.tokenizer(
            premises, hypotheses,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        outputs = self.session.run(
            ["logits"],
            {
                "input_ids":      enc["input_ids"].astype("int64"),
                "attention_mask": enc["attention_mask"].astype("int64"),
            },
        )
        return outputs[0]
