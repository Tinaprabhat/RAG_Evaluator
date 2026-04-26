"""
Config — central runtime settings.

All thresholds, model choices, and engine toggles in one place.
Designed for "set up once, run many times".
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- Engine toggles ----
    use_math_engine: bool = True
    use_nli_engine: bool = True
    use_ann_validator: bool = True
    use_logit_uncertainty: bool = False  # off by default — needs ollama
    use_self_consistency: bool = False    # off by default — slow

    # ---- Models ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_onnx_dir: str = ".ragwatch_cache/nli_onnx"  # used if model_quantized.onnx exists
    ollama_model: str = "qwen2.5:0.5b"
    ollama_host: str = "http://localhost:11434"

    # ---- Thresholds ----
    # Lazy-NLI gating: skip NLI when cosine similarity is decisive
    cosine_high_threshold: float = 0.80   # > this → entailed, skip NLI
    cosine_low_threshold: float = 0.20    # < this → not entailed, skip NLI
    redundancy_threshold: float = 0.85    # chunk-pair similarity above = redundant
    ann_atomicity_threshold: float = 0.50 # ANN output below = non-atomic

    # ---- Self-consistency ----
    n_consistency_samples: int = 3
    consistency_temperature: float = 0.3

    # ---- Composite weights ----
    weights: dict[str, float] = field(default_factory=lambda: {
        "context_relevance": 0.15,
        "context_precision": 0.10,
        "faithfulness": 0.30,
        "answer_relevance": 0.15,
        "completeness": 0.15,
        "correctness": 0.15,
    })

    # ---- I/O ----
    embedding_cache_dir: str = ".ragwatch_cache"
    ann_weights_path: str = ".ragwatch_cache/ann_weights.npz"

    @classmethod
    def cpu_safe(cls) -> "Config":
        """Preset for low-RAM CPU-only environments (Lenovo ThinkBook style)."""
        return cls(
            use_logit_uncertainty=False,
            use_self_consistency=False,
        )

    @classmethod
    def full(cls) -> "Config":
        """All engines on. Needs ollama running."""
        return cls(
            use_logit_uncertainty=True,
            use_self_consistency=True,
        )
