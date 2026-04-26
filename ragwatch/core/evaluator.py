"""
Evaluator — main orchestrator.

Runs all enabled engines on a single (query, context, answer, gt?) tuple
and combines per-parameter scores into a final EvalResult with composite + CI.
"""

from __future__ import annotations
import time

from ragwatch.core.schemas import EvalInput, EvalResult, ScoreWithCI
from ragwatch.core.config import Config
from ragwatch.core.scorer import composite_score
from ragwatch.engines.math_engine import MathEngine
from ragwatch.engines.nli_engine import NLIEngine
from ragwatch.engines.ann_validator import ANNValidator
from ragwatch.engines.self_consistency import SelfConsistencyEngine
from ragwatch.engines.logit_uncertainty import LogitUncertaintyEngine
from ragwatch.utils.embeddings import EmbeddingCache
from ragwatch.utils.preprocessor import decompose_into_claims


class Evaluator:
    """Top-level evaluator. Wire engines together; produce one EvalResult per input."""

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config.cpu_safe()
        self.embedder = EmbeddingCache(self.cfg.embedding_model)

        self.math = MathEngine(
            self.embedder,
            redundancy_threshold=self.cfg.redundancy_threshold,
        ) if self.cfg.use_math_engine else None

        self.nli = NLIEngine(
            self.embedder,
            model_name=self.cfg.nli_model,
            cosine_high=self.cfg.cosine_high_threshold,
            cosine_low=self.cfg.cosine_low_threshold,
            onnx_dir=self.cfg.nli_onnx_dir,
        ) if self.cfg.use_nli_engine else None

        self.ann = ANNValidator(
            self.embedder,
            weights_path=self.cfg.ann_weights_path,
        ) if self.cfg.use_ann_validator else None

        self.consistency = SelfConsistencyEngine(self.embedder) \
            if self.cfg.use_self_consistency else None

        self.logit = LogitUncertaintyEngine(
            model=self.cfg.ollama_model,
            host=self.cfg.ollama_host,
        ) if self.cfg.use_logit_uncertainty else None

    # ---------- public API ----------

    def evaluate(self, inp: EvalInput) -> EvalResult:
        """Run all enabled engines on a single input."""
        start = time.time()
        result = EvalResult()
        engines_used: list[str] = []

        # ---- Math engine (always cheap, always run if enabled) ----
        if self.math is not None:
            result.context_relevance = self.math.context_relevance(inp)
            result.context_precision = self.math.context_precision(inp)
            result.context_redundancy = self.math.context_redundancy(inp)
            result.answer_relevance = self.math.answer_relevance(inp)
            result.correctness = self.math.correctness(inp)
            engines_used.append("math")

        # ---- Claim decomposition (used downstream) ----
        raw_claims = decompose_into_claims(inp.answer)
        result.n_claims = len(raw_claims)

        atomic_claims = raw_claims
        if self.ann is not None and raw_claims:
            atomic_claims, _ = self.ann.filter_atomic(
                raw_claims, threshold=self.cfg.ann_atomicity_threshold
            )
            engines_used.append("ann_validator")
        result.n_atomic_claims = len(atomic_claims)

        # ---- NLI engine ----
        if self.nli is not None and atomic_claims and inp.context:
            faith, flagged = self.nli.faithfulness(atomic_claims, inp.context)
            result.faithfulness = faith
            result.flagged_claims = flagged
            # hallucination = 1 - faithfulness, with same uncertainty
            result.hallucination_score = ScoreWithCI(
                score=1.0 - faith.score,
                std=faith.std,
                n_samples=faith.n_samples,
                notes="1 - faithfulness",
            )
            result.completeness = self.nli.completeness(inp.query, inp.answer)
            engines_used.append("nli")

        # ---- Self-consistency ----
        if self.consistency is not None and atomic_claims:
            result.self_consistency = self.consistency.score(atomic_claims)
            engines_used.append("self_consistency")

        # ---- Logit uncertainty (slow, optional) ----
        if self.logit is not None and inp.context:
            slm_score = self.logit.score(inp.context, inp.answer)
            # if SLM-confidence is high, it strengthens our faithfulness signal
            if result.faithfulness is not None and slm_score.n_samples > 0:
                # blend: weighted average tilted toward NLI (more trustworthy)
                blended = 0.7 * result.faithfulness.score + 0.3 * slm_score.score
                result.faithfulness = ScoreWithCI(
                    score=float(blended),
                    std=max(result.faithfulness.std, slm_score.std),
                    n_samples=result.faithfulness.n_samples + slm_score.n_samples,
                    notes="NLI(0.7) blended with SLM-self-consistency(0.3)",
                )
            engines_used.append("logit_uncertainty")

        # ---- Composite ----
        scores_for_composite = {
            "context_relevance": result.context_relevance,
            "context_precision": result.context_precision,
            "faithfulness": result.faithfulness,
            "answer_relevance": result.answer_relevance,
            "completeness": result.completeness,
            "correctness": result.correctness,
        }
        result.composite = composite_score(scores_for_composite, self.cfg)

        result.latency_seconds = time.time() - start
        result.engines_used = engines_used
        return result

    def evaluate_batch(self, inputs: list[EvalInput]) -> list[EvalResult]:
        """Run evaluation on a list of inputs."""
        return [self.evaluate(i) for i in inputs]
