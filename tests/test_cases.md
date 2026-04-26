# RAGWatch — Test Cases & Expected Outputs

This document defines **checkable test cases** for verifying RAGWatch behavior end-to-end.
Each case lists the input, the expected qualitative output, and pass/fail criteria.

> Ranges are intentionally loose because the deterministic dummy embedder used in the
> unit suite differs from the real `all-MiniLM-L6-v2` model. Real-model thresholds are
> noted in parentheses where they meaningfully diverge.

---

## Case 1 — Well-supported, fully grounded answer

**Input**
```json
{
  "query": "Who developed the theory of general relativity, and in what year?",
  "context": [
    "Albert Einstein developed the theory of general relativity in 1915.",
    "The theory describes gravity as a curvature of spacetime."
  ],
  "answer": "The theory of general relativity was developed by Albert Einstein in 1915.",
  "ground_truth": "Albert Einstein developed general relativity in 1915."
}
```

**Expected output**

| Parameter             | Expected range | Notes                                    |
|-----------------------|----------------|------------------------------------------|
| context_relevance     | ≥ 0.40         | answer terms appear in context           |
| context_precision     | ≥ 0.40         | answer drawn from context                |
| context_redundancy    | ≥ 0.40         | two distinct chunks                      |
| faithfulness          | ≥ 0.60         | claim entailed by chunk 1                |
| answer_relevance      | ≥ 0.40         | answer addresses the question            |
| completeness          | ≥ 0.40         | both sub-questions covered               |
| hallucination_score   | ≤ 0.40         | = 1 − faithfulness                       |
| correctness           | ≥ 0.50         | matches ground truth closely             |
| composite             | ≥ 0.45         | overall healthy                          |
| flagged_claims        | empty          | no flagged claims                        |

**Pass criteria** — composite ≥ 0.45, faithfulness ≥ 0.6, no flagged claims.

---

## Case 2 — Hallucinated answer (factually wrong)

**Input**
```json
{
  "query": "How many planets are in our solar system?",
  "context": [
    "Our solar system contains eight planets.",
    "Pluto was reclassified as a dwarf planet in 2006."
  ],
  "answer": "There are nine planets in our solar system, including Pluto.",
  "ground_truth": "Eight planets."
}
```

**Expected output**

| Parameter             | Expected range | Notes                                       |
|-----------------------|----------------|---------------------------------------------|
| faithfulness          | < Case 1's     | answer contradicts context                  |
| hallucination_score   | > Case 1's     | inverse of faithfulness                     |
| correctness           | < 0.7          | wrong answer                                |
| flagged_claims        | non-empty      | the "nine planets" claim should be flagged  |

**Pass criteria** — Case 2 faithfulness must be **strictly less than Case 1 faithfulness**.

---

## Case 3 — Off-topic answer

**Input**
```json
{
  "query": "What is the capital of France?",
  "context": ["Paris is the capital of France."],
  "answer": "Bananas grow on trees in tropical climates.",
  "ground_truth": "Paris."
}
```

**Expected output**

| Parameter             | Expected     |
|-----------------------|--------------|
| answer_relevance      | very low     |
| context_precision     | very low     |
| faithfulness          | very low     |
| flagged_claims        | non-empty    |
| composite             | < 0.4        |

**Pass criteria** — answer_relevance and faithfulness both clearly low.

---

## Case 4 — Empty context (retrieval failure)

**Input**
```json
{
  "query": "Who painted the Mona Lisa?",
  "context": [],
  "answer": "Leonardo da Vinci painted the Mona Lisa."
}
```

**Expected behavior**

- Evaluator does **not crash**.
- `context_relevance.n_samples == 0`.
- `faithfulness` is None (NLI engine skipped — no context to entail against).
- `composite` is computed only from `answer_relevance` (and `correctness` if GT provided).

**Pass criteria** — no exceptions thrown; result object well-formed.

---

## Case 5 — Single-word answer

**Input**
```json
{
  "query": "Capital of France?",
  "context": ["Paris is the capital of France."],
  "answer": "Paris"
}
```

**Expected behavior**

- Evaluator handles minimal answer length.
- `n_claims` may be 1 or 0 depending on splitter.
- `composite` is still produced.

**Pass criteria** — no exceptions; composite present.

---

## Case 6 — Highly redundant context

**Input**
```json
{
  "query": "Who developed relativity?",
  "context": [
    "Einstein developed relativity in 1915.",
    "Einstein developed relativity in 1915.",
    "Einstein developed relativity in 1915."
  ],
  "answer": "Einstein developed relativity in 1915."
}
```

**Expected output**

| Parameter             | Expected     |
|-----------------------|--------------|
| context_redundancy    | ≤ 0.5        | identical chunks → high redundancy → low diversity score |

**Pass criteria** — context_redundancy score ≤ 0.5.

---

## Case 7 — Diverse context

**Input**
```json
{
  "query": "Tell me about a few different topics.",
  "context": [
    "Apples grow on trees in orchards across temperate climates.",
    "Quantum mechanics describes the behavior of subatomic particles.",
    "The Eiffel Tower stands in Paris, France, completed in 1889."
  ],
  "answer": "Apples grow on trees, quantum mechanics studies particles, and the Eiffel Tower is in Paris."
}
```

**Expected output**

| Parameter             | Expected     |
|-----------------------|--------------|
| context_redundancy    | ≥ 0.7        | unrelated chunks → low redundancy → high diversity |

**Pass criteria** — context_redundancy ≥ 0.7.

---

## Case 8 — ANN claim validator on cold start

**Setup** — fresh `ANNValidator` with random init, no training.

**Expected behavior**

- `predict([sentences])` returns scores in [0, 1].
- `filter_atomic([s])` with `threshold=0.99` falls back to keeping all sentences
  (cold-start safety).

**Pass criteria** — no empty list returned; safety fallback engages.

---

## Case 9 — ANN claim validator after training

**Setup**

```python
ann = ANNValidator(embedder)
ann.train_synthetic(
    clean_paragraphs=[
        "Einstein developed relativity in 1915. The theory describes gravity.",
        "Paris is the capital of France. The Seine flows through Paris.",
        "Photosynthesis converts sunlight. Plants release oxygen as a byproduct.",
    ],
    epochs=15,
    lr=2e-2,
)
```

**Expected output**

```python
clean = ann.predict(["Albert Einstein developed the theory of relativity."])
fragment = ann.predict(["Einstein developed"])
```

**Pass criteria** — `clean.mean() > fragment.mean()` after training.

Also: `losses[-1] < losses[0]` — i.e. backprop reduces loss across epochs.

---

## Case 10 — Confidence intervals are non-zero on multi-evidence parameters

**Setup** — Case 1 input, ≥ 2 chunks.

**Expected output**

- `context_relevance.std > 0` (variance across multiple chunks).
- `faithfulness.std ≥ 0` (variance across multiple claims).

**Pass criteria** — std field reports positive variance whenever n_samples ≥ 2.

---

## Case 11 — Composite score uncertainty propagates correctly

**Verification**

Given:
- context_relevance: `0.8 ± 0.1` (weight 0.15)
- faithfulness:     `0.9 ± 0.2` (weight 0.30)
- all other params: None

Expected composite mean = `(0.15*0.8 + 0.30*0.9) / (0.15+0.30) ≈ 0.8667`
Expected composite std  = `sqrt((0.15² × 0.1²) + (0.30² × 0.2²)) / 0.45 ≈ 0.1374`

**Pass criteria** — values match within `1e-6`. Tested in `test_scorer.py::TestCompositeScore::test_uncertainty_propagates`.

---

## Case 12 — Round-trip persistence (ANN weights)

**Steps**

1. Create ANN, predict on a sample.
2. Save weights to disk.
3. Load weights into a fresh ANN instance.
4. Predict on the same sample.

**Pass criteria** — predictions match within `1e-9`. Tested in `test_ann_validator.py::TestPersistence::test_save_and_load_roundtrip`.

---

## Running the full check

```bash
pytest -v tests/
```

Expected: **all tests pass**, no skips, no warnings about missing models
(unit suite uses dummy embedder; no model downloads required).

For a smoke test against real models (downloads ~250 MB):

```bash
python -m ragwatch.cli.main demo
```

This runs Case 1 against the real `all-MiniLM-L6-v2` + `nli-deberta-v3-small` and
prints a console report.

---

# v0.2 — Meta-evaluation (Option A) Test Cases

## Case 13 — Synthetic case set is well-formed

**Verification**

```python
from ragwatch.meta_eval.synthetic_cases import (
    get_synthetic_cases, get_known_good_cases, get_known_bad_cases, case_count_summary
)
cases = get_synthetic_cases()
```

**Pass criteria**
- `len(cases) >= 25` — at least 25 hand-crafted cases
- All "faithful" tag cases have `is_good=True` and `true_faithfulness=1.0`
- All "hallucinated" tag cases have `is_good=False` and `true_faithfulness=0.0`
- `case_count_summary()["TOTAL"]` equals `len(cases)`

Tested in `test_meta_eval.py::TestSyntheticCases`.

---

## Case 14 — Pearson correlation behaves correctly

**Verification**

```python
from ragwatch.meta_eval.correlation import pearson
assert pearson([1,2,3,4,5], [2,4,6,8,10]) == 1.0  # perfect positive
assert pearson([1,2,3,4,5], [10,8,6,4,2]) == -1.0 # perfect negative
assert pearson([1,1,1,1], [2,4,6,8]) == 0.0       # zero variance → 0
```

**Pass criteria** — values match within `1e-9`.

Tested in `test_correlation.py::TestPearson`.

---

## Case 15 — AUROC perfectly separates labeled cases

**Verification**

```python
from ragwatch.meta_eval.correlation import auroc
assert auroc([0.9, 0.8, 0.7, 0.2, 0.1], [1,1,1,0,0]) == 1.0
```

**Pass criteria** — AUROC = 1.0 exactly when scores perfectly separate classes.

Tested in `test_correlation.py::TestAUROC`.

---

## Case 16 — Calibration error is zero on perfect predictions

**Verification**

```python
from ragwatch.meta_eval.correlation import expected_calibration_error
d = expected_calibration_error([0.1, 0.3, 0.5, 0.7, 0.9],
                               [0.1, 0.3, 0.5, 0.7, 0.9], n_bins=5)
assert d["ece"] < 1e-9
```

**Pass criteria** — ECE < 1e-9 when predictions equal truth.

Tested in `test_correlation.py::TestECE`.

---

## Case 17 — MetaEvaluator distinguishes good from bad cases

**Setup** — patched evaluator (DummyEmbedder + token-overlap NLI), full case set.

**Pass criteria**
- `meta.per_tag_composite["faithful"] > meta.per_tag_composite["off_topic"]`
- `meta.trust_score` is in `[0, 1]`
- `meta.trust_label ∈ {TRUSTED, ACCEPTABLE, MARGINAL, UNRELIABLE}`

Tested in `test_meta_eval.py::TestMetaEvaluator::test_distinguishes_good_from_bad`.

---

## Case 18 — Self-trust report renders correctly

**Verification**

```python
from ragwatch.meta_eval.report import to_console
out = to_console(meta_result)
assert "TRUST SCORE" in out
assert "Faithfulness Agreement" in out
```

**Pass criteria** — console output contains expected sections.

Tested in `test_meta_eval.py::TestReports`.

---

# v0.2 — ONNX Quantization (Option B) Test Cases

## Case 19 — ONNX export module imports cleanly

**Verification**

```python
from ragwatch.utils.onnx_export import (
    DEFAULT_MODEL, DEFAULT_OUT_DIR, ONNXNLIRunner, export_and_quantize
)
```

**Pass criteria** — no import errors. `DEFAULT_MODEL` references deberta.

Tested in `test_onnx_export.py::TestImports`.

---

## Case 20 — ONNX runner raises clear error when model file missing

**Verification**

```python
from ragwatch.utils.onnx_export import ONNXNLIRunner
ONNXNLIRunner(model_dir="/nonexistent/path")
# expected: FileNotFoundError mentioning 'quantize'
```

**Pass criteria** — `FileNotFoundError` with message guiding user to run quantize command.

Tested in `test_onnx_export.py::TestONNXRunnerLazyLoad`.

---

## Case 21 — NLIEngine accepts onnx_dir without crashing

**Verification**

```python
from ragwatch.engines.nli_engine import NLIEngine
eng = NLIEngine(embedder, onnx_dir="/some/path")
assert eng.backend == "none"  # before _load() is called
```

**Pass criteria** — engine constructs successfully with `onnx_dir` flag.

Tested in `test_onnx_export.py::TestNLIEngineWithOnnxFlag`.

---

## Case 22 — NLIEngine falls back to PyTorch when ONNX missing

**Setup** — `onnx_dir` provided but no `model_quantized.onnx` in it.

**Pass criteria**
- `_load()` does not raise on the ONNX-missing path.
- `backend` becomes `"pytorch"` (or import-skips if sentence-transformers unavailable).

Tested in `test_onnx_export.py::TestNLIEngineWithOnnxFlag::test_engine_falls_back_to_pytorch_when_onnx_missing`.

---

## Case 23 — Setup check exits 0 on a valid environment

**Verification**

```bash
python setup_check.py
echo $?  # expected: 0 if numpy + scipy installed and ragwatch importable
```

**Pass criteria** — script reports OK for required deps and returns 0.

This is a manual integration check, not a pytest case.

---

# Real-Model Integration (opt-in)

These cases require ~250 MB of model downloads. They are NOT in the default
suite. Run manually:

```bash
python -m ragwatch.cli.main demo       # 1 case
python -m ragwatch.cli.main meta       # 30 cases
python -m ragwatch.cli.main quantize   # one-time export
python -m ragwatch.cli.main demo       # re-run; should now use ONNX backend
```

**Pass criteria**
- `demo` completes without exception.
- `meta` produces a `TRUST SCORE >= 0.65` (ACCEPTABLE or better) on real models.
- After `quantize`: `.ragwatch_cache/nli_onnx/model_quantized.onnx` exists,
  size between 40-80 MB.
- After `quantize`: subsequent `demo` runs are noticeably faster
  (typically 2-3× speedup on CPU).

---

# v0.3 — Pytest Plugin Test Cases

## Case 24 — `evaluate_rag` happy path

**Setup**
```python
from ragwatch.pytest_plugin import evaluate_rag

cases = [{"query": "Q", "context": ["c"], "answer": "A"}]
report = evaluate_rag(cases, threshold=0.0)
```

**Pass criteria**
- `report.passed is True` when threshold=0
- `report.n_cases == len(cases)`
- `report.composite_mean` in [0, 1]

Tested in `test_pytest_plugin.py::TestEvaluateRag::test_returns_report`.

---

## Case 25 — `evaluate_rag` empty/non-list rejection

**Pass criteria**
- `evaluate_rag([])` raises ValueError
- `evaluate_rag({...})` (dict, not list) raises TypeError

Tested in `test_pytest_plugin.py::TestEvaluateRag::test_empty_cases_rejected`,
`test_non_list_rejected`.

---

## Case 26 — Threshold breach populates failures

**Setup**
```python
report = evaluate_rag([GOOD_CASE], threshold=0.99)  # impossible
```

**Pass criteria**
- `report.passed is False`
- `report.failures` is non-empty
- `report.summary()` contains "FAILED"

Tested in `test_pytest_plugin.py::TestEvaluateRag::test_failures_populated_on_low_threshold`.

---

## Case 27 — `case_from_dict` field tolerance

**Verification** — accepts both canonical and alternative field names:
- `query` / `question`
- `context` / `contexts` / `chunks`
- `answer` / `response` / `output`
- `ground_truth` / `gt` / `reference`

Also auto-promotes single string context to a list.

Tested in `test_pytest_plugin.py::TestCaseFromDict`.

---

## Case 28 — `Thresholds.strict` / `Thresholds.permissive` presets

**Verification**
```python
Thresholds.strict()      # composite=0.80, faithfulness=0.75
Thresholds.permissive()  # composite=0.5
```

Tested in `test_pytest_plugin.py::TestThresholds`.

---

## Case 29 — Per-case `evaluate_one` works in pytest.parametrize

**Setup**
```python
@pytest.mark.parametrize("case", RAG_CASES, ids=lambda c: c["query"][:30])
def test_individual_case(case):
    result = evaluate_one(case)
    assert result.composite is not None
```

**Pass criteria**
- Each case shows up by its query in pytest output.
- Per-case granular pass/fail.

Tested in `test_pytest_plugin.py::TestPytestWorkflow::test_parametrize_pattern`.

---

## Case 30 — Evaluator cache reuse

**Verification**

```python
from ragwatch.pytest_plugin.api import _get_evaluator, reset_evaluator_cache

ev1 = _get_evaluator(mode="cpu_safe")
ev2 = _get_evaluator(mode="cpu_safe")
assert ev1 is ev2  # cached

reset_evaluator_cache()
ev3 = _get_evaluator(mode="cpu_safe")
assert ev1 is not ev3  # cache cleared
```

**Pass criteria** — cache works; reset clears it.

Tested in `test_pytest_plugin.py::TestEvaluatorCache`.

---

## Case 31 — `assert report.passed, report.summary()` pattern

**The canonical user pattern**:
```python
def test_my_rag():
    report = evaluate_rag(MY_CASES, threshold=0.7)
    assert report.passed, report.summary()
```

**Pass criteria** — `report.summary()` produces a multi-line string with
parameter values, status, and (if any) failure list.

Tested in `test_pytest_plugin.py::TestPytestWorkflow::test_simple_assertion_pattern`.

---

## Case 32 — `report.__bool__` works for truthy checks

**Verification**
```python
report = evaluate_rag(cases, threshold=0.0)
if report:
    print("ok")
```

**Pass criteria** — `bool(report)` reflects `report.passed`.

Tested in `test_pytest_plugin.py::TestEvaluateRag::test_bool_dunder`.
