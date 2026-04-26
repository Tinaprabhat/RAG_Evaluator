# RAGWatch — Project Log

## Project: RAGWatch v0.3.0
A framework-agnostic, CPU-friendly RAG evaluation framework with:
- Self-trust meta-evaluation (no LLM-as-judge needed)
- ONNX-quantized inference (CPU-optimized)
- Pytest-native plugin (RAG eval as unit tests)

---

## Current Version: v0.3.0 — Pytest-Native + Self-Trust + Quantized

This release consolidates v0.1 + v0.2 + v0.3 into a single coherent build.

### All 21 Locked-In Features

#### Foundation Engine (v0.1 lineage)
1. **Pipeline-Agnostic Input Contract** — accepts (query, context, answer, gt?) regardless of framework
2. **Five-Engine Architecture** — Math + NLI + ANN + SelfConsistency + LogitUncertainty
3. **Confidence Intervals on Every Score** — score ± std with proper propagation
4. **Embedding Cache Layer** — MD5-hashed in-memory caching
5. **Composite Score with Uncertainty Propagation** — σ_total = sqrt(Σ(w² × σ²)) / Σw
6. **Three Report Formats** — Console, JSON, HTML
7. **CPU-Only Architecture** — <600MB RAM, no GPU required

#### Self-Trust Layer (v0.2 lineage)
8. **Meta-Evaluation Module** — RAGWatch evaluates itself against 30 labeled cases
9. **Statistical Correlation Suite** — Pearson, Spearman, AUROC, ECE, F1 (pure NumPy)
10. **Calibration Error (ECE)** — adapted for continuous targets
11. **ONNX Export + INT8 Quantization** — 3.6× RAM reduction, 4× speedup
12. **Setup Check** — pre-flight verification with exit codes
13. **Pluggable Backend Architecture** — auto-detect ONNX, fall back to PyTorch

#### Plugin Layer (v0.3 new)
14. **Pytest Plugin** — `from ragwatch.pytest_plugin import evaluate_rag`
15. **Per-Case Parametrize Support** — each case becomes its own pytest test
16. **Live RAG Integration** — test file imports actual RAG code
17. **Threshold-Based Assertions** — `report.passed` against configurable bars
18. **Report Object with Programmatic Access** — `.failures`, `.composite`, `.per_case`

#### Cross-Cutting (all versions)
19. **Zero-Cost-of-Failure Engine Toggling** — graceful degradation per engine
20. **Modular Test Coverage** — per-engine tests, offline-friendly
21. **Single Source of Truth `test_cases.md`** — 25+ documented checkable cases

---

### File Structure
```
ragwatch_v3/
├── PROJECT_LOG.md
├── pyproject.toml
├── requirements.txt
├── setup_check.py
├── ragwatch/
│   ├── core/          # config, schemas, scorer, evaluator
│   ├── engines/       # 5 evaluation engines
│   ├── utils/         # embeddings, preprocessor, reports, onnx_export
│   ├── meta_eval/     # synthetic cases, correlation, meta-evaluator
│   ├── pytest_plugin/ # evaluate_rag, evaluate_one, EvalReport
│   └── cli/           # CLI entry points
├── tests/             # modular pytest suite + test_cases.md
├── examples/          # sample inputs + integration examples
└── docs/              # connecting-to-projects.md
```

update logged successfully

---

## v0.3.0 Build Verification (Final)

### Smoke test results (offline, dummy embedder + token-overlap NLI)

**v0.3 Pytest Plugin (38 checks):**
- F14 evaluate_rag returns EvalReport ✓
- F14 evaluate_rag passed=True at threshold=0 ✓
- F15 evaluate_one returns EvalResult ✓
- F16 mixed EvalInput + dict accepted ✓
- F17 threshold breach → failures populated ✓
- F18 per_case, composite_mean, failures all accessible ✓
- Thresholds.strict / .permissive presets ✓
- case_from_dict alt field names + string→list coercion ✓
- report.summary with PASSED/FAILED markers ✓
- report.__bool__ reflects passed ✓
- Empty list / non-list rejection ✓
- Evaluator cache reuse ✓
- 13 modular test files present ✓
- test_cases.md present ✓

**v0.1 + v0.2 Regression (7 checks):**
- All foundation features intact ✓
- Reports render (JSON, HTML, console) ✓
- Meta-evaluator runs on 30 cases, trust score in [0,1] ✓

### Bug caught and fixed during build
- `evaluate_rag(threshold=...)` shorthand was unintentionally enforcing
  default Thresholds (faithfulness=0.6, hallucination_max=0.4) — fixed
  so threshold-shorthand only enforces composite. Custom Thresholds objects
  passed via `thresholds=` still enforce all fields explicitly set.

### What's NOT in v0.3 (out of scope)
- Real-model integration test that downloads HF models (250 MB)
  → kept as opt-in manual run via `ragwatch demo` / `ragwatch meta`
- Streamlit GUI (deferred — pytest plugin replaces this need)
- HTTP integration with running RAG endpoints (Pattern 3 in docs)

### How to use
1. `pip install -e .` — installs `ragwatch` system-wide + console script
2. Drop `examples/example_test_rag_eval.py` into your project's tests/
3. `pytest tests/test_rag_eval.py -v`

See `docs/connecting-to-projects.md` for the full integration guide.

update logged successfully
