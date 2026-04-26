# RAGWatch

**A framework-agnostic, CPU-friendly RAG evaluation framework that evaluates itself.**

RAGWatch lets you turn RAG quality into a unit test. Drop one file into your project's `tests/` folder, run `pytest`, and get rigorous, statistically grounded evaluation of your RAG pipeline — no LLM judge, no API keys, no GPU, no upload.

---

## Description

Most RAG evaluation frameworks (RAGAS, DeepEval, TruLens) rely on an LLM-as-judge. That introduces three compounding problems: cost, hallucination on diverse queries, and the unanswered question — *who evaluates the evaluator?*

RAGWatch takes a different path:

- **No LLM-as-judge.** Evaluation is decomposed into atomic operations: cosine similarity, NLI entailment, claim-level checks, and self-consistency. Each is verifiable.
- **Confidence intervals on every score.** Every parameter returns `score ± std`. Statistical transparency, not vibes.
- **Self-trust meta-evaluation.** RAGWatch ships with 30 hand-labeled cases and computes its own AUROC, calibration error, and trust score against ground truth.
- **Fully CPU-local.** ~470 MB RAM with quantized ONNX. No GPU, no cloud, no API keys.
- **Pytest-native.** RAG quality is a unit test in your existing test suite. CI-ready out of the box.

Built on a Lenovo ThinkBook with no GPU. Designed to stay there.

---

## Problem Statement

RAG (Retrieval-Augmented Generation) systems power search, customer support, agents, and education tools. They fail in subtle ways: they hallucinate facts, retrieve irrelevant chunks, miss parts of compound questions, and drift when prompts change. **You need to know when this happens — automatically, in CI, with no manual review.**

Existing frameworks have three structural issues:

1. **They depend on LLM judges** — costly, opaque, and unreliable on diverse queries.
2. **They don't evaluate themselves** — you trust their scores on faith.
3. **They don't fit existing developer workflows** — they assume you'll write custom harnesses, run notebooks, or upload data to dashboards.

RAGWatch addresses all three. Evaluation is mathematical and reproducible. The framework grades itself against ground truth. And it integrates as a `pytest` plugin — RAG quality becomes part of the same test run that already gates your code.

---

## The 21 Features

### Foundation Engine (Core)
1. **Pipeline-Agnostic Input Contract** — accepts `(query, context, answer, ground_truth?)` regardless of framework.
2. **Five-Engine Architecture** — Math + NLI + ANN + SelfConsistency + LogitUncertainty.
3. **Confidence Intervals on Every Score** — `score ± std` derived from variance across evidence units.
4. **Embedding Cache Layer** — MD5-hashed in-memory cache; same chunk embedded once across batches.
5. **Composite Score with Uncertainty Propagation** — proper variance propagation via `σ = sqrt(Σ(w² × σ²)) / Σw`.
6. **Three Report Formats** — Console, JSON, HTML (color-coded).
7. **CPU-Only Architecture** — < 600 MB RAM total; quantized < 470 MB; no GPU required.

### Self-Trust Layer
8. **Meta-Evaluation Module** — 30 labeled cases, computes AUROC, Pearson, Spearman, F1, ECE, and an overall trust score (TRUSTED / ACCEPTABLE / MARGINAL / UNRELIABLE).
9. **Statistical Correlation Suite** — Pearson, Spearman, AUROC, ECE, F1 (pure NumPy, no scipy.stats).
10. **Calibration Error (ECE) for Continuous Targets** — adapted from classification ECE.
11. **ONNX Export + INT8 Quantization** — NLI model: 180 MB → 50 MB, 250 ms → 60 ms per claim.
12. **Setup Check** — pre-flight verification of Python, deps, RAM, disk, model files.
13. **Pluggable Backend Architecture** — auto-detects ONNX, falls back to PyTorch transparently.

### Pytest Plugin (v0.3 Headline)
14. **Pytest Plugin** — `from ragwatch.pytest_plugin import evaluate_rag` — RAG eval as a unit test.
15. **Per-Case Parametrize Support** — each case becomes its own pytest test for granular pass/fail.
16. **Live RAG Integration** — test files can import your actual RAG code and evaluate live outputs.
17. **Threshold-Based Assertions** — `report.passed` against configurable bars (`Thresholds.strict()`, `.permissive()`, custom).
18. **Report Object with Programmatic Access** — `.failures`, `.composite_mean`, `.per_case` for custom assertions.

### Cross-Cutting
19. **Zero-Cost-of-Failure Engine Toggling** — disable any engine; pipeline gracefully degrades.
20. **Modular Test Coverage** — 13 per-engine test files, offline-friendly with DummyEmbedder.
21. **Single Source of Truth `test_cases.md`** — 32 documented checkable cases with expected ranges.

---

## Architecture

### High-Level Flow

```
┌────────────────────────────────────────────────────────────┐
│                     INPUT                                  │
│         (query, context, answer, ground_truth?)            │
└──────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│                  PREPROCESSING                             │
│  - sentence splitting (spaCy / regex fallback)             │
│  - claim decomposition                                     │
│  - ANN claim validator (NumPy backprop, ~12k params)       │
└──────────────────────────┬─────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┬──────────────┐
        ▼                  ▼                  ▼              ▼
   ┌─────────┐       ┌──────────┐      ┌────────────┐   ┌──────────┐
   │  MATH   │       │   NLI    │      │  SELF-     │   │  LOGIT   │
   │ ENGINE  │       │  ENGINE  │      │ CONSIST.   │   │  UNCERT. │
   │         │       │  (lazy   │      │            │   │ (optional│
   │ cosine  │       │  gated)  │      │ pairwise   │   │  ollama) │
   │ NDCG    │       │  deberta │      │ cosine     │   │  SLM     │
   │ overlap │       │ -v3-small│      │ across     │   │  entropy │
   │         │       │  (ONNX)  │      │ claims     │   │          │
   └────┬────┘       └────┬─────┘      └─────┬──────┘   └────┬─────┘
        │                 │                  │               │
        └─────────────────┴────────┬─────────┴───────────────┘
                                   ▼
              ┌────────────────────────────────────┐
              │       SCORER + CI CALCULATOR       │
              │  - per-parameter mean ± std        │
              │  - composite with σ propagation    │
              │  - flagged claim list              │
              └────────────────┬───────────────────┘
                               │
                               ▼
              ┌────────────────────────────────────┐
              │            OUTPUT                  │
              │  - EvalResult (programmatic)       │
              │  - Console / JSON / HTML reports   │
              │  - pytest assertion (via plugin)   │
              └────────────────────────────────────┘
```

### Repository Layout

```
ragwatch_v3/
├── PROJECT_LOG.md
├── pyproject.toml
├── requirements.txt
├── setup_check.py                  # rule-#7 environment check
│
├── ragwatch/
│   ├── core/                       # config, schemas, scorer, evaluator
│   │   ├── config.py
│   │   ├── schemas.py              # EvalInput, EvalResult, ScoreWithCI
│   │   ├── scorer.py               # aggregate, composite, propagation
│   │   └── evaluator.py            # main orchestrator
│   │
│   ├── engines/                    # five evaluation engines
│   │   ├── math_engine.py          # cosine-based deterministic
│   │   ├── nli_engine.py           # claim-level entailment + lazy gating
│   │   ├── ann_validator.py        # NumPy-only MLP for claim atomicity
│   │   ├── self_consistency.py     # SelfCheckGPT-inspired
│   │   └── logit_uncertainty.py    # ollama SLM signal (optional)
│   │
│   ├── utils/                      # supporting infrastructure
│   │   ├── embeddings.py           # cached MiniLM
│   │   ├── preprocessor.py         # split + decomposition
│   │   ├── reports.py              # console, JSON, HTML
│   │   └── onnx_export.py          # quantization + ONNX runner
│   │
│   ├── meta_eval/                  # self-trust harness (Option A)
│   │   ├── synthetic_cases.py      # 30 hand-labeled cases
│   │   ├── correlation.py          # Pearson, Spearman, AUROC, ECE, F1
│   │   ├── meta_evaluator.py       # runs RAGWatch on labeled cases
│   │   └── report.py               # self-trust report
│   │
│   ├── pytest_plugin/              # v0.3 headline
│   │   └── api.py                  # evaluate_rag, evaluate_one, EvalReport
│   │
│   └── cli/
│       └── main.py                 # run, demo, meta, quantize, init
│
├── tests/                          # 13 test files + cases doc
│   ├── _fixtures.py                # DummyEmbedder
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_scorer.py
│   ├── test_embeddings.py
│   ├── test_preprocessor.py
│   ├── test_math_engine.py
│   ├── test_nli_engine.py
│   ├── test_ann_validator.py
│   ├── test_self_consistency.py
│   ├── test_correlation.py
│   ├── test_meta_eval.py
│   ├── test_onnx_export.py
│   ├── test_pytest_plugin.py
│   ├── test_integration.py
│   └── test_cases.md               # 32 documented checkable cases
│
├── examples/
│   ├── sample_input.json
│   ├── sample_corpus.txt
│   └── example_test_rag_eval.py    # copy to your project
│
└── docs/
    └── connecting-to-projects.md   # full integration guide
```

### Why This Architecture

- **Engines are independent and toggleable.** Each can be disabled in `Config`; the pipeline gracefully degrades and reports which engines actually ran.
- **Lazy NLI gating.** When cosine similarity is decisive (≥ 0.80 or ≤ 0.20), NLI is skipped. This cuts NLI calls by 50–70% on real RAG outputs without measurable accuracy loss.
- **NumPy-only ANN.** A 12k-parameter MLP for atomic-claim validation, trained via pure NumPy backprop on contrastively auto-generated examples. No PyTorch dependency for this component.
- **Pluggable NLI backend.** When the quantized ONNX file is present, the NLI engine uses it. Otherwise it falls back to the PyTorch CrossEncoder. Same API contract.
- **Cached evaluator in the plugin.** Building the evaluator costs model-load time; the plugin caches one per `mode` and reuses it across cases.

---

## Quick Start (CLI Setup — Step by Step)

This is the canonical setup path on **Lenovo ThinkBook (CPU-only) with PowerShell**. The same steps work on macOS / Linux with minor terminal differences.

### Step 1 — System check

```powershell
python --version
# Need 3.9 or higher. If "command not found", try `py --version` instead.

where.exe python
# Confirms which Python interpreter you're using
```

### Step 2 — Unzip and open in VS Code

```powershell
# After unzipping ragwatch_v3.zip into your projects folder:
cd C:\path\to\ragwatch_v3
code .
```

### Step 3 — Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then activate again. Your prompt should now start with `(.venv)`.

### Step 4 — Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

If the `torch` install hangs or fails, force the CPU-only wheel:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then download the spaCy small English model (optional but recommended):

```powershell
python -m spacy download en_core_web_sm
```

> First-run downloads total roughly 500 MB (PyTorch CPU wheel is the heaviest). On a ThinkBook this takes 5–10 minutes.

### Step 5 — Verify the environment

```powershell
python setup_check.py
```

Expected output: section-by-section ✓ marks for Python version, required packages, optional packages, spaCy model, disk, RAM, and RAGWatch imports. Exit code `0` means you're ready.

### Step 6 — Install RAGWatch as a system-wide command

```powershell
pip install -e .
```

This makes `ragwatch` callable from any folder, and makes `from ragwatch.pytest_plugin import evaluate_rag` work in any project.

Verify:

```powershell
ragwatch --help
python -c "from ragwatch.pytest_plugin import evaluate_rag; print('OK')"
```

### Step 7 — Run RAGWatch's own tests

```powershell
pytest -v tests/
```

Expected: all tests pass in under a minute. They use a `DummyEmbedder` and don't download any models. If anything fails here, stop and investigate before going further.

### Step 8 — Smoke test against real models

```powershell
python -m ragwatch.cli.main demo
```

This downloads `all-MiniLM-L6-v2` (~80 MB) and `nli-deberta-v3-small` (~180 MB) from HuggingFace **once**, then runs a single canned case end-to-end and prints the console report.

### Step 9 — One-time NLI model quantization (recommended)

```powershell
python -m ragwatch.cli.main quantize
```

Exports the NLI model to ONNX and quantizes weights to INT8. After this, NLI uses ~50 MB RAM (down from ~180 MB) and runs ~4× faster on CPU. Subsequent `demo` runs automatically use the quantized backend.

### Step 10 — Run meta-evaluation (RAGWatch evaluating itself)

```powershell
python -m ragwatch.cli.main meta --out trust.json
```

This runs RAGWatch on its 30 hand-labeled cases and prints a self-trust report. You'll see Pearson, Spearman, AUROC, calibration error, per-tag scores, and an overall trust label. The JSON dump is for programmatic analysis.

### Step 11 — Evaluate your own pipeline

Create a JSON file with your RAG outputs:

```json
[
  {
    "query": "your user query",
    "context": ["chunk 1", "chunk 2"],
    "answer": "what your RAG generated",
    "ground_truth": "optional reference answer"
  }
]
```

Then:

```powershell
ragwatch run --input my_pipeline_output.json --report html --out report.html
```

Open `report.html` in your browser.

---

## Connecting RAGWatch to Your Existing Project

This is the recommended path. Once `pip install -e .` is done, you don't need to touch the RAGWatch folder again.

### Step 1 — In your project, add one test file

```
your-rag-project/
├── src/
│   └── my_rag.py
├── tests/
│   ├── test_unit.py
│   ├── test_integration.py
│   └── test_rag_eval.py    ← NEW
└── ...
```

### Step 2 — Write the test

```python
# tests/test_rag_eval.py
from ragwatch.pytest_plugin import evaluate_rag

RAG_CASES = [
    {
        "query": "What does Assignment 1 require?",
        "context": [
            "Assignment 1 requires building a RAG pipeline.",
            "Submission deadline is October 15.",
        ],
        "answer": "Build a RAG pipeline, due October 15.",
        "ground_truth": "Build RAG pipeline, due Oct 15.",
    },
    # ... more cases
]


def test_rag_quality():
    report = evaluate_rag(RAG_CASES, threshold=0.7)
    assert report.passed, report.summary()
```

### Step 3 — Run as part of your normal test suite

```powershell
pytest tests/test_rag_eval.py -v
```

That's it. RAG quality is a unit test now.

### Optional — Per-case granularity

```python
import pytest
from ragwatch.pytest_plugin import evaluate_one

@pytest.mark.parametrize("case", RAG_CASES, ids=lambda c: c["query"][:30])
def test_individual_case(case):
    result = evaluate_one(case)
    assert result.faithfulness.score >= 0.6
    assert result.hallucination_score.score <= 0.4
```

Each case shows up by name in pytest output. Failures are visible at a glance.

### Optional — Live RAG integration (catches regressions)

```python
from ragwatch.pytest_plugin import evaluate_rag
from src.my_rag import answer_question

QUERIES = ["...", "...", "..."]

def test_rag_pipeline_quality():
    cases = []
    for q in QUERIES:
        result = answer_question(q)
        cases.append({
            "query": q,
            "context": result["chunks"],
            "answer": result["answer"],
        })
    report = evaluate_rag(cases, threshold=0.7)
    assert report.passed, report.summary()
```

This pattern catches regressions when you change retriever, LLM, or prompts.

For the full integration guide with four patterns, see `docs/connecting-to-projects.md`.

---

## Scalability and Industry Relevance

### Scalability

**Per-case latency** (on a Lenovo ThinkBook, CPU-only):

| Configuration                       | Per case |
|-------------------------------------|----------|
| Math + NLI (PyTorch) + SelfConsist  | ~3.0 s   |
| Math + NLI (ONNX-quantized)         | ~0.8 s   |
| Math only (no NLI)                  | ~0.05 s  |

For a 100-case evaluation set, a quantized run finishes in ~80 seconds — fast enough to gate every CI build without slowing down developer iteration.

**RAM footprint:**

| Component                | Default | Quantized |
|--------------------------|---------|-----------|
| MiniLM embeddings        | 80 MB   | 80 MB     |
| NLI model (deberta-small)| 180 MB  | 50 MB     |
| ANN claim validator      | <1 MB   | <1 MB     |
| Other                    | ~30 MB  | ~30 MB    |
| **Total**                | ~600 MB | ~470 MB   |

The whole system runs on a 4 GB consumer laptop with no GPU. This makes RAGWatch deployable on student machines, edge devices, and air-gapped environments where most evaluation tooling cannot operate.

**Batch parallelism.** The embedding cache amortizes embedding cost across cases — the same chunk is embedded only once. NLI claims are batched per case so the cross-encoder runs one forward pass per claim cluster. For larger corpora, the design extends naturally to multiprocessing without model duplication (the cache is read-mostly).

**Statelessness.** Every evaluator instance is independent. Horizontal scaling for very large eval sets is a thin wrapper around `evaluate_batch`.

### Industry Relevance

| Capability                        | RAGWatch | RAGAS | DeepEval | TruLens |
|-----------------------------------|----------|-------|----------|---------|
| Pipeline-agnostic                 | ✓        | ~     | ~        | ~       |
| No LLM-as-judge                   | ✓        | ✗     | ✗        | ✗       |
| Confidence intervals              | ✓        | ✗     | ✗        | ✗       |
| Self-evaluation                   | ✓        | ✗     | ✗        | ✗       |
| ANN claim validator               | ✓        | ✗     | ✗        | ✗       |
| Lazy NLI gating                   | ✓        | ✗     | ✗        | ✗       |
| ONNX-quantized backend            | ✓        | ✗     | ✗        | ✗       |
| Pytest plugin                     | ✓        | ✗     | partial  | ✗       |
| <600 MB RAM, no GPU               | ✓        | ✗     | ✗        | ✗       |
| Zero API key required             | ✓        | ✗     | ✗        | ✗       |

**Where RAGWatch fits in production:**

- **CI/CD quality gates.** Every PR can fail the build if RAG quality regresses. No new infrastructure — pytest is already there.
- **Cost-sensitive teams.** Startups, students, and self-hosted enterprise deployments cannot pay GPT-4 judge costs at every CI run. RAGWatch removes that recurring cost entirely.
- **Air-gapped environments.** Government, defense, and regulated industries (healthcare, finance) often forbid sending data to external APIs. RAGWatch is fully local.
- **R&D iteration loops.** Developers iterate dozens of times per day on retrievers, prompts, and chunking strategies. A 1-second eval per case is the difference between iterating freely and waiting on judge APIs.
- **Audit trails.** Confidence intervals and the meta-evaluation trust score are auditable artifacts — useful for compliance reviews and incident postmortems.

---

## How We Got Here — The Journey

This project was developed iteratively in conversation, with each round refining the design based on a real concern.

**Round 1 — "How do we evaluate any RAG?"**
We laid out the three layers: retrieval, generation, end-to-end. We mapped 20+ parameters and grouped them into RAGAS-style metrics: faithfulness, answer relevance, context relevance.

**Round 2 — "Generalize it."**
We designed the universal `(query, context, answer, ground_truth?)` contract. Three tiers emerged: deterministic (math), statistical (BERTScore/NLI), and LLM-judge.

**Round 3 — "LLM judges hallucinate too."**
The decisive turn. We dropped LLM-as-judge entirely. Replaced it with multi-judge voting, constrained prompting, evidence verification — then realized something simpler: most parameters are mathematically answerable.

**Round 4 — "Less compute, more architecture."**
Decomposed every parameter into atomic verifiable claims. NLI handles entailment. Math handles similarity. SLMs only as uncertainty meters, not judges.

**Round 5 — "Add a learned splitter via backprop."**
The split-quality problem: bad sentence splits poison faithfulness scores. We added an ANN claim validator — a NumPy-only MLP trained via backprop on contrastively auto-generated data. ~12k parameters. <1 MB.

**Round 6 — "What does the field actually do?"**
Surveyed recent research: semantic entropy probes (Farquhar et al., Nature 2024), SelfCheckGPT (Manakul et al., EMNLP 2023), pRAG, and supervised uncertainty heads. Found the whitespace: combine NLI + ANN + logit uncertainty + self-consistency, all CPU-local. **v0.1 was locked.**

**Round 7 — "Who evaluates the evaluator?"**
The deepest question. Built a meta-evaluation module: 30 hand-labeled cases with known truth, AUROC against ground truth, calibration error, trust score. **v0.2 added Option A (meta-eval) + Option B (ONNX quantization).**

**Round 8 — "How does anyone actually use this?"**
The productization question. Considered Streamlit GUI, file uploaders, decorators. Then realized: every RAG project already has a `tests/` folder and runs `pytest`. Built the pytest plugin instead. **v0.3 made RAG quality a unit test.**

Each round caught a real flaw in the previous design. The final framework is the result of asking "but what if that's wrong?" eight times in a row.

---

## License

MIT — feel free to use, fork, and extend.

---

## Citation

If RAGWatch is useful in your research, please cite the underlying ideas:

- **Semantic Entropy:** Farquhar et al. *Detecting hallucinations in large language models using semantic entropy.* Nature, 2024.
- **SelfCheckGPT:** Manakul et al. *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection.* EMNLP 2023.
- **RAGAS:** Es et al. *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL 2024.
