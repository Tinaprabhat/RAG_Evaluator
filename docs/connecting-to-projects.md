# Connecting RAGWatch to Your Project

This guide walks through wiring RAGWatch into any RAG project so that
evaluation runs as part of `pytest` — no GUI, no upload, no separate tool.

---

## TL;DR — Three Steps

```powershell
# 1. Install RAGWatch system-wide (once, ever)
pip install -e <path-to-ragwatch_v3>

# 2. Drop one file into your project's tests/ folder
#    → tests/test_rag_eval.py

# 3. Run as part of your normal test suite
pytest tests/test_rag_eval.py -v
```

That's it. RAG quality is now a unit test.

---

## Step 1 — Install RAGWatch

### Option A — Editable install (development)
From the directory containing the unzipped `ragwatch_v3/`:

```powershell
cd C:\where-you-unzipped\ragwatch_v3
pip install -e .
```

This makes `ragwatch` importable from any project and installs the `ragwatch`
console script.

### Option B — Wheel install (production)
```powershell
cd C:\where-you-unzipped\ragwatch_v3
python -m build
pip install dist/ragwatch-0.3.0-py3-none-any.whl
```

### Verify
```powershell
ragwatch --help
python -c "from ragwatch.pytest_plugin import evaluate_rag; print('OK')"
```

---

## Step 2 — Add a test file to YOUR project

Your project structure stays as it is. Just add **one** file:

```
your-rag-project/
├── src/
│   └── my_rag.py             ← your existing RAG code
├── tests/
│   ├── test_unit.py          ← your existing tests
│   ├── test_integration.py   ← your existing tests
│   └── test_rag_eval.py      ← NEW — RAGWatch eval lives here
└── ...
```

### Pattern 1 — Static test cases (simplest)

Hardcode known-good cases. Best for regression tests:

```python
# tests/test_rag_eval.py
"""RAG quality evaluation via RAGWatch."""

from ragwatch.pytest_plugin import evaluate_rag

RAG_CASES = [
    {
        "query": "What does Assignment 1 require?",
        "context": [
            "Assignment 1 requires students to build a RAG pipeline.",
            "Submission deadline is October 15.",
        ],
        "answer": "Assignment 1 requires building a RAG pipeline, due October 15.",
        "ground_truth": "Build RAG pipeline, due Oct 15.",
    },
    {
        "query": "When is the midterm?",
        "context": ["Midterm is scheduled for week 7."],
        "answer": "The midterm is in week 7.",
        "ground_truth": "Week 7.",
    },
]


def test_rag_quality():
    """Pass if every case meets the quality bar."""
    report = evaluate_rag(RAG_CASES, threshold=0.7)
    assert report.passed, report.summary()
```

Run:
```powershell
pytest tests/test_rag_eval.py -v
```

### Pattern 2 — Per-case parametrize (granular pass/fail)

Each case becomes its own pytest test:

```python
import pytest
from ragwatch.pytest_plugin import evaluate_one

RAG_CASES = [
    {"query": "...", "context": [...], "answer": "...", "ground_truth": "..."},
    # ... more cases
]


@pytest.mark.parametrize("case", RAG_CASES, ids=lambda c: c["query"][:30])
def test_individual_case(case):
    """Each case shows up by name in pytest output."""
    result = evaluate_one(case)
    assert result.faithfulness.score >= 0.6, \
        f"hallucinated: {case['query']!r} → {result.faithfulness.score:.3f}"
    assert result.hallucination_score.score <= 0.4
```

Pytest output:
```
test_individual_case[What does Assignment 1 require...]  PASSED
test_individual_case[When is the midterm...]             PASSED
test_individual_case[Who developed relativity...]        FAILED
```

### Pattern 3 — Live RAG integration (catches regressions)

Import your actual RAG code, evaluate the live output:

```python
# tests/test_rag_eval.py
from ragwatch.pytest_plugin import evaluate_rag
from src.my_rag import answer_question  # your real RAG entry point


QUERIES = [
    "What does Assignment 1 require?",
    "When is the midterm?",
    "How is the final graded?",
]

EXPECTED = {
    "What does Assignment 1 require?": "Build RAG pipeline, due Oct 15.",
    "When is the midterm?": "Week 7.",
    "How is the final graded?": "30% project + 70% exam.",
}


def test_rag_pipeline_quality():
    """Run the actual RAG, evaluate the live outputs."""
    cases = []
    for q in QUERIES:
        result = answer_question(q)  # your RAG runs
        cases.append({
            "query": q,
            "context": result["chunks"],
            "answer": result["answer"],
            "ground_truth": EXPECTED.get(q),
        })
    report = evaluate_rag(cases, threshold=0.7)
    assert report.passed, report.summary()
```

This pattern catches regressions when you change retriever/LLM/prompts.

### Pattern 4 — Custom thresholds

Use `Thresholds` for fine-grained control:

```python
from ragwatch.pytest_plugin import evaluate_rag, Thresholds

# strict bar for production-critical RAG
def test_production_rag_quality():
    report = evaluate_rag(RAG_CASES, thresholds=Thresholds.strict())
    assert report.passed, report.summary()


# relaxed bar for early-stage R&D RAG
def test_experimental_rag_quality():
    report = evaluate_rag(RAG_CASES, thresholds=Thresholds.permissive())
    assert report.passed, report.summary()


# fully custom
def test_custom_thresholds():
    custom = Thresholds(
        composite=0.65,
        faithfulness=0.70,
        hallucination_max=0.30,
        answer_relevance=0.50,
    )
    report = evaluate_rag(RAG_CASES, thresholds=custom)
    assert report.passed, report.summary()
```

---

## Step 3 — Load test cases from a file (optional)

For larger eval sets, store cases in a JSON/JSONL file checked into git:

```python
# tests/test_rag_eval.py
import json
from pathlib import Path
from ragwatch.pytest_plugin import evaluate_rag

EVAL_DATA = Path(__file__).parent / "rag_eval_cases.json"


def test_rag_quality():
    cases = json.loads(EVAL_DATA.read_text(encoding="utf-8"))
    report = evaluate_rag(cases, threshold=0.7)
    assert report.passed, report.summary()
```

```
your-rag-project/
└── tests/
    ├── rag_eval_cases.json    ← your test cases (committed)
    └── test_rag_eval.py
```

Format of `rag_eval_cases.json`:
```json
[
  {
    "query": "...",
    "context": ["...", "..."],
    "answer": "...",
    "ground_truth": "..."
  }
]
```

---

## Step 4 — CI/CD Integration

Because it's just pytest, RAG quality runs in CI for free:

### GitHub Actions
```yaml
# .github/workflows/test.yml
- name: Install dependencies
  run: |
    pip install -r requirements.txt
    pip install ragwatch

- name: Run RAG quality tests
  run: pytest tests/test_rag_eval.py -v
```

### GitLab CI
```yaml
test:
  script:
    - pip install ragwatch
    - pytest tests/test_rag_eval.py
```

Every PR fails the build if RAG quality regresses. Real production guardrails.

---

## Tuning: First Run, Iterate, Lock

### First run — see where you stand

Use a permissive threshold to get a baseline:
```python
def test_baseline():
    report = evaluate_rag(RAG_CASES, threshold=0.0)  # any score passes
    print(report.summary())
    # always passes — but prints aggregate metrics
    assert True
```

Run:
```powershell
pytest tests/test_rag_eval.py -v -s
```

### Look at the numbers

```
RAGWatch quality report — 10 case(s)
  composite     : 0.732 ± 0.094
  faithfulness  : 0.811
  hallucination : 0.189
```

Now you know your real baseline.

### Set a realistic threshold

Drop the threshold ~5% below your baseline to catch regressions
without false alarms:

```python
def test_rag_quality():
    report = evaluate_rag(RAG_CASES, threshold=0.68)  # baseline 0.73 - 0.05
    assert report.passed, report.summary()
```

---

## Advanced: Connecting via the Standalone CLI

If you want the report as HTML/JSON outside of pytest:

```powershell
# from your project root
ragwatch run --input tests/rag_eval_cases.json --report html --out report.html
```

```powershell
# meta-evaluation: how trustworthy is RAGWatch on your data?
ragwatch meta --out trust.json
```

```powershell
# one-time: quantize the NLI model for 4× faster CPU inference
ragwatch quantize
```

---

## Troubleshooting

**`ImportError: ragwatch.pytest_plugin`**
→ Reinstall: `pip install -e <path-to-ragwatch_v3>`

**Tests are slow (>30 s per case)**
→ Run `ragwatch quantize` once. Drops NLI from 250 ms to ~60 ms per claim.

**Dummy `0.5` faithfulness scores**
→ NLI model not loaded yet. Check `python -m ragwatch.cli.main demo` works.

**`spaCy model not found`**
→ Optional. Run `python -m spacy download en_core_web_sm` or ignore — regex
splitter works fine.

**Want to disable specific engines**
→ Pass a custom Config:
```python
from ragwatch.core.config import Config
from ragwatch.pytest_plugin import evaluate_rag

cfg = Config.cpu_safe()
cfg.use_self_consistency = False
report = evaluate_rag(cases, config=cfg)
```

---

## What This Replaces

| Before RAGWatch                   | With RAGWatch pytest plugin       |
|-----------------------------------|-----------------------------------|
| Manual eval scripts in notebooks  | `pytest tests/test_rag_eval.py`   |
| Run, eyeball, hope for the best   | Hard fail on threshold breach     |
| "I'll evaluate later"             | Runs every commit                 |
| LLM-judge API calls (cost $$)     | Fully local, free, deterministic  |
| Eval data in random folders       | `tests/rag_eval_cases.json` (git) |
| No team consistency               | Same metrics, same thresholds     |
