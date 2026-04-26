"""
Synthetic labeled RAG cases for meta-evaluation.

Each case has:
    - input: the (query, context, answer) we'd send to the evaluator
    - true_faithfulness: 1.0 if the answer is fully grounded; 0.0 if it hallucinates
    - true_relevance:    1.0 if the answer addresses the query; 0.0 if off-topic
    - is_good:           True if the answer is overall good
    - tag:               category for sub-group analysis

These are built by hand so we know the ground truth without depending on any
external benchmark or LLM judge.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from ragwatch.core.schemas import EvalInput


@dataclass
class LabeledCase:
    eval_input: EvalInput
    true_faithfulness: float   # in [0, 1] — known ground truth
    true_relevance: float      # in [0, 1]
    is_good: bool              # binary label
    tag: str = ""              # e.g. "faithful", "hallucinated", "off_topic"
    notes: str = ""


# ---------- helper ----------

def _case(query, context, answer, tf, tr, good, tag, notes="", gt=None):
    return LabeledCase(
        eval_input=EvalInput(query=query, context=context, answer=answer, ground_truth=gt),
        true_faithfulness=tf,
        true_relevance=tr,
        is_good=good,
        tag=tag,
        notes=notes,
    )


# ============================================================
# Group 1 — Faithful, well-grounded answers  (n=10)
# ============================================================

GOOD_CASES: list[LabeledCase] = [
    _case(
        "Who developed the theory of general relativity, and in what year?",
        ["Albert Einstein developed the theory of general relativity in 1915.",
         "The theory describes gravity as a curvature of spacetime."],
        "The theory of general relativity was developed by Albert Einstein in 1915.",
        tf=1.0, tr=1.0, good=True, tag="faithful",
        gt="Einstein developed general relativity in 1915.",
    ),
    _case(
        "What is the capital of France?",
        ["Paris is the capital of France.",
         "The Seine river flows through Paris."],
        "Paris is the capital of France.",
        tf=1.0, tr=1.0, good=True, tag="faithful",
        gt="Paris.",
    ),
    _case(
        "How many planets orbit the Sun?",
        ["Our solar system contains eight planets.",
         "Pluto was reclassified as a dwarf planet in 2006."],
        "There are eight planets in our solar system.",
        tf=1.0, tr=1.0, good=True, tag="faithful",
        gt="Eight planets.",
    ),
    _case(
        "What converts sunlight into chemical energy in plants?",
        ["Photosynthesis converts sunlight into chemical energy in plants.",
         "Chlorophyll absorbs light energy."],
        "Photosynthesis converts sunlight into chemical energy in plants.",
        tf=1.0, tr=1.0, good=True, tag="faithful",
        gt="Photosynthesis.",
    ),
    _case(
        "When was the Eiffel Tower completed?",
        ["The Eiffel Tower in Paris was completed in 1889.",
         "It was built for the 1889 Exposition Universelle."],
        "The Eiffel Tower was completed in 1889.",
        tf=1.0, tr=1.0, good=True, tag="faithful",
        gt="1889.",
    ),
    _case(
        "Who wrote Romeo and Juliet?",
        ["Romeo and Juliet was written by William Shakespeare.",
         "It was first published in 1597."],
        "William Shakespeare wrote Romeo and Juliet.",
        tf=1.0, tr=1.0, good=True, tag="faithful",
        gt="Shakespeare.",
    ),
    _case(
        "What is the chemical symbol for gold?",
        ["Gold has the chemical symbol Au.",
         "Its atomic number is 79."],
        "The chemical symbol for gold is Au.",
        tf=1.0, tr=1.0, good=True, tag="faithful",
        gt="Au.",
    ),
    _case(
        "What is the largest organ in the human body?",
        ["The skin is the largest organ in the human body.",
         "It performs many protective functions."],
        "The skin is the largest organ in the human body.",
        tf=1.0, tr=1.0, good=True, tag="faithful",
        gt="The skin.",
    ),
    _case(
        "When did World War II end?",
        ["World War II ended in 1945.",
         "Japan formally surrendered on September 2, 1945."],
        "World War II ended in 1945.",
        tf=1.0, tr=1.0, good=True, tag="faithful",
        gt="1945.",
    ),
    _case(
        "What is the speed of light in vacuum?",
        ["The speed of light in vacuum is approximately 299,792 kilometers per second.",
         "It is denoted by c in physics."],
        "The speed of light in vacuum is approximately 299,792 km/s.",
        tf=1.0, tr=1.0, good=True, tag="faithful",
        gt="About 299,792 km/s.",
    ),
]


# ============================================================
# Group 2 — Hallucinated answers (contradict context)  (n=8)
# ============================================================

HALLUCINATED_CASES: list[LabeledCase] = [
    _case(
        "How many planets are in our solar system?",
        ["Our solar system contains eight planets.",
         "Pluto was reclassified as a dwarf planet in 2006."],
        "There are nine planets in our solar system, including Pluto.",
        tf=0.0, tr=1.0, good=False, tag="hallucinated",
        notes="answer contradicts context (eight vs nine)",
        gt="Eight planets.",
    ),
    _case(
        "When was the Eiffel Tower completed?",
        ["The Eiffel Tower was completed in 1889."],
        "The Eiffel Tower was completed in 1925.",
        tf=0.0, tr=1.0, good=False, tag="hallucinated",
        gt="1889.",
    ),
    _case(
        "Who developed the theory of general relativity?",
        ["Albert Einstein developed the theory of general relativity in 1915."],
        "Isaac Newton developed the theory of general relativity in 1687.",
        tf=0.0, tr=1.0, good=False, tag="hallucinated",
        gt="Einstein.",
    ),
    _case(
        "What is the capital of France?",
        ["Paris is the capital of France."],
        "Berlin is the capital of France.",
        tf=0.0, tr=1.0, good=False, tag="hallucinated",
        gt="Paris.",
    ),
    _case(
        "Who wrote Romeo and Juliet?",
        ["Romeo and Juliet was written by William Shakespeare."],
        "Romeo and Juliet was written by Charles Dickens.",
        tf=0.0, tr=1.0, good=False, tag="hallucinated",
        gt="Shakespeare.",
    ),
    _case(
        "When did World War II end?",
        ["World War II ended in 1945."],
        "World War II ended in 1939.",
        tf=0.0, tr=1.0, good=False, tag="hallucinated",
        gt="1945.",
    ),
    _case(
        "What is the chemical symbol for gold?",
        ["Gold has the chemical symbol Au."],
        "The chemical symbol for gold is Gd.",
        tf=0.0, tr=1.0, good=False, tag="hallucinated",
        gt="Au.",
    ),
    _case(
        "What is the largest planet in our solar system?",
        ["Jupiter is the largest planet in our solar system."],
        "Mars is the largest planet in our solar system.",
        tf=0.0, tr=1.0, good=False, tag="hallucinated",
        gt="Jupiter.",
    ),
]


# ============================================================
# Group 3 — Off-topic answers (relevant context, wrong answer)  (n=6)
# ============================================================

OFF_TOPIC_CASES: list[LabeledCase] = [
    _case(
        "What is the capital of France?",
        ["Paris is the capital of France."],
        "Bananas grow on trees in tropical climates.",
        tf=0.0, tr=0.0, good=False, tag="off_topic",
        gt="Paris.",
    ),
    _case(
        "Who painted the Mona Lisa?",
        ["Leonardo da Vinci painted the Mona Lisa around 1503."],
        "The mitochondria is the powerhouse of the cell.",
        tf=0.0, tr=0.0, good=False, tag="off_topic",
        gt="Leonardo da Vinci.",
    ),
    _case(
        "What is photosynthesis?",
        ["Photosynthesis is the process by which plants convert sunlight into chemical energy."],
        "Football is a popular sport played around the world.",
        tf=0.0, tr=0.0, good=False, tag="off_topic",
        gt="Plants converting sunlight to energy.",
    ),
    _case(
        "When did World War II end?",
        ["World War II ended in 1945."],
        "Pizza was invented in Italy.",
        tf=0.0, tr=0.0, good=False, tag="off_topic",
        gt="1945.",
    ),
    _case(
        "Who wrote Hamlet?",
        ["Hamlet was written by William Shakespeare around 1600."],
        "The Pacific Ocean is the largest ocean.",
        tf=0.0, tr=0.0, good=False, tag="off_topic",
        gt="Shakespeare.",
    ),
    _case(
        "What is the boiling point of water?",
        ["Water boils at 100 degrees Celsius at standard atmospheric pressure."],
        "Cats often sleep more than 12 hours per day.",
        tf=0.0, tr=0.0, good=False, tag="off_topic",
        gt="100°C.",
    ),
]


# ============================================================
# Group 4 — Partial / incomplete answers  (n=6)
# ============================================================

PARTIAL_CASES: list[LabeledCase] = [
    _case(
        "Who developed relativity, and in what year?",
        ["Albert Einstein developed the theory of general relativity in 1915."],
        "Albert Einstein developed the theory.",
        tf=1.0, tr=0.5, good=False, tag="partial",
        notes="missing the year — incomplete",
        gt="Einstein, 1915.",
    ),
    _case(
        "What is the capital of France and what river flows through it?",
        ["Paris is the capital of France.",
         "The Seine river flows through Paris."],
        "Paris is the capital of France.",
        tf=1.0, tr=0.5, good=False, tag="partial",
        notes="missing the river — incomplete",
        gt="Paris and the Seine.",
    ),
    _case(
        "Who wrote Romeo and Juliet, and when was it published?",
        ["Romeo and Juliet was written by William Shakespeare and published in 1597."],
        "Romeo and Juliet was written by Shakespeare.",
        tf=1.0, tr=0.5, good=False, tag="partial",
        notes="missing publication year",
        gt="Shakespeare, 1597.",
    ),
    _case(
        "What converts sunlight to energy in plants and where does it occur?",
        ["Photosynthesis converts sunlight to chemical energy and occurs in chloroplasts."],
        "Photosynthesis converts sunlight to energy.",
        tf=1.0, tr=0.5, good=False, tag="partial",
        notes="missing 'in chloroplasts'",
        gt="Photosynthesis in chloroplasts.",
    ),
    _case(
        "When did World War II start and end?",
        ["World War II started in 1939 and ended in 1945."],
        "World War II ended in 1945.",
        tf=1.0, tr=0.5, good=False, tag="partial",
        gt="1939 to 1945.",
    ),
    _case(
        "What is the chemical symbol and atomic number of gold?",
        ["Gold has the chemical symbol Au and atomic number 79."],
        "Gold's chemical symbol is Au.",
        tf=1.0, tr=0.5, good=False, tag="partial",
        gt="Au, 79.",
    ),
]


# ============================================================
# Public API
# ============================================================

def get_synthetic_cases() -> list[LabeledCase]:
    """All labeled cases (n=30)."""
    return GOOD_CASES + HALLUCINATED_CASES + OFF_TOPIC_CASES + PARTIAL_CASES


def get_known_good_cases() -> list[LabeledCase]:
    return list(GOOD_CASES)


def get_known_bad_cases() -> list[LabeledCase]:
    return HALLUCINATED_CASES + OFF_TOPIC_CASES + PARTIAL_CASES


def case_count_summary() -> dict[str, int]:
    """Count of cases per tag — useful for sub-group analysis."""
    cases = get_synthetic_cases()
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.tag] = counts.get(c.tag, 0) + 1
    counts["TOTAL"] = len(cases)
    return counts
