"""Tests for NLIEngine.

The real cross-encoder (~180MB) is heavy, so we monkey-patch
`_entailment_score` to avoid downloading models in unit tests.
The integration test (test_integration.py) opt-ins to the real model.
"""

import pytest

from ragwatch.engines.nli_engine import NLIEngine, _decompose_query


@pytest.fixture
def nli(dummy_embedder):
    eng = NLIEngine(dummy_embedder, cosine_high=0.80, cosine_low=0.20)

    # patch _entailment_score: 1.0 if hypothesis tokens overlap >= 50% with premise
    def fake_entailment(premise, hypothesis):
        p_toks = set(premise.lower().split())
        h_toks = set(hypothesis.lower().split())
        if not h_toks:
            return 0.5
        overlap = len(p_toks & h_toks) / len(h_toks)
        return float(min(1.0, overlap))

    eng._entailment_score = fake_entailment  # type: ignore[method-assign]
    return eng


class TestQueryDecomposition:
    def test_simple_query_no_split(self):
        parts = _decompose_query("What is the capital?")
        # filtered to len(words) >= 3 → ["What is the capital"] (sans ?)
        assert len(parts) == 1

    def test_compound_question(self):
        parts = _decompose_query(
            "Who developed relativity, and in what year was it developed?"
        )
        assert len(parts) >= 2

    def test_empty(self):
        assert _decompose_query("") == []


class TestFaithfulness:
    def test_empty_claims(self, nli):
        s, flagged = nli.faithfulness([], ["some context"])
        assert s.score == 0.0
        assert flagged == []

    def test_empty_context(self, nli):
        s, flagged = nli.faithfulness(["some claim"], [])
        assert s.score == 0.0

    def test_well_supported_claim(self, nli):
        claims = ["Einstein developed relativity in 1915"]
        context = ["Albert Einstein developed the theory of relativity in 1915 in Germany"]
        s, flagged = nli.faithfulness(claims, context)
        assert s.score > 0.5
        assert claims[0] not in flagged

    def test_unsupported_claim_flagged(self, nli):
        claims = ["Bananas grow on Mars in zero gravity environments"]
        context = ["Einstein developed relativity in 1915 in Germany"]
        s, flagged = nli.faithfulness(claims, context)
        # disjoint topic → should be flagged
        assert s.score < 0.5
        assert len(flagged) >= 1


class TestCompleteness:
    def test_returns_score(self, nli):
        s = nli.completeness(
            "Who developed relativity and what year?",
            "Einstein developed relativity in 1915",
        )
        assert 0.0 <= s.score <= 1.0
