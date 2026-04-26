"""Tests for SelfConsistencyEngine — internal claim coherence."""

import pytest

from ragwatch.engines.self_consistency import SelfConsistencyEngine


@pytest.fixture
def consistency(dummy_embedder):
    return SelfConsistencyEngine(dummy_embedder)


class TestSelfConsistency:
    def test_empty_claims(self, consistency):
        s = consistency.score([])
        assert s.score == 0.0
        assert s.n_samples == 0

    def test_single_claim_trivially_consistent(self, consistency):
        s = consistency.score(["A single claim."])
        assert s.score == 1.0
        assert s.n_samples == 1

    def test_in_range(self, consistency):
        claims = [
            "Einstein developed relativity in 1915.",
            "The theory describes spacetime curvature.",
            "Einstein won the Nobel Prize in 1921.",
        ]
        s = consistency.score(claims)
        assert 0.0 <= s.score <= 1.0

    def test_n_samples_is_pair_count(self, consistency):
        # 4 claims → C(4,2) = 6 pairs
        claims = ["a sentence one", "a sentence two", "a sentence three", "a sentence four"]
        s = consistency.score(claims)
        assert s.n_samples == 6

    def test_topical_claims_more_consistent_than_random(self, consistency):
        topical = [
            "Einstein developed relativity",
            "Einstein won the Nobel Prize",
            "Einstein was born in Germany",
        ]
        random = [
            "Apples grow on trees",
            "Quantum mechanics is hard",
            "The Eiffel Tower is in Paris",
        ]
        s_top = consistency.score(topical)
        s_rand = consistency.score(random)
        # all-Einstein claims should be more internally consistent than disjoint topics
        assert s_top.score >= s_rand.score
