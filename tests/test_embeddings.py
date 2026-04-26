"""Tests for embedding utilities (cosine, cosine_matrix, cache)."""

import math
import numpy as np

from ragwatch.utils.embeddings import cosine, cosine_matrix


class TestCosine:
    def test_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert math.isclose(cosine(v, v), 1.0, abs_tol=1e-9)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert math.isclose(cosine(a, b), 0.0, abs_tol=1e-9)

    def test_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert math.isclose(cosine(a, b), -1.0, abs_tol=1e-9)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert cosine(a, b) == 0.0

    def test_scale_invariance(self):
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 4.0])
        assert math.isclose(cosine(a, b), 1.0, abs_tol=1e-9)


class TestCosineMatrix:
    def test_shape(self):
        A = np.random.rand(3, 5)
        B = np.random.rand(4, 5)
        M = cosine_matrix(A, B)
        assert M.shape == (3, 4)

    def test_self(self):
        A = np.random.rand(4, 8)
        M = cosine_matrix(A, A)
        # diagonal should be 1.0
        np.testing.assert_allclose(np.diag(M), np.ones(4), atol=1e-6)

    def test_empty_input(self):
        empty = np.zeros((0, 5))
        other = np.random.rand(3, 5)
        assert cosine_matrix(empty, other).shape == (0, 3)
        assert cosine_matrix(other, empty).shape == (3, 0)


class TestEmbeddingCache:
    def test_dummy_cache_reuse(self, dummy_embedder):
        v1 = dummy_embedder.embed("hello world")
        v2 = dummy_embedder.embed("hello world")
        np.testing.assert_array_equal(v1, v2)

    def test_dummy_token_overlap_signal(self, dummy_embedder):
        # strings sharing many tokens should be more similar than unrelated ones
        a = dummy_embedder.embed("the cat sat on the mat")
        b = dummy_embedder.embed("the cat sat on the rug")
        c = dummy_embedder.embed("quantum chromodynamics symmetry breaking")
        sim_ab = cosine(a, b)
        sim_ac = cosine(a, c)
        assert sim_ab > sim_ac
