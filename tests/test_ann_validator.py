"""Tests for ANNValidator — pure-NumPy MLP with backprop."""

import os
import tempfile

import numpy as np
import pytest

from ragwatch.engines.ann_validator import ANNValidator


@pytest.fixture
def ann(dummy_embedder):
    return ANNValidator(dummy_embedder, weights_path=None)


class TestANNArchitecture:
    def test_weight_shapes(self, ann):
        assert ann.W1.shape == (ann.INPUT_DIM, ann.H1)
        assert ann.W2.shape == (ann.H1, ann.H2)
        assert ann.W3.shape == (ann.H2, 1)
        assert ann.b1.shape == (ann.H1,)
        assert ann.b2.shape == (ann.H2,)
        assert ann.b3.shape == (1,)

    def test_total_params_under_30k(self, ann):
        total = (
            ann.W1.size + ann.b1.size +
            ann.W2.size + ann.b2.size +
            ann.W3.size + ann.b3.size
        )
        assert total < 30_000

    def test_input_dim_matches_embed_plus_hand(self, ann):
        assert ann.INPUT_DIM == ann.EMBED_DIM + ann.HAND_DIM


class TestForwardPass:
    def test_predict_returns_probabilities(self, ann):
        sentences = ["A clean atomic claim about something.", "fragment cut"]
        out = ann.predict(sentences)
        assert out.shape == (2,)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_empty_input(self, ann):
        out = ann.predict([])
        assert out.shape == (0,)


class TestTraining:
    def test_loss_decreases(self, ann):
        corpus = [
            "Albert Einstein developed relativity in 1915. The theory describes gravity. He won the Nobel Prize in 1921.",
            "Paris is the capital of France. The Seine flows through Paris. The Eiffel Tower is iconic.",
            "Photosynthesis converts sunlight into chemical energy. Plants use chlorophyll. Oxygen is released as a byproduct.",
        ]
        losses = ann.train_synthetic(corpus, epochs=8, batch_size=8, lr=1e-2, seed=1)
        assert len(losses) == 8
        # final epoch loss should be lower than first
        assert losses[-1] < losses[0]

    def test_atomic_vs_fragment_separation(self, ann):
        """After training, real sentences should score higher than mid-cut fragments."""
        corpus = [
            "Albert Einstein developed the theory of relativity. The theory describes gravity.",
            "Paris is the capital of France. The Seine flows through Paris.",
            "Photosynthesis converts sunlight. Plants use chlorophyll.",
            "Machine learning is a subset of AI. Neural networks are popular.",
        ]
        ann.train_synthetic(corpus, epochs=15, batch_size=8, lr=2e-2, seed=1)

        clean = ann.predict([
            "Albert Einstein developed the theory of relativity.",
            "Paris is the capital of France.",
        ])
        fragments = ann.predict([
            "Einstein developed",
            "the theory of",
        ])
        # mean atomic score on clean > mean atomic score on fragments
        assert float(clean.mean()) > float(fragments.mean())

    def test_training_requires_corpus(self, ann):
        with pytest.raises(ValueError, match="clean sentences"):
            ann.train_synthetic([], epochs=1)


class TestPersistence:
    def test_save_and_load_roundtrip(self, dummy_embedder):
        ann1 = ANNValidator(dummy_embedder)
        sample = ["A clean atomic claim about something useful."]
        out_before = ann1.predict(sample)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ann.npz")
            ann1.save(path)
            assert os.path.exists(path)

            ann2 = ANNValidator(dummy_embedder, weights_path=path)
            out_after = ann2.predict(sample)
        np.testing.assert_allclose(out_before, out_after, atol=1e-9)


class TestFilterAtomic:
    def test_returns_pairs(self, ann):
        sents = ["A real sentence about science.", "fragment", "Another full sentence here."]
        kept, scores = ann.filter_atomic(sents, threshold=0.0)  # threshold=0 keeps all
        assert len(kept) == len(scores) == 3

    def test_empty(self, ann):
        kept, scores = ann.filter_atomic([])
        assert kept == [] and scores == []

    def test_cold_start_safety(self, ann):
        # untrained ANN may reject everything; the engine should fall back to all
        sents = ["A reasonable sentence about a topic."]
        kept, scores = ann.filter_atomic(sents, threshold=0.99)
        # fall-back: should not return empty
        assert len(kept) >= 1
