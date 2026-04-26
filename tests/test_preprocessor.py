"""Tests for sentence splitting, claim decomposition, and hand features."""

from ragwatch.utils.preprocessor import (
    decompose_into_claims,
    hand_features,
    split_sentences,
)


class TestSplitSentences:
    def test_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_single_sentence(self):
        out = split_sentences("Albert Einstein developed relativity.")
        assert len(out) == 1

    def test_multiple_sentences(self):
        text = "Einstein developed relativity. He was born in 1879. He won the Nobel Prize."
        out = split_sentences(text)
        assert len(out) == 3
        assert all("Einstein" in s or "He" in s for s in out)

    def test_preserves_content(self):
        text = "The cat sat on the mat. The dog ate the bone."
        out = split_sentences(text)
        joined = " ".join(out)
        assert "cat sat" in joined
        assert "dog ate" in joined


class TestDecomposeIntoClaims:
    def test_short_sentences_stay_intact(self):
        text = "Einstein discovered relativity. Newton discovered gravity."
        claims = decompose_into_claims(text)
        assert len(claims) == 2

    def test_long_sentence_splits(self):
        text = (
            "Einstein developed the theory of general relativity in 1915, "
            "and he later won the Nobel Prize in Physics in 1921 for the photoelectric effect."
        )
        claims = decompose_into_claims(text)
        # the long sentence should be split into at least 2 sub-claims
        assert len(claims) >= 2

    def test_empty(self):
        assert decompose_into_claims("") == []


class TestHandFeatures:
    def test_returns_three_features(self):
        feats = hand_features("The cat sat on the mat.")
        assert len(feats) == 3
        assert all(0.0 <= f <= 1.0 for f in feats)

    def test_word_count_normalization(self):
        short = hand_features("hi")
        long = hand_features(" ".join(["word"] * 50))
        # length feature is the third one
        assert long[2] >= short[2]

    def test_subject_signal(self):
        # capital-start with multi-word
        with_subj = hand_features("Albert wrote the paper")
        # single token, no subject
        without_subj = hand_features("running")
        assert with_subj[1] >= without_subj[1]
