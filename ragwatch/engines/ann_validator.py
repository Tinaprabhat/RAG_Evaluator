"""
ANN Validator — pure-NumPy MLP that scores 'is this an atomic claim?'.

Trained via backprop on contrastively auto-generated examples:
  - Positive: well-formed sentences from clean paragraphs (label 1)
  - Negative: random sub-spans, mid-sentence cuts (label 0)

Architecture:
    input:  384-dim embedding + 3 hand-features = 387
    h1:     387 → 64 (ReLU)
    h2:     64  → 32 (ReLU)
    out:    32  → 1  (sigmoid)

Total params ≈ 27k. RAM footprint < 1 MB. Pure NumPy — no PyTorch.
"""

from __future__ import annotations
import os
import random
from typing import Any

import numpy as np

from ragwatch.utils.embeddings import EmbeddingCache
from ragwatch.utils.preprocessor import hand_features, split_sentences


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[neg])
    out[neg] = e / (1.0 + e)
    return out


class ANNValidator:
    """Tiny NumPy MLP for atomic-claim validation."""

    EMBED_DIM = 384  # all-MiniLM-L6-v2
    HAND_DIM = 3
    INPUT_DIM = EMBED_DIM + HAND_DIM
    H1 = 64
    H2 = 32

    def __init__(self, embedder: EmbeddingCache, weights_path: str | None = None, seed: int = 42):
        self.embedder = embedder
        self.weights_path = weights_path
        rng = np.random.default_rng(seed)

        # Xavier-ish init
        self.W1 = rng.normal(0, np.sqrt(2.0 / self.INPUT_DIM), (self.INPUT_DIM, self.H1))
        self.b1 = np.zeros(self.H1)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.H1), (self.H1, self.H2))
        self.b2 = np.zeros(self.H2)
        self.W3 = rng.normal(0, np.sqrt(2.0 / self.H2), (self.H2, 1))
        self.b3 = np.zeros(1)

        if weights_path and os.path.exists(weights_path):
            self.load(weights_path)

    # ---------- forward / backward ----------

    def _featurize(self, sentences: list[str]) -> np.ndarray:
        emb = self.embedder.embed_batch(sentences)  # [n, 384]
        hand = np.array([hand_features(s) for s in sentences], dtype=np.float64)  # [n, 3]
        return np.hstack([emb, hand])

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _relu(z2)
        z3 = a2 @ self.W3 + self.b3
        out = _sigmoid(z3)
        cache = dict(X=X, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3, out=out)
        return out, cache

    def _backward(
        self, cache: dict[str, np.ndarray], y: np.ndarray, lr: float
    ) -> float:
        N = cache["X"].shape[0]
        out = cache["out"]
        # binary cross-entropy with sigmoid → simple gradient
        eps = 1e-9
        loss = float(-np.mean(y * np.log(out + eps) + (1 - y) * np.log(1 - out + eps)))

        dz3 = (out - y) / N                                 # [N,1]
        dW3 = cache["a2"].T @ dz3                           # [H2,1]
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.W3.T                               # [N,H2]
        dz2 = da2 * _relu_grad(cache["z2"])
        dW2 = cache["a1"].T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * _relu_grad(cache["z1"])
        dW1 = cache["X"].T @ dz1
        db1 = dz1.sum(axis=0)

        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        return loss

    # ---------- public API ----------

    def predict(self, sentences: list[str]) -> np.ndarray:
        """Return atomic-claim probability for each sentence (0..1)."""
        if not sentences:
            return np.zeros(0)
        X = self._featurize(sentences)
        out, _ = self._forward(X)
        return out.flatten()

    def filter_atomic(
        self, sentences: list[str], threshold: float = 0.5
    ) -> tuple[list[str], list[float]]:
        """Keep only sentences scored as atomic; return (claims, scores)."""
        if not sentences:
            return [], []
        scores = self.predict(sentences)
        kept = [(s, float(p)) for s, p in zip(sentences, scores) if p >= threshold]
        if not kept:
            # fall back to all sentences if ANN rejects everything (cold-start safety)
            return sentences, scores.tolist()
        return [s for s, _ in kept], [p for _, p in kept]

    # ---------- training ----------

    def train_synthetic(
        self,
        clean_paragraphs: list[str],
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 1e-2,
        seed: int = 7,
    ) -> list[float]:
        """
        Self-supervised training:
          - Positives = full sentences from `clean_paragraphs`
          - Negatives = mid-sentence cuts and short fragments
        Returns per-epoch loss.
        """
        rnd = random.Random(seed)
        positives: list[str] = []
        for p in clean_paragraphs:
            for s in split_sentences(p):
                if len(s.split()) >= 4:
                    positives.append(s)
        if len(positives) < 4:
            raise ValueError("need at least a handful of clean sentences to train")

        # synthesize negatives: mid-sentence cuts
        negatives: list[str] = []
        for s in positives:
            words = s.split()
            if len(words) >= 6:
                cut = rnd.randint(2, len(words) - 2)
                negatives.append(" ".join(words[:cut]))           # truncated head
                negatives.append(" ".join(words[cut:]))           # tailless body
            elif len(words) >= 4:
                negatives.append(" ".join(words[:2]))

        X_pos = self._featurize(positives)
        X_neg = self._featurize(negatives)
        X = np.vstack([X_pos, X_neg])
        y = np.vstack([
            np.ones((X_pos.shape[0], 1)),
            np.zeros((X_neg.shape[0], 1)),
        ])

        N = X.shape[0]
        idx = np.arange(N)
        losses: list[float] = []
        for ep in range(epochs):
            np.random.default_rng(seed + ep).shuffle(idx)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                bi = idx[start:start + batch_size]
                Xb, yb = X[bi], y[bi]
                _, cache = self._forward(Xb)
                ep_loss += self._backward(cache, yb, lr)
                n_batches += 1
            losses.append(ep_loss / max(1, n_batches))
        return losses

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
        )

    def load(self, path: str) -> None:
        data: Any = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W3, self.b3 = data["W3"], data["b3"]
