"""
Preprocessor — sentence splitting and claim decomposition.

Tries spaCy if available; falls back to a robust regex splitter so the
package still runs on minimal installs.
"""

from __future__ import annotations
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _regex_split(text: str) -> list[str]:
    """Robust regex-based sentence splitter (fallback)."""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Uses spaCy if installed, regex otherwise."""
    if not text or not text.strip():
        return []

    try:
        import spacy
        # cache the loaded model on the function object
        nlp = getattr(split_sentences, "_nlp", None)
        if nlp is None:
            try:
                nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            except OSError:
                # model not downloaded
                return _regex_split(text)
            split_sentences._nlp = nlp  # type: ignore[attr-defined]
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    except ImportError:
        return _regex_split(text)


def decompose_into_claims(answer: str) -> list[str]:
    """
    Split an answer into atomic claims.
    Step 1: sentence-level split.
    Step 2: split on coordinating conjunctions for very long sentences.
    """
    sents = split_sentences(answer)
    claims: list[str] = []
    for s in sents:
        if len(s.split()) > 20:
            # try to split on ", and " / "; " for very long sentences
            sub = re.split(r",\s+and\s+|;\s+", s)
            claims.extend([x.strip() for x in sub if x.strip()])
        else:
            claims.append(s)
    return claims


def hand_features(sentence: str) -> list[float]:
    """
    Hand-crafted features for the ANN claim validator.
    Cheap to compute, useful as a prior.

    Features:
        - has_verb (heuristic): contains a common verb form ending
        - has_subject (heuristic): starts with a capital, has > 1 word
        - normalized word count (clipped to [0, 1])
    """
    s = sentence.strip()
    words = s.split()
    n = len(words)

    verb_endings = ("ed", "ing", "es", "is", "are", "was", "were", "has", "have")
    has_verb = 0.0
    for w in words:
        wl = w.lower()
        if wl in verb_endings or any(wl.endswith(end) for end in verb_endings):
            has_verb = 1.0
            break

    has_subject = 1.0 if (n > 1 and s[:1].isalpha()) else 0.0
    norm_len = min(1.0, n / 30.0)

    return [has_verb, has_subject, norm_len]
