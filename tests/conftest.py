"""Pytest fixtures — thin wrappers around tests/_fixtures.py."""

import pytest

from tests._fixtures import (
    DummyEmbedder,
    make_good_input,
    make_hallucinated_input,
)


@pytest.fixture
def dummy_embedder():
    return DummyEmbedder()


@pytest.fixture
def good_input():
    return make_good_input()


@pytest.fixture
def hallucinated_input():
    return make_hallucinated_input()
