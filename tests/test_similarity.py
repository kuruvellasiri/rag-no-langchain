
import pytest

from src.retrieval.similarity_search import cosine_similarity


def test_identical_vectors():
    vector = [1, 2, 3]

    score = cosine_similarity(vector, vector)

    assert score == pytest.approx(1.0)


def test_orthogonal_vectors():
    vector_a = [1, 0]
    vector_b = [0, 1]

    score = cosine_similarity(vector_a, vector_b)

    assert score == pytest.approx(0.0)


def test_opposite_vectors():
    vector_a = [1, 0]
    vector_b = [-1, 0]

    score = cosine_similarity(vector_a, vector_b)

    assert score == pytest.approx(-1.0)


def test_different_dimensions():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 2])
