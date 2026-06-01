"""Test cases for the graph module."""
import numpy as np
import pytest

from ge_molsg import (
    adaptive_bandwidth,
    build_laplacian,
    compute_affinity,
    graph_laplacian,
    knn_distance_graph,
)
from tests.store import sphere_surface


def test_affinity_is_symmetric_and_nonnegative():
    """The affinity matrix is symmetric with non-negative entries."""
    points = sphere_surface(n=80).augmented_points()
    W = compute_affinity(points, n_neighbors=20)
    assert W.shape == (80, 80)
    assert np.allclose((W - W.T).toarray(), 0.0, atol=1e-12)
    assert W.data.min() >= 0.0


def test_affinity_diagonal_is_zero():
    """Self-affinity is excluded (zero diagonal)."""
    points = sphere_surface(n=50).augmented_points()
    W = compute_affinity(points, n_neighbors=15)
    assert np.allclose(W.diagonal(), 0.0)


def test_adaptive_bandwidth_matches_median_neighbor():
    """Bandwidth equals the floor(k/2)-th sorted neighbor distance."""
    points = sphere_surface(n=60).augmented_points()
    n_neighbors = 20
    K = knn_distance_graph(points, n_neighbors)
    sigma = adaptive_bandwidth(K, n_neighbors)

    median_k = n_neighbors // 2
    expected = np.empty(K.shape[0])
    for i in range(K.shape[0]):
        row = np.sort(K.data[K.indptr[i] : K.indptr[i + 1]])
        expected[i] = row[median_k - 1]
    assert np.allclose(sigma, expected)


def test_normalized_laplacian_is_symmetric():
    """The normalized Laplacian is symmetric."""
    points = sphere_surface(n=70).augmented_points()
    L = build_laplacian(points, n_neighbors=20, laplacian_type="normalized")
    assert np.allclose((L - L.T).toarray(), 0.0, atol=1e-12)


def test_unnormalized_laplacian_rows_sum_to_zero():
    """For L = D - W each row sums to zero."""
    points = sphere_surface(n=50).augmented_points()
    W = compute_affinity(points, n_neighbors=15)
    L = graph_laplacian(W, laplacian_type="unnormalized")
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 0.0, atol=1e-10)


def test_graph_laplacian_fails_for_unknown_type():
    """It raises ValueError for an unknown laplacian_type."""
    points = sphere_surface(n=30).augmented_points()
    W = compute_affinity(points, n_neighbors=10)
    with pytest.raises(ValueError):
        graph_laplacian(W, laplacian_type="banana")
