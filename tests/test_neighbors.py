"""Test cases for the neighbor backends."""
import numpy as np
import pytest

from ge_molsg import get_neighbor_backend, make_ckdtree_backend, make_sklearn_backend
from ge_molsg.exception import BackendError
from tests.store import sphere_surface


def test_ckdtree_backend_shapes_and_self_inclusion():
    """The default cKDTree backend returns k neighbors with the self-match first."""
    points = sphere_surface(n=60).augmented_points()
    idx, dist = make_ckdtree_backend()(points, k=10)
    assert idx.shape == (60, 10)
    assert dist.shape == (60, 10)
    assert np.array_equal(idx[:, 0], np.arange(60))
    assert np.allclose(dist[:, 0], 0.0)


def test_sklearn_backend_shapes_and_self_inclusion():
    """The sklearn backend returns k neighbors with the self-match first."""
    points = sphere_surface(n=60).augmented_points()
    idx, dist = make_sklearn_backend()(points, k=10)
    assert idx.shape == (60, 10)
    assert dist.shape == (60, 10)
    assert np.array_equal(idx[:, 0], np.arange(60))
    assert np.allclose(dist[:, 0], 0.0)


def test_exact_backends_agree():
    """cKDTree and sklearn (both exact) return the same neighbor distances."""
    points = sphere_surface(n=80).augmented_points()
    _, d_ck = make_ckdtree_backend()(points, k=12)
    _, d_sk = make_sklearn_backend()(points, k=12)
    assert np.allclose(np.sort(d_ck, axis=1), np.sort(d_sk, axis=1), atol=1e-9)


def test_neighbor_distances_are_sorted_ascending():
    """Returned distances increase with neighbor rank."""
    points = sphere_surface(n=40).augmented_points()
    _, dist = make_ckdtree_backend()(points, k=8)
    assert np.all(np.diff(dist, axis=1) >= -1e-9)


def test_default_backend_is_ckdtree():
    """The 'ckdtree' name resolves to a callable backend."""
    assert callable(get_neighbor_backend("ckdtree"))


def test_get_neighbor_backend_resolves_sklearn():
    """The 'sklearn' backend is still resolvable by name."""
    assert callable(get_neighbor_backend("sklearn"))


def test_get_neighbor_backend_passes_through_callable():
    """A custom callable is returned unchanged."""

    def custom(points, k):  # noqa: D401
        return np.zeros((len(points), k), int), np.zeros((len(points), k))

    assert get_neighbor_backend(custom) is custom


def test_get_neighbor_backend_fails_for_unknown_name():
    """It raises BackendError for an unknown backend name."""
    with pytest.raises(BackendError):
        get_neighbor_backend("does_not_exist")
