"""Test cases for the eigendecomposition module."""
import numpy as np
import pytest

from ge_molsg import build_laplacian, compute_eigenpairs
from tests.store import sphere_surface


def _laplacian(n=80, k=20):
    points = sphere_surface(n=n).augmented_points()
    return build_laplacian(points, n_neighbors=k, laplacian_type="normalized")


def test_eigensystem_shapes():
    """It returns n_components eigenvalues and matching eigenvectors."""
    L = _laplacian()
    evals, evecs = compute_eigenpairs(L, n_components=15)
    assert evals.shape == (15,)
    assert evecs.shape == (L.shape[0], 15)


def test_eigenvalues_are_ascending():
    """Eigenvalues are returned in ascending order."""
    L = _laplacian()
    evals, _ = compute_eigenpairs(L, n_components=15)
    assert np.all(np.diff(evals) >= -1e-9)


def test_drop_first_removes_trivial_mode():
    """drop_first=True yields a larger smallest eigenvalue than drop_first=False."""
    L = _laplacian()
    evals_drop, _ = compute_eigenpairs(L, n_components=10, drop_first=True)
    evals_keep, _ = compute_eigenpairs(L, n_components=10, drop_first=False)
    # The kept spectrum starts at the trivial (~0) mode.
    assert evals_keep[0] <= evals_drop[0] + 1e-9
    assert abs(evals_keep[0]) < 1e-6


def test_arpack_matches_dense_spectrum():
    """ARPACK and the dense solver agree on the eigenvalues."""
    L = _laplacian(n=60, k=15)
    ev_arpack, _ = compute_eigenpairs(L, n_components=10, eigensolver="arpack")
    ev_dense, _ = compute_eigenpairs(L, n_components=10, eigensolver="dense")
    assert np.allclose(ev_arpack, ev_dense, atol=1e-8)


def test_eigenvectors_are_l2_normalized():
    """Each eigenvector column has unit L2 norm."""
    L = _laplacian()
    _, evecs = compute_eigenpairs(L, n_components=12, normalize_vectors=True)
    norms = np.linalg.norm(evecs, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_eigensystem_fails_for_unknown_solver():
    """It raises ValueError for an unknown eigensolver."""
    L = _laplacian(n=40, k=10)
    with pytest.raises(ValueError):
        compute_eigenpairs(L, n_components=5, eigensolver="nope")
