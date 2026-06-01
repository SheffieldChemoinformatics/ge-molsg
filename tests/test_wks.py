"""Test cases for the WKS descriptor module."""
import numpy as np

from ge_molsg import build_laplacian, compute_eigenpairs, wks, wks_descriptor
from tests.store import sphere_surface


def _eigensystem(n=80, k=20, n_components=15):
    points = sphere_surface(n=n).augmented_points()
    L = build_laplacian(points, n_neighbors=k, laplacian_type="normalized")
    return compute_eigenpairs(L, n_components=n_components)


def test_wks_descriptor_shape():
    """The descriptor has shape (n_vertices, evals)."""
    eigensystem = _eigensystem(n=80)
    desc = wks_descriptor(eigensystem, evals=40)
    assert desc.shape == (80, 40)


def test_wks_descriptor_is_finite_and_nonnegative():
    """WKS values are finite and non-negative (sum of squared modes)."""
    desc = wks_descriptor(_eigensystem(), evals=30)
    assert np.isfinite(desc).all()
    assert desc.min() >= 0.0


def test_wks_l2_normalizes_rows():
    """wks(l2=True) returns unit-norm rows."""
    eigensystem = _eigensystem()
    desc = wks(eigensystem, evals=30, l2=True)
    norms = np.linalg.norm(desc, axis=1)
    # Rows are either unit norm or zero (degenerate vertices).
    nonzero = norms > 0
    assert np.allclose(norms[nonzero], 1.0, atol=1e-6)


def test_wks_unnormalized_matches_core():
    """wks(l2=False) equals the raw wks_descriptor output."""
    eigensystem = _eigensystem()
    raw = wks_descriptor(eigensystem, evals=25)
    via_wks = wks(eigensystem, evals=25, l2=False)
    assert np.allclose(raw, via_wks)


def test_wks_invariant_to_eigenvector_sign():
    """WKS depends on squared eigenvectors, so sign flips do not change it."""
    evals, evecs = _eigensystem()
    flipped = evecs * np.array([1, -1] * (evecs.shape[1] // 2) + [1] * (evecs.shape[1] % 2))
    a = wks_descriptor((evals, evecs), evals=20)
    b = wks_descriptor((evals, flipped), evals=20)
    assert np.allclose(a, b)
