"""Bottom eigendecomposition of the graph Laplacian.

Takes the smallest eigenpairs of a precomputed Laplacian (see
:mod:`ge_molsg.graph`) and drops the trivial pair.

The trivial (smallest) eigenpair is dropped because the WKS energy grid is
anchored at ``log(|lambda_0|)``; for a connected graph ``lambda_0`` is ~0, so
``log(0)`` would clamp to a large negative value and distort the descriptor
scale. Dropping it guarantees the returned spectrum begins at the first
informative eigenvalue.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, lobpcg

EIGEN_SOLVERS = ("arpack", "lobpcg", "dense")


def compute_eigenpairs(
    L: csr_matrix,
    n_components: int,
    drop_first: bool = True,
    normalize_vectors: bool = True,
    eigensolver: str = "arpack",
    eigen_tol: float = 0.0,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the smallest eigenpairs of a symmetric Laplacian.

    Parameters
    ----------
    L : (N, N) sparse symmetric matrix
        Graph Laplacian.
    n_components : int
        Number of non-trivial eigenpairs to return.
    drop_first : bool, default True
        Discard the trivial smallest pair before returning.
    normalize_vectors : bool, default True
        L2-normalise each eigenvector column.
    eigensolver : {'arpack', 'lobpcg', 'dense'}, default 'arpack'
        Backend solver. 'arpack' is the default; 'dense' is exact
        but only suitable for small matrices.
    eigen_tol : float, default 0.0
        Solver convergence tolerance; 0 uses machine precision.
    random_state : int, optional
        Seed for the lobpcg initial guess.

    Returns
    -------
    eigenvalues : (n_components,) array, ascending order.
    eigenvectors : (N, n_components) array.
    """
    if eigensolver not in EIGEN_SOLVERS:
        raise ValueError(
            f"Unknown eigensolver '{eigensolver}'. Choose from {EIGEN_SOLVERS}."
        )

    n = L.shape[0]
    k = n_components + 1 if drop_first else n_components

    # eigsh/lobpcg require k < N; fall back to a dense solve for small matrices.
    if eigensolver == "dense" or k >= n:
        from numpy.linalg import eigh

        evals, evecs = eigh(L.toarray())
    elif eigensolver == "arpack":
        # Smallest-magnitude eigenpairs.
        evals, evecs = eigsh(L, k=k, which="SM", tol=eigen_tol, maxiter=n * 5)
    else:  # lobpcg
        rng = np.random.default_rng(random_state)
        X0 = rng.standard_normal((n, k))
        evals, evecs = lobpcg(L, X0, largest=False, tol=eigen_tol or 1e-8, maxiter=n // 5)

    order = np.argsort(np.real(evals))
    evals = np.real(evals)[order]
    evecs = np.real(evecs)[:, order]

    if drop_first:
        evals = evals[1 : n_components + 1]
        evecs = evecs[:, 1 : n_components + 1]
    else:
        evals = evals[:n_components]
        evecs = evecs[:, :n_components]

    if normalize_vectors:
        norms = np.linalg.norm(evecs, axis=0, keepdims=True)
        evecs = evecs / (norms + 1e-12)

    return evals, evecs
