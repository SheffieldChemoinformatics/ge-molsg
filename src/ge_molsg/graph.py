"""Adaptive-bandwidth affinity matrix and graph Laplacian.

Builds a sparse affinity graph over a point set using a Gaussian kernel with a
per-point adaptive bandwidth, then forms the corresponding graph Laplacian. The
neighbor search is delegated to a pluggable Euclidean backend (see
:mod:`ge_molsg.neighbors`).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix, find, diags as sp_diags

from .neighbors import get_neighbor_backend


def adaptive_bandwidth(K: csr_matrix, n_neighbors: int) -> np.ndarray:
    """Per-point local scale from the kNN distance graph.

    For each row of ``K``, returns the distance to the
    ``floor(n_neighbors / 2)``-th nearest neighbor. ``K`` is a CSR matrix whose
    stored values are neighbor distances.
    """
    median_k = int(np.floor(n_neighbors / 2))
    adap_sd = np.zeros(K.shape[0])
    for i in range(K.shape[0]):
        row = np.sort(K.data[K.indptr[i] : K.indptr[i + 1]])
        if row.size == 0:
            adap_sd[i] = 0.0
        else:
            adap_sd[i] = row[min(median_k - 1, row.size - 1)]
    return adap_sd


def knn_distance_graph(
    points: np.ndarray,
    n_neighbors: int,
    backend: str | Callable = "ckdtree",
    backend_kwargs: Optional[dict] = None,
) -> csr_matrix:
    """Build the kNN graph as a CSR distance matrix (self-matches excluded)."""
    nn = get_neighbor_backend(backend, **(backend_kwargs or {}))
    knn_idx, knn_dist = nn(points, n_neighbors)
    n = points.shape[0]
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = knn_idx.ravel()
    data = knn_dist.ravel()
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def compute_affinity(
    points: np.ndarray,
    n_neighbors: int,
    backend: str | Callable = "ckdtree",
    backend_kwargs: Optional[dict] = None,
    adaptive_bw: bool = True,
    sigma: Optional[float] = None,
    square_distances: bool = True,
    symmetrize: bool = True,
    eps: float = 1e-10,
) -> csr_matrix:
    """Adaptive-bandwidth affinity matrix.

    Builds the kNN distance graph, scales each edge distance by the source
    point's local bandwidth (then symmetrizes by averaging), and applies a
    Gaussian: ``W = exp(-d_scaled)``. With ``adaptive_bw=False`` a fixed
    ``sigma`` is used instead.
    """
    n = points.shape[0]
    K = knn_distance_graph(points, n_neighbors, backend, backend_kwargs)

    x, y, dists = find(K)
    dists = np.maximum(dists, 0.0)

    if adaptive_bw:
        adap_sd = adaptive_bandwidth(K, n_neighbors)
        d_scaled = dists / (adap_sd[x] + eps)
    else:
        s = sigma if sigma not in (None, 0) else eps
        d_scaled = dists / s
    if square_distances:
        d_scaled = d_scaled ** 2

    W = csr_matrix((np.exp(-d_scaled), (x, y)), shape=(n, n))

    # Discard non-finite and negative entries before symmetrizing.
    W.data = np.where(np.isfinite(W.data), W.data, 0.0)
    W.data = np.maximum(W.data, 0.0)

    if symmetrize:
        W = (W + W.T) / 2
    W.data = np.where(np.isfinite(W.data), W.data, 0.0)
    return W.tocsr()


def graph_laplacian(W: csr_matrix, laplacian_type: str = "normalized") -> csr_matrix:
    """Graph Laplacian of an affinity matrix.

    - 'unnormalized': ``L = D - W``
    - 'normalized':   ``L = D^{-1/2} (D - W) D^{-1/2}``
    - 'random_walk':  ``L = I - D^{-1} W``
    """
    N = W.shape[0]
    degree = np.ravel(W.sum(axis=1))

    if laplacian_type == "unnormalized":
        D = sp_diags(degree)
        return (D - W).tocsr()

    if laplacian_type == "normalized":
        D = sp_diags(degree)
        L = D - W
        d_inv_sqrt = degree.copy()
        nz = d_inv_sqrt != 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(d_inv_sqrt[nz])
        Dinvs = sp_diags(d_inv_sqrt)
        return Dinvs.dot(L).dot(Dinvs).tocsr()

    if laplacian_type == "random_walk":
        from scipy.sparse import identity

        d_inv = degree.copy()
        nz = d_inv != 0
        d_inv[nz] = 1.0 / d_inv[nz]
        Dinv = sp_diags(d_inv)
        I = identity(N, format="csr")
        return (I - Dinv.dot(W)).tocsr()

    raise ValueError(
        f"Unknown laplacian_type '{laplacian_type}'. "
        "Use 'unnormalized', 'normalized', or 'random_walk'."
    )


def build_laplacian(
    points: np.ndarray,
    n_neighbors: int,
    backend: str | Callable = "ckdtree",
    backend_kwargs: Optional[dict] = None,
    adaptive_bw: bool = True,
    sigma: Optional[float] = None,
    square_distances: bool = True,
    symmetrize: bool = True,
    laplacian_type: str = "normalized",
) -> csr_matrix:
    """Build the graph Laplacian directly from a point set."""
    W = compute_affinity(
        points,
        n_neighbors=n_neighbors,
        backend=backend,
        backend_kwargs=backend_kwargs,
        adaptive_bw=adaptive_bw,
        sigma=sigma,
        square_distances=square_distances,
        symmetrize=symmetrize,
    )
    return graph_laplacian(W, laplacian_type=laplacian_type)
