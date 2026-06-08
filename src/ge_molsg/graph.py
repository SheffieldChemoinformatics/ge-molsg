"""Adaptive-bandwidth affinity matrix and graph Laplacian.

Builds a sparse affinity graph over a point set using a Gaussian kernel with a
per-point adaptive bandwidth, then forms the corresponding graph Laplacian. The
neighbor search is delegated to a pluggable Euclidean backend (see
:mod:`ge_molsg.neighbors`).

References
----------
The symmetric normalised Laplacian follows Chung, *Spectral Graph Theory*
(AMS, 1997), and the per-point adaptive bandwidth follows the self-tuning
approach of Zelnik-Manor & Perona, "Self-Tuning Spectral Clustering"
(NeurIPS, 2004). The adaptive-bandwidth affinity and normalised-Laplacian
formulation was inspired by TopOMetry (https://github.com/davisidarta/topometry, Sidarta-Oliveira et al., 2022).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix, find, diags as sp_diags, eye as sp_eye

from .neighbors import get_neighbor_backend


def adaptive_bandwidth(
    K: csr_matrix, n_neighbors: int, median_k: Optional[int] = None
) -> np.ndarray:

    """Parameters
    ----------
    median_k : int, optional
        Rank of the neighbour distance used as the local scale. If ``None``
        (default), uses ``floor(n_neighbors / 2)``. Pass an explicit
        value to pin the rank (e.g. ``median_k=100``); each row must then store
        at least ``median_k`` distances.

    When every row has the same number of stored entries (the usual case for a
    kNN graph), the computation is vectorized with ``np.partition`` (O(k) per row);
    ``partition`` returns the same element as ``sort`` at that index, so the result
    is identical to a sort-based implementation. A per-row fallback handles
    ragged graphs.
    """
    rank = int(median_k) if median_k is not None else int(np.floor(n_neighbors / 2))
    counts = np.diff(K.indptr)

    # Fast path: uniform row lengths. partition finds the rank-th smallest
    # regardless of intra-row ordering, so CSR canonicalization is harmless.
    if counts.size and np.all(counts == counts[0]) and 1 <= rank <= counts[0]:
        data = K.data.reshape(K.shape[0], counts[0])
        return np.partition(data, rank - 1, axis=1)[:, rank - 1]

    # General fallback for ragged rows.
    adap_sd = np.zeros(K.shape[0])
    for i in range(K.shape[0]):
        row = K.data[K.indptr[i] : K.indptr[i + 1]]
        if row.size:
            j = min(rank - 1, row.size - 1)
            adap_sd[i] = np.partition(row, j)[j]
    return adap_sd


def knn_distance_graph(
    points: np.ndarray,
    n_neighbors: int,
    backend: str | Callable = "ckdtree",
    backend_kwargs: Optional[dict] = None,
) -> csr_matrix:
    """Build the kNN graph as a CSR distance matrix.

    """
    nn = get_neighbor_backend(backend, **(backend_kwargs or {}))
    knn_idx, knn_dist = nn(points, n_neighbors)   # n_neighbors cols: self (col 0) + (n-1) real
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
    square_distances: bool = False,
    symmetrize: bool = True,
    eps: float = 1e-10,
    median_k: Optional[int] = None,
) -> csr_matrix:
    """Adaptive-bandwidth affinity matrix.

    Builds the kNN distance graph, then applies a source-only adaptive-bandwidth
    Gaussian affinity kernel:

        square_distances=False (default):  W_ij = exp(-(d_ij / σ_i))
        square_distances=True:             W_ij = exp(-(d_ij / σ_i)²)

	With ``adaptive_bw=False`` a fixed ``sigma`` is used.

    ``median_k`` (optional) pins the adaptive-bandwidth rank; if ``None``
    (default) it falls back to ``floor(n_neighbors / 2)``.
    """
    n = points.shape[0]
    K = knn_distance_graph(points, n_neighbors, backend, backend_kwargs)

    x, y, dists = find(K)
    dists = np.maximum(dists, 0.0)

    if adaptive_bw:
        adap_sd = adaptive_bandwidth(K, n_neighbors, median_k=median_k)
        # Source-only adaptive bandwidth.
        d_scaled = dists / (adap_sd[x] + eps)
        if square_distances:
            d_scaled = d_scaled ** 2
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
    - 'normalized':   ``L = I - D^{-1/2} W D^{-1/2}``
    - 'random_walk':  ``L = I - D^{-1} W``

    The normalised form is the symmetric normalised Laplacian; on a connected
    graph it equals ``D^{-1/2} (D - W) D^{-1/2}``.
    """
    N = W.shape[0]
    degree = np.ravel(W.sum(axis=1))

    if laplacian_type == "unnormalized":
        return (sp_diags(degree) - W).tocsr()

    if laplacian_type == "normalized":
        d_inv_sqrt = np.zeros_like(degree)
        nz = degree != 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(degree[nz])
        Dinvs = sp_diags(d_inv_sqrt)
        return (sp_eye(N, format="csr") - Dinvs @ W @ Dinvs).tocsr()

    if laplacian_type == "random_walk":
        d_inv = np.zeros_like(degree)
        nz = degree != 0
        d_inv[nz] = 1.0 / degree[nz]
        return (sp_eye(N, format="csr") - sp_diags(d_inv) @ W).tocsr()

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
    square_distances: bool = False,
    symmetrize: bool = True,
    laplacian_type: str = "normalized",
    median_k: Optional[int] = None,
) -> csr_matrix:
    """Build the graph Laplacian directly from a point set.
    """
    W = compute_affinity(
        points,
        n_neighbors=n_neighbors,
        backend=backend,
        backend_kwargs=backend_kwargs,
        adaptive_bw=adaptive_bw,
        sigma=sigma,
        square_distances=square_distances,
        symmetrize=symmetrize,
        median_k=median_k,
    )
    return graph_laplacian(W, laplacian_type=laplacian_type)
