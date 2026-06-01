"""Codebook construction from a sampled set of surfaces.

A codebook is a set of ``n_codewords`` cluster centers in WKS descriptor
space, fit with MiniBatchKMeans over per-vertex descriptors pooled across a
sample of molecules. ``sample_descriptor_pool`` either keeps all vertices or
farthest-point-subsamples each molecule to ``n_per_mol`` for a smaller,
more uniform pool.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def farthest_point_sample(X: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k farthest-point-sampled rows of X."""
    n = X.shape[0]
    k = min(k, n)
    idxs = np.empty(k, dtype=np.int64)
    selected = np.zeros(n, dtype=bool)
    start = int(np.random.randint(0, n))
    idxs[0] = start
    selected[start] = True
    dists = np.einsum("ij,ij->i", X - X[start], X - X[start])
    for i in range(1, k):
        best = int(np.argmax(dists))
        if selected[best]:
            unsel = np.flatnonzero(~selected)
            best = int(unsel[0]) if len(unsel) else best
        idxs[i] = best
        selected[best] = True
        diff = X - X[best]
        dists = np.minimum(dists, np.einsum("ij,ij->i", diff, diff))
    return idxs


def sample_descriptor_pool(
    descriptor_list: Sequence[np.ndarray],
    n_per_mol: Optional[int] = None,
) -> np.ndarray:
    """Pool per-vertex descriptors across molecules.

    Parameters
    ----------
    descriptor_list : list of (N_i, D) arrays
        Per-molecule per-vertex descriptors.
    n_per_mol : int, optional
        If given, farthest-point-subsample each molecule to this many vertices
        before pooling. If None, keep all vertices.
    """
    if n_per_mol is None:
        return np.vstack(list(descriptor_list))
    pool: List[np.ndarray] = []
    for desc in descriptor_list:
        if desc.shape[0] <= n_per_mol:
            pool.append(desc)
        else:
            pool.append(desc[farthest_point_sample(desc, n_per_mol)])
    return np.vstack(pool)


def build_codebook(
    descriptor_pool: np.ndarray,
    n_codewords: int,
    random_state: int = 42,
    **kmeans_kwargs,
) -> np.ndarray:
    """Fit a MiniBatchKMeans codebook over a pooled descriptor space.

    Returns
    -------
    codebook : (n_codewords, D) array of cluster centers.
    """
    pool = np.asarray(descriptor_pool)
    pool = np.nan_to_num(pool, nan=0.0, posinf=0.0, neginf=0.0)
    if pool.ndim != 2:
        raise ValueError(f"descriptor_pool must be 2-D, got {pool.shape}")
    km = MiniBatchKMeans(
        n_clusters=n_codewords, random_state=random_state, **kmeans_kwargs
    )
    return km.fit(pool).cluster_centers_
