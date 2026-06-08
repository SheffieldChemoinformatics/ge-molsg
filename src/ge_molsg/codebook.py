"""Codebook construction from a set of surfaces.

A codebook is a set of ``n_codewords`` cluster centres in WKS descriptor
space, fit with MiniBatchKMeans over per-vertex descriptors pooled across a
set of molecules. ``build_descriptor_pool`` pools all vertices. To reduce the
pool size, optionally pass the descriptors through ``sample_descriptor_pool``
first, which farthest-point-subsamples each molecule.
"""

from __future__ import annotations

from typing import List, Sequence

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


def build_descriptor_pool(descriptor_list: Sequence[np.ndarray]) -> np.ndarray:
    """Pool per-vertex descriptors across molecules into one array.

    Retains every vertex of every molecule. To reduce the pool size, pass the
    descriptors through :func:`sample_descriptor_pool` before pooling.

    Parameters
    ----------
    descriptor_list : list of (N_i, D) arrays
        Per-molecule per-vertex descriptors.

    Returns
    -------
    pool : (sum(N_i), D) array
        All vertex descriptors stacked.
    """
    return np.vstack(list(descriptor_list))


def sample_descriptor_pool(
    descriptor_list: Sequence[np.ndarray], n_per_mol: int
) -> List[np.ndarray]:
    """Farthest-point-subsample each molecule to at most ``n_per_mol`` vertices.

    Returns a new descriptor list (one array per molecule) suitable for passing
    to :func:`build_descriptor_pool`. Molecules with no more than ``n_per_mol``
    vertices are returned unchanged.

    Parameters
    ----------
    descriptor_list : list of (N_i, D) arrays
        Per-molecule per-vertex descriptors.
    n_per_mol : int
        Maximum vertices to keep per molecule.

    Returns
    -------
    list of arrays
        Subsampled per-molecule descriptors, in input order.
    """
    out: List[np.ndarray] = []
    for desc in descriptor_list:
        if desc.shape[0] <= n_per_mol:
            out.append(desc)
        else:
            out.append(desc[farthest_point_sample(desc, n_per_mol)])
    return out


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
