"""Pluggable Euclidean k-nearest-neighbor backends.

Each backend maps ``(points, k) -> (indices, distances)`` with self-matches
removed, so graph construction is independent of the search implementation. The
default backend is SciPy's ``cKDTree``, which is fast and exact for the
low-dimensional augmented point clouds used here; scikit-learn's
``NearestNeighbors`` is also available. Additional backends can be registered
or passed as a callable with the same signature.
"""

from __future__ import annotations

from typing import Callable, Protocol, Tuple

import numpy as np

from .exception import BackendError


class NeighborBackend(Protocol):
    """Callable returning ``(indices, distances)`` with self-matches excluded."""

    def __call__(
        self, points: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]: ...


def _strip_self(idx: np.ndarray, dist: np.ndarray):
    """Drop the self-match (column 0 when a point set is queried on itself)."""
    return idx[:, 1:], dist[:, 1:]


def make_ckdtree_backend(workers: int = -1) -> NeighborBackend:
    """Build an exact SciPy ``cKDTree`` backend (default).

    Parameters
    ----------
    workers : int, default -1
        Worker threads for the query; -1 uses all available cores.
    """

    def _backend(points: np.ndarray, k: int):
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        dist, idx = tree.query(points, k=k + 1, workers=workers)
        return _strip_self(idx, dist)

    return _backend


def make_sklearn_backend(algorithm: str = "auto", n_jobs: int = 1) -> NeighborBackend:
    """Build an exact scikit-learn ``NearestNeighbors`` backend.

    Parameters
    ----------
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default 'auto'
        Search algorithm passed to ``sklearn.neighbors.NearestNeighbors``.
    n_jobs : int, default 1
        Worker threads for the neighbor query.
    """

    def _backend(points: np.ndarray, k: int):
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(
            n_neighbors=k + 1, algorithm=algorithm, metric="euclidean", n_jobs=n_jobs
        )
        nn.fit(points)
        dist, idx = nn.kneighbors(points)
        return _strip_self(idx, dist)

    return _backend


_REGISTRY = {
    "ckdtree": make_ckdtree_backend,
    "sklearn": make_sklearn_backend,
}


def get_neighbor_backend(name: str | Callable, **kwargs) -> NeighborBackend:
    """Resolve a backend by name, or pass through a custom callable.

    Examples
    --------
    >>> backend = get_neighbor_backend("ckdtree")            # default
    >>> backend = get_neighbor_backend("sklearn", algorithm="kd_tree")
    >>> backend = get_neighbor_backend(my_callable)  # (points, k) -> (idx, dist)
    """
    if callable(name):
        return name
    key = name.lower()
    if key not in _REGISTRY:
        raise BackendError(
            f"Unknown neighbor backend '{name}'. "
            f"Choose from {sorted(_REGISTRY)} or pass a callable."
        )
    return _REGISTRY[key](**kwargs)
