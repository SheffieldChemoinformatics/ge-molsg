"""
cov_cp_dense.py
===============
Dense patch-based covariance descriptor extraction for molecular surfaces.

Computes a covariance descriptor for every vertex on the surface, treating
each vertex as a patch centre.  Optimised for the N-centre case:

  * No FPS — every vertex index 0..N-1 is a centre.
  * query_ball_tree — queries all N centres in one batched call instead of
    looping over query_ball_point N times.  For large surfaces this is the
    dominant saving.
  * Single query at r_max — distances computed once; smaller radii are
    resolved by boolean thresholding on the cached distance values, not by
    additional tree queries.
  * Joblib parallelism — the covariance loop (unavoidably O(N)) is split
    across CPU cores via joblib.  Falls back gracefully to n_jobs=1.

Output
------
compute_dense_patches returns (N, T * n_radii) where T = D*(D+1)/2.
Row i is the descriptor for vertex i.
Columns: [desc_r0 | desc_r1 | ... | desc_rK].
"""

import numpy as np
from scipy.spatial import cKDTree

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except ImportError:
    _JOBLIB = False


# ---------------------------------------------------------------------------
# Helpers shared with cov_cp_v3
# ---------------------------------------------------------------------------

def _regularised_covariance_old(X, eps=1e-6, normalise=False):
    if normalise:
        std = X.std(axis=0)
        std[std < 1e-10] = 1.0
        X = (X - X.mean(axis=0)) / std
    if X.shape[0] < 2:
        return np.eye(X.shape[1])
    C = np.cov(X, rowvar=False)
    return C + eps * np.eye(C.shape[0])


def _regularised_covariance(X, eps=1e-6, normalise=True):
    if X.shape[0] < 2:
        return np.eye(X.shape[1])
    C = np.cov(X, rowvar=False)
    C += eps * np.eye(C.shape[0])
    #C += (eps / max(X.shape[0], 1)) * np.eye(C.shape[0])
    if normalise:
        t = np.trace(C)
        if t > 1e-10:
            C /= t
    return C

def _log_map_spd(C):
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-10)
    # (eigvecs * log_eigvals) @ eigvecs.T avoids allocating the D×D diag matrix
    return (eigvecs * np.log(eigvals)) @ eigvecs.T


def _make_triu_meta(d):
    """Precompute upper-triangle indices and off-diagonal sqrt(2) weights for dimension d."""
    rows, cols = np.triu_indices(d)
    weights    = np.where(rows == cols, 1.0, np.sqrt(2))
    return rows, cols, weights

# Module-level cache — populated once per unique D, reused across all vertices/molecules.
_TRIU_CACHE: dict = {}

def _weighted_upper_triangle(M):
    """Vectorised upper-triangle extraction — replaces the O(D²) Python double loop."""
    d = M.shape[0]
    if d not in _TRIU_CACHE:
        _TRIU_CACHE[d] = _make_triu_meta(d)
    rows, cols, weights = _TRIU_CACHE[d]
    return M[rows, cols] * weights


def compute_patch_log_covariance(feature_matrix, eps=1e-6, normalise=False):
    C     = _regularised_covariance(feature_matrix, eps=eps, normalise=normalise)
    log_C = _log_map_spd(C)
    return _weighted_upper_triangle(log_C)


# ---------------------------------------------------------------------------
# Batch neighbourhood lookup — the key efficiency gain
# ---------------------------------------------------------------------------

def query_all_radii(vertices, radii):
    """
    For every vertex, return neighbour indices partitioned by radius.

    Improvements over v5:
    - query_ball_point with workers=-1 releases the GIL and parallelises
      the tree query across all cores (scipy >= 1.9).
    - Fast path for single-radius case: skips all partitioning overhead.
    - Distances are computed once via the tree's squared-distance return
      rather than re-running np.linalg.norm per vertex in a Python loop.
    """
    radii  = sorted(radii)
    r_max  = radii[-1]
    tree   = cKDTree(vertices)

    # Fast path — single radius, no partitioning needed
    if len(radii) == 1:
        raw = tree.query_ball_point(vertices, r_max, workers=-1)
        neighbours = []
        for i, nbr_list in enumerate(raw):
            members = np.asarray(nbr_list, dtype=np.intp)
            if i not in members:
                members = np.append(members, i)
            neighbours.append([members])
        return neighbours

    # Multi-radius: query at r_max once, partition by thresholding squared distances.
    # query_ball_point with return_sorted=False is faster and we don't need order.
    raw = tree.query_ball_point(vertices, r_max, workers=-1, return_sorted=False)

    neighbours = []
    for i, nbr_list in enumerate(raw):
        nbr_idx = np.asarray(nbr_list, dtype=np.intp)
        diff    = vertices[nbr_idx] - vertices[i]
        d2      = np.einsum('ij,ij->i', diff, diff)   # avoids sqrt entirely
        per_radius = []
        for r in radii:
            mask    = d2 <= r * r
            members = nbr_idx[mask]
            if i not in members:
                members = np.append(members, i)
            per_radius.append(members)
        neighbours.append(per_radius)

    return neighbours


# ---------------------------------------------------------------------------
# Per-vertex descriptor computation (parallelisable unit)
# ---------------------------------------------------------------------------

def _vertex_descriptor(i, neighbours_i, patch_features, radii, T, eps, normalise):
    """
    Compute the concatenated multi-radius descriptor for a single vertex.
    Inlined covariance chain to avoid Python function-call overhead per vertex.
    """
    n_radii = len(neighbours_i)
    out     = np.empty(T * n_radii)
    offset  = 0

    for members in neighbours_i:
        X = patch_features[members]
        # ── Regularised covariance ────────────────────────────────────────
        if X.shape[0] < 2:
            C = np.eye(X.shape[1])
        else:
            C = np.cov(X, rowvar=False)
            C += eps * np.eye(C.shape[0])
            if normalise:
                t = np.trace(C)
                if t > 1e-10:
                    C /= t
        # ── Log-map (eigh + broadcasting multiply — no np.diag alloc) ────
        eigvals, eigvecs = np.linalg.eigh(C)
        np.maximum(eigvals, 1e-10, out=eigvals)
        log_C = (eigvecs * np.log(eigvals)) @ eigvecs.T
        # ── Weighted upper triangle (cached index gather) ─────────────────
        d = log_C.shape[0]
        if d not in _TRIU_CACHE:
            _TRIU_CACHE[d] = _make_triu_meta(d)
        rows, cols, weights = _TRIU_CACHE[d]
        out[offset:offset + T] = log_C[rows, cols] * weights
        offset += T

    return out


# ---------------------------------------------------------------------------
# Top-level dense extraction
# ---------------------------------------------------------------------------

def compute_patches(vertices, patch_features,
                          radius    = [6.0, 9.0, 12.0],
                          eps       = 1e-6,
                          normalise = True,
                          n_jobs    = 1):
    """
    Compute a covariance patch descriptor for every vertex on the surface.

    Parameters
    ----------
    vertices       : (N, 3) surface vertex coordinates
    patch_features : (N, D) per-vertex scalar features
    radius         : float or list of floats, radii in Angstrom (default [6., 9., 12.])
    eps            : float, covariance regularisation (default 1e-6)
    normalise      : bool, trace-normalise covariance within each patch (default True)
    n_jobs         : int, reserved for future use (default 1).
                     Do not set > 1 when called from inside a ProcessPoolExecutor worker.

    Returns
    -------
    descriptors : (N, T * n_radii)
                  Row i is the descriptor for vertex i.
                  T = D*(D+1)/2
    """
    vertices       = np.asarray(vertices,       dtype=np.float64)
    patch_features = np.asarray(patch_features, dtype=np.float64)

    assert vertices.shape[0] == patch_features.shape[0], (
        f"vertices ({vertices.shape[0]}) and patch_features "
        f"({patch_features.shape[0]}) must have the same number of rows"
    )

    if isinstance(radius, (int, float)):
        radius = [float(radius)]
    radius = sorted(radius)

    N = vertices.shape[0]
    D = patch_features.shape[1]
    T = D * (D + 1) // 2

    # Prewarm the triu cache for this D so the first vertex doesn't pay the cost.
    if D not in _TRIU_CACHE:
        _TRIU_CACHE[D] = _make_triu_meta(D)

    # ------------------------------------------------------------------
    # 1. Batch neighbourhood lookup
    # ------------------------------------------------------------------
    neighbours = query_all_radii(vertices, radius)

    # ------------------------------------------------------------------
    # 2. Compute N covariance descriptors — serial within worker process
    # ------------------------------------------------------------------
    descriptors = np.empty((N, T * len(radius)), dtype=np.float64)
    for i in range(N):
        descriptors[i] = _vertex_descriptor(
            i, neighbours[i], patch_features, radius, T, eps, normalise
        )

    return descriptors
