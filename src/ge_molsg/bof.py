"""Bag-of-Features aggregation.

Four aggregation strategies are provided:

``hq_bof``
    Hard-quantisation BoF (HQ). Each vertex is assigned to its single nearest
    codeword; the histogram is a pure codeword frequency count. Fast and
    simple; no softmax, no k-NN spread.

``knn_histogram`` (alias: ``hard_knn_bof``)
    Hard k-NN BoF. Each vertex votes equally for its ``knn`` nearest codewords
    (by squared Euclidean distance); votes are summed and L1-normalised.
    No softmax weighting.

``knn_bof``
    k-NN soft BoF. Each vertex is assigned to its ``knn`` nearest codewords
    (by squared Euclidean distance). Gaussian-softmax weights are row-normalised
    per vertex so each vertex contributes a soft probability distribution over
    its k neighbours; these are summed and L1-normalised to form the histogram.

``soft_bof``
    Fully-soft assignment parameterised by a temperature ``tau``. Every
    codeword receives a weighted contribution from every vertex.

All codeword-distance computations route through :func:`_bof_sqdist`, which
evaluates the squared-Euclidean distance matrix via a single BLAS ``gemm``
(``||x||^2 + ||c||^2 - 2 x.c^T``). This is numerically identical to
``scipy.spatial.distance.cdist(..., 'sqeuclidean')`` up to floating-point
reassociation (~1e-12) but several times faster, since ``cdist`` does not use
BLAS for this metric.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster import vq
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize as sk_normalize


def _bof_sqdist(descriptors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between descriptors and codewords via BLAS.

    Returns an ``(N, K)`` matrix where entry ``(i, k)`` is
    ``||descriptors[i] - codebook[k]||^2``, computed as
    ``||x||^2 + ||c||^2 - 2 x.c^T`` so the dominant term is a single ``gemm``.
    Equivalent to ``cdist(codebook, descriptors, 'sqeuclidean').T`` up to
    floating-point reassociation. Tiny negative values from cancellation are
    clamped to zero.

    Parameters
    ----------
    descriptors : (N, D) array
    codebook : (K, D) array

    Returns
    -------
    sq_dist : (N, K) array
    """
    d = np.ascontiguousarray(descriptors, dtype=np.float64)
    c = np.ascontiguousarray(codebook, dtype=np.float64)
    dn = np.einsum("ij,ij->i", d, d)[:, None]      # (N, 1)
    cn = np.einsum("ij,ij->i", c, c)[None, :]      # (1, K)
    sq = dn + cn - 2.0 * (d @ c.T)                 # (N, K)
    np.maximum(sq, 0.0, out=sq)
    return sq


def _topk_indices(sq_dist: np.ndarray, knn: int) -> np.ndarray:
    """Indices of the ``knn`` smallest entries per row (unordered).

    Uses ``argpartition`` (O(K) per row) rather than a full ``argsort``. The
    selected set is identical to ``argsort(...)[:, :knn]``; only the intra-row
    order differs, which is irrelevant for hard 0/1 voting and for the
    row-normalised soft mask (both are order-invariant over the selected set).
    """
    k = min(knn, sq_dist.shape[1])
    if k >= sq_dist.shape[1]:
        return np.broadcast_to(np.arange(sq_dist.shape[1]), sq_dist.shape)
    return np.argpartition(sq_dist, k - 1, axis=1)[:, :k]


def _codebook_sigma(codebook: np.ndarray) -> float:
    """Global bandwidth: sqrt(2 * median pairwise Euclidean distance between codewords)."""
    return float(np.sqrt(2.0 * np.median(pdist(codebook, "euclidean"))))


def hq_bof(
    descriptors: np.ndarray,
    codebook: np.ndarray,
    norm: str = "l1",
) -> np.ndarray:
    """Hard-quantisation Bag-of-Features (HQ-BoF).

    Each vertex descriptor is assigned to its single nearest codeword, then the
    codeword frequency counts are normalised.

    Parameters
    ----------
    descriptors : (N, D) array
        Per-vertex descriptors for one molecule.
    codebook : (K, D) array
        Cluster centres from a fitted codebook.
    norm : {'l1', 'l2'}, default 'l1'
        Normalisation applied to the frequency histogram.

    Returns
    -------
    histogram : (K,) array
        Normalised codeword frequency vector.
    """
    descriptors = np.nan_to_num(
        np.asarray(descriptors, dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    # Single-nearest assignment: scipy VQ is faster here than materialising the
    # full (N, K) gemm distance matrix, so hq_bof intentionally does not use
    # _bof_sqdist (unlike the k-NN / soft paths, which need all K distances).
    cb = np.ascontiguousarray(codebook, dtype=np.float64)
    labels, _ = vq.vq(descriptors, cb)
    K = codebook.shape[0]
    frequency = np.bincount(labels, minlength=K).astype(np.float64)
    return sk_normalize(frequency.reshape(1, -1), norm=norm)[0]


def knn_histogram(
    descriptors: np.ndarray,
    codebook: np.ndarray,
    knn: int = 3,
    norm: str = "l1",
) -> np.ndarray:
    """Hard k-NN Bag-of-Features (knn_histogram).

    Each vertex votes equally for its ``knn`` nearest codewords (by squared
    Euclidean distance). Votes are summed and normalised. No softmax weighting.

    Parameters
    ----------
    descriptors : (N, D) array
        Per-vertex descriptors for one molecule.
    codebook : (K, D) array
        Cluster centres from a fitted codebook.
    knn : int, default 3
        Number of nearest codewords each vertex votes for.
    norm : {'l1', 'l2'}, default 'l1'
        Normalisation applied to the final histogram.

    Returns
    -------
    histogram : (K,) array
        Normalised hard-assignment histogram.
    """
    descriptors = np.nan_to_num(
        np.asarray(descriptors, dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    sq_dist = _bof_sqdist(descriptors, codebook)                   # (N, K)
    nn_idx = _topk_indices(sq_dist, knn)                           # (N, knn)
    mask = np.zeros((descriptors.shape[0], codebook.shape[0]))     # (N, K)
    np.put_along_axis(mask, nn_idx, 1.0, axis=1)
    return sk_normalize(mask.sum(axis=0).reshape(1, -1), norm=norm)[0]


# ``hard_knn_bof`` is kept as an alias of ``knn_histogram`` for back-compatibility.
hard_knn_bof = knn_histogram


def knn_bof(
    descriptors: np.ndarray,
    codebook: np.ndarray,
    knn: int = 3,
    sigma: float | None = None,
    norm: str = "l1",
) -> np.ndarray:
    """k-NN soft Bag-of-Features (KNN-BoF).

    Each vertex is assigned to its ``knn`` nearest codewords (by squared
    Euclidean distance). Gaussian-softmax weights are row-normalised per vertex
    so each vertex contributes a soft probability distribution over its k
    neighbours. The per-vertex distributions are summed and normalised to form
    the histogram.

    Parameters
    ----------
    descriptors : (N, D) array
        Per-vertex descriptors for one molecule.
    codebook : (K, D) array
        Cluster centres from a fitted codebook.
    knn : int, default 3
        Number of nearest codewords each vertex votes for.
    sigma : float, optional
        Gaussian bandwidth. Defaults to ``sqrt(2 * median pairwise codeword
        distance)`` if not supplied.
    norm : {'l1', 'l2'}, default 'l1'
        Normalisation applied to the final histogram.

    Returns
    -------
    histogram : (K,) array
        Normalised soft-assignment histogram.
    """
    descriptors = np.nan_to_num(
        np.asarray(descriptors, dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    if sigma is None:
        sigma = _codebook_sigma(codebook)

    sq_dist = _bof_sqdist(descriptors, codebook)                   # (N, K)
    softmax = np.exp(-0.5 * sq_dist / (sigma ** 2 + 1e-12))        # (N, K)

    nn_idx = _topk_indices(sq_dist, knn)                           # (N, knn)
    mask = np.zeros_like(softmax)                                  # (N, K)
    np.put_along_axis(mask, nn_idx, 1.0, axis=1)

    # Row-normalise per vertex: each vertex becomes a soft distribution.
    encoding = sk_normalize(softmax * mask, norm="l1")             # (N, K)

    histogram = encoding.sum(axis=0)                               # (K,)
    return sk_normalize(histogram.reshape(1, -1), norm=norm)[0]


def soft_bof(
    descriptors: np.ndarray,
    codebook: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Fully-soft Bag-of-Features with temperature ``tau``.

    Every codeword receives a weighted contribution from every vertex via a
    temperature-scaled softmax. L1-normalised.

    Parameters
    ----------
    descriptors : (N, D) array
        Per-vertex descriptors for one molecule.
    codebook : (K, D) array
        Cluster centres from a fitted codebook.
    tau : float
        Temperature; smaller values sharpen the assignment.

    Returns
    -------
    histogram : (K,) array
        L1-normalised soft-assignment histogram.
    """
    descriptors = np.nan_to_num(
        np.asarray(descriptors, dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    dists = _bof_sqdist(descriptors, codebook)
    logits = -dists / tau
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    hist = weights.sum(axis=0)
    total = hist.sum()
    return hist / total if total > 0 else hist
