"""Bag-of-Features aggregation.

Aggregates per-vertex descriptors into a fixed-length per-molecule histogram
against a codebook. ``knn_histogram`` (the GE-MolSG default) uses a hard k-NN
assignment masked by a global-bandwidth softmax weight; ``soft_bof`` is a
fully-soft alternative parameterized by a temperature.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist


def knn_histogram(
    descriptors: np.ndarray, codebook: np.ndarray, knn: int = 3
) -> np.ndarray:
    """Hard k-NN BoF with a global-sigma softmax mask.

    Each descriptor votes for its ``knn`` nearest codewords, weighted by a
    Gaussian of the descriptor-codeword distance with a global bandwidth
    sigma = sqrt(2 * median pairwise codeword distance). The histogram is L1
    normalized.
    """
    descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.sqrt(2 * np.median(pdist(codebook, "euclidean")))
    sq_dist = cdist(codebook, descriptors, metric="sqeuclidean").T  # (N, K)
    softmax = np.exp(-0.5 * sq_dist / (sigma ** 2 + 1e-12))
    nn_idx = np.argsort(sq_dist, axis=1)[:, :knn]
    mask = np.zeros_like(softmax)
    for row, ix in zip(mask, nn_idx):
        row[ix] = 1.0
    weighted = (mask * softmax).sum(axis=0)
    total = weighted.sum()
    return weighted / total if total > 0 else weighted


def soft_bof(
    descriptors: np.ndarray, codebook: np.ndarray, tau: float
) -> np.ndarray:
    """Fully-soft BoF with temperature ``tau``. L1-normalised histogram."""
    descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
    z_sq = (descriptors ** 2).sum(axis=1, keepdims=True)
    e_sq = (codebook ** 2).sum(axis=1)
    dists = z_sq + e_sq - 2 * descriptors @ codebook.T
    logits = -dists / tau
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    hist = weights.sum(axis=0)
    total = hist.sum()
    return hist / total if total > 0 else hist
