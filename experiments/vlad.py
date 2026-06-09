"""
vlad.py
=======
Vector of Locally Aggregated Descriptors (VLAD) for patch-based
molecular surface descriptors.

Produces a fixed-length vector per molecule
regardless of how many patches it has.

References
----------
Jégou, H. et al. "Aggregating local descriptors into a compact image
representation." CVPR 2010.

Arandjelović, R. & Zisserman, A. "All about VLAD." CVPR 2013.
"""

import numpy as np
from sklearn.preprocessing import normalize as sk_normalize


def vlad_encode(descriptors, codebook, normalize="ssr"):
    """
    VLAD encoding of a set of descriptors against a codebook.

    For each codeword k, sum the residuals (x_i - c_k) of all
    descriptors assigned to k, then flatten and normalise.

    Parameters
    ----------
    descriptors : (N, D) array
        Local descriptors for one molecule (e.g. patch log-covariances).
    codebook : (K, D) array
        Codebook centres (from k-means).
    normalize : str, one of {"ssr", "l2", "none"}
        Normalisation strategy applied to the final VLAD vector.
        - "ssr"  : signed-square-root (power normalisation, alpha=0.5)
                   followed by L2 normalisation.  Best for downstream
                   linear classifiers and chi2/cosine kernels.  This is
                   the standard from Arandjelović & Zisserman 2013.
        - "l2"   : L2 normalisation only.
        - "none" : raw VLAD (not recommended for k-means codebooks).

    Returns
    -------
    vlad_vec : (K * D,) array
        Flattened VLAD vector.
    """
    descriptors = np.asarray(descriptors, dtype=np.float64)
    codebook    = np.asarray(codebook,    dtype=np.float64)

    K, D = codebook.shape
    N    = descriptors.shape[0]
    assert descriptors.shape[1] == D, (
        f"descriptor dim ({descriptors.shape[1]}) != codebook dim ({D})"
    )

    # ── Hard assignment ──────────────────────────────────────────────
    # (N, K) squared distances
    dists   = (
        (descriptors ** 2).sum(axis=1, keepdims=True)
        + (codebook ** 2).sum(axis=1)
        - 2.0 * descriptors @ codebook.T
    )
    assigns = np.argmin(dists, axis=1)          # (N,)

    # ── Accumulate residuals ─────────────────────────────────────────
    V = np.zeros((K, D), dtype=np.float64)
    for i in range(N):
        k = assigns[i]
        V[k] += descriptors[i] - codebook[k]

    # ── Intra-normalise (per-cluster L2) ─────────────────────────────
    # Reduces burstiness: a single dominant cluster can't swamp the rest.
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    V = V / norms

    # ── Flatten ──────────────────────────────────────────────────────
    vlad_vec = V.ravel()                        # (K * D,)

    # ── Global normalisation ─────────────────────────────────────────
    if normalize == "ssr":
        vlad_vec = np.sign(vlad_vec) * np.sqrt(np.abs(vlad_vec))
        norm = np.linalg.norm(vlad_vec)
        if norm > 1e-12:
            vlad_vec /= norm
    elif normalize == "l2":
        norm = np.linalg.norm(vlad_vec)
        if norm > 1e-12:
            vlad_vec /= norm
    # else: "none" — return raw

    return vlad_vec


# ──────────────────────────────────────────────────────────────────────
# Soft VLAD  (optional — uses soft assignment weights instead of hard)
# ──────────────────────────────────────────────────────────────────────

def soft_vlad_encode(descriptors, codebook, tau=None, normalize="ssr"):
    """
    Soft-assignment VLAD.

    Instead of hard nearest-neighbour assignment, each descriptor
    contributes its residual to every codeword, weighted by the
    softmax assignment probability.  This is smoother and often
    works better with small descriptor counts (e.g. 32 patches).

    Parameters
    ----------
    descriptors : (N, D) array
    codebook    : (K, D) array
    tau         : float or None
        Softmax temperature.  If None, uses the mean nearest-neighbour
        distance (a reasonable automatic default).
    normalize   : str, {"ssr", "l2", "none"}

    Returns
    -------
    vlad_vec : (K * D,) array
    """
    descriptors = np.asarray(descriptors, dtype=np.float64)
    codebook    = np.asarray(codebook,    dtype=np.float64)

    K, D = codebook.shape
    N    = descriptors.shape[0]

    # ── Squared distances ────────────────────────────────────────────
    dists = (
        (descriptors ** 2).sum(axis=1, keepdims=True)
        + (codebook ** 2).sum(axis=1)
        - 2.0 * descriptors @ codebook.T
    )                                            # (N, K)

    # ── Auto-calibrate tau if not given ──────────────────────────────
    if tau is None:
        tau = float(np.mean(np.min(dists, axis=1)))
        if tau < 1e-12:
            tau = 1.0

    # ── Soft assignment weights ──────────────────────────────────────
    logits  = -dists / tau
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)  # (N, K)

    # ── Weighted residual accumulation ───────────────────────────────
    # V[k] = sum_i  w_{ik} * (x_i - c_k)
    residuals = descriptors[:, None, :] - codebook[None, :, :]  # (N, K, D)
    V = np.einsum("nk,nkd->kd", weights, residuals)            # (K, D)

    # ── Intra-normalise ──────────────────────────────────────────────
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    V = V / norms

    # ── Flatten + global normalisation ───────────────────────────────
    vlad_vec = V.ravel()

    if normalize == "ssr":
        vlad_vec = np.sign(vlad_vec) * np.sqrt(np.abs(vlad_vec))
        norm = np.linalg.norm(vlad_vec)
        if norm > 1e-12:
            vlad_vec /= norm
    elif normalize == "l2":
        norm = np.linalg.norm(vlad_vec)
        if norm > 1e-12:
            vlad_vec /= norm

    return vlad_vec


# ──────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────────────────────────────

def vlad_bof(input, soft=True):
    """
    Usage
    -----
    In desc_gen, replace:
        patch_bof = soft_bof([patch_desc, patch_cb, patch_tau])
    with:
        patch_bof = vlad_bof([patch_desc, patch_cb, patch_tau])

    Parameters
    ----------
    input : [descriptors, codebook, tau]
    soft  : bool
        If True, use soft VLAD (recommended for patches).
        If False, use hard VLAD.

    Returns
    -------
    vlad_vec : (K * D,) array, L2-normalised
    """
    descriptors, codebook, tau = input

    if soft:
        return soft_vlad_encode(descriptors, codebook, tau=tau,
                                normalize="ssr")
    else:
        return vlad_encode(descriptors, codebook, normalize="ssr")
