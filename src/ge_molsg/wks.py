"""Wave Kernel Signature (WKS) descriptor.

``wks_descriptor`` implements the WKS of Aubry et al. (2011), ported from the
reference MATLAB implementation [1]. Edge-case guards are noted inline.

The eigensystem passed in must already have its trivial mode dropped (see
:func:`ge_molsg.eigen.compute_eigenpairs`): the WKS log-energy grid is anchored
at ``log(|lambda_0|)``, and a near-zero trivial eigenvalue would distort the
descriptor scale.

[1] http://imagine.enpc.fr/~aubrym/projects/wks/index.html
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import normalize as l2_normalize


def wks_descriptor(
    eigensystem: Tuple[np.ndarray, np.ndarray],
    evals: int = 100,
    variance: int = 7,
    sample: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Compute the Wave Kernel Signature for a Laplace-Beltrami eigensystem.

    Faithful to the molsg/Aubry implementation.

    Parameters
    ----------
    eigensystem : (lambda, phi)
        ``lambda`` : (K,) eigenvalues (ascending, trivial mode already dropped).
        ``phi``    : (N, K) eigenvectors.
    evals : int
        Number of WKS energy evaluations (descriptor dimensionality per vertex).
    variance : int
        Gaussian variance multiplier; sigma = variance * (energy-grid spacing).
    sample : sequence of int, optional
        If given, compute WKS only at these vertex indices.

    Returns
    -------
    descriptor : (N, evals) array  (or (len(sample), evals) if sampled)
    """
    lam, phi = eigensystem
    lam = np.asarray(lam, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64)
    if sample is not None:
        phi = phi[sample]

    # Log-energy axis; the clamp guards against residual zero eigenvalues.
    log_E = np.log(np.maximum(np.abs(lam), 1e-6))  # (K,)

    # Sort so the grid spans [min, max] energy regardless of input ordering.
    log_E_sorted = np.sort(log_E)
    e = np.linspace(log_E_sorted[0], log_E_sorted[-1] / 1.02, evals)

    # Guard the degenerate single-evaluation case where spacing is undefined.
    spacing = (e[1] - e[0]) if evals > 1 else 1.0
    sigma = spacing * variance
    if sigma <= 0:
        sigma = 1e-6

    phi_squared = phi * phi  # (N, K)

    # WKS(e) = sum_k exp(-(e - log_E_k)^2 / 2 sigma^2) * phi_k^2.
    E = np.tile(e, (lam.shape[0], 1)).T              # (evals, K)
    Tau = np.exp(-((E - log_E) ** 2) / (2 * sigma ** 2))  # (evals, K)
    WKS = np.tensordot(Tau, phi_squared, (1, 1)).T   # (N, evals)

    # Per-energy normalisation (sum of Gaussian weights).
    C = (
        np.exp(
            -((np.tile(e, (lam.shape[0], 1)) - np.tile(log_E, (evals, 1)).T) ** 2)
            / (2 * sigma ** 2)
        )
    ).sum(axis=0)  # (evals,)

    descriptor = WKS / np.maximum(C, 1e-12)
    return descriptor


def wks(
    eigensystem: Tuple[np.ndarray, np.ndarray],
    evals: int = 50,
    variance: int = 7,
    l2: bool = True,
) -> np.ndarray:
    """Per-vertex WKS descriptor, optionally L2-normalized."""
    desc = wks_descriptor(eigensystem, evals=evals, variance=variance)
    if l2:
        desc = l2_normalize(desc)
    return np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
