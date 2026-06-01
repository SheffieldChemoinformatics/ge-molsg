"""Extension point for covariance patch descriptors.

The patch descriptor is a separate channel that aggregates per-vertex chemical
fields (ESP, logP, SASA, curvature, and similar) into local covariance patches::

    features    = stack of per-vertex chemical fields
    patch_desc  = compute_patches(vertices, features, radius, normalise)
    patch_bof   = soft_bof(patch_desc, codebook, tau)        # or VLAD

This module exposes the interface but ships no implementation: it is not fused
with the WKS pipeline and remains inert until a ``compute_patches`` callable
(and, optionally, a VLAD aggregator) is supplied to :class:`PatchDescriptor`.
Methods raise :class:`NotImplementedError` until then.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np


# Supported per-vertex field names for the patch channel.
VALID_PATCH_FEATURES = [
    "esp", "logp", "sdc", "sdx", "sa",
    "si", "mean_curv", "gauss_curv", "phi_vdw",
]


@dataclass
class PatchConfig:
    """Configuration for the (future) patch descriptor channel."""

    features: Sequence[str] = field(default_factory=lambda: ["esp"])
    radii: Sequence[float] = field(default_factory=lambda: [2.5])
    normalise: bool = True
    feature_weights: Optional[dict] = None  # Optional per-feature weight, keyed by name.


class PatchDescriptor:
    """Pluggable covariance-patch descriptor.

    Parameters
    ----------
    config : PatchConfig
        Which per-vertex fields to stack and the patch radii/normalisation.
    compute_patches : callable, optional
        ``(vertices, features, radius, normalise) -> (M, D) array``. Supply a
        real implementation to activate the channel.
    vlad_fn : callable, optional
        ``([patch_desc, codebook, tau], soft=bool) -> vector`` for the VLAD
        aggregation variant. Optional.
    """

    def __init__(
        self,
        config: Optional[PatchConfig] = None,
        compute_patches: Optional[Callable] = None,
        vlad_fn: Optional[Callable] = None,
    ):
        self.config = config or PatchConfig()
        self._compute_patches = compute_patches
        self._vlad_fn = vlad_fn

    @property
    def is_active(self) -> bool:
        return self._compute_patches is not None

    def _require_backend(self):
        if self._compute_patches is None:
            raise NotImplementedError(
                "Patch descriptors are inactive. Supply a `compute_patches` "
                "callable to PatchDescriptor(...) to enable this channel."
            )

    def stack_features(self, field_pool: dict) -> np.ndarray:
        """Stack the configured per-vertex fields into a (N, F) array.

        ``field_pool`` maps feature name -> (N, 1) column, e.g. the per-vertex
        chemical fields produced upstream (esp, logp, sa, ...). Applies manual
        feature weights if configured.
        """
        cols: List[np.ndarray] = []
        for name in self.config.features:
            if name not in field_pool:
                raise KeyError(f"field '{name}' missing from field_pool")
            cols.append(np.asarray(field_pool[name]).reshape(-1, 1))
        feat = np.hstack(cols) if cols else np.empty((0, 0))
        if self.config.feature_weights:
            w = np.array(
                [self.config.feature_weights.get(n, 1.0)
                 for n in self.config.features],
                dtype=np.float64,
            )
            feat = feat * w
        return feat

    def per_vertex(self, vertices: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Per-vertex patch descriptors via the supplied ``compute_patches``."""
        self._require_backend()
        desc = self._compute_patches(
            vertices, features,
            radius=self.config.radii, normalise=self.config.normalise,
        )
        return np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)

    def bag_of_features(self, patch_desc, codebook, tau, mode: str = "soft"):
        """Aggregate patch descriptors into a per-molecule vector.

        ``mode='soft'`` uses the shared ``bof.soft_bof``; ``mode='vlad'``
        requires a ``vlad_fn`` to have been supplied.
        """
        self._require_backend()
        if mode == "soft":
            from .bof import soft_bof
            return soft_bof(patch_desc, codebook, tau)
        if mode == "vlad":
            if self._vlad_fn is None:
                raise NotImplementedError(
                    "VLAD aggregation needs a `vlad_fn` (e.g. vlad.vlad_bof)."
                )
            return self._vlad_fn([patch_desc, codebook, tau], soft=False)
        raise ValueError(f"Unknown patch BoF mode '{mode}'")
