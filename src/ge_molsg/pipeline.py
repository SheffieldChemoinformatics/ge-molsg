"""High-level convenience wrapper.

:class:`GEMolSG` composes the free functions in this package, holding a config
and a fitted codebook so they need not be passed repeatedly. It is entirely
optional: every stage it calls (``compute_wks``, ``compute_wks_batch``,
``build_codebook``, ``hq_bof``, ``knn_bof``, ``soft_bof``) can be used directly.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .surface import MolSurface
from .descriptor import GEMolSGConfig, compute_wks, compute_wks_batch
from .codebook import build_codebook, build_descriptor_pool, sample_descriptor_pool
from .bof import hq_bof, knn_bof, soft_bof


class GEMolSG:
    """WKS descriptor, codebook, and Bag-of-Features pipeline."""

    def __init__(
        self,
        config: Optional[GEMolSGConfig] = None,
        n_codewords: int = 1000,
        bof_mode: str = "knn_bof",         # 'hq' | 'knn_bof' | 'soft'
        bof_knn: int = 3,
        bof_tau: Optional[float] = None,
        sample_n_per_mol: Optional[int] = None,   # None = retain all vertices
        random_state: int = 42,
        n_jobs: int = 1,
        progress: bool = False,
    ):
        self.config = config or GEMolSGConfig()
        self.n_codewords = n_codewords
        self.bof_mode = bof_mode
        self.bof_knn = bof_knn
        self.bof_tau = bof_tau
        self.sample_n_per_mol = sample_n_per_mol
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.progress = progress
        self.codebook_: Optional[np.ndarray] = None

    # Per-vertex descriptor -------------------------------------------------
    def descriptor(self, surface: MolSurface) -> np.ndarray:
        """Per-vertex WKS descriptor for one surface."""
        return compute_wks(surface, self.config)

    def descriptors(self, surfaces: Sequence[MolSurface]) -> List[np.ndarray]:
        """Per-vertex WKS descriptors for many surfaces (parallel if n_jobs>1)."""
        return compute_wks_batch(
            surfaces, self.config, n_jobs=self.n_jobs, progress=self.progress
        )

    # Codebook --------------------------------------------------------------
    def fit_codebook(
        self,
        sample_surfaces: Sequence[MolSurface],
        descriptors: Optional[Sequence[np.ndarray]] = None,
    ) -> "GEMolSG":
        """Build the codebook from a sample of surfaces.

        If ``descriptors`` are supplied they are used directly; otherwise they
        are generated from ``sample_surfaces``. The KMeans fit itself is serial.
        """
        if descriptors is None:
            descriptors = self.descriptors(sample_surfaces)
        if self.sample_n_per_mol is not None:
            descriptors = sample_descriptor_pool(descriptors, self.sample_n_per_mol)
        pool = build_descriptor_pool(descriptors)
        self.codebook_ = build_codebook(
            pool, self.n_codewords, random_state=self.random_state
        )
        return self

    # Bag-of-Features transform --------------------------------------------
    def _aggregate(self, per_vertex: np.ndarray) -> np.ndarray:
        if self.codebook_ is None:
            raise RuntimeError("Call fit_codebook(...) before transform(...).")
        if self.bof_mode == "hq":
            return hq_bof(per_vertex, self.codebook_)
        if self.bof_mode == "knn_bof":
            return knn_bof(per_vertex, self.codebook_, knn=self.bof_knn)
        if self.bof_mode == "soft":
            if self.bof_tau is None:
                raise ValueError("bof_mode='soft' requires bof_tau.")
            return soft_bof(per_vertex, self.codebook_, self.bof_tau)
        raise ValueError(
            f"Unknown bof_mode '{self.bof_mode}'. Choose from 'hq', 'knn_bof', 'soft'."
        )

    def transform(self, surface: MolSurface) -> np.ndarray:
        """One surface to a fixed-length Bag-of-Features vector."""
        return self._aggregate(self.descriptor(surface))

    def transform_many(self, surfaces: Sequence[MolSurface]) -> np.ndarray:
        """Many surfaces to a stacked Bag-of-Features matrix."""
        per_vertex = self.descriptors(surfaces)
        return np.vstack([self._aggregate(pv) for pv in per_vertex])
