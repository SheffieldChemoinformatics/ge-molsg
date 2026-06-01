"""Per-vertex WKS descriptor and batch generation.

Orchestrates the per-surface pipeline:

    surface -> augmented point cloud [x, y, z, esp * elec_weight]
            -> adaptive-bandwidth affinity     (graph.compute_affinity)
            -> graph Laplacian                  (graph.graph_laplacian)
            -> bottom eigenpairs, trivial dropped (eigen.compute_eigenpairs)
            -> WKS                              (wks.wks)

``compute_wks`` produces a single descriptor; ``compute_wks_batch`` runs it over
many surfaces with optional process-based parallelism and a progress bar. Each
stage is independently importable, so portions of the workflow can be used
without the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Sequence

import numpy as np

from .surface import MolSurface
from .graph import compute_affinity, graph_laplacian
from .eigen import compute_eigenpairs
from .wks import wks
from .parallel import parallel_map


@dataclass
class GEMolSGConfig:
    """Configuration for the per-vertex WKS descriptor."""

    # Graph construction
    n_components: int = 100           # Eigenpairs retained (after dropping trivial).
    n_neighbors: int = 100            # kNN graph connectivity.
    elec_weight: float = 0.3          # ESP scale in the augmented point cloud.
    adaptive_bw: bool = True          # Per-point adaptive bandwidth.
    sigma: Optional[float] = None     # Fixed bandwidth; used only if adaptive_bw is False.
    square_distances: bool = True
    symmetrize: bool = True
    laplacian_type: str = "normalized"  # 'normalized' | 'unnormalized' | 'random_walk'.

    # Neighbor backend
    backend: str = "ckdtree"
    backend_kwargs: dict = field(default_factory=dict)

    # Eigensolver
    eigensolver: str = "arpack"       # 'arpack' | 'lobpcg' | 'dense'.
    eigen_tol: float = 0.0            # 0 = machine precision.
    random_state: Optional[int] = None

    # WKS
    evals: int = 50                   # Descriptor width (energy evaluations).
    variance: int = 7

    # Descriptor assembly
    l2_normalize: bool = True


def compute_wks(surface: MolSurface, config: Optional[GEMolSGConfig] = None) -> np.ndarray:
    """Compute the per-vertex WKS descriptor for one surface."""
    cfg = config or GEMolSGConfig()

    points = surface.augmented_points(elec_weight=cfg.elec_weight)

    W = compute_affinity(
        points,
        n_neighbors=cfg.n_neighbors,
        backend=cfg.backend,
        backend_kwargs=cfg.backend_kwargs,
        adaptive_bw=cfg.adaptive_bw,
        sigma=cfg.sigma,
        square_distances=cfg.square_distances,
        symmetrize=cfg.symmetrize,
    )
    L = graph_laplacian(W, laplacian_type=cfg.laplacian_type)

    eigensystem = compute_eigenpairs(
        L,
        n_components=cfg.n_components,
        drop_first=True,
        eigensolver=cfg.eigensolver,
        eigen_tol=cfg.eigen_tol,
        random_state=cfg.random_state,
    )

    return wks(
        eigensystem,
        evals=cfg.evals,
        variance=cfg.variance,
        l2=cfg.l2_normalize,
    )


def compute_wks_batch(
    surfaces: Sequence[MolSurface],
    config: Optional[GEMolSGConfig] = None,
    n_jobs: int = 1,
    progress: bool = False,
) -> List[np.ndarray]:
    """Compute per-vertex WKS descriptors for many surfaces.

    Parameters
    ----------
    surfaces : sequence of MolSurface
    config : GEMolSGConfig, optional
    n_jobs : int, default 1
        Worker processes; ``1`` runs serially, ``-1`` uses all CPUs.
    progress : bool, default False
        Display a progress bar.

    Returns
    -------
    list of arrays
        Per-vertex descriptors, in input order.
    """
    cfg = config or GEMolSGConfig()
    worker = partial(_compute_wks_worker, config=cfg)
    return parallel_map(
        worker, surfaces, n_jobs=n_jobs, progress=progress, desc="Descriptors"
    )


def _compute_wks_worker(surface: MolSurface, config: GEMolSGConfig) -> np.ndarray:
    """Top-level worker (picklable) for process-based parallelism."""
    return compute_wks(surface, config)
