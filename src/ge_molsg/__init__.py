"""GE-MolSG: WKS surface descriptors with a graph-Laplacian spectral embedding.

The graph Laplacian whose bottom spectrum feeds the Wave Kernel Signature is
built with scipy: an adaptive-bandwidth Euclidean affinity matrix, a normalised
Laplacian, and the bottom eigenpairs with the trivial mode dropped. The
neighbor search is pluggable (scikit-learn by default).

Each stage is an importable free function; :class:`GEMolSG` is an optional
convenience wrapper that composes them. The covariance-patch channel is kept
open via :mod:`ge_molsg.patches` but is not part of the WKS pipeline.
"""

from ge_molsg._version import __version__
from ge_molsg.exception import GEMolSGError, SurfaceError, BackendError
from ge_molsg.surface import MolSurface, load_surface_npy
from ge_molsg.neighbors import (
    get_neighbor_backend,
    make_ckdtree_backend,
    make_sklearn_backend,
)
from ge_molsg.graph import (
    adaptive_bandwidth,
    knn_distance_graph,
    compute_affinity,
    graph_laplacian,
    build_laplacian,
)
from ge_molsg.eigen import compute_eigenpairs
from ge_molsg.wks import wks_descriptor, wks
from ge_molsg.descriptor import (
    GEMolSGConfig,
    compute_wks,
    compute_wks_batch,
)
from ge_molsg.codebook import (
    build_codebook,
    build_descriptor_pool,
    sample_descriptor_pool,
    farthest_point_sample,
)
from .bof import hq_bof, knn_bof, soft_bof, knn_histogram, hard_knn_bof
from ge_molsg.parallel import parallel_map
from ge_molsg.pipeline import GEMolSG
from ge_molsg.patches import PatchDescriptor, PatchConfig, VALID_PATCH_FEATURES

__all__ = [
    "__version__",
    # exceptions
    "GEMolSGError",
    "SurfaceError",
    "BackendError",
    # surface
    "MolSurface",
    "load_surface_npy",
    # neighbours
    "get_neighbor_backend",
    "make_ckdtree_backend",
    "make_sklearn_backend",
    # graph
    "adaptive_bandwidth",
    "knn_distance_graph",
    "compute_affinity",
    "graph_laplacian",
    "build_laplacian",
    # spectral
    "compute_eigenpairs",
    # wks
    "wks_descriptor",
    "wks",
    # descriptor orchestration
    "GEMolSGConfig",
    "compute_wks",
    "compute_wks_batch",
    # codebook
    "build_codebook",
    "build_descriptor_pool",
    "sample_descriptor_pool",
    "farthest_point_sample",
    # bof
    "hq_bof",
    "knn_histogram",
    "hard_knn_bof",
    "knn_bof",
    "soft_bof",
    # parallelism
    "parallel_map",
    # pipeline
    "GEMolSG",
    # patch extension (open hook, not part of the WKS pipeline)
    "PatchDescriptor",
    "PatchConfig",
    "VALID_PATCH_FEATURES",
]
