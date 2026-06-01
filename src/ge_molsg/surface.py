"""Molecular surface container and loaders.

A surface is a triangulated mesh (vertices and optional faces) carrying a
per-vertex electrostatic potential (ESP). Surfaces are stored as ``.npy``
object arrays ``[vertices, faces, charges]``; :class:`MolSurface` wraps the
loaded arrays so downstream code does not index into a raw object array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .exception import SurfaceError


@dataclass
class MolSurface:
    """Triangulated molecular surface with a per-vertex ESP field.

    Parameters
    ----------
    vertices : (N, 3) float array
        Cartesian coordinates of surface vertices.
    faces : (M, 3) int array
        Triangle vertex indices. May be ``None`` when only the point cloud and
        ESP are required, as the descriptor pipeline does not use faces.
    esp : (N,) float array
        Per-vertex electrostatic potential (partial-charge projection).
    name : str, optional
        Identifier, typically the source filename.
    """

    vertices: np.ndarray
    faces: Optional[np.ndarray]
    esp: np.ndarray
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        self.esp = np.asarray(self.esp, dtype=np.float64).ravel()
        if self.faces is not None:
            self.faces = np.ascontiguousarray(self.faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise SurfaceError(
                f"vertices must be (N, 3), got {self.vertices.shape}"
            )
        if self.esp.shape[0] != self.vertices.shape[0]:
            raise SurfaceError(
                f"esp length {self.esp.shape[0]} != n_vertices "
                f"{self.vertices.shape[0]}"
            )

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    def augmented_points(self, elec_weight: float = 0.3) -> np.ndarray:
        """Return the 4-D graph input ``[x, y, z, esp * elec_weight]``.

        This is the point set the affinity graph is built on. ESP is scaled by
        ``elec_weight`` so its spread is commensurate with the spatial
        coordinates before the (Euclidean) neighbor search.
        """
        col = (self.esp * elec_weight).reshape(-1, 1)
        return np.concatenate([self.vertices, col], axis=1)


def load_surface_npy(path: str, name: Optional[str] = None) -> MolSurface:
    """Load a ``[vertices, faces, charges]`` object-array ``.npy`` file."""
    data = np.load(path, allow_pickle=True)
    vertices = np.asarray(data[0], dtype=np.float64)
    faces = np.asarray(data[1], dtype=np.int64) if data[1] is not None else None
    esp = np.asarray(data[2], dtype=np.float64).ravel()
    return MolSurface(vertices=vertices, faces=faces, esp=esp, name=name or path)
