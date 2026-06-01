"""Molecular surface store for unit tests.

Provides a real reference surface (the thalidomide R enantiomer, shipped as
``tests/data/thalidomide_R.npy``) alongside deterministic synthetic surfaces.
The real surface is used where a representative molecular surface matters; the
lightweight synthetic surfaces keep the fast invariant tests quick.
"""
# tests/store.py
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from ge_molsg import MolSurface, load_surface_npy

DATA_DIR = Path(__file__).parent / "data"
THALIDOMIDE_R = DATA_DIR / "thalidomide_R.npy"


@functools.lru_cache(maxsize=1)
def thalidomide_r() -> MolSurface:
    """Return the thalidomide R-enantiomer reference surface.

    Loaded once and cached for the test session.
    """
    return load_surface_npy(str(THALIDOMIDE_R), name="thalidomide_R")


def sphere_surface(n: int = 200, seed: int = 0, shift: float = 0.0) -> MolSurface:
    """Return a noisy unit-sphere surface with a smooth ESP field.

    Parameters
    ----------
    n : int
        Number of surface vertices.
    seed : int
        Random seed for reproducibility.
    shift : float
        Rigid translation applied to all vertices (used to make dissimilar
        surfaces for separation tests).
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    vertices = u * (1.0 + 0.05 * rng.normal(size=(n, 1))) + shift
    esp = np.sin(3 * vertices[:, 0]) + 0.5 * vertices[:, 2] + 0.1 * rng.normal(size=n)
    return MolSurface(vertices=vertices, faces=None, esp=esp, name=f"sphere_{seed}")


def small_surface(seed: int = 0) -> MolSurface:
    """Return a small synthetic surface for fast tests."""
    return sphere_surface(n=80, seed=seed)


def surface_batch(count: int = 4, n: int = 120, seed: int = 0):
    """Return a list of distinct synthetic surfaces."""
    return [sphere_surface(n=n, seed=seed + i) for i in range(count)]
