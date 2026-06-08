"""Exceptions for the ge_molsg package."""

from __future__ import annotations


class GEMolSGError(Exception):
    """Base exception for all ge_molsg errors."""


class SurfaceError(GEMolSGError):
    """Raised when a molecular surface is malformed or inconsistent."""


class BackendError(GEMolSGError):
    """Raised when a neighbour backend cannot be resolved or executed."""
