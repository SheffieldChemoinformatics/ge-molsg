"""Test cases for the surface module."""
import numpy as np
import pytest

from ge_molsg import MolSurface, load_surface_npy
from ge_molsg.exception import SurfaceError
from tests.store import THALIDOMIDE_R, sphere_surface, thalidomide_r


def test_load_thalidomide_reference_surface():
    """It loads the thalidomide R-enantiomer surface with expected layout."""
    surface = thalidomide_r()
    assert surface.name == "thalidomide_R"
    assert surface.n_vertices == 5738
    assert surface.vertices.shape == (5738, 3)
    assert surface.faces.shape == (11472, 3)
    assert surface.esp.shape == (5738,)


def test_thalidomide_augmented_points_layout():
    """The augmented cloud of the real surface has a scaled ESP column."""
    surface = thalidomide_r()
    points = surface.augmented_points(elec_weight=0.3)
    assert points.shape == (5738, 4)
    assert np.allclose(points[:, 3], surface.esp * 0.3)


def test_load_surface_npy_round_trip():
    """load_surface_npy reads the on-disk object array directly."""
    surface = load_surface_npy(str(THALIDOMIDE_R))
    assert surface.n_vertices == 5738


def test_surface_construction_sets_shapes():
    """It stores vertices and ESP with consistent shapes."""
    surface = sphere_surface(n=50)
    assert surface.vertices.shape == (50, 3)
    assert surface.esp.shape == (50,)
    assert surface.n_vertices == 50


def test_augmented_points_appends_scaled_esp():
    """It appends a fourth column equal to esp * elec_weight."""
    surface = sphere_surface(n=30)
    weight = 0.3
    points = surface.augmented_points(elec_weight=weight)
    assert points.shape == (30, 4)
    assert np.allclose(points[:, :3], surface.vertices)
    assert np.allclose(points[:, 3], surface.esp * weight)


def test_augmented_points_zero_weight_zeroes_field():
    """A zero weight removes the ESP contribution."""
    surface = sphere_surface(n=20)
    points = surface.augmented_points(elec_weight=0.0)
    assert np.allclose(points[:, 3], 0.0)


def test_surface_fails_for_wrong_vertex_shape():
    """It raises SurfaceError for non (N, 3) vertices."""
    with pytest.raises(SurfaceError):
        MolSurface(vertices=np.zeros((10, 2)), faces=None, esp=np.zeros(10))


def test_surface_fails_for_mismatched_esp_length():
    """It raises SurfaceError when ESP length differs from vertex count."""
    with pytest.raises(SurfaceError):
        MolSurface(vertices=np.zeros((10, 3)), faces=None, esp=np.zeros(9))
