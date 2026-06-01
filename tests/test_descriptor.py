"""Test cases for descriptor orchestration."""
import numpy as np

from ge_molsg import GEMolSGConfig, compute_wks, compute_wks_batch
from tests.store import sphere_surface, surface_batch, thalidomide_r

CONFIG = GEMolSGConfig(n_components=15, n_neighbors=20, evals=20)


def test_compute_wks_on_real_surface():
    """The full pipeline produces a finite descriptor for thalidomide R."""
    surface = thalidomide_r()
    config = GEMolSGConfig(n_components=30, n_neighbors=50, evals=30)
    desc = compute_wks(surface, config)
    assert desc.shape == (surface.n_vertices, 30)
    assert np.isfinite(desc).all()


def test_compute_wks_shape():
    """A single surface yields a (n_vertices, evals) descriptor."""
    surface = sphere_surface(n=80)
    desc = compute_wks(surface, CONFIG)
    assert desc.shape == (80, 20)
    assert np.isfinite(desc).all()


def test_compute_wks_default_config():
    """compute_wks runs with the default configuration."""
    surface = sphere_surface(n=150)
    desc = compute_wks(surface)
    assert desc.shape[0] == 150


def test_batch_returns_one_descriptor_per_surface():
    """The batch helper returns results aligned to the input order."""
    surfaces = surface_batch(count=4, n=100)
    descs = compute_wks_batch(surfaces, CONFIG, n_jobs=1)
    assert len(descs) == 4
    assert all(d.shape == (100, 20) for d in descs)


def test_batch_serial_and_parallel_agree():
    """Serial and parallel execution produce identical descriptors."""
    surfaces = surface_batch(count=3, n=90)
    serial = compute_wks_batch(surfaces, CONFIG, n_jobs=1)
    parallel = compute_wks_batch(surfaces, CONFIG, n_jobs=2)
    assert all(np.allclose(a, b) for a, b in zip(serial, parallel))
