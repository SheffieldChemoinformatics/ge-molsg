"""Test cases for the patch descriptor extension hook."""
import numpy as np
import pytest

from ge_molsg import PatchConfig, PatchDescriptor
from tests.store import sphere_surface


def test_patch_descriptor_inactive_by_default():
    """Without a backend the descriptor reports itself inactive."""
    patch = PatchDescriptor(PatchConfig(features=["esp"]))
    assert patch.is_active is False


def test_patch_per_vertex_raises_without_backend():
    """per_vertex raises NotImplementedError until a backend is supplied."""
    patch = PatchDescriptor(PatchConfig(features=["esp"]))
    surface = sphere_surface(n=40)
    with pytest.raises(NotImplementedError):
        patch.per_vertex(surface.vertices, surface.esp.reshape(-1, 1))


def test_patch_activates_with_backend():
    """Supplying compute_patches activates the channel."""

    def fake_patches(vertices, features, radius, normalise):
        return np.column_stack([vertices.mean(axis=1), features.ravel()])

    patch = PatchDescriptor(
        PatchConfig(features=["esp"]), compute_patches=fake_patches
    )
    surface = sphere_surface(n=40)
    assert patch.is_active is True
    desc = patch.per_vertex(surface.vertices, surface.esp.reshape(-1, 1))
    assert desc.shape == (40, 2)


def test_stack_features_orders_columns():
    """stack_features assembles columns in the configured order."""
    patch = PatchDescriptor(PatchConfig(features=["esp", "logp"]))
    pool = {"esp": np.arange(5), "logp": np.arange(5) * 2}
    feat = patch.stack_features(pool)
    assert feat.shape == (5, 2)
    assert np.allclose(feat[:, 0], np.arange(5))
    assert np.allclose(feat[:, 1], np.arange(5) * 2)
