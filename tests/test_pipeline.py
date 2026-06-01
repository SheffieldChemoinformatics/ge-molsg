"""Test cases for the GEMolSG pipeline wrapper."""
import numpy as np
import pytest

from ge_molsg import GEMolSG, GEMolSGConfig
from tests.store import sphere_surface, surface_batch

CONFIG = GEMolSGConfig(n_components=12, n_neighbors=20, evals=16)


def _fitted_model(**kwargs):
    model = GEMolSG(CONFIG, n_codewords=20, sample_n_per_mol=40, **kwargs)
    model.fit_codebook(surface_batch(count=4, n=100))
    return model


def test_transform_returns_codebook_length_vector():
    """transform produces one BoF vector per surface."""
    model = _fitted_model()
    vector = model.transform(sphere_surface(n=90))
    assert vector.shape == (20,)
    assert np.isclose(vector.sum(), 1.0)


def test_transform_many_stacks_vectors():
    """transform_many returns a (n_surfaces, n_codewords) matrix."""
    model = _fitted_model()
    surfaces = surface_batch(count=5, n=90)
    matrix = model.transform_many(surfaces)
    assert matrix.shape == (5, 20)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_transform_before_fit_raises():
    """transform before fit_codebook raises a clear error."""
    model = GEMolSG(CONFIG, n_codewords=20)
    with pytest.raises(RuntimeError):
        model.transform(sphere_surface(n=50))


def test_fit_codebook_accepts_precomputed_descriptors():
    """A precomputed descriptor list can be passed to fit_codebook."""
    surfaces = surface_batch(count=4, n=90)
    model = GEMolSG(CONFIG, n_codewords=20, sample_n_per_mol=40)
    descs = model.descriptors(surfaces)
    model.fit_codebook(surfaces, descriptors=descs)
    assert model.codebook_.shape == (20, descs[0].shape[1])
