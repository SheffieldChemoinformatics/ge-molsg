"""Test cases for the codebook and Bag-of-Features modules."""
import numpy as np

from ge_molsg import (
    GEMolSGConfig,
    build_codebook,
    compute_wks_batch,
    farthest_point_sample,
    knn_histogram,
    sample_descriptor_pool,
    soft_bof,
)
from tests.store import surface_batch

CONFIG = GEMolSGConfig(n_components=12, n_neighbors=20, evals=16)


def _descriptors(count=4, n=100):
    return compute_wks_batch(surface_batch(count=count, n=n), CONFIG, n_jobs=1)


def test_sample_pool_keeps_all_vertices_by_default():
    """Without subsampling the pool stacks every vertex descriptor."""
    descs = _descriptors(count=3, n=80)
    pool = sample_descriptor_pool(descs)
    assert pool.shape[0] == sum(d.shape[0] for d in descs)


def test_sample_pool_subsamples_per_molecule():
    """With n_per_mol the pool is capped per molecule."""
    descs = _descriptors(count=3, n=120)
    pool = sample_descriptor_pool(descs, n_per_mol=40)
    assert pool.shape[0] == 3 * 40


def test_farthest_point_sample_returns_unique_indices():
    """FPS returns the requested number of distinct indices."""
    X = np.random.default_rng(0).normal(size=(100, 5))
    idx = farthest_point_sample(X, 20)
    assert len(idx) == 20
    assert len(np.unique(idx)) == 20


def test_build_codebook_shape():
    """The codebook has n_codewords rows in descriptor space."""
    pool = sample_descriptor_pool(_descriptors(count=3, n=100), n_per_mol=50)
    codebook = build_codebook(pool, n_codewords=24)
    assert codebook.shape == (24, pool.shape[1])


def test_knn_histogram_is_normalized():
    """The k-NN BoF histogram sums to one."""
    descs = _descriptors(count=3, n=100)
    pool = sample_descriptor_pool(descs, n_per_mol=50)
    codebook = build_codebook(pool, n_codewords=20)
    hist = knn_histogram(descs[0], codebook, knn=3)
    assert hist.shape == (20,)
    assert np.isclose(hist.sum(), 1.0)


def test_soft_bof_is_normalized():
    """The soft BoF histogram sums to one."""
    descs = _descriptors(count=3, n=100)
    pool = sample_descriptor_pool(descs, n_per_mol=50)
    codebook = build_codebook(pool, n_codewords=20)
    hist = soft_bof(descs[0], codebook, tau=0.1)
    assert hist.shape == (20,)
    assert np.isclose(hist.sum(), 1.0)
