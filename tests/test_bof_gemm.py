"""BoF equivalence + gemm-distance regression.

Asserts the BLAS-gemm distance path (``_bof_sqdist``) and vectorised voting
produce output identical to the reference cdist/argsort/loop implementations,
and that ``hard_knn_bof`` is exactly ``knn_histogram``.
"""
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize as sk_normalize

import ge_molsg as gm
from ge_molsg.bof import _bof_sqdist


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    return rng.randn(800, 60), rng.randn(150, 60)


def test_sqdist_matches_cdist(data):
    desc, cb = data
    ref = cdist(cb, desc, "sqeuclidean").T
    got = _bof_sqdist(desc, cb)
    assert np.max(np.abs(ref - got)) < 1e-9


def _ref_hard(desc, cb, knn=3, norm="l1"):
    sq = cdist(cb, desc, "sqeuclidean").T
    idx = np.argsort(sq, axis=1)[:, :knn]
    mask = np.zeros((desc.shape[0], cb.shape[0]))
    for row, ix in zip(mask, idx):
        row[ix] = 1.0
    return sk_normalize(mask.sum(0).reshape(1, -1), norm=norm)[0]


def test_hard_knn_matches_reference(data):
    desc, cb = data
    assert np.max(np.abs(_ref_hard(desc, cb) - gm.knn_histogram(desc, cb))) < 1e-9


def test_hard_knn_is_knn_histogram(data):
    desc, cb = data
    assert gm.hard_knn_bof is gm.knn_histogram
    assert np.array_equal(gm.hard_knn_bof(desc, cb), gm.knn_histogram(desc, cb))


def test_all_histograms_normalised(data):
    desc, cb = data
    for out in (gm.hq_bof(desc, cb), gm.knn_histogram(desc, cb),
                gm.knn_bof(desc, cb), gm.soft_bof(desc, cb, 1.0)):
        assert out.shape == (cb.shape[0],)
        assert abs(out.sum() - 1.0) < 1e-9
