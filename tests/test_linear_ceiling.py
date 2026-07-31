"""Tests for `linear_ceiling`, the interpretability anchor for reconstruction MSE."""

from __future__ import annotations

import numpy as np
import pytest


def _adata(n=120, p=40, rank=5, seed=0):
    ad = pytest.importorskip("anndata")
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((n, rank))
    loadings = rng.standard_normal((rank, p))
    x = scores @ loadings + 0.1 * rng.standard_normal((n, p))
    return ad.AnnData(x.astype(np.float64))


def test_matches_explicit_pca_variance_ratio():
    """Must agree with the textbook definition computed from a full SVD."""
    pytest.importorskip("scipy")
    from structboost import linear_ceiling

    adata = _adata()
    x = np.asarray(adata.X)
    centered = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    for k in (1, 3, 7):
        expected = (singular[:k] ** 2).sum() / (singular**2).sum()
        assert linear_ceiling(adata, k) == pytest.approx(expected, rel=1e-6)


def test_is_monotone_and_bounded():
    pytest.importorskip("scipy")
    from structboost import linear_ceiling

    adata = _adata()
    values = [linear_ceiling(adata, k) for k in range(1, 10)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:], strict=False))


def test_recovers_known_rank():
    """A rank-5 signal plus small noise is nearly fully explained by 5 components."""
    pytest.importorskip("scipy")
    from structboost import linear_ceiling

    adata = _adata(rank=5)
    assert linear_ceiling(adata, 5) > 0.99
    assert linear_ceiling(adata, 2) < 0.9


def test_sparse_and_dense_agree_without_densifying():
    """Sparse input must give the same answer; centering must not densify."""
    pytest.importorskip("scipy")
    import scipy.sparse as sp

    from structboost import linear_ceiling

    ad = pytest.importorskip("anndata")
    dense = _adata(n=80, p=30, rank=4)
    x = np.asarray(dense.X)
    x[np.abs(x) < 0.5] = 0.0  # make it genuinely sparse
    dense = ad.AnnData(x)
    sparse = ad.AnnData(sp.csr_matrix(x))
    for k in (2, 5):
        assert linear_ceiling(sparse, k) == pytest.approx(linear_ceiling(dense, k), rel=1e-6)


def test_rejects_out_of_range_n_components():
    pytest.importorskip("scipy")
    from structboost import linear_ceiling

    adata = _adata(n=20, p=10)
    with pytest.raises(ValueError, match="n_components must be in"):
        linear_ceiling(adata, 11)
    with pytest.raises(ValueError, match="n_components must be in"):
        linear_ceiling(adata, 0)
