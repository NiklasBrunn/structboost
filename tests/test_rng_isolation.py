"""Tests that a seeded fit does not leak into the caller's RNG streams.

Skipped automatically if BAE dependencies (torch, anndata) are missing.
"""

from __future__ import annotations

import numpy as np
import pytest


def _require_deps():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")


def _adata(seed=0, n=60, n_genes=24):
    import anndata as ad

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_genes)).astype(np.float32)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return ad.AnnData(x)


def _config(**overrides):
    from structboost import BAEConfig

    params = {
        "latent_dim": 3,
        "decoder_hidden_dims": (8,),
        "boosting_stepno": 5,
        "max_iterations": 3,
        "enable_early_stopping": False,
        "device": "cpu",
        "seed": 7,
    }
    params.update(overrides)
    return BAEConfig(**params)


def _fitted_model(adata=None):
    from structboost import BAE

    adata = _adata() if adata is None else adata
    model = BAE(adata.n_vars, _config())
    model.fit(adata, verbose=False)
    return model, adata


def test_seeded_fit_restores_the_callers_torch_stream():
    _require_deps()
    import torch

    from structboost import BAE

    adata = _adata()
    # Constructed first: building any nn.Module draws from the global generator,
    # and that is ordinary torch behaviour, not the leak under test.
    model = BAE(adata.n_vars, _config())

    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    model.fit(adata, verbose=False)

    assert torch.equal(before, torch.get_rng_state())


def test_seeded_fit_does_not_change_the_callers_next_draw():
    _require_deps()
    import torch

    from structboost import BAE

    adata = _adata()
    model = BAE(adata.n_vars, _config())

    torch.manual_seed(1234)
    expected = torch.randn(4)

    torch.manual_seed(1234)
    model.fit(adata, verbose=False)
    assert torch.equal(expected, torch.randn(4))


def test_fit_never_touches_the_global_numpy_generator():
    """Nothing in the package reads it, so nothing may reseed it either."""
    _require_deps()
    from structboost import BAE

    adata = _adata()
    model = BAE(adata.n_vars, _config())

    np.random.seed(4321)
    before = np.random.get_state()[1].copy()
    model.fit(adata, verbose=False)

    assert np.array_equal(before, np.random.get_state()[1])


def test_explicit_seed_argument_is_isolated_too():
    _require_deps()
    import torch

    from structboost import BAE

    adata = _adata()
    model = BAE(adata.n_vars, _config(seed=None))

    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    model.fit(adata, seed=11, verbose=False)

    assert torch.equal(before, torch.get_rng_state())


def test_seeded_stability_selection_is_isolated():
    _require_deps()
    import torch

    model, adata = _fitted_model()

    torch.manual_seed(99)
    before = torch.get_rng_state().clone()
    model.stability_selection(adata, n_runs=2, seed=0, verbose=False)

    assert torch.equal(before, torch.get_rng_state())


def test_unseeded_fits_still_differ():
    """Isolation must not turn an unseeded fit into a deterministic one.

    Restoring state around an unseeded call would make successive fits on one
    model repeat themselves, silently removing the variability the caller asked
    for by not passing a seed.
    """
    _require_deps()
    from structboost import BAE

    adata = _adata()
    model = BAE(adata.n_vars, _config(seed=None))

    model.fit(adata, verbose=False)
    first = adata.obsm["X_bae"].copy()
    model.fit(adata, verbose=False)

    assert not np.array_equal(first, adata.obsm["X_bae"])


def test_seeded_fits_remain_reproducible_across_models():
    """The isolation must not cost reproducibility, which is its whole point."""
    _require_deps()
    from structboost import BAE

    a1 = _adata()
    BAE(a1.n_vars, _config()).fit(a1, verbose=False)
    a2 = _adata()
    BAE(a2.n_vars, _config()).fit(a2, verbose=False)

    np.testing.assert_array_equal(a1.obsm["X_bae"], a2.obsm["X_bae"])
