"""Tests for reading expression from ``adata.layers`` instead of ``adata.X``.

Skipped automatically if BAE dependencies (torch, anndata) are missing.
"""

from __future__ import annotations

import numpy as np
import pytest


def _require_deps():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")


def _matrix(seed=0, n=60, n_genes=24):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_genes)).astype(np.float32)
    return (x - x.mean(axis=0)) / x.std(axis=0)


def _adata(seed=0, n=60, n_genes=24):
    """AnnData whose ``X`` and ``scaled`` layer hold *different* matrices.

    Different on purpose: if the two agreed, every assertion here would pass
    whether or not the layer argument was honoured.
    """
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed + 100)
    adata = ad.AnnData(_matrix(seed, n, n_genes))
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.layers["scaled"] = _matrix(seed + 1, n, n_genes)
    adata.obs["batch"] = pd.Categorical(rng.choice(["a", "b"], size=n))
    return adata


def _config(**overrides):
    from structboost import BAEConfig

    params = {
        "latent_dim": 3,
        "decoder_hidden_dims": (8,),
        "boosting_stepno": 5,
        "max_iterations": 3,
        "enable_early_stopping": False,
        "device": "cpu",
        "seed": 0,
    }
    params.update(overrides)
    return BAEConfig(**params)


def test_fitting_on_a_layer_matches_fitting_the_same_matrix_as_x():
    """`layer=` must change *which* matrix is read and nothing else."""
    _require_deps()
    import anndata as ad

    from structboost import BAE

    layered = _adata()
    on_layer = BAE(layered.n_vars, _config())
    on_layer.fit(layered, layer="scaled", verbose=False)

    # The same numbers, but sitting in X.
    direct_adata = ad.AnnData(layered.layers["scaled"].copy())
    direct_adata.var_names = list(layered.var_names)
    direct = BAE(direct_adata.n_vars, _config())
    direct.fit(direct_adata, verbose=False)

    np.testing.assert_array_equal(layered.obsm["X_bae"], direct_adata.obsm["X_bae"])
    np.testing.assert_array_equal(
        layered.varm["BAE_encoder_weights"], direct_adata.varm["BAE_encoder_weights"]
    )


def test_fitting_on_a_layer_differs_from_fitting_on_x():
    """Guards the guard: the fixture's two matrices really are different."""
    _require_deps()
    from structboost import BAE

    a = _adata()
    on_x = BAE(a.n_vars, _config())
    on_x.fit(a, verbose=False)
    from_x = a.obsm["X_bae"].copy()

    b = _adata()
    on_layer = BAE(b.n_vars, _config())
    on_layer.fit(b, layer="scaled", verbose=False)

    assert not np.array_equal(from_x, b.obsm["X_bae"])


def test_transform_and_reconstruct_default_to_the_fit_time_layer():
    _require_deps()
    from structboost import BAE

    a = _adata()
    model = BAE(a.n_vars, _config())
    model.fit(a, layer="scaled", verbose=False)

    # Default: the fit-time layer, not X.
    np.testing.assert_array_equal(model.transform(a), model.transform(a, layer="scaled"))
    np.testing.assert_array_equal(model.reconstruct(a), model.reconstruct(a, layer="scaled"))

    # Explicit None forces X, which is a different matrix here.
    assert not np.array_equal(model.transform(a), model.transform(a, layer=None))


def test_layer_survives_save_and_load(tmp_path):
    _require_deps()
    from structboost import BAE

    a = _adata()
    model = BAE(a.n_vars, _config())
    model.fit(a, layer="scaled", verbose=False)
    latent = model.transform(a)

    loaded = BAE.load(model.save(tmp_path / "model.pt"))

    assert loaded._layer == "scaled"
    np.testing.assert_array_equal(loaded.transform(a), latent)


def test_a_format_1_checkpoint_loads_as_x():
    """Checkpoints written before layer support were fitted on ``adata.X``."""
    _require_deps()
    import torch

    from structboost import BAE
    from structboost._persistence import CHECKPOINT_FORMAT, build_payload, restore_payload

    a = _adata()
    model = BAE(a.n_vars, _config())
    model.fit(a, layer="scaled", verbose=False)

    payload = build_payload(model)
    payload.pop("layer")  # as an older structboost would have written it
    payload["format_version"] = 1
    assert CHECKPOINT_FORMAT > 1

    restored = restore_payload(BAE, payload, torch.device("cpu"))
    assert restored._layer is None


def test_missing_layer_raises_naming_the_available_layers():
    _require_deps()
    from structboost import BAE

    a = _adata()
    model = BAE(a.n_vars, _config())
    with pytest.raises(KeyError, match="scaled"):
        model.fit(a, layer="lognorm", verbose=False)


def test_layer_is_recorded_in_uns():
    _require_deps()
    from structboost import BAE

    a = _adata()
    BAE(a.n_vars, _config()).fit(a, layer="scaled", verbose=False)
    assert a.uns["bae"]["layer"] == "scaled"

    b = _adata()
    BAE(b.n_vars, _config()).fit(b, verbose=False)
    # No key rather than a None that AnnData cannot write.
    assert "layer" not in b.uns["bae"]


def test_downstream_methods_follow_the_fit_time_layer():
    """Stability selection and reconstruction stats must read the fitted matrix."""
    _require_deps()
    from structboost import BAE

    a = _adata()
    model = BAE(a.n_vars, _config())
    model.fit(a, layer="scaled", batch_key="batch", batch_integration_mode="decoder", verbose=False)

    # Would raise or silently use X if `self._layer` were not threaded through.
    model.stability_selection(a, n_runs=2, seed=0, verbose=False)
    assert "BAE_iteration_frequency" in a.varm

    del a.X  # nothing below may fall back to it
    a.X = np.zeros_like(a.layers["scaled"])
    np.testing.assert_array_equal(model.transform(a), model.transform(a, layer="scaled"))
