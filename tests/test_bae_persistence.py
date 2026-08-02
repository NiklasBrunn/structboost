"""Tests for BAE checkpoint persistence (``BAE.save`` / ``BAE.load``).

Skipped automatically if BAE dependencies (torch, anndata) are missing.
"""

from __future__ import annotations

import numpy as np
import pytest


def _require_deps():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")


def _adata(n=60, n_genes=24, seed=0):
    """Small standardized AnnData with named genes and two obs covariates."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_genes)).astype(np.float32)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    adata = ad.AnnData(x)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs["batch"] = pd.Categorical(rng.choice(["a", "b"], size=n))
    adata.obs["donor"] = pd.Categorical(rng.choice(["d1", "d2", "d3"], size=n))
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


def _fit(adata=None, config=None, **fit_kwargs):
    from structboost import BAE

    adata = _adata() if adata is None else adata
    model = BAE(adata.n_vars, config or _config())
    model.fit(adata, verbose=False, **fit_kwargs)
    return model, adata


def test_round_trip_preserves_transform_and_reconstruct(tmp_path):
    _require_deps()
    from structboost import BAE

    model, adata = _fit()
    latent = model.transform(adata)
    reconstruction = model.reconstruct(adata)

    path = model.save(tmp_path / "model.pt")
    loaded = BAE.load(path)

    np.testing.assert_array_equal(loaded.transform(adata), latent)
    np.testing.assert_array_equal(loaded.reconstruct(adata), reconstruction)


def test_round_trip_preserves_config_including_tuple_and_device(tmp_path):
    _require_deps()
    from structboost import BAE

    model, _ = _fit(config=_config(decoder_hidden_dims=(8, 12)))
    loaded = BAE.load(model.save(tmp_path / "model.pt"))

    assert loaded.config == model.config
    # A list would compare unequal to the original tuple and would propagate
    # into any later `dataclasses.replace` round trip.
    assert isinstance(loaded.config.decoder_hidden_dims, tuple)
    assert str(loaded.config.device) == "cpu"
    assert loaded.n_genes == model.n_genes


def test_round_trip_preserves_diagnostics(tmp_path):
    _require_deps()
    from structboost import BAE

    model, _ = _fit(config=_config(diagnostics=True, max_iterations=4))
    loaded = BAE.load(model.save(tmp_path / "model.pt"))

    assert loaded.training_history == model.training_history
    assert loaded._latent_init == model._latent_init
    original = model.training_report.to_dict()
    restored = loaded.training_report.to_dict()
    assert set(restored) == set(original)
    for field, values in original.items():
        np.testing.assert_allclose(restored[field], values, err_msg=field)
        assert restored[field].dtype == values.dtype, field


def test_round_trip_preserves_conditioning(tmp_path):
    _require_deps()
    from structboost import BAE

    adata = _adata()
    model, _ = _fit(adata=adata, batch_key=["batch", "donor"])
    reconstruction = model.reconstruct(adata)

    loaded = BAE.load(model.save(tmp_path / "model.pt"))

    np.testing.assert_array_equal(loaded.reconstruct(adata), reconstruction)
    assert loaded._batch_encoding.obs_columns == ["batch", "donor"]
    assert loaded._batch_encoding.encoded_columns == model._batch_encoding.encoded_columns
    assert loaded._batch_integration_mode == model._batch_integration_mode == "both"
    np.testing.assert_allclose(loaded._batch_weights, model._batch_weights)
    # The training design matrix is deliberately not persisted.
    assert loaded._batch_encoding.encoded.shape == (0, model._batch_encoding.n_columns)


@pytest.mark.parametrize("mode", ["encoder", "decoder", "both"])
def test_round_trip_preserves_every_mode(tmp_path, mode):
    """Each mode must restore to a model that reconstructs identically."""
    _require_deps()
    from structboost import BAE

    adata = _adata()
    model, _ = _fit(adata=adata, batch_key="batch", batch_integration_mode=mode)
    reconstruction = model.reconstruct(adata)

    loaded = BAE.load(model.save(tmp_path / f"{mode}.pt"))

    assert loaded._batch_integration_mode == mode
    np.testing.assert_array_equal(loaded.reconstruct(adata), reconstruction)


def test_non_string_categorical_levels_survive(tmp_path):
    _require_deps()
    import pandas as pd

    from structboost import BAE

    adata = _adata()
    rng = np.random.default_rng(1)
    adata.obs["plate"] = pd.Categorical(rng.choice([1, 2, 3], size=adata.n_obs))
    adata.obs["treated"] = rng.choice([True, False], size=adata.n_obs)

    model, _ = _fit(adata=adata, batch_key=["plate", "treated"], batch_integration_mode="decoder")
    reconstruction = model.reconstruct(adata)

    loaded = BAE.load(model.save(tmp_path / "model.pt"))

    # If the levels came back as strings, `transform_obs_covariates` would either
    # raise "levels not seen during fitting" or dummy-encode every cell to zero.
    np.testing.assert_array_equal(loaded.reconstruct(adata), reconstruction)
    assert loaded._batch_encoding.column_info["plate"]["categories"] == [1, 2, 3]
    assert loaded._batch_encoding.column_info["treated"]["categories"] == [
        bool(v) for v in model._batch_encoding.column_info["treated"]["categories"]
    ]


def test_unpersistable_categorical_level_type_is_rejected_at_save(tmp_path):
    _require_deps()
    import pandas as pd

    adata = _adata()
    adata.obs["collected"] = pd.Categorical(
        pd.to_datetime(["2026-01-01", "2026-02-01"] * (adata.n_obs // 2))
    )

    model, _ = _fit(adata=adata, batch_key=["collected"], batch_integration_mode="decoder")

    with pytest.raises(ValueError, match="collected"):
        model.save(tmp_path / "model.pt")


def test_transfer_state_round_trips_and_loaded_model_can_be_a_reference(tmp_path):
    _require_deps()
    from structboost import BAE

    reference, _ = _fit()
    target = _adata(seed=3)
    transferred = BAE.from_reference(
        reference, target, n_additional_dims=2, config=_config(latent_dim=5)
    )
    transferred.fit(target, verbose=False)
    latent = transferred.transform(target)

    loaded = BAE.load(transferred.save(tmp_path / "transfer.pt"))

    np.testing.assert_array_equal(loaded.prior_weights, transferred.prior_weights)
    np.testing.assert_array_equal(loaded.fitted_prior_weights, transferred.fitted_prior_weights)
    np.testing.assert_array_equal(loaded.novel_weights, transferred.novel_weights)
    assert loaded.n_prior_dims == transferred.n_prior_dims
    assert loaded._prior_info["n_additional_dims"] == 2
    np.testing.assert_allclose(loaded._latent_scaling["mean"], transferred._latent_scaling["mean"])
    np.testing.assert_allclose(
        loaded._latent_scaling["scale"], transferred._latent_scaling["scale"]
    )
    np.testing.assert_array_equal(loaded.transform(target), latent)

    # `_var_names` is what makes a model usable as a prior; it must survive.
    downstream = BAE.from_reference(
        loaded, _adata(seed=4), n_additional_dims=1, config=_config(latent_dim=6)
    )
    assert downstream.n_prior_dims == loaded.config.latent_dim


def test_split_softmax_model_round_trips(tmp_path):
    _require_deps()
    from structboost import BAE

    adata = _adata()
    model, _ = _fit(adata=adata, config=_config(split_softmax=True))
    reconstruction = model.reconstruct(adata)

    loaded = BAE.load(model.save(tmp_path / "model.pt"))

    assert loaded.split_softmax_layer is not None
    np.testing.assert_array_equal(loaded.reconstruct(adata), reconstruction)


def test_batch_norm_decoder_buffers_round_trip(tmp_path):
    _require_deps()
    from structboost import BAE

    adata = _adata()
    model, _ = _fit(adata=adata, config=_config(decoder_use_batch_norm=True))
    reconstruction = model.reconstruct(adata)

    loaded = BAE.load(model.save(tmp_path / "model.pt"))

    np.testing.assert_array_equal(loaded.reconstruct(adata), reconstruction)


def test_per_dimension_mandatory_genes_round_trip(tmp_path):
    _require_deps()
    from structboost import BAE

    adata = _adata()
    mandatory = [["gene_0", "gene_1"], ["gene_2"], ["gene_3", "gene_4", "gene_5"]]
    model, _ = _fit(adata=adata, mandatory_genes=mandatory)

    loaded = BAE.load(model.save(tmp_path / "model.pt"))

    assert loaded._mandatory_genes == mandatory


def test_checkpoint_loads_without_arbitrary_code_execution(tmp_path):
    _require_deps()
    import torch

    model, _ = _fit(
        batch_key="batch", batch_integration_mode="decoder", config=_config(diagnostics=True)
    )
    path = model.save(tmp_path / "model.pt")

    # Pins the safety property: the payload must stay free of pickled objects
    # (NumPy arrays and scalars included), so loading cannot execute code.
    payload = torch.load(path, weights_only=True)
    assert payload["magic"] == "structboost.bae"


def test_saving_an_unfitted_model_is_refused(tmp_path):
    _require_deps()
    from structboost import BAE

    model = BAE(24, _config())
    with pytest.raises(RuntimeError, match="not fitted"):
        model.save(tmp_path / "model.pt")


def test_loading_a_missing_file_raises(tmp_path):
    _require_deps()
    from structboost import BAE

    with pytest.raises(FileNotFoundError, match="No BAE checkpoint"):
        BAE.load(tmp_path / "absent.pt")


def test_loading_a_foreign_torch_file_is_refused(tmp_path):
    _require_deps()
    import torch

    from structboost import BAE

    path = tmp_path / "foreign.pt"
    torch.save({"weights": torch.zeros(3)}, path)
    with pytest.raises(ValueError, match="not a structboost BAE checkpoint"):
        BAE.load(path)


def test_loading_a_newer_checkpoint_format_is_refused(tmp_path):
    _require_deps()
    import torch

    from structboost import BAE
    from structboost._persistence import CHECKPOINT_FORMAT

    model, _ = _fit()
    path = model.save(tmp_path / "model.pt")
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = CHECKPOINT_FORMAT + 1
    torch.save(payload, path)

    with pytest.raises(ValueError, match="Upgrade structboost"):
        BAE.load(path)


def test_explicit_device_argument_is_honoured(tmp_path):
    _require_deps()
    from structboost import BAE

    model, _ = _fit()
    loaded = BAE.load(model.save(tmp_path / "model.pt"), device="cpu")
    assert str(loaded.config.device) == "cpu"
    assert str(loaded.encoder.linear.weight.device) == "cpu"


def test_transform_warns_when_the_gene_panel_does_not_match(tmp_path):
    _require_deps()
    from structboost import BAE

    model, adata = _fit()
    loaded = BAE.load(model.save(tmp_path / "model.pt"))

    permuted = adata[:, ::-1].copy()
    with pytest.warns(UserWarning, match="gene panel"):
        loaded.transform(permuted)

    # The matching panel must stay silent.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        loaded.transform(adata)
