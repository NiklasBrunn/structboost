"""Batch-integration regression tests.

These pin the property the feature exists to deliver: after conditioning on a
covariate, the latent space should carry *less* of that covariate's signal than
an uncorrected fit. Without a test at this level, a conditioning mechanism can
be wired up correctly, type-check, round-trip through AnnData, and still not
integrate anything. The removed ``"additive"`` conditioning mode was exactly that
failure: a structurally sound covariate term the optimizer simply declined to use,
which left *more* batch in the latent than no correction at all; the 0.9.0.0
CHANGELOG entry records the measurements.

The fixture is deliberately small so the whole module runs in a few seconds.
"""

from __future__ import annotations

import numpy as np
import pytest


def _require_deps():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")
    pytest.importorskip("pandas")


def _sim():
    """Small simulation with a strong, known batch effect."""
    from structboost import sim_scrnaseq_anndata

    return sim_scrnaseq_anndata(
        n=400,
        n_genes=150,
        stageno=3,
        stagep=15,
        stageoverlap=1,
        hierarchy=None,
        n_batches=2,
        batch_effect_sd=1.0,
        seed=1,
    )


def _batch_r2(latent: np.ndarray, batch) -> float:
    """Mean fraction of latent variance explained by the batch label.

    0 means the latent carries no batch signal; 1 means it is pure batch.
    """
    import pandas as pd

    design = pd.get_dummies(pd.Series(np.asarray(batch)), drop_first=True).to_numpy(float)
    design = (design - design.mean(0)) / design.std(0)
    centered = latent - latent.mean(0, keepdims=True)
    fitted = design @ np.linalg.lstsq(design, centered, rcond=None)[0]
    ss_total = (centered**2).sum(0)
    ss_residual = ((centered - fitted) ** 2).sum(0)
    explained = 1 - np.divide(ss_residual, ss_total, out=np.ones_like(ss_total), where=ss_total > 0)
    return float(np.mean(explained))


def _fit_and_score(**fit_kwargs) -> float:
    from structboost import BAE, BAEConfig

    adata = _sim()
    config = BAEConfig(
        latent_dim=3,
        max_iterations=30,
        enable_early_stopping=False,
        seed=0,
        boosting_nu=0.1,
        boosting_stepno=25,
    )
    BAE(adata.n_vars, config).fit(adata, verbose=False, **fit_kwargs)
    return _batch_r2(adata.obsm["X_bae"], adata.obs["batch"])


@pytest.fixture(scope="module")
def uncorrected_batch_r2() -> float:
    """Batch signal left in the latent when no correction is applied."""
    _require_deps()
    baseline = _fit_and_score()
    # Guard the guard: if the simulated batch effect stopped reaching the latent,
    # every comparison below would pass vacuously.
    assert baseline > 0.15, f"fixture no longer has a batch effect to remove ({baseline})"
    return baseline


@pytest.mark.parametrize(
    "fit_kwargs",
    [
        {"batch_key": "batch", "batch_integration_mode": "decoder"},
        {"batch_key": "batch", "batch_integration_mode": "both"},
        {"batch_key": "batch"},
    ],
    ids=["decoder", "both", "default"],
)
def test_supported_paths_remove_batch_from_latent(fit_kwargs, uncorrected_batch_r2):
    """Each correction path must leave less batch signal than no correction."""
    _require_deps()
    corrected = _fit_and_score(**fit_kwargs)
    assert corrected < uncorrected_batch_r2, (
        f"{fit_kwargs} left batch R^2 {corrected:.4f}, "
        f"no better than the uncorrected {uncorrected_batch_r2:.4f}"
    )


def test_batch_key_alone_integrates(uncorrected_batch_r2):
    """Passing `batch_key` alone must integrate, with no mode named.

    The default mode is "both". If a future change makes the default inert, or
    reintroduces a mode that defaults to off, this fails.
    """
    _require_deps()
    corrected = _fit_and_score(batch_key="batch")
    assert corrected < uncorrected_batch_r2


def _quick_fit(adata, **fit_kwargs):
    """A few iterations only: these tests check wiring, not integration quality."""
    from structboost import BAE, BAEConfig

    config = BAEConfig(latent_dim=3, max_iterations=3, enable_early_stopping=False, seed=0)
    model = BAE(adata.n_vars, config)
    model.fit(adata, verbose=False, **fit_kwargs)
    return model


def test_encoder_mode_reaches_only_the_boosting_fit():
    """`"encoder"` must protect gene selection without conditioning the decoder.

    The distinction is observable: a decoder that was never given the covariate
    cannot demand it back at reconstruction time, which is what keeps a fitted
    model usable on data carrying no covariate annotation.
    """
    _require_deps()
    adata = _sim()
    model = _quick_fit(adata, batch_key="batch", batch_integration_mode="encoder")

    assert adata.uns["bae"]["batch_integration_mode"] == "encoder"
    assert "batch_weights" in adata.uns["bae"]

    stripped = adata.copy()
    del stripped.obs["batch"]
    model.reconstruct(stripped)


def test_decoder_mode_reaches_only_the_decoder():
    """`"decoder"` conditions reconstruction and leaves the boosting design alone."""
    _require_deps()
    adata = _sim()
    _quick_fit(adata, batch_key="batch", batch_integration_mode="decoder")

    assert adata.uns["bae"]["batch_integration_mode"] == "decoder"
    # No nuisance regression, so no coefficients for it.
    assert "batch_weights" not in adata.uns["bae"]


def test_mode_without_batch_key_raises():
    """A mode with nothing to integrate over is a mistake, not a silent no-op."""
    _require_deps()
    from structboost import BAE, BAEConfig

    adata = _sim()
    model = BAE(adata.n_vars, BAEConfig(latent_dim=2, max_iterations=2, seed=0))
    with pytest.raises(ValueError, match="without a batch_key"):
        model.fit(adata, batch_integration_mode="decoder", verbose=False)


def test_str_and_list_batch_key_agree():
    """`batch_key="batch"` and `batch_key=["batch"]` must be the same fit."""
    _require_deps()
    a1, a2 = _sim(), _sim()
    _quick_fit(a1, batch_key="batch")
    _quick_fit(a2, batch_key=["batch"])
    np.testing.assert_array_equal(a1.obsm["X_bae"], a2.obsm["X_bae"])
