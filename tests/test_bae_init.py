"""Tests for BAE latent-state initialization (warm start)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from structboost import BAEConfig
from structboost._utils import _pca_scores


def _require_deps():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")


def _adata(n=300, n_genes=100, stageno=4):
    from structboost import sim_scrnaseq_anndata

    adata = sim_scrnaseq_anndata(
        n=n, n_genes=n_genes, stageno=stageno, stagep=8, stageoverlap=2, hierarchy=None, seed=0
    )
    rng = np.random.default_rng(0)
    adata.obsm["X_emb4"] = rng.normal(size=(adata.n_obs, 4))
    adata.obsm["X_emb7"] = rng.normal(size=(adata.n_obs, 7))
    adata.obs["batch"] = ["b0", "b1"] * (adata.n_obs // 2)
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    return adata


def _fit(adata=None, *, n_iter=8, cfg_kwargs=None, **fit_kwargs):
    from structboost import BAE

    adata = (_adata() if adata is None else adata).copy()
    defaults = {
        "latent_dim": 4,
        "boosting_stepno": 20,
        "max_iterations": n_iter,
        "enable_early_stopping": False,
        "seed": 0,
    }
    config = BAEConfig(**{**defaults, **(cfg_kwargs or {})})
    model = BAE(adata.n_vars, config)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input data")
        model.fit(adata, verbose=False, **fit_kwargs)
    return model, adata


class TestPcaScores:
    def test_shape_and_variance_ordering(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 30))
        scores = _pca_scores(X, 5)
        assert scores.shape == (200, 5)
        variances = scores.var(axis=0)
        assert np.all(np.diff(variances) <= 1e-9)

    def test_components_are_orthogonal(self):
        rng = np.random.default_rng(1)
        scores = _pca_scores(rng.normal(size=(200, 30)), 5)
        gram = scores.T @ scores
        off_diagonal = gram - np.diag(np.diag(gram))
        assert np.abs(off_diagonal).max() < 1e-6 * np.abs(np.diag(gram)).max()

    def test_centering_is_a_noop_on_standardized_input(self):
        """The whole point of skipping centering for z-scored data."""
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 30))
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        np.testing.assert_allclose(
            _pca_scores(X, 4, center=True), _pca_scores(X, 4, center=False), atol=1e-8
        )

    def test_centering_matters_on_uncentered_input(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(200, 30)) + 25.0
        centered = _pca_scores(X, 4, center=True)
        uncentered = _pca_scores(X, 4, center=False)
        assert not np.allclose(centered, uncentered, atol=1e-6)

    def test_never_implicitly_standardizes(self):
        """Scaling one feature must change the scores; if the helper divided by
        column stds it would be invariant to that."""
        rng = np.random.default_rng(4)
        X = rng.normal(size=(200, 30))
        X_scaled = X.copy()
        X_scaled[:, 0] *= 100.0
        assert not np.allclose(_pca_scores(X, 3), _pca_scores(X_scaled, 3), atol=1e-6)

    @pytest.mark.parametrize("n_components", [0, -1, 31])
    def test_invalid_n_components(self, n_components):
        rng = np.random.default_rng(5)
        with pytest.raises(ValueError, match="n_components must be in"):
            _pca_scores(rng.normal(size=(200, 30)), n_components)


class TestDefaultIsUnchanged:
    def test_zero_start_is_reproducible(self):
        _require_deps()
        _, a = _fit()
        _, b = _fit()
        np.testing.assert_array_equal(a.varm["BAE_encoder_weights"], b.varm["BAE_encoder_weights"])

    def test_default_records_zero_provenance(self):
        _require_deps()
        _, adata = _fit()
        assert adata.uns["bae"]["latent_init"] == {"method": "zero", "pretrain_epochs": 0}


class TestWarmStart:
    def test_pca_changes_the_trajectory(self):
        _require_deps()
        _, zero = _fit()
        _, warm = _fit(init_pca=True)
        assert not np.array_equal(
            zero.varm["BAE_encoder_weights"], warm.varm["BAE_encoder_weights"]
        )

    def test_warm_start_is_reproducible(self):
        _require_deps()
        _, a = _fit(init_pca=True)
        _, b = _fit(init_pca=True)
        np.testing.assert_array_equal(a.varm["BAE_encoder_weights"], b.varm["BAE_encoder_weights"])

    def test_obsm_start_differs_from_pca_start(self):
        _require_deps()
        _, from_obsm = _fit(init_obsm="X_emb4")
        _, from_pca = _fit(init_pca=True)
        assert not np.array_equal(
            from_obsm.varm["BAE_encoder_weights"], from_pca.varm["BAE_encoder_weights"]
        )

    def test_provenance_is_recorded(self):
        _require_deps()
        _, pca = _fit(init_pca=True)
        assert pca.uns["bae"]["latent_init"]["method"] == "pca"
        _, obsm = _fit(init_obsm="X_emb4")
        assert obsm.uns["bae"]["latent_init"]["method"] == "obsm:X_emb4"

    def test_warm_start_lowers_the_initial_loss(self):
        _require_deps()
        zero, _ = _fit(n_iter=8, cfg_kwargs={"diagnostics": True})
        warm, _ = _fit(n_iter=8, cfg_kwargs={"diagnostics": True}, init_pca=True)
        assert warm.training_report.loss_post_decoder[0] < zero.training_report.loss_post_decoder[0]

    def test_applied_once_only(self):
        """From iteration 1 onward z comes from the encoder, so a longer run must
        not simply keep re-imposing the initialization."""
        _require_deps()
        short, _ = _fit(n_iter=2, cfg_kwargs={"diagnostics": True}, init_pca=True)
        long, _ = _fit(n_iter=10, cfg_kwargs={"diagnostics": True}, init_pca=True)
        np.testing.assert_allclose(
            short.training_report.loss_pre_boost[:2],
            long.training_report.loss_pre_boost[:2],
            rtol=1e-6,
        )


class TestLatentDimOverride:
    def test_mismatched_obsm_overrides_and_warns(self):
        _require_deps()
        from structboost import BAE

        adata = _adata()
        model = BAE(
            adata.n_vars,
            BAEConfig(
                latent_dim=4,
                boosting_stepno=20,
                max_iterations=5,
                enable_early_stopping=False,
                seed=0,
            ),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(adata, verbose=False, init_obsm="X_emb7")
        messages = [str(w.message) for w in caught]
        assert any("latent_dim was changed from 4 to 7" in m for m in messages)
        assert model.config.latent_dim == 7
        assert adata.obsm["X_bae"].shape[1] == 7
        assert adata.varm["BAE_encoder_weights"].shape[1] == 7

    def test_matching_obsm_does_not_warn(self):
        _require_deps()
        model, adata = _fit(init_obsm="X_emb4")
        assert model.config.latent_dim == 4
        assert adata.obsm["X_bae"].shape[1] == 4

    @pytest.mark.parametrize(
        "cfg_kwargs, fit_kwargs",
        [
            ({}, {"batch_key": "batch"}),
            ({"split_softmax": True}, {}),
            ({"split_softmax": True}, {"batch_key": "batch"}),
        ],
    )
    def test_override_composes_with_decoder_input_changes(self, cfg_kwargs, fit_kwargs):
        """latent_dim, split_softmax and cVAE conditioning all drive the decoder
        input width; they must compose."""
        _require_deps()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, adata = _fit(cfg_kwargs=cfg_kwargs, init_obsm="X_emb7", **fit_kwargs)
        assert model.config.latent_dim == 7
        assert adata.obsm["X_bae"].shape[1] == 7


class TestDecoderPretraining:
    def test_pretraining_lowers_the_initial_loss(self):
        """Regression test for placement: `fit` re-initializes the decoder under the
        seed, so pre-training done before that point would be silently discarded and
        this assertion would fail."""
        _require_deps()
        plain, _ = _fit(cfg_kwargs={"diagnostics": True}, init_pca=True)
        pre, _ = _fit(cfg_kwargs={"diagnostics": True}, init_pca=True, init_pretrain_epochs=20)
        assert pre.training_report.loss_post_decoder[0] < plain.training_report.loss_post_decoder[0]

    def test_more_epochs_lower_the_initial_loss_further(self):
        _require_deps()
        losses = []
        for epochs in (0, 10, 40):
            model, _ = _fit(
                cfg_kwargs={"diagnostics": True},
                init_pca=True,
                init_pretrain_epochs=epochs,
            )
            losses.append(model.training_report.loss_post_decoder[0])
        assert losses[0] > losses[1] > losses[2]

    def test_recorded_in_provenance(self):
        _require_deps()
        _, adata = _fit(init_pca=True, init_pretrain_epochs=7)
        assert adata.uns["bae"]["latent_init"]["pretrain_epochs"] == 7

    def test_works_with_obs_covariates(self):
        _require_deps()
        _, adata = _fit(
            init_obsm="X_emb4",
            batch_key="batch",
            init_pretrain_epochs=5,
        )
        assert adata.obsm["X_bae"].shape[1] == 4


class TestValidation:
    def test_both_init_options_rejected(self):
        _require_deps()
        with pytest.raises(ValueError, match="mutually exclusive"):
            _fit(init_obsm="X_emb4", init_pca=True)

    def test_missing_obsm_key(self):
        _require_deps()
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            _fit(init_obsm="does_not_exist")

    def test_non_finite_representation(self):
        _require_deps()
        adata = _adata()
        bad = np.zeros((adata.n_obs, 4))
        bad[0, 0] = np.nan
        adata.obsm["X_bad"] = bad
        with pytest.raises(ValueError, match="non-finite"):
            _fit(adata, init_obsm="X_bad")

    def test_pca_with_too_many_components(self):
        _require_deps()
        adata = _adata(n=20, n_genes=40, stageno=2)
        with pytest.raises(ValueError, match="init_pca needs latent_dim"):
            _fit(adata, cfg_kwargs={"latent_dim": 40}, init_pca=True)

    def test_negative_pretrain_epochs(self):
        _require_deps()
        with pytest.raises(ValueError, match="init_pretrain_epochs must be >= 0"):
            _fit(init_pca=True, init_pretrain_epochs=-1)

    def test_pretrain_without_warm_start_rejected(self):
        """Pre-training is only meaningful for a non-zero initialization."""
        _require_deps()
        with pytest.raises(ValueError, match="requires a warm start"):
            _fit(init_pretrain_epochs=10)
