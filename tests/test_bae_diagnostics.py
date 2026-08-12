"""Tests for BAE training diagnostics (TrainingReport)."""

from __future__ import annotations

import numpy as np
import pytest

from structboost import BAEConfig, TrainingReport

REPORT_FIELDS = tuple(TrainingReport.__dataclass_fields__)


def _require_deps():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")


def _fit(diagnostics, *, n_iter=15, seed=0, **cfg_kwargs):
    from structboost import BAE, sim_scrnaseq_anndata

    adata = sim_scrnaseq_anndata(
        n=200, n_genes=60, stageno=4, stagep=8, stageoverlap=2, hierarchy=None, seed=0
    )
    config = BAEConfig(
        latent_dim=4,
        boosting_stepno=20,
        max_iterations=n_iter,
        enable_early_stopping=False,
        seed=seed,
        **cfg_kwargs,
    )
    model = BAE(adata.n_vars, config)
    model.fit(adata, verbose=False, diagnostics=diagnostics)
    return model, adata


class TestDiagnosticsAreReadOnly:
    """Enabling diagnostics must never change the fitted model."""

    def test_fitted_model_is_identical(self):
        _require_deps()
        _, a_off = _fit(False)
        _, a_on = _fit(True)
        np.testing.assert_array_equal(
            a_off.varm["BAE_encoder_weights"], a_on.varm["BAE_encoder_weights"]
        )
        np.testing.assert_array_equal(a_off.obsm["X_bae"], a_on.obsm["X_bae"])

    def test_train_loss_is_identical(self):
        _require_deps()
        m_off, _ = _fit(False)
        m_on, _ = _fit(True)
        assert m_off.training_history["train_loss"] == m_on.training_history["train_loss"]


class TestProgressBar:
    """Two of the diagnostic series are surfaced live in the tqdm postfix."""

    def _run(self, diagnostics, capfd):
        from structboost import BAE, sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=200, n_genes=60, stageno=4, stagep=8, seed=0)
        config = BAEConfig(
            latent_dim=4,
            boosting_stepno=20,
            max_iterations=5,
            enable_early_stopping=False,
            seed=0,
        )
        BAE(adata.n_vars, config).fit(adata, verbose=True, diagnostics=diagnostics)
        return capfd.readouterr().err

    def test_shows_weight_change_and_sparsity(self, capfd):
        _require_deps()
        out = self._run(True, capfd)
        assert "dW=" in out
        assert "n_sel=" in out
        assert "train_loss=" in out

    def test_first_iteration_reports_no_weight_change(self, capfd):
        """dW compares against the previous iteration, which does not exist yet."""
        _require_deps()
        assert "dW=n/a" in self._run(True, capfd)

    def test_default_bar_is_unchanged(self, capfd):
        _require_deps()
        out = self._run(False, capfd)
        assert "train_loss=" in out
        assert "dW=" not in out
        assert "n_sel=" not in out


class TestReportStructure:
    def test_absent_by_default(self):
        _require_deps()
        model, _ = _fit(False)
        assert model.training_report is None

    def test_config_flag_enables_collection(self):
        _require_deps()
        from structboost import BAE, sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=200, n_genes=60, stageno=4, stagep=8, seed=0)
        config = BAEConfig(
            latent_dim=4,
            boosting_stepno=20,
            max_iterations=5,
            enable_early_stopping=False,
            seed=0,
            diagnostics=True,
        )
        model = BAE(adata.n_vars, config).fit(adata, verbose=False)
        assert model.training_report is not None

    def test_fit_argument_overrides_config(self):
        _require_deps()
        from structboost import BAE, sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=200, n_genes=60, stageno=4, stagep=8, seed=0)
        config = BAEConfig(
            latent_dim=4,
            boosting_stepno=20,
            max_iterations=5,
            enable_early_stopping=False,
            seed=0,
            diagnostics=True,
        )
        model = BAE(adata.n_vars, config).fit(adata, verbose=False, diagnostics=False)
        assert model.training_report is None

    def test_all_series_have_one_entry_per_iteration(self):
        _require_deps()
        model, _ = _fit(True, n_iter=12)
        report = model.training_report
        assert report.n_iterations == 12
        for field in REPORT_FIELDS:
            assert getattr(report, field).shape[0] == 12, field

    def test_per_dimension_fields_are_2d(self):
        _require_deps()
        model, _ = _fit(True, n_iter=12)
        report = model.training_report
        assert report.n_selected_per_dim.shape == (12, 4)
        assert report.latent_var_per_dim.shape == (12, 4)

    def test_iteration_is_a_range(self):
        _require_deps()
        model, _ = _fit(True, n_iter=12)
        np.testing.assert_array_equal(model.training_report.iteration, np.arange(12))


class TestLossDecomposition:
    """The three-point decomposition of the alternating optimizer."""

    def test_deltas_match_their_definitions(self):
        _require_deps()
        r = _fit(True, n_iter=20)[0].training_report
        np.testing.assert_allclose(r.encoder_delta, r.loss_post_boost - r.loss_pre_boost)
        np.testing.assert_allclose(r.decoder_delta, r.loss_post_decoder - r.loss_post_boost)

    def test_iteration_start_equals_previous_end(self):
        """Nothing changes between the decoder update and the next target step."""
        _require_deps()
        r = _fit(True, n_iter=20)[0].training_report
        np.testing.assert_allclose(r.loss_pre_boost[1:], r.loss_post_decoder[:-1], rtol=1e-6)

    def test_losses_are_finite_and_positive(self):
        _require_deps()
        r = _fit(True, n_iter=20)[0].training_report
        for field in ("loss_pre_boost", "loss_post_boost", "loss_post_decoder"):
            values = getattr(r, field)
            assert np.isfinite(values).all(), field
            assert (values > 0).all(), field

    def test_encoder_step_descends_at_a_small_target_step(self):
        """The boosting target is a descent direction, for a small enough step.

        `-dL_target/dz` points downhill, so re-fitting the encoder onto
        `z - lr * dL_target/dz` reduces the loss while `lr` stays inside the
        radius where the linearisation holds.

        Changed in 0.4.0.0: the target gradient now sums over cells instead of
        averaging, so it is `n_cells` times larger and the default
        `target_optim_lr=1.0` sits well outside that radius — the step overshoots
        and the encoder half of the alternation raises the loss about 60% of the
        time on this dataset (the decoder half then recovers it). This test
        therefore pins the mathematical property at a small step rather than at
        the default; `test_encoder_step_at_default_lr_is_recorded` covers the
        default. `lr=0.005 == 1/200` reproduces the pre-0.4.0.0 effective step for
        this 200-cell fixture.
        """
        _require_deps()
        r = _fit(True, n_iter=20, target_optim_lr=0.005)[0].training_report
        assert np.mean(r.encoder_delta <= 0) > 0.9

    def test_encoder_step_at_default_lr_is_recorded(self):
        """At the default step the encoder delta is finite and training still works.

        Overshoot is expected here (see the test above); what must hold is that
        the diagnostic stays well-defined and the alternation as a whole descends.
        """
        _require_deps()
        r = _fit(True, n_iter=20)[0].training_report
        assert np.isfinite(r.encoder_delta).all()
        assert r.loss_post_decoder[-1] < r.loss_post_decoder[0]


class TestConvergenceMetrics:
    def test_first_iteration_is_nan_for_pairwise_metrics(self):
        """These compare against the previous iteration, which does not exist."""
        _require_deps()
        r = _fit(True, n_iter=12)[0].training_report
        for field in ("weight_change_rel", "support_jaccard", "min_dim_cosine"):
            assert np.isnan(getattr(r, field)[0]), field
            assert np.isfinite(getattr(r, field)[1:]).all(), field

    def test_weight_change_is_not_degenerate(self):
        """Guards the aliasing trap: numpy() shares storage with the weight tensor,
        so an uncopied snapshot would report zero change forever."""
        _require_deps()
        r = _fit(True, n_iter=20)[0].training_report
        assert np.nanstd(r.weight_change_rel) > 0
        assert np.nanmax(r.weight_change_rel) > 1e-6

    def test_support_jaccard_and_cosine_are_in_range(self):
        _require_deps()
        r = _fit(True, n_iter=20)[0].training_report
        assert np.all((r.support_jaccard[1:] >= 0) & (r.support_jaccard[1:] <= 1))
        assert np.all((r.min_dim_cosine[1:] >= -1.001) & (r.min_dim_cosine[1:] <= 1.001))

    def test_selected_gene_counts_are_consistent(self):
        _require_deps()
        model, adata = _fit(True, n_iter=12)
        r = model.training_report
        assert np.all(r.n_selected > 0)
        assert np.all(r.n_selected <= adata.n_vars)
        assert np.all(r.n_selected_per_dim <= adata.n_vars)

    def test_boosting_r2_is_sane(self):
        _require_deps()
        r = _fit(True, n_iter=20)[0].training_report
        assert np.isfinite(r.boosting_r2).all()
        assert r.boosting_r2.max() <= 1.0 + 1e-9

    def test_gradient_norms_are_finite_and_nonnegative(self):
        _require_deps()
        r = _fit(True, n_iter=12)[0].training_report
        for field in ("target_grad_norm", "decoder_grad_norm", "encoder_weight_norm"):
            values = getattr(r, field)
            assert np.isfinite(values).all(), field
            assert (values >= 0).all(), field


class TestStorageAndRoundTrip:
    def test_written_to_uns(self):
        _require_deps()
        _, adata = _fit(True, n_iter=10)
        stored = adata.uns["bae"]["training_report"]
        assert set(stored) == set(REPORT_FIELDS)

    def test_train_loss_history_remains_backward_compatible(self):
        _require_deps()
        _, adata = _fit(True, n_iter=10)
        assert len(adata.uns["bae"]["training_history"]["train_loss"]) == 10

    def test_uns_absent_when_diagnostics_off(self):
        _require_deps()
        _, adata = _fit(False, n_iter=5)
        assert "training_report" not in adata.uns["bae"]

    def test_h5ad_round_trip(self, tmp_path):
        _require_deps()
        import anndata as ad

        _, adata = _fit(True, n_iter=10)
        path = tmp_path / "bae.h5ad"
        adata.write_h5ad(path)
        loaded = ad.read_h5ad(path)
        report = TrainingReport.from_dict(loaded.uns["bae"]["training_report"])
        np.testing.assert_allclose(
            report.loss_post_decoder, adata.uns["bae"]["training_report"]["loss_post_decoder"]
        )
        assert report.n_iterations == 10

    def test_to_dict_from_dict_round_trip(self):
        _require_deps()
        r = _fit(True, n_iter=10)[0].training_report
        rebuilt = TrainingReport.from_dict(r.to_dict())
        for field in REPORT_FIELDS:
            np.testing.assert_array_equal(getattr(rebuilt, field), getattr(r, field))

    def test_from_dict_reports_missing_fields(self):
        _require_deps()
        r = _fit(True, n_iter=5)[0].training_report
        data = r.to_dict()
        del data["boosting_r2"]
        with pytest.raises(KeyError, match="boosting_r2"):
            TrainingReport.from_dict(data)


class TestPlotting:
    def test_plot_accepts_report_model_and_adata(self):
        _require_deps()
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from structboost import plot_training_diagnostics

        model, adata = _fit(True, n_iter=10)
        for source in (model.training_report, model, adata):
            fig = plot_training_diagnostics(source)
            assert len(fig.axes) == 6
            matplotlib.pyplot.close(fig)

    def test_plot_raises_without_diagnostics(self):
        _require_deps()
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from structboost import plot_training_diagnostics

        model, _ = _fit(False, n_iter=5)
        with pytest.raises(ValueError, match="diagnostics=True"):
            plot_training_diagnostics(model)
