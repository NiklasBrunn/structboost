"""Tests for transferring a prior encoder weight matrix onto new data.

The properties pinned here are the ones that fail *silently* if broken: a frozen
prior that quietly drifts, an anchored prior that accumulates drift across
iterations, a checkpoint restore that discards every novel dimension, and a
stability readout that reports certainty it did not measure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from structboost import BAEConfig, allboost


def _require_deps():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")


def _adata(n=300, n_genes=200, stageno=4, seed=0):
    from structboost import sim_scrnaseq_anndata

    return sim_scrnaseq_anndata(n=n, n_genes=n_genes, stageno=stageno, seed=seed, standardize=True)


def _config(**overrides):
    defaults = {
        "latent_dim": 4,
        "boosting_stepno": 10,
        "max_iterations": 15,
        "enable_early_stopping": False,
        "seed": 0,
    }
    return BAEConfig(**{**defaults, **overrides})


def _transfer_config(n_additional, **overrides):
    """Config sized for a transfer, so the latent_dim override warning stays quiet."""
    return _config(latent_dim=4 + n_additional, **overrides)


def _reference(adata=None):
    """A fitted BAE usable as a prior, plus the data it was fitted on."""
    from structboost import BAE

    adata = _adata() if adata is None else adata
    model = BAE(adata.n_vars, _config())
    model.fit(adata.copy(), verbose=False)
    return model, adata


# --------------------------------------------------------------------------
# allboost offset (beta_init)
# --------------------------------------------------------------------------


def test_beta_init_zeros_is_identical_to_omitting_it():
    """The offset path must not perturb the existing zero-start behaviour."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 40))
    Y = X[:, :4] @ rng.normal(size=(4, 3)) + 0.3 * rng.normal(size=(120, 3))

    without = allboost(X, Y, stepno=15)
    with_zeros = allboost(X, Y, stepno=15, beta_init=np.zeros((3, 40)))
    assert np.array_equal(without, with_zeros)


def test_beta_init_fits_the_residual_at_the_offset():
    """Starting from the OLS solution leaves nothing to fit, so beta must not move."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 20))
    Y = rng.normal(size=(200, 2))
    ols = np.linalg.lstsq(X, Y, rcond=None)[0].T

    fitted = allboost(X, Y, stepno=25, beta_init=ols)
    assert np.abs(fitted - ols).max() < 1e-8


def test_beta_init_drift_depends_on_stepno_not_on_repetition():
    """Re-anchoring is idempotent: the anchor is fixed, so drift cannot accumulate."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(150, 30))
    Y = rng.normal(size=(150, 2))
    anchor = np.zeros((2, 30))
    anchor[:, :4] = rng.normal(size=(2, 4))

    first = allboost(X, Y, stepno=20, beta_init=anchor)
    again = allboost(X, Y, stepno=20, beta_init=anchor)
    assert np.array_equal(first, again)


def test_beta_init_shape_is_validated():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 10))
    Y = rng.normal(size=(50, 2))
    with pytest.raises(ValueError, match="beta_init must have shape"):
        allboost(X, Y, stepno=5, beta_init=np.zeros((3, 10)))


# --------------------------------------------------------------------------
# from_reference construction and gene alignment
# --------------------------------------------------------------------------


def test_from_reference_sets_latent_dim_from_the_prior():
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    model = BAE.from_reference(ref, adata, n_additional_dims=3, config=_transfer_config(3))
    assert model.config.latent_dim == 4 + 3
    assert model.prior_weights.shape == (adata.n_vars, 4)


def test_n_additional_dims_defaults_to_five():
    """A user choice with a pragmatic default, not a required argument."""
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    model = BAE.from_reference(ref, adata, config=_transfer_config(5))
    assert model.n_prior_dims == 4
    assert model.config.latent_dim == 4 + 5


def test_overridden_latent_dim_warns_rather_than_silently_resizing():
    """latent_dim is derived from the reference, so a passed value is not honoured."""
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    with pytest.warns(UserWarning, match="latent_dim was changed from 20 to 6"):
        model = BAE.from_reference(ref, adata, n_additional_dims=2, config=_config(latent_dim=20))
    assert model.config.latent_dim == 6


def test_matching_latent_dim_does_not_warn():
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        BAE.from_reference(ref, adata, n_additional_dims=2, config=_config(latent_dim=6))


def test_from_reference_reports_the_hyperparameters_it_did_not_inherit():
    """Only `latent_dim` comes from the prior, and the rest going back to defaults
    is otherwise invisible: it is what made a transfer fit run 1000 iterations
    behind a reference tuned to 15.
    """
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    with pytest.warns(UserWarning, match="used BAEConfig defaults") as caught:
        BAE.from_reference(ref, adata, n_additional_dims=2)

    message = str(caught[0].message)
    # The fields, not the exact sentence. `seed` matters most: an unseeded
    # transfer off a seeded reference is silently irreproducible.
    for field in ("boosting_stepno", "max_iterations", "seed"):
        assert field in message
    # `latent_dim` is genuinely inherited, so it must never appear as dropped.
    assert "latent_dim" not in message


def test_no_warning_when_the_reference_config_is_already_the_default():
    """Pins the field diff rather than the `isinstance` gate.

    Warning off the reference *type* alone would fire here too, even though
    nothing was actually dropped. The config is swapped after fitting so the
    fixture stays fast: `BAEConfig()` would fit at `max_iterations=1000`.
    """
    _require_deps()
    from dataclasses import replace

    from structboost import BAE

    ref, adata = _reference()
    ref.config = replace(BAEConfig(), latent_dim=ref.config.latent_dim)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        BAE.from_reference(ref, adata, n_additional_dims=2)


def test_a_file_reference_never_warns_about_inheritance(tmp_path):
    """The asymmetry that ruled out inheriting: a file carries no config.

    Warning here would report a difference against nothing.
    """
    _require_deps()
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from structboost import BAE, write_encoder_weights

    ref, adata = _reference()
    path = write_encoder_weights(
        ref.get_encoder_weights(),
        tmp_path / "prior.parquet",
        gene_ids=list(adata.var_names),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BAE.from_reference(path, adata.copy(), n_additional_dims=2)
    assert not [w for w in caught if "used BAEConfig defaults" in str(w.message)]


def test_bare_array_is_rejected_as_a_prior():
    _require_deps()
    from structboost import BAE

    _, adata = _reference()
    with pytest.raises(TypeError, match="bare array"):
        BAE.from_reference(np.zeros((adata.n_vars, 4)), adata, n_additional_dims=2)


def test_prior_aligns_by_gene_name_not_position():
    """A shuffled target panel must relocate weights, never reuse row order."""
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    order = np.random.default_rng(0).permutation(adata.n_vars)
    shuffled = adata[:, order].copy()

    model = BAE.from_reference(ref, shuffled, n_additional_dims=1, config=_transfer_config(1))
    expected = ref.get_encoder_weights()[order]
    assert np.allclose(model.prior_weights, expected)


def test_low_coverage_raises_and_names_the_threshold():
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    renamed = adata.copy()
    renamed.var_names = [f"unmatched_{i}" for i in range(renamed.n_vars)]
    with pytest.raises(ValueError, match="min_coverage"):
        BAE.from_reference(ref, renamed, n_additional_dims=2)


def test_partial_coverage_warns_and_is_recorded():
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    weights = ref.get_encoder_weights()
    # Drop the genes carrying the most mass in dimension 0 until it is attenuated.
    ranked = np.argsort(-np.abs(weights[:, 0]))
    dropped = set(ranked[:3].tolist())
    partial = adata.copy()
    partial.var_names = [
        f"gone_{i}" if i in dropped else name for i, name in enumerate(adata.var_names)
    ]
    with pytest.warns(UserWarning, match="weight mass"):
        model = BAE.from_reference(
            ref, partial, n_additional_dims=1, min_coverage=0.1, config=_transfer_config(1)
        )
    coverage = model._prior_info["prior_coverage"]
    assert coverage[0] < 1.0


# --------------------------------------------------------------------------
# The two prior modes
# --------------------------------------------------------------------------


def test_frozen_mode_leaves_the_prior_bitwise_unchanged():
    """The guarantee the mode exists to provide."""
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    model = BAE.from_reference(
        ref, work, n_additional_dims=2, prior_mode="frozen", config=_transfer_config(2)
    )
    prior = model.prior_weights.copy()
    model.fit(work, verbose=False, decoder_warmup_epochs=3)

    fitted = work.varm["BAE_encoder_weights"][:, :4]
    assert np.array_equal(fitted.astype(np.float64), prior)
    assert work.uns["bae_transfer"]["prior_weights_unchanged"] is True


def test_frozen_guarantee_holds_for_a_float64_prior(tmp_path):
    """A Parquet prior is float64; the encoder stores float32.

    Values like 2.6 are not float32-representable, so a float64 comparison would
    report drift for essentially every real prior read from disk even though
    nothing moved. Regression for that false negative.
    """
    _require_deps()
    pytest.importorskip("pyarrow")
    from structboost import BAE, write_encoder_weights

    _, adata = _reference()
    # A hand-authored prior, as a blueprint would declare it: blunt one-decimal
    # coefficients on a handful of genes per dimension.
    rng = np.random.default_rng(0)
    weights = np.zeros((adata.n_vars, 3), dtype=np.float64)
    for j in range(3):
        rows = rng.choice(adata.n_vars, size=6, replace=False)
        weights[rows, j] = [2.6, 1.4, -1.2, 1.7, -2.0, 1.1]
    assert not np.array_equal(weights, weights.astype(np.float32).astype(np.float64))

    path = write_encoder_weights(
        weights, tmp_path / "p.parquet", gene_symbols=list(adata.var_names)
    )
    work = adata.copy()
    model = BAE.from_reference(
        path, work, n_additional_dims=2, prior_mode="frozen", config=_config(latent_dim=5)
    )
    model.fit(work, verbose=False, decoder_warmup_epochs=3)

    assert work.uns["bae_transfer"]["prior_weights_unchanged"] is True
    assert np.array_equal(model.fitted_prior_weights.astype(np.float32), weights.astype(np.float32))


def test_anchored_mode_moves_the_prior_but_does_not_accumulate():
    """Drift is bounded by stepno and is re-anchored every iteration.

    This is what separates an anchored prior from an ordinary warm start: a
    starting point that is never re-applied is forgotten as the support
    random-walks, so a long run would end unrelated to the reference.
    """
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()

    def drift(max_iterations):
        work = adata.copy()
        model = BAE.from_reference(
            ref,
            work,
            n_additional_dims=2,
            prior_mode="anchored",
            config=_transfer_config(2, max_iterations=max_iterations),
        )
        prior = model.prior_weights.copy()
        model.fit(work, verbose=False, decoder_warmup_epochs=3)
        return np.linalg.norm(work.varm["BAE_encoder_weights"][:, :4] - prior)

    short, long = drift(10), drift(80)
    assert short > 0.0, "anchored mode must actually boost the prior columns"
    # Eight times the iterations must not mean more distance from the anchor.
    assert long <= short * 2.0


def test_zero_additional_dims_adapts_only_the_decoder():
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    model = BAE.from_reference(ref, work, n_additional_dims=0, config=_transfer_config(0))
    model.fit(work, verbose=False, decoder_warmup_epochs=3)

    assert model.config.latent_dim == 4
    assert work.uns["bae_transfer"]["prior_weights_unchanged"] is True


# --------------------------------------------------------------------------
# Issue A: the phase-1 checkpoint must not win
# --------------------------------------------------------------------------


def test_warmup_checkpoint_cannot_erase_the_novel_dimensions():
    """The regression that would otherwise be silent.

    Phase 1 converges the decoder against the prior programs alone. If it were
    tracked by checkpoint selection its loss would frequently beat the early
    phase-2 iterations, where the new dimensions switch on and briefly worsen the
    reconstruction — and `fit` would restore an encoder whose novel columns are
    all zero, returning the prior unchanged with no error raised.

    A single phase-2 iteration with a long warm-up is the sharp case: whatever
    that iteration's loss, the novel columns must survive.
    """
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    model = BAE.from_reference(
        ref,
        work,
        n_additional_dims=3,
        config=_transfer_config(3, max_iterations=1, enable_early_stopping=True),
    )
    model.fit(work, verbose=False, decoder_warmup_epochs=40)

    novel = work.varm["BAE_encoder_weights"][:, 4:]
    assert np.abs(novel).sum() > 0, "novel dimensions were erased by checkpoint restore"
    assert work.uns["bae_transfer"]["n_selected_novel"] > 0


def test_warmup_loss_is_recorded_separately_from_train_loss():
    """Keeping it out of train_loss is what keeps it out of checkpoint selection."""
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    model = BAE.from_reference(
        ref, work, n_additional_dims=2, config=_transfer_config(2, max_iterations=12)
    )
    model.fit(work, verbose=False, decoder_warmup_epochs=5)

    history = model.training_history
    assert len(history["warmup_loss"]) == 1
    assert len(history["train_loss"]) == 12


def test_decoder_warmup_without_a_prior_is_rejected():
    _require_deps()
    from structboost import BAE

    adata = _adata()
    model = BAE(adata.n_vars, _config())
    with pytest.raises(ValueError, match="only applies to a model built by"):
        model.fit(adata.copy(), verbose=False, decoder_warmup_epochs=5)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "fit_kwargs", "match"),
    [
        ({"config": _config(latent_dim=6, split_softmax=True)}, {}, "split_softmax"),
        ({"config": _transfer_config(2)}, {"init_pca": True}, "init_pca"),
        ({"config": _transfer_config(2)}, {"init_obsm": "X_emb"}, "init_pca"),
    ],
)
def test_conflicting_settings_raise(kwargs, fit_kwargs, match):
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    work.obsm["X_emb"] = np.zeros((work.n_obs, 6))
    model = BAE.from_reference(ref, work, n_additional_dims=2, **kwargs)
    with pytest.raises(ValueError, match=match):
        model.fit(work, verbose=False, **fit_kwargs)


def test_panel_mismatch_at_fit_time_raises():
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    model = BAE.from_reference(ref, adata, n_additional_dims=2, config=_transfer_config(2))
    smaller = adata[:, :50].copy()
    with pytest.raises(ValueError, match="aligned to a"):
        model.fit(smaller, verbose=False)


@pytest.mark.parametrize("n_additional_dims", [-1, -5])
def test_negative_additional_dims_rejected(n_additional_dims):
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    with pytest.raises(ValueError, match="n_additional_dims"):
        BAE.from_reference(ref, adata, n_additional_dims=n_additional_dims)


# --------------------------------------------------------------------------
# Batch integration and stability selection
# --------------------------------------------------------------------------


def test_batch_nuisance_applies_to_the_novel_dimensions():
    """Novel dimensions are new target columns, so they inherit nuisance regression."""
    _require_deps()
    import pandas as pd

    from structboost import BAE

    adata = _adata()
    adata.obs["batch"] = ["b0", "b1"] * (adata.n_obs // 2)
    ref = BAE(adata.n_vars, _config())
    ref.fit(adata.copy(), verbose=False, batch_key="batch")

    work = adata.copy()
    model = BAE.from_reference(ref, work, n_additional_dims=2, config=_transfer_config(2))
    model.fit(
        work,
        verbose=False,
        decoder_warmup_epochs=3,
        batch_key="batch",
    )

    z = work.obsm["X_bae"][:, 4:]
    design = pd.get_dummies(work.obs["batch"], drop_first=True).to_numpy(float)
    design = np.hstack([np.ones((design.shape[0], 1)), design])
    centered = z - z.mean(axis=0, keepdims=True)
    residual = centered - design @ np.linalg.lstsq(design, centered, rcond=None)[0]
    total = (centered**2).sum(axis=0)
    r2 = np.where(total > 0, 1.0 - (residual**2).sum(axis=0) / np.where(total > 0, total, 1), 0.0)
    assert r2.max() < 0.5


def test_frozen_prior_reports_no_stability_rather_than_certainty():
    """`|W| > 0` is tautological on the prior support; movement is the real signal.

    A frozen dimension cannot vary, so its honest selection frequency is zero.
    Reporting 1.0 would read as overwhelming evidence for genes that were never
    re-selected at all.
    """
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    # A wider support than the module default: iteration-mode stability matches
    # dimensions across runs by cosine similarity, and with only two free
    # dimensions and `boosting_stepno=10` there is too little support to match on
    # (quality ~0.44, below the 0.5 warning threshold). This is governed by
    # `boosting_stepno`, not by the iteration count.
    model = BAE.from_reference(
        ref,
        work,
        n_additional_dims=2,
        prior_mode="frozen",
        config=_transfer_config(2, boosting_stepno=30),
    )
    model.fit(work, verbose=False, decoder_warmup_epochs=3)

    result = model.stability_selection(work, n_runs=8, threshold=0.7)
    assert result.frequency[:, :4].max() == 0.0
    assert result.frequency[:, 4:].max() > 0.0


def test_anchored_prior_reports_movement_off_the_anchor():
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    model = BAE.from_reference(
        ref, work, n_additional_dims=2, prior_mode="anchored", config=_transfer_config(2)
    )
    model.fit(work, verbose=False, decoder_warmup_epochs=3)

    result = model.stability_selection(work, n_runs=8, threshold=0.7)
    assert result.frequency[:, :4].max() > 0.0


def test_stability_selection_leaves_a_transfer_model_unchanged():
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    model = BAE.from_reference(ref, work, n_additional_dims=2, config=_transfer_config(2))
    model.fit(work, verbose=False, decoder_warmup_epochs=3)

    before = model.get_encoder_weights().copy()
    model.stability_selection(work, n_runs=5, threshold=0.7)
    assert np.array_equal(model.get_encoder_weights(), before)


# --------------------------------------------------------------------------
# Diagnostics and round-trip
# --------------------------------------------------------------------------


def test_transfer_diagnostics_split_prior_and_novel_support():
    """A combined count would be dominated by the prior, not by this fit."""
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    model = BAE.from_reference(ref, work, n_additional_dims=2, config=_transfer_config(2))
    model.fit(work, verbose=False, decoder_warmup_epochs=3)

    info = work.uns["bae_transfer"]
    assert info["n_selected_prior"] > 0
    assert info["n_selected_novel"] > 0
    assert info["prior_mode"] == "frozen"
    assert info["n_prior_dims"] == 4
    # Zeroing the novel columns cannot improve the reconstruction.
    assert info["variance_explained"] >= info["variance_explained_prior_only"]


def test_block_accessors_split_prior_and_novel():
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    model = BAE.from_reference(ref, work, n_additional_dims=3, config=_transfer_config(3))
    model.fit(work, verbose=False, decoder_warmup_epochs=3)

    W = model.get_encoder_weights()
    assert model.n_prior_dims == 4
    assert model.fitted_prior_weights.shape == (adata.n_vars, 4)
    assert model.novel_weights.shape == (adata.n_vars, 3)
    assert np.array_equal(model.fitted_prior_weights, W[:, :4])
    assert np.array_equal(model.novel_weights, W[:, 4:])
    # Frozen: the fitted prior block is the anchor it started from.
    assert np.array_equal(model.fitted_prior_weights.astype(np.float64), model.prior_weights)


def test_block_accessors_are_none_without_a_prior():
    _require_deps()
    from structboost import BAE

    adata = _adata()
    model = BAE(adata.n_vars, _config())
    assert model.n_prior_dims == 0
    assert model.novel_weights is None
    assert model.fitted_prior_weights is None
    assert model.prior_weights is None


def test_per_dimension_variance_shares_are_reported():
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    model = BAE.from_reference(ref, work, n_additional_dims=3, config=_transfer_config(3))
    model.fit(work, verbose=False, decoder_warmup_epochs=3)

    info = work.uns["bae_transfer"]
    assert len(info["novel_variance_share_per_dim"]) == 3
    assert len(info["n_selected_novel_per_dim"]) == 3
    # Dropping a dimension cannot improve the reconstruction.
    assert min(info["novel_variance_share_per_dim"]) >= -1e-6


def test_ensembl_prior_joins_a_symbol_indexed_dataset(tmp_path):
    """The standard real-world layout: symbols in var_names, accessions in var.

    A reference written with Ensembl accessions as the join key must still align,
    since both identifier sets are present on both sides.
    """
    _require_deps()
    pytest.importorskip("pyarrow")
    from structboost import BAE, write_encoder_weights

    ref, adata = _reference()
    target = adata.copy()
    accessions = [f"ENSMUSG{i:011d}" for i in range(target.n_vars)]
    target.var["gene_ids"] = accessions

    path = write_encoder_weights(
        ref.get_encoder_weights(),
        tmp_path / "prior.parquet",
        gene_ids=accessions,
        gene_symbols=list(adata.var_names),
    )
    model = BAE.from_reference(path, target, n_additional_dims=2)

    assert model._prior_info["join_key"] == "gene_id->gene_ids"
    assert model._prior_info["n_matched_genes"] == target.n_vars
    assert np.allclose(model.prior_weights, ref.get_encoder_weights())


def test_join_on_forces_a_column_and_validates_it():
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    target = adata.copy()
    target.var["my_symbols"] = list(adata.var_names)

    model = BAE.from_reference(
        ref, target, n_additional_dims=1, join_on="my_symbols", config=_transfer_config(1)
    )
    assert model._prior_info["join_key"] == "var_names->my_symbols"

    with pytest.raises(ValueError, match="not a column of adata.var"):
        BAE.from_reference(ref, target, n_additional_dims=1, join_on="absent")


# --------------------------------------------------------------------------
# Scaled latent (X_bae_scaled)
# --------------------------------------------------------------------------


def _fitted_transfer(n_additional_dims=3):
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    model = BAE.from_reference(
        ref,
        work,
        n_additional_dims=n_additional_dims,
        config=_transfer_config(n_additional_dims),
    )
    model.fit(work, verbose=False, decoder_warmup_epochs=5)
    return model, work, adata


def test_scaled_latent_is_written_without_touching_the_raw_one():
    """`X_bae` must stay exactly `X @ W`; the scaled view is a separate key.

    That identity is what makes the encoder auditable and the frozen-prior
    guarantee checkable, so the fix for the block scale disparity is an extra
    key rather than a rescaled one.
    """
    _require_deps()
    model, work, _ = _fitted_transfer()

    raw = work.obsm["X_bae"]
    scaled = work.obsm["X_bae_scaled"]
    assert raw.shape == scaled.shape
    assert np.allclose(raw, np.asarray(work.X) @ work.varm["BAE_encoder_weights"], atol=1e-4)
    assert np.abs(scaled.mean(axis=0)).max() < 1e-5
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-5)


def test_ordinary_model_gets_no_scaled_latent():
    _require_deps()
    from structboost import BAE

    adata = _adata()
    work = adata.copy()
    BAE(adata.n_vars, _config()).fit(work, verbose=False)
    assert "X_bae_scaled" not in work.obsm


def test_transform_reuses_training_statistics():
    """New cells must go through the fitted map, not standardize themselves.

    Re-estimating on a query set makes the transformation differ per call and
    estimates the moments badly when the set is small.
    """
    _require_deps()
    model, work, adata = _fitted_transfer()
    mean = work.uns["bae_transfer"]["latent_mean"]
    scale = work.uns["bae_transfer"]["latent_scale"]

    query = adata[:60].copy()
    model.transform(query)
    assert np.allclose(query.obsm["X_bae_scaled"], (query.obsm["X_bae"] - mean) / scale)
    # A self-standardized query would be exactly centred; this one must not be.
    assert np.abs(query.obsm["X_bae_scaled"].mean(axis=0)).max() > 1e-6


def test_apply_encoder_refreshes_the_scaled_latent_and_its_statistics():
    """Otherwise the scaled key silently describes a superseded encoder."""
    _require_deps()
    model, work, _ = _fitted_transfer()
    before = work.obsm["X_bae_scaled"].copy()
    before_scale = np.asarray(work.uns["bae_transfer"]["latent_scale"]).copy()

    result = model.stability_selection(work, n_runs=6, threshold=0.7)
    model.apply_encoder(result.stable_encoder(), work)

    assert not np.array_equal(before, work.obsm["X_bae_scaled"])
    assert not np.allclose(before_scale, work.uns["bae_transfer"]["latent_scale"])
    assert np.abs(work.obsm["X_bae_scaled"].mean(axis=0)).max() < 1e-5


def test_scaled_latent_equalizes_the_block_scale_disparity():
    """The point of the key: prior and novel blocks contribute comparably.

    Untransformed, a hand-authored prior's coefficients dwarf boosted ones, so
    the novel block contributes almost nothing to Euclidean distance and any
    neighbour graph over the raw latent is effectively a prior-only graph.
    """
    _require_deps()
    model, work, _ = _fitted_transfer()
    k0 = model.n_prior_dims

    def novel_share(matrix):
        variances = matrix.var(axis=0)
        return variances[k0:].sum() / variances.sum()

    assert novel_share(work.obsm["X_bae_scaled"]) > novel_share(work.obsm["X_bae"])
    # After standardization every dimension contributes equally by construction.
    n_novel = work.obsm["X_bae"].shape[1] - k0
    assert np.isclose(
        novel_share(work.obsm["X_bae_scaled"]), n_novel / work.obsm["X_bae"].shape[1], atol=1e-6
    )


def test_a_transferred_model_is_itself_a_valid_prior():
    """Chaining must work: the result carries gene names and a weight matrix."""
    _require_deps()
    from structboost import BAE

    ref, adata = _reference()
    work = adata.copy()
    first = BAE.from_reference(ref, work, n_additional_dims=2, config=_transfer_config(2))
    first.fit(work, verbose=False, decoder_warmup_epochs=3)

    second_data = adata.copy()
    second = BAE.from_reference(
        work, second_data, n_additional_dims=1, config=_config(latent_dim=7)
    )
    assert second.config.latent_dim == 6 + 1
    second.fit(second_data, verbose=False, decoder_warmup_epochs=2)
    assert second_data.uns["bae_transfer"]["prior_weights_unchanged"] is True


@pytest.mark.parametrize("suffix", [".parquet", ".csv", ".tsv"])
def test_encoder_weight_files_round_trip(tmp_path, suffix):
    pytest.importorskip("pandas")
    if suffix == ".parquet":
        pytest.importorskip("pyarrow")
    from structboost import read_encoder_weights, write_encoder_weights

    rng = np.random.default_rng(0)
    weights = np.zeros((6, 3))
    weights[[0, 3, 5]] = rng.normal(size=(3, 3))
    ids = [f"ENSMUSG{i:011d}" for i in range(6)]
    # Symbols a spreadsheet would silently rewrite as dates.
    symbols = ["Snap25", "SEPT2", "MARCH1", "DEC1", "Pvalb", "Sst"]

    path = write_encoder_weights(
        weights,
        tmp_path / f"w{suffix}",
        gene_ids=ids,
        gene_symbols=symbols,
        metadata={"species": "mus_musculus", "annotation_release": "GRCm39.110"},
    )
    loaded = read_encoder_weights(path)

    assert np.allclose(loaded.to_numpy(), weights)
    assert loaded.attrs["join_key"] == "gene_id"
    assert list(loaded.index) == ids
    assert list(loaded.attrs["gene_symbol"]) == symbols
    assert loaded.attrs["metadata"]["annotation_release"] == "GRCm39.110"


def test_dimension_columns_order_numerically_not_lexicographically(tmp_path):
    """dim_10 must not sort before dim_2, which would silently permute the matrix."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from structboost import read_encoder_weights, write_encoder_weights

    weights = np.arange(4 * 12, dtype=float).reshape(4, 12)
    path = write_encoder_weights(
        weights, tmp_path / "w.parquet", gene_ids=[f"g{i}" for i in range(4)]
    )
    assert np.allclose(read_encoder_weights(path).to_numpy(), weights)


def test_symbol_only_file_joins_on_symbol(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from structboost import read_encoder_weights, write_encoder_weights

    weights = np.ones((3, 2))
    path = write_encoder_weights(
        weights, tmp_path / "s.parquet", gene_symbols=["Snap25", "Gad1", "Sst"]
    )
    loaded = read_encoder_weights(path)
    assert loaded.attrs["join_key"] == "gene_symbol"
    assert list(loaded.index) == ["Snap25", "Gad1", "Sst"]


def test_writing_without_identifiers_is_rejected(tmp_path):
    pytest.importorskip("pandas")
    from structboost import write_encoder_weights

    with pytest.raises(ValueError, match="gene_ids or gene_symbols"):
        write_encoder_weights(np.ones((3, 2)), tmp_path / "w.csv")


def test_prior_can_be_loaded_from_a_file(tmp_path):
    """The standalone path: write the reference matrix, transfer from the file."""
    _require_deps()
    pytest.importorskip("pyarrow")
    from structboost import BAE, write_encoder_weights

    ref, adata = _reference()
    path = write_encoder_weights(
        ref.get_encoder_weights(),
        tmp_path / "prior.parquet",
        gene_ids=list(adata.var_names),
    )

    work = adata.copy()
    model = BAE.from_reference(path, work, n_additional_dims=2, config=_transfer_config(2))
    assert np.allclose(model.prior_weights, ref.get_encoder_weights())
    model.fit(work, verbose=False, decoder_warmup_epochs=3)
    assert work.uns["bae_transfer"]["prior_weights_unchanged"] is True


def test_transfer_recovers_a_held_out_population():
    """The point of the extension: novel dimensions pick up structure the prior lacks.

    The reference sees four of five populations; the target adds the fifth. Genes
    selected by the novel dimensions should be enriched for that population's
    markers relative to the genome-wide marker rate.
    """
    _require_deps()
    from structboost import BAE, sim_scrnaseq_anndata

    full = sim_scrnaseq_anndata(
        n=500, n_genes=300, stageno=5, seed=3, standardize=True, hierarchy=None
    )
    held_out = sorted(set(full.obs["stage"]))[-1]
    reference_cells = full[full.obs["stage"] != held_out].copy()

    ref = BAE(reference_cells.n_vars, _config(latent_dim=4, max_iterations=25))
    ref.fit(reference_cells, verbose=False)

    work = full.copy()
    model = BAE.from_reference(
        ref, work, n_additional_dims=2, config=_config(latent_dim=6, max_iterations=25)
    )
    model.fit(work, verbose=False, decoder_warmup_epochs=10)

    novel = work.varm["BAE_encoder_weights"][:, 4:]
    selected = (np.abs(novel) > 0).any(axis=1)
    is_marker = np.asarray(work.var["is_marker"]).astype(bool)
    assert selected.sum() > 0
    assert is_marker[selected].mean() > is_marker.mean()
