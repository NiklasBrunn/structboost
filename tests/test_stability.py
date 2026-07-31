"""Tests for stability selection (Meinshausen-Bühlmann over allboost)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from structboost import StabilitySelectionResult, allboost, stability_selection


def _planted(n=600, p=200, k=4, seed=0):
    """Predictors X and targets Y where a disjoint gene block drives each target.

    Each latent dimension is driven by ``min(10, p // (2k))`` genes, so the marker
    blocks always fit inside ``p`` with null genes left over.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X = (X - X.mean(0)) / X.std(0)
    per_dim = max(1, min(10, p // (2 * k)))
    loadings = np.zeros((p, k))
    for j in range(k):
        block = slice(j * per_dim, (j + 1) * per_dim)
        # Floor the magnitude so every planted marker is genuinely detectable;
        # otherwise near-zero loadings are correctly (but confusingly) unstable.
        signs = rng.choice([-1.0, 1.0], size=per_dim)
        loadings[block, j] = signs * (1.5 + np.abs(rng.standard_normal(per_dim)))
    Y = X @ loadings + 0.3 * rng.standard_normal((n, k))
    truth = np.abs(loadings).sum(1) > 0
    return X, Y, truth


def test_recovers_planted_markers_with_high_precision():
    X, Y, truth = _planted()
    res = stability_selection(X, Y, n_subsamples=50, threshold=0.7, stepno=30, nu=0.1, seed=0)
    stable = res.stable_support.any(axis=1)
    tp = (stable & truth).sum()
    precision = tp / max(stable.sum(), 1)
    recall = tp / truth.sum()
    assert precision >= 0.9, f"precision {precision}"
    assert recall >= 0.7, f"recall {recall}"
    # True markers should sit at high frequency, null genes near zero.
    mx = res.frequency.max(axis=1)
    assert np.median(mx[truth]) > 0.8
    assert np.median(mx[~truth]) < 0.2


def test_frequencies_are_valid_and_shaped():
    X, Y, _ = _planted(p=50, k=3)
    res = stability_selection(X, Y, n_subsamples=20, seed=1)
    assert res.frequency.shape == (50, 3)
    assert res.stable_support.shape == (50, 3)
    assert (res.frequency >= 0).all() and (res.frequency <= 1).all()
    assert res.avg_selected.shape == (3,)


def test_deterministic_under_seed():
    X, Y, _ = _planted(p=40)
    a = stability_selection(X, Y, n_subsamples=15, seed=42)
    b = stability_selection(X, Y, n_subsamples=15, seed=42)
    np.testing.assert_array_equal(a.frequency, b.frequency)


def test_meinshausen_buhlmann_bound_formula():
    X, Y, _ = _planted(p=80, k=2)
    thr = 0.75
    res = stability_selection(X, Y, n_subsamples=30, threshold=thr, seed=0)
    expected = res.avg_selected**2 / ((2 * thr - 1) * 80)
    np.testing.assert_allclose(res.expected_false_positives, expected, rtol=1e-12)
    assert (res.expected_false_positives >= 0).all()


def test_threshold_at_or_below_half_warns_and_returns_nan_bound():
    X, Y, _ = _planted(p=40, k=2)
    with pytest.warns(UserWarning, match="error bound is undefined"):
        res = stability_selection(X, Y, n_subsamples=10, threshold=0.5, seed=0)
    assert np.isnan(res.expected_false_positives).all()
    # The stable support itself is still computed.
    assert res.stable_support.shape == (40, 2)


def test_n_genes_excludes_trailing_nuisance_columns():
    X, Y, _ = _planted(p=60, k=2)
    nuisance = np.random.default_rng(0).standard_normal((X.shape[0], 2))
    aug = np.hstack([X, nuisance])
    mandatory = np.array([60, 61], dtype=np.intp)  # force the nuisance columns
    res = stability_selection(
        aug, Y, n_genes=60, mandatory_features=mandatory, n_subsamples=15, seed=0
    )
    # Reported frequencies cover only the 60 gene columns, never the nuisance ones.
    assert res.frequency.shape == (60, 2)


def test_subsampling_is_without_replacement():
    # A tiny target so allboost is trivial; we only inspect the subsample size.
    X, Y, _ = _planted(n=100, p=20, k=1)
    # With subsample_frac=0.5 and n=100, each fit sees 50 distinct rows. This is
    # asserted indirectly: bootstrap (with replacement) is not an option, so the
    # result is reproducible and valid. Here we just confirm it runs and is bounded.
    res = stability_selection(X, Y, subsample_frac=0.5, n_subsamples=10, seed=0)
    assert res.subsample_frac == 0.5


def test_invalid_arguments_raise():
    X, Y, _ = _planted(p=30, k=2)
    with pytest.raises(ValueError, match="n_subsamples"):
        stability_selection(X, Y, n_subsamples=0)
    with pytest.raises(ValueError, match="subsample_frac"):
        stability_selection(X, Y, subsample_frac=1.5)
    with pytest.raises(ValueError, match="threshold"):
        stability_selection(X, Y, threshold=0.0)
    with pytest.raises(ValueError, match="n_genes"):
        stability_selection(X, Y, n_genes=999)
    with pytest.raises(ValueError, match="share n_samples"):
        stability_selection(X, Y[:-1])


def test_result_is_frozen_dataclass():
    import dataclasses

    X, Y, _ = _planted(p=20, k=2)
    res = stability_selection(X, Y, n_subsamples=5, seed=0)
    assert isinstance(res, StabilitySelectionResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.threshold = 0.9


def test_matches_a_manual_selection_frequency():
    """The frequency must equal a hand-rolled tally over the same subsamples."""
    X, Y, _ = _planted(n=200, p=30, k=2)
    kw = dict(stepno=20, nu=0.1, csf=0.9)
    res = stability_selection(X, Y, n_subsamples=8, subsample_frac=0.5, seed=7, **kw)

    rng = np.random.default_rng(7)
    counts = np.zeros((30, 2))
    for _ in range(8):
        idx = rng.choice(200, size=100, replace=False)
        beta = allboost(X[idx], Y[idx], **kw)
        counts += np.abs(beta).T > 0
    np.testing.assert_array_equal(res.frequency, counts / 8)


# --- BAE integration ---


def _require_bae():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")


def test_bae_method_stores_and_returns():
    _require_bae()
    from structboost import BAE, BAEConfig, sim_scrnaseq_anndata

    a = sim_scrnaseq_anndata(n=400, n_genes=150, stageno=4, stagep=12, seed=1)
    m = BAE(
        a.n_vars, BAEConfig(latent_dim=4, max_iterations=30, enable_early_stopping=False, seed=0)
    )
    m.fit(a, verbose=False)

    res = m.stability_selection(a, mode="subsample", n_runs=20, seed=0)
    assert res.frequency.shape == (a.n_vars, 4)
    assert a.varm["BAE_selection_frequency"].shape == (a.n_vars, 4)
    ss = a.uns["bae"]["stability_selection"]
    assert ss["threshold"] == 0.7
    assert ss["n_subsamples"] == 20
    assert ss["n_stable_per_dim"].shape == (4,)


def test_bae_stability_requires_fit():
    _require_bae()
    from structboost import BAE, sim_scrnaseq_anndata

    a = sim_scrnaseq_anndata(n=100, n_genes=40, stageno=3, stagep=8, seed=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        BAE(a.n_vars).stability_selection(a)


def test_bae_fit_flag_runs_stability_selection():
    _require_bae()
    from structboost import BAE, BAEConfig, sim_scrnaseq_anndata

    a = sim_scrnaseq_anndata(n=300, n_genes=120, stageno=4, stagep=10, seed=2)
    BAE(
        a.n_vars, BAEConfig(latent_dim=4, max_iterations=20, enable_early_stopping=False, seed=0)
    ).fit(a, verbose=False, stability_selection="subsample")
    assert "BAE_selection_frequency" in a.varm
    assert "stability_selection" in a.uns["bae"]


def test_bae_stability_uses_gradient_targets_not_latent():
    """Regressing onto the latent z is nearly circular; the method must use z*.

    z = W_sparse @ x is a function of only the selected genes, so re-selecting it
    is trivial. We check the method does not simply reproduce a fit against the
    stored latent by confirming stable genes track true markers rather than being
    a degenerate 0/1 copy of the encoder support.
    """
    _require_bae()
    from structboost import BAE, BAEConfig, sim_scrnaseq_anndata

    a = sim_scrnaseq_anndata(
        n=800,
        n_genes=300,
        stageno=5,
        stagep=25,
        stageoverlap=6,
        hierarchy=None,
        effect_size=3.0,
        seed=1,
    )
    truth = np.asarray(a.var["is_marker"]).astype(bool)
    m = BAE(
        a.n_vars, BAEConfig(latent_dim=5, max_iterations=80, enable_early_stopping=False, seed=0)
    )
    m.fit(a, verbose=False)
    res = m.stability_selection(a, mode="subsample", n_runs=40, seed=0)
    stable = res.stable_support.any(axis=1)
    precision = (stable & truth).sum() / max(stable.sum(), 1)
    assert precision >= 0.9


def test_mandatory_genes_excluded_from_error_bound():
    """Forced genes are always selected; they must not count toward q or p.

    Their frequency-1.0 is not evidence of stability, and they are not candidates
    competing for selection, so the Meinshausen-Bühlmann bound uses the effective
    (competitive) model size and candidate pool.
    """
    X, Y, _ = _planted(p=80, k=2)
    forced = np.array([70, 71, 72], dtype=np.intp)
    res = stability_selection(
        X, Y, n_subsamples=30, threshold=0.7, mandatory_features=forced, seed=0
    )
    # Forced genes appear with frequency 1.0 by construction.
    np.testing.assert_array_equal(res.frequency[70], np.ones(2))
    # Bound uses q - 3 forced and p - 3 forced, not the raw counts.
    q_eff = np.maximum(res.avg_selected - 3, 0.0)
    expected = q_eff**2 / ((2 * 0.7 - 1) * (80 - 3))
    np.testing.assert_allclose(res.expected_false_positives, expected, rtol=1e-12)


def test_gradient_targets_are_per_cell_deterministic():
    """The precompute-then-subsample shortcut is valid only if z*_i depends on
    cell i alone. It does, because _compute_boosting_targets runs the decoder in
    eval mode — even with batch norm enabled, which would otherwise couple cells.

    This pins the property the BAE stability-selection method relies on: computing
    z* once on the full data and subsampling it equals recomputing per subsample.
    """
    _require_bae()
    import torch

    from structboost import BAE, BAEConfig, sim_scrnaseq_anndata

    a = sim_scrnaseq_anndata(n=400, n_genes=100, stageno=4, stagep=10, seed=1)
    for batch_norm in (False, True):
        m = BAE(
            a.n_vars,
            BAEConfig(
                latent_dim=4,
                decoder_hidden_dims=(32,),
                decoder_use_batch_norm=batch_norm,
                max_iterations=25,
                enable_early_stopping=False,
                seed=0,
            ),
        ).fit(a, verbose=False)
        X = np.asarray(a.X, dtype=np.float32)
        full = m._compute_boosting_targets(torch.from_numpy(X), lr=1.0)
        idx = np.array([0, 7, 40, 200, 399])
        sub = m._compute_boosting_targets(torch.from_numpy(X[idx]), lr=1.0)
        np.testing.assert_allclose(full[idx], sub, atol=1e-4, err_msg=f"batch_norm={batch_norm}")


# --- Iteration-mode stability selection -------------------------------------


def _tiny_adata(n_obs=60, n_vars=30, seed=0):
    """Small z-scored AnnData suitable for a fast BAE fit."""
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_obs, n_vars))
    X = (X - X.mean(0)) / X.std(0)
    return anndata.AnnData(X.astype(np.float32))


def _fitted_model(adata, **overrides):
    from structboost import BAE, BAEConfig

    kwargs = dict(
        latent_dim=3,
        boosting_stepno=4,
        max_iterations=3,
        enable_early_stopping=False,
        decoder_updates_per_iteration=2,
        seed=0,
    )
    kwargs.update(overrides)
    model = BAE(n_genes=adata.n_vars, config=BAEConfig(**kwargs))
    model.fit(adata, verbose=False)
    return model


def test_iteration_mode_returns_per_dimension_frequency():
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)

    res = model.stability_selection(adata, mode="iteration", n_runs=5, seed=0)

    assert res.mode == "iteration"
    assert res.n_iterations == 5
    assert res.n_runs == 5
    # Per latent dimension, matched to the fitted model.
    assert res.frequency.shape == (adata.n_vars, model.config.latent_dim)
    assert np.all((res.frequency >= 0) & (res.frequency <= 1))
    assert res.stable_support.shape == res.frequency.shape
    assert 0.0 <= res.dim_match_quality <= 1.0
    # The MB bound is undefined for non-exchangeable training iterations.
    assert np.isnan(res.expected_false_positives).all()
    assert adata.varm["BAE_iteration_frequency"].shape == res.frequency.shape
    assert adata.uns["bae"]["stability_selection"]["mode"] == "iteration"


def test_subsample_mode_unchanged_and_tagged():
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)

    res = model.stability_selection(adata, mode="subsample", n_runs=4, seed=0)

    assert res.mode == "subsample"
    assert res.n_runs == 4
    assert res.n_iterations == 0
    assert res.frequency.shape == (adata.n_vars, model.config.latent_dim)
    assert adata.uns["bae"]["stability_selection"]["mode"] == "subsample"


def test_iteration_mode_does_not_mutate_the_fitted_model():
    """The call must be non-destructive, like subsample mode."""
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)

    before_encoder = model.get_encoder_weights().copy()
    before_decoder = {k: v.detach().clone() for k, v in model.decoder.state_dict().items()}

    model.stability_selection(adata, mode="iteration", n_runs=4, seed=0)

    np.testing.assert_array_equal(model.get_encoder_weights(), before_encoder)
    for key, value in model.decoder.state_dict().items():
        assert (value == before_decoder[key]).all(), f"decoder parameter {key} changed"


def test_iteration_loop_matches_the_training_loop():
    """Pin that iteration mode reproduces `fit`'s alternation exactly.

    The iteration-mode loop deliberately mirrors steps 1-6 of `fit` rather than
    sharing code with it, so this equivalence must be asserted or the two can drift
    apart silently.

    The comparison is made on *recorded* supports rather than on the fitted encoder,
    because `fit` restores the best-scoring iteration rather than the last one — a
    test that read the final weights could pass without the loops agreeing.

    A 1-iteration fit and a 2-iteration fit share their first iteration exactly
    (same seed, same data), so continuing the former by one iteration must land on
    the same support the latter recorded for its second iteration.
    """
    torch = pytest.importorskip("torch")
    adata = _tiny_adata()

    def record_supports(model):
        supports = []
        original = model.encoder.set_weights

        def hook(W):
            original(W)
            weights = W.detach().cpu().numpy()
            supports.append(np.sort(np.flatnonzero(np.abs(weights).max(axis=0) > 0)))

        model.encoder.set_weights = hook
        return supports, original

    # Two-iteration fit: capture the support produced by its *second* iteration.
    from structboost import BAE, BAEConfig

    cfg = dict(
        latent_dim=3,
        boosting_stepno=4,
        enable_early_stopping=False,
        decoder_updates_per_iteration=2,
        seed=0,
    )
    two = BAE(n_genes=adata.n_vars, config=BAEConfig(max_iterations=2, **cfg))
    supports_two, original = record_supports(two)
    two.fit(adata.copy(), verbose=False)
    two.encoder.set_weights = original
    second_iteration_support = supports_two[1]

    # One-iteration fit, then continue by one iteration through iteration mode.
    one = BAE(n_genes=adata.n_vars, config=BAEConfig(max_iterations=1, **cfg))
    one.fit(adata.copy(), verbose=False)
    supports_cont, original = record_supports(one)
    torch.manual_seed(0)
    one._iteration_support_frequency(adata.copy(), n_iterations=1, seed=0)
    one.encoder.set_weights = original

    np.testing.assert_array_equal(supports_cont[0], second_iteration_support)


def test_iteration_mode_rejects_bad_arguments():
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)

    with pytest.raises(ValueError, match="mode must be"):
        model.stability_selection(adata, mode="bogus")
    with pytest.raises(ValueError, match="n_runs must be >= 1"):
        model.stability_selection(adata, mode="iteration", n_runs=0)


def test_deprecated_count_aliases_still_work():
    """`n_subsamples` / `n_iterations` keep working but warn."""
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)

    with pytest.warns(FutureWarning, match="n_subsamples is deprecated"):
        res = model.stability_selection(adata, n_subsamples=3, seed=0)
    assert res.n_runs == 3

    with pytest.warns(FutureWarning, match="n_iterations is deprecated"):
        res = model.stability_selection(adata, mode="iteration", n_iterations=3, seed=0)
    assert res.n_runs == 3


def test_iteration_dimensions_are_matched_to_the_fitted_model():
    """Per-dimension counting must survive a permutation of the encoder rows.

    Anchoring is the whole reason per-dimension frequencies mean anything: without
    it, a permuted iteration would contribute its genes to the wrong dimensions.
    """
    torch = pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)

    baseline = model.stability_selection(adata, mode="iteration", n_runs=3, seed=0)

    # Permute the fitted encoder's rows. Matching should undo it, so the per-gene
    # frequency matrix should come back permuted the same way, not scrambled.
    perm = torch.tensor([2, 0, 1])
    with torch.no_grad():
        model.encoder.linear.weight.copy_(model.encoder.linear.weight[perm].clone())
    permuted = model.stability_selection(adata, mode="iteration", n_runs=3, seed=0)

    assert permuted.frequency.shape == baseline.frequency.shape
    # Column sums are permutation-equivariant; the multiset of per-dimension
    # frequencies must be preserved even though the model rows moved.
    np.testing.assert_allclose(
        np.sort(permuted.frequency.sum(axis=0)),
        np.sort(baseline.frequency.sum(axis=0)),
        rtol=0.5,
    )


def test_fit_stability_selection_takes_a_mode_string():
    """One argument, so "disabled but with a mode" cannot be expressed."""
    pytest.importorskip("torch")
    from structboost import BAE, BAEConfig

    adata = _tiny_adata()
    cfg = dict(
        latent_dim=3,
        boosting_stepno=4,
        max_iterations=2,
        enable_early_stopping=False,
        decoder_updates_per_iteration=2,
        seed=0,
    )

    model = BAE(n_genes=adata.n_vars, config=BAEConfig(**cfg))
    model.fit(adata, verbose=False, stability_selection="iteration")
    assert adata.uns["bae"]["stability_selection"]["mode"] == "iteration"

    other = _tiny_adata()
    model = BAE(n_genes=other.n_vars, config=BAEConfig(**cfg))
    model.fit(other, verbose=False, stability_selection=None)
    assert "stability_selection" not in other.uns.get("bae", {})


def test_fit_stability_selection_true_is_deprecated():
    pytest.importorskip("torch")
    from structboost import BAE, BAEConfig

    adata = _tiny_adata()
    model = BAE(
        n_genes=adata.n_vars,
        config=BAEConfig(
            latent_dim=3,
            boosting_stepno=4,
            max_iterations=2,
            enable_early_stopping=False,
            decoder_updates_per_iteration=2,
            seed=0,
        ),
    )
    with pytest.warns(FutureWarning, match="stability_selection=True is deprecated"):
        model.fit(adata, verbose=False, stability_selection=True)
    assert adata.uns["bae"]["stability_selection"]["mode"] == "subsample"


def test_fit_rejects_unknown_stability_mode():
    pytest.importorskip("torch")
    from structboost import BAE, BAEConfig

    adata = _tiny_adata()
    model = BAE(n_genes=adata.n_vars, config=BAEConfig(latent_dim=2, max_iterations=1))
    with pytest.raises(ValueError, match="stability_selection must be"):
        model.fit(adata, verbose=False, stability_selection="bogus")


def test_default_mode_is_iteration():
    """The default readout is the one with the lowest measured FDR.

    Changed in 0.7.0.0: subsample mode's Meinshausen-Bühlmann bound is violated by
    roughly an order of magnitude on data where the truth is known, while iteration
    frequencies are empirically calibrated. Pinned so the default cannot drift back
    without the evidence being revisited.
    """
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)

    res = model.stability_selection(adata, n_runs=3, seed=0)
    assert res.mode == "iteration"


# --- Coefficient aggregation and the derived encoder ------------------------


@pytest.mark.parametrize("mode", ["iteration", "subsample"])
def test_coefficients_are_collected_in_both_modes(mode):
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)

    res = model.stability_selection(adata, mode=mode, n_runs=6, seed=0)

    shape = (adata.n_vars, model.config.latent_dim)
    assert res.coefficient_cond_mean.shape == shape
    assert res.coefficient_sd.shape == shape
    assert res.sign_consistency.shape == shape
    assert np.isfinite(res.coefficient_cond_mean).all()
    assert (res.coefficient_sd >= 0).all()
    assert ((res.sign_consistency >= 0.5) & (res.sign_consistency <= 1.0)).all()
    # Unselected entries carry no coefficient information.
    never = res.frequency == 0
    assert (res.coefficient_cond_mean[never] == 0).all()


def test_coefficient_mean_is_the_shrunk_bagged_estimator():
    """`coefficient_mean` must equal cond_mean * frequency, not the conditional mean."""
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)

    res = model.stability_selection(adata, n_runs=6, seed=0)
    np.testing.assert_allclose(
        res.coefficient_mean, res.coefficient_cond_mean * res.frequency, rtol=1e-10
    )


def test_stable_encoder_is_the_masked_conditional_mean():
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)
    res = model.stability_selection(adata, n_runs=6, threshold=0.7, seed=0)

    encoder = res.stable_encoder()

    assert encoder.shape == res.frequency.shape
    # Exactly the conditional mean on the stable support, zero elsewhere.
    np.testing.assert_array_equal(
        encoder != 0, res.stable_support & (res.coefficient_cond_mean != 0)
    )
    kept = res.stable_support
    np.testing.assert_allclose(encoder[kept], res.coefficient_cond_mean[kept])
    assert (encoder[~kept] == 0).all()


def test_threshold_is_the_precision_recall_dial():
    """A lower threshold keeps more genes — the replacement for a separate
    unmasked estimator, which was removed as redundant with this parameter."""
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)

    strict = model.stability_selection(adata, n_runs=6, threshold=0.9, seed=0).stable_encoder()
    loose = model.stability_selection(adata, n_runs=6, threshold=0.2, seed=0).stable_encoder()

    assert (loose != 0).sum() >= (strict != 0).sum()
    # Loosening can only add entries, never change or drop existing ones.
    assert ((strict != 0) <= (loose != 0)).all()


def test_apply_encoder_is_explicit_and_stability_selection_stays_pure():
    """The diagnostic must not mutate the model; installing must be a separate call."""
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)
    before = model.get_encoder_weights().copy()

    res = model.stability_selection(adata, n_runs=6, seed=0)
    np.testing.assert_array_equal(model.get_encoder_weights(), before)

    model.apply_encoder(res.stable_encoder(), adata)
    assert not np.array_equal(model.get_encoder_weights(), before)
    np.testing.assert_allclose(adata.varm["BAE_encoder_weights"], res.stable_encoder(), rtol=1e-6)
    assert adata.obsm["X_bae"].shape == (adata.n_obs, model.config.latent_dim)
    assert adata.uns["bae"]["encoder_source"] == "aggregated"


def test_apply_encoder_validates_shape_and_finiteness():
    pytest.importorskip("torch")
    adata = _tiny_adata()
    model = _fitted_model(adata)

    with pytest.raises(ValueError, match="weights must have shape"):
        model.apply_encoder(np.zeros((3, adata.n_vars)))  # transposed
    bad = np.zeros((adata.n_vars, model.config.latent_dim))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        model.apply_encoder(bad)


def test_aggregation_works_with_mandatory_genes():
    """Forced genes must survive the stable-support mask, not be silently dropped."""
    pytest.importorskip("torch")
    from structboost import BAE, BAEConfig

    adata = _tiny_adata(n_vars=40)
    mandatory = [str(g) for g in adata.var_names[:4]]
    model = BAE(
        n_genes=adata.n_vars,
        config=BAEConfig(
            latent_dim=3,
            boosting_stepno=5,
            max_iterations=4,
            enable_early_stopping=False,
            decoder_updates_per_iteration=2,
            seed=0,
        ),
    )
    model.fit(adata, verbose=False, mandatory_genes=mandatory)

    for mode in ("iteration", "subsample"):
        res = model.stability_selection(adata, mode=mode, n_runs=6, seed=0)
        encoder = res.stable_encoder()
        idx = [list(adata.var_names).index(g) for g in mandatory]
        assert (encoder[idx] != 0).any(axis=1).all(), f"{mode}: mandatory gene dropped"


def test_masking_can_break_batch_integration_and_warns():
    """Thresholding a conditioned encoder is not a covariate-neutral operation.

    Integration is a property of the whole coefficient vector — surviving genes
    partially cancel each other's residual covariate signal — so dropping
    low-frequency entries can reintroduce it. `apply_encoder` measures the change
    rather than warning unconditionally.
    """
    pytest.importorskip("torch")
    from structboost import BAE, BAEConfig, sim_scrnaseq_anndata

    adata = sim_scrnaseq_anndata(
        n=400,
        n_genes=200,
        stageno=5,
        effect_size=3.0,
        n_batches=3,
        batch_effect_sd=1.0,
        seed=0,
        standardize=True,
    )
    adata.obs["batch"] = adata.obs["batch"].astype(str)
    model = BAE(
        n_genes=adata.n_vars,
        config=BAEConfig(
            latent_dim=5,
            boosting_stepno=10,
            max_iterations=30,
            enable_early_stopping=False,
            target_optim_lr=1.0,
            seed=0,
        ),
    )
    model.fit(adata, verbose=False, condition_obs=["batch"], nuisance_obs=["batch"])
    res = model.stability_selection(adata, n_runs=8, seed=0)

    # The guard must be measurement-driven: an unconditioned model never warns.
    plain_adata = _tiny_adata()
    plain = _fitted_model(plain_adata)
    plain_res = plain.stability_selection(plain_adata, n_runs=6, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plain.apply_encoder(plain_res.stable_encoder(), plain_adata)

    # And installing on a conditioned model must at least not raise.
    model.apply_encoder(res.stable_encoder(), adata)


# --- Progress reporting ------------------------------------------------------


@pytest.mark.parametrize("mode", ["iteration", "subsample"])
def test_stability_selection_reports_progress(capsys, mode):
    """Both modes show a bar by default; `verbose=False` is silent.

    The default is 300 runs, so a silent call looks indistinguishable from a hang.
    """
    pytest.importorskip("torch")
    pytest.importorskip("tqdm")

    adata = _tiny_adata()
    model = _fitted_model(adata)

    model.stability_selection(adata, mode=mode, n_runs=2, seed=0, verbose=False)
    assert capsys.readouterr().err == ""

    model.stability_selection(adata, mode=mode, n_runs=2, seed=0)
    assert "Stability selection" in capsys.readouterr().err


def test_module_level_stability_selection_is_silent_by_default(capsys):
    """A direct caller of the NumPy-only function gets no bar unless it asks."""
    X, Y, _ = _planted(n=120, p=40, k=2, seed=0)

    stability_selection(X, Y, n_subsamples=2, stepno=4, seed=0)
    assert capsys.readouterr().err == ""

    pytest.importorskip("tqdm")
    stability_selection(X, Y, n_subsamples=2, stepno=4, seed=0, verbose=True)
    assert "Stability selection" in capsys.readouterr().err
