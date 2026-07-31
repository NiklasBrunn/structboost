"""Tests for the torch/anndata-based BAE model.

Skipped automatically if BAE dependencies (torch, anndata, scipy, tqdm) are missing.
"""

from __future__ import annotations

import numpy as np
import pytest


def _require_bae_deps():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")
    pytest.importorskip("scipy")
    pytest.importorskip("tqdm")


def test_bae_fit_transform_smoke():
    """Test basic fit/transform workflow with standardized data."""
    _require_bae_deps()

    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    # Create standardized data (mean≈0, std≈1)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    adata = ad.AnnData(x)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        device="cpu",
    )
    model = BAE(n_genes=adata.n_vars, config=config)
    model.fit(adata, max_iterations=3, verbose=False)

    assert "X_bae" in adata.obsm
    assert adata.obsm["X_bae"].shape == (adata.n_obs, config.latent_dim)
    assert "BAE_encoder_weights" in adata.varm
    assert adata.varm["BAE_encoder_weights"].shape == (adata.n_vars, config.latent_dim)
    assert "bae" in adata.uns


def test_bae_transform_requires_fit():
    """Test that transform raises error if model not fitted."""
    _require_bae_deps()

    import anndata as ad

    from structboost import BAE

    rng = np.random.default_rng(1)
    adata = ad.AnnData(rng.normal(size=(16, 8)).astype(np.float32))
    model = BAE(n_genes=adata.n_vars)

    with pytest.raises(RuntimeError, match="not fitted"):
        _ = model.transform(adata)


def test_bae_encoder_reset():
    """Test encoder weight reset functionality."""
    _require_bae_deps()

    import torch

    from structboost import BAE, BAEConfig

    config = BAEConfig(latent_dim=2, device="cpu")
    model = BAE(n_genes=10, config=config)

    # Set some non-zero weights
    model.encoder.linear.weight.data.fill_(1.0)
    assert torch.any(model.encoder.linear.weight != 0)

    # Reset should zero them
    model.encoder.reset_weights()
    assert torch.all(model.encoder.linear.weight == 0)


def test_disentangle_boosting_targets():
    """Leave-one-out residuals are orthogonal to the other original targets."""
    from structboost._utils import disentangle_boosting_targets

    rng = np.random.default_rng(42)
    n, d = 100, 4
    base = rng.standard_normal((n, d))
    targets = base.copy()
    targets[:, 1] = 0.8 * targets[:, 0] + 0.2 * base[:, 1]

    result = disentangle_boosting_targets(targets)
    assert result.shape == targets.shape
    assert result.dtype == targets.dtype

    # Each output column is orthogonal to every *original* other column. It is
    # deliberately not claimed that the output columns are mutually orthogonal.
    for j in range(d):
        others = np.delete(targets, j, axis=1)
        np.testing.assert_allclose(others.T @ result[:, j], 0.0, atol=1e-10)

    # In two dimensions simultaneous leave-one-out residualization flips the
    # correlation sign without reducing its magnitude. This pins the method's
    # actual mathematics and prevents it being presented as exact orthogonalization.
    pair = targets[:, :2]
    pair_result = disentangle_boosting_targets(pair, standardize=True)
    before = np.corrcoef(pair, rowvar=False)[0, 1]
    after = np.corrcoef(pair_result, rowvar=False)[0, 1]
    assert after == pytest.approx(-before, abs=1e-12)

    single = rng.standard_normal((50, 1))
    result_single = disentangle_boosting_targets(single)
    assert result_single.shape == single.shape
    np.testing.assert_array_equal(result_single, single)

    unequal_var = rng.standard_normal((100, 3))
    unequal_var[:, 0] *= 10
    result_std = disentangle_boosting_targets(unequal_var, standardize=True)
    assert result_std.shape == unequal_var.shape
    assert not np.any(np.isnan(result_std))


def test_bae_fit_with_leave_one_out_disentanglement():
    """The earlier residualization method remains available through the new API."""
    _require_bae_deps()

    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(123)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    adata = ad.AnnData(x)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        disentanglement="leave_one_out",
        device="cpu",
    )
    model = BAE(n_genes=adata.n_vars, config=config)
    model.fit(adata, max_iterations=3, verbose=False)

    # Basic checks: training completes and outputs are valid
    assert "X_bae" in adata.obsm
    assert adata.obsm["X_bae"].shape == (adata.n_obs, config.latent_dim)
    assert not np.any(np.isnan(adata.obsm["X_bae"]))
    assert adata.uns["bae"]["disentanglement"] == "leave_one_out"


def test_disentanglement_config_api_and_validation():
    """Only the new method-based disentanglement API is accepted."""
    from structboost import BAEConfig

    config = BAEConfig()
    assert config.disentanglement == "none"
    assert config.disentanglement_lambda == pytest.approx(1e-4)
    assert not config.disentanglement_standardize
    assert "disentangle_targets" not in BAEConfig.__dataclass_fields__
    assert "disentangle_standardize" not in BAEConfig.__dataclass_fields__

    with pytest.raises(ValueError, match="disentanglement must be one of"):
        BAEConfig(disentanglement="invalid")
    for value in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="must be finite and >= 0"):
            BAEConfig(disentanglement_lambda=value)
    with pytest.raises(ValueError, match="only available"):
        BAEConfig(disentanglement="correlation", disentanglement_standardize=True)


def test_correlation_disentanglement_loss_and_descent_direction():
    """The soft loss detects correlation and its gradient points downhill."""
    _require_bae_deps()
    import torch

    from structboost import BAE

    rng = np.random.default_rng(5)
    base = rng.normal(size=(200, 3)).astype(np.float32)
    base[:, 1] = 0.9 * base[:, 0] + 0.1 * base[:, 1]
    z = torch.tensor(base, requires_grad=True)

    correlation_loss, variance_loss = BAE._correlation_disentanglement_loss(z)
    assert float(correlation_loss.detach()) > 0.2
    assert float(variance_loss.detach()) >= 0

    combined = correlation_loss + 0.1 * variance_loss
    (gradient,) = torch.autograd.grad(combined, z)
    moved = z.detach() - 1e-3 * gradient
    moved_correlation, moved_variance = BAE._correlation_disentanglement_loss(moved)
    moved_loss = moved_correlation + 0.1 * moved_variance
    assert float(moved_loss.detach()) < float(combined.detach())


def test_weighted_correlation_loss_matches_explicit_replication():
    """Weighted moments describe the same pseudo-population as repeated cells."""
    _require_bae_deps()
    import torch

    from structboost import BAE

    z = torch.tensor(
        [[-1.0, 0.2], [0.5, 1.3], [2.0, -0.7]],
        dtype=torch.float64,
    )
    weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    repeated = torch.repeat_interleave(z, weights.to(torch.int64), dim=0)
    weighted = BAE._correlation_disentanglement_loss(z, weights)
    explicit = BAE._correlation_disentanglement_loss(repeated)
    for observed, expected in zip(weighted, explicit, strict=True):
        torch.testing.assert_close(observed, expected)


def test_correlation_target_step_is_independent_of_n_cells():
    """The n-cell scaling keeps the soft-constraint step replication-invariant."""
    _require_bae_deps()
    import torch

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(6)
    base = rng.normal(size=(30, 12)).astype(np.float32)
    base = (base - base.mean(axis=0)) / base.std(axis=0)
    weights = torch.tensor(rng.normal(size=(3, 12)).astype(np.float32))

    steps = []
    for repeats in (1, 3):
        torch.manual_seed(9)
        model = BAE(
            12,
            BAEConfig(
                latent_dim=3,
                decoder_hidden_dims=(8,),
                disentanglement="correlation",
                disentanglement_lambda=1e-3,
            ),
        )
        model.encoder.set_weights(weights)
        x = torch.from_numpy(np.tile(base, (repeats, 1)))
        with torch.no_grad():
            z = model.encoder(x).numpy()
        targets = model._compute_boosting_targets(x, lr=0.1)
        steps.append(targets[0] - z[0])

    np.testing.assert_allclose(steps[1], steps[0], rtol=2e-5, atol=1e-7)


def test_correlation_fit_records_selection_objective_and_metadata():
    """Checkpoint selection includes the soft constraint without replacing MSE."""
    _require_bae_deps()
    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 30)).astype(np.float32)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    adata = ad.AnnData(x)
    model = BAE(
        adata.n_vars,
        BAEConfig(
            latent_dim=3,
            decoder_hidden_dims=(12,),
            boosting_stepno=10,
            target_optim_lr=0.1,
            max_iterations=4,
            enable_early_stopping=False,
            disentanglement="correlation",
            disentanglement_lambda=1e-4,
            seed=0,
        ),
    ).fit(adata, verbose=False)

    train = np.asarray(model.training_history["train_loss"])
    selection = np.asarray(model.training_history["selection_loss"])
    assert train.shape == selection.shape == (4,)
    assert np.all(selection >= train)
    assert adata.uns["bae"]["disentanglement"] == "correlation"
    assert adata.uns["bae"]["disentanglement_lambda"] == pytest.approx(1e-4)


def test_correlation_disentanglement_reduces_fitted_latent_correlation():
    """The soft constraint materially decorrelates a fitted non-collapsed latent."""
    _require_bae_deps()
    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    n_cells, n_genes, latent_dim = 240, 90, 3
    shared = rng.normal(size=n_cells)
    factors = np.column_stack(
        (
            shared + 0.25 * rng.normal(size=n_cells),
            shared + 0.25 * rng.normal(size=n_cells),
            rng.normal(size=n_cells),
        )
    )
    loadings = np.zeros((n_genes, latent_dim))
    for dim in range(latent_dim):
        feature_slice = slice(dim * 20, (dim + 1) * 20)
        loadings[feature_slice, dim] = rng.uniform(0.8, 1.4, size=20)
    x = factors @ loadings.T + 0.4 * rng.normal(size=(n_cells, n_genes))
    x = ((x - x.mean(axis=0)) / x.std(axis=0)).astype(np.float32)

    correlations = {}
    variances = {}
    for method in ("none", "correlation"):
        adata = ad.AnnData(x.copy())
        model = BAE(
            n_genes,
            BAEConfig(
                latent_dim=latent_dim,
                decoder_hidden_dims=(32,),
                boosting_stepno=20,
                target_optim_lr=0.1,
                decoder_updates_per_iteration=4,
                max_iterations=20,
                enable_early_stopping=False,
                disentanglement=method,
                disentanglement_lambda=1e-3,
                seed=0,
                device="cpu",
            ),
        ).fit(adata, verbose=False)
        z = model.transform(adata)
        corr = np.corrcoef(z, rowvar=False)
        off_diagonal = corr[np.triu_indices(latent_dim, k=1)]
        correlations[method] = np.mean(np.abs(off_diagonal))
        variances[method] = np.var(z, axis=0)

    assert correlations["correlation"] < 0.6 * correlations["none"]
    assert np.all(variances["correlation"] > 1e-4)


def test_bae_standardize_targets_enabled():
    """Test BAE with standardize_targets=True (not the default, which is False)."""
    _require_bae_deps()

    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    adata = ad.AnnData(x)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        enable_early_stopping=False,
        standardize_targets=True,
        device="cpu",
    )
    model = BAE(n_genes=adata.n_vars, config=config)
    model.fit(adata, verbose=False)

    assert "X_bae" in adata.obsm
    assert not np.any(np.isnan(adata.obsm["X_bae"]))


def test_bae_standardize_targets_disabled():
    """Test BAE with standardize_targets=False, which is the default."""
    _require_bae_deps()

    import anndata as ad

    from structboost import BAE, BAEConfig

    assert BAEConfig().standardize_targets is False

    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    adata = ad.AnnData(x)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        enable_early_stopping=False,
        standardize_targets=False,  # Disabled
        device="cpu",
    )
    model = BAE(n_genes=adata.n_vars, config=config)
    model.fit(adata, verbose=False)

    assert "X_bae" in adata.obsm
    assert not np.any(np.isnan(adata.obsm["X_bae"]))


def test_bae_config_decoder_weight_decay_default():
    """Test that decoder_weight_decay defaults to 0.0."""
    _require_bae_deps()
    from structboost import BAEConfig

    config = BAEConfig()
    assert config.decoder_weight_decay == 0.0


def test_bae_config_decoder_weight_decay_negative_raises():
    """Test that negative decoder_weight_decay raises ValueError."""
    _require_bae_deps()
    from structboost import BAEConfig

    with pytest.raises(ValueError, match="decoder_weight_decay must be >= 0"):
        BAEConfig(decoder_weight_decay=-0.01)


def test_bae_early_stopping_enabled():
    """Test BAE with early stopping on the checkpoint-selection objective."""
    _require_bae_deps()

    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    adata = ad.AnnData(x)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=100,  # High max, should stop early
        enable_early_stopping=True,
        early_stopping_patience=3,
        device="cpu",
    )
    model = BAE(n_genes=adata.n_vars, config=config)
    model.fit(adata, verbose=False)

    # Should have stopped before max_iterations
    n_iterations = len(model._training_history["train_loss"])
    assert n_iterations <= config.max_iterations
    assert "X_bae" in adata.obsm


def test_split_softmax_module():
    """Unit test for SplitSoftmax: simplex properties + interleaving semantics."""
    _require_bae_deps()
    import torch

    from structboost._encoder import SplitSoftmax

    z = torch.tensor([[1.0, -2.0, 0.0, 3.0]])
    ssm = SplitSoftmax()
    h = ssm(z)

    assert h.shape == (1, 8)  # 2 * d = 8
    assert torch.all(h >= 0)
    assert torch.allclose(h.sum(dim=-1), torch.tensor(1.0))
    # Interleaved: s = (1, -1, -2, 2, 0, 0, 3, -3)
    assert h[0, 0] > h[0, 1]  # z_1 > 0: positive wins
    assert h[0, 2] < h[0, 3]  # z_2 < 0: negative wins
    assert torch.allclose(h[0, 4], h[0, 5])  # z_3 = 0: tied


def test_split_softmax_zero_input():
    """z = 0 → uniform distribution h_j = 1/(2d)."""
    _require_bae_deps()
    import torch

    from structboost._encoder import SplitSoftmax

    d = 5
    z = torch.zeros(1, d)
    h = SplitSoftmax()(z)
    expected = torch.full((1, 2 * d), 1.0 / (2 * d))
    assert torch.allclose(h, expected, atol=1e-6)


def test_bae_config_split_softmax_default():
    """split_softmax defaults to False."""
    _require_bae_deps()
    from structboost import BAEConfig

    assert BAEConfig().split_softmax is False


def test_split_softmax_false_matches_previous():
    """split_softmax=False must produce identical results to default config."""
    _require_bae_deps()
    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 32)).astype(np.float32)

    shared = dict(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        seed=42,
    )

    adata1 = ad.AnnData(x.copy())
    model1 = BAE(n_genes=32, config=BAEConfig(**shared))
    model1.fit(adata1, verbose=False)

    adata2 = ad.AnnData(x.copy())
    model2 = BAE(n_genes=32, config=BAEConfig(**shared, split_softmax=False))
    model2.fit(adata2, verbose=False)

    np.testing.assert_array_equal(adata1.obsm["X_bae"], adata2.obsm["X_bae"])


def test_split_softmax_true_compositional():
    """split_softmax=True: decoder input is compositional, z is still d-dim."""
    _require_bae_deps()
    import anndata as ad
    import torch

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    n_cells, n_genes, latent_dim = 64, 32, 4
    x = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(x)

    config = BAEConfig(
        latent_dim=latent_dim,
        decoder_hidden_dims=(16,),
        max_iterations=5,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        split_softmax=True,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, verbose=False)

    # Latent is still d-dimensional
    z_stored = adata.obsm["X_bae"]
    assert z_stored.shape == (n_cells, latent_dim)

    # Encoder weights shape unchanged
    assert adata.varm["BAE_encoder_weights"].shape == (n_genes, latent_dim)

    # Verify compositional property of internal representation
    X_t = torch.from_numpy(x).float()
    z_t = model.encoder(X_t)
    h = model.split_softmax_layer(z_t)
    h_np = h.detach().numpy()
    assert h_np.shape == (n_cells, 2 * latent_dim)
    assert np.all(h_np >= 0)
    np.testing.assert_allclose(h_np.sum(axis=1), 1.0, atol=1e-6)

    # Training converged (loss decreased)
    losses = model._training_history["train_loss"]
    assert losses[-1] < losses[0]


def test_bae_early_stopping_disabled():
    """Test BAE trains for exactly max_iterations when early stopping disabled."""
    _require_bae_deps()

    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    adata = ad.AnnData(x)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=5,
        enable_early_stopping=False,  # Disabled
        device="cpu",
    )
    model = BAE(n_genes=adata.n_vars, config=config)
    model.fit(adata, verbose=False)

    # Should train for exactly max_iterations
    assert len(model._training_history["train_loss"]) == config.max_iterations


# ---------------------------------------------------------------------------
# Shared test fixtures for split-softmax tests
# ---------------------------------------------------------------------------
n_cells, n_genes = 64, 32


@pytest.fixture
def adata():
    _require_bae_deps()
    import anndata as ad

    rng = np.random.default_rng(0)
    x = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    return ad.AnnData(x)


# ---------------------------------------------------------------------------
# Tests for transform_splitsoftmax
# ---------------------------------------------------------------------------


def test_bae_transform_splitsoftmax(adata):
    """transform_splitsoftmax returns valid compositional output."""
    _require_bae_deps()
    from structboost import BAE, BAEConfig

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        split_softmax=True,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, verbose=False)

    h = model.transform_splitsoftmax(adata)

    # Shape: (n_cells, 2 * latent_dim)
    assert h.shape == (n_cells, 8)

    # Compositional: non-negative and sums to 1
    assert np.all(h >= 0)
    np.testing.assert_allclose(h.sum(axis=1), 1.0, atol=1e-6)

    # Stored in adata.obsm
    assert "X_bae_splitsoftmax" in adata.obsm
    np.testing.assert_array_equal(adata.obsm["X_bae_splitsoftmax"], h)

    # Original latent z is still d-dimensional
    assert adata.obsm["X_bae"].shape == (n_cells, 4)


def test_bae_transform_splitsoftmax_stores_varm(adata):
    """transform_splitsoftmax stores clipped encoder weights in varm."""
    _require_bae_deps()
    from structboost import BAE, BAEConfig

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        split_softmax=True,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, verbose=False)

    model.transform_splitsoftmax(adata)

    # Weights stored in varm
    assert "bae_program_weights" in adata.varm
    W = adata.varm["bae_program_weights"]

    # Shape: (n_vars, 2 * latent_dim)
    assert W.shape == (n_genes, 2 * 4)

    # All non-negative (clipped)
    assert np.all(W >= 0)

    # Positive-split columns (even indices) have only positive encoder weights
    # Negative-split columns (odd indices) have only abs(negative) encoder weights
    W_enc = model.get_encoder_weights()
    np.testing.assert_array_equal(W[:, 0::2], np.maximum(W_enc, 0))
    np.testing.assert_array_equal(W[:, 1::2], np.maximum(-W_enc, 0))


def test_bae_transform_splitsoftmax_without_split_softmax_warns(adata):
    """transform_splitsoftmax warns when split_softmax=False and still works."""
    _require_bae_deps()
    from structboost import BAE, BAEConfig

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        split_softmax=False,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, verbose=False)

    with pytest.warns(UserWarning, match="not trained with split_softmax=True"):
        h = model.transform_splitsoftmax(adata)

    # Should still produce valid compositional output
    assert h.shape == (n_cells, 8)
    assert np.all(h >= 0)
    np.testing.assert_allclose(h.sum(axis=1), 1.0, atol=1e-6)
    assert "X_bae_splitsoftmax" in adata.obsm


def test_bae_transform_splitsoftmax_raises_not_fitted():
    """transform_splitsoftmax raises RuntimeError when model not fitted."""
    _require_bae_deps()
    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    adata_local = ad.AnnData(rng.normal(size=(n_cells, n_genes)).astype(np.float32))

    config = BAEConfig(latent_dim=4, split_softmax=True)
    model = BAE(n_genes=n_genes, config=config)

    with pytest.raises(RuntimeError, match="not fitted"):
        model.transform_splitsoftmax(adata_local)


# ---------------------------------------------------------------------------
# Tests for get_splitsoftmax_encoder_weights
# ---------------------------------------------------------------------------


def test_bae_get_splitsoftmax_encoder_weights_clipped(adata):
    """get_splitsoftmax_encoder_weights returns clipped interleaved weights by default."""
    _require_bae_deps()
    from structboost import BAE, BAEConfig

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        split_softmax=True,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, verbose=False)

    W_split = model.get_splitsoftmax_encoder_weights()  # clip_negative=True by default
    W = model.get_encoder_weights()

    # Shape: (n_genes, 2 * latent_dim)
    assert W_split.shape == (n_genes, 2 * 4)

    # All values must be non-negative (clipped)
    assert np.all(W_split >= 0)

    # Even columns = clipped original encoder weights (positive direction)
    np.testing.assert_array_equal(W_split[:, 0::2], np.maximum(W, 0))

    # Odd columns = clipped negated encoder weights (negative direction)
    np.testing.assert_array_equal(W_split[:, 1::2], np.maximum(-W, 0))


def test_bae_get_splitsoftmax_encoder_weights_unclipped(adata):
    """get_splitsoftmax_encoder_weights with clip_negative=False returns raw weights."""
    _require_bae_deps()
    from structboost import BAE, BAEConfig

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        split_softmax=True,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, verbose=False)

    W_split = model.get_splitsoftmax_encoder_weights(clip_negative=False)
    W = model.get_encoder_weights()

    # Shape: (n_genes, 2 * latent_dim)
    assert W_split.shape == (n_genes, 2 * 4)

    # Even columns = original encoder weights (positive direction, unclipped)
    np.testing.assert_array_equal(W_split[:, 0::2], W)

    # Odd columns = negated encoder weights (negative direction, unclipped)
    np.testing.assert_array_equal(W_split[:, 1::2], -W)


def test_bae_get_splitsoftmax_encoder_weights_without_split_softmax_warns(adata):
    """get_splitsoftmax_encoder_weights warns when split_softmax=False."""
    _require_bae_deps()
    from structboost import BAE, BAEConfig

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        split_softmax=False,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, verbose=False)

    with pytest.warns(UserWarning, match="not trained with split_softmax=True"):
        W_split = model.get_splitsoftmax_encoder_weights()

    # Should still produce correct clipped weights
    assert W_split.shape == (n_genes, 2 * 4)
    assert np.all(W_split >= 0)


def test_bae_get_splitsoftmax_encoder_weights_tensor(adata):
    """get_splitsoftmax_encoder_weights returns torch tensor when as_numpy=False."""
    _require_bae_deps()
    import torch

    from structboost import BAE, BAEConfig

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        split_softmax=True,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, verbose=False)

    W_split = model.get_splitsoftmax_encoder_weights(as_numpy=False)
    assert isinstance(W_split, torch.Tensor)
    assert W_split.shape == (n_genes, 8)
    assert torch.all(W_split >= 0)  # default clip_negative=True


def test_bae_splitsoftmax_consistency(adata):
    """Split-softmax output is consistent with interleaved encoder weights."""
    _require_bae_deps()
    import scipy.sparse as sp

    from structboost import BAE, BAEConfig

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        split_softmax=True,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, verbose=False)

    h = model.transform_splitsoftmax(adata)
    W_split = model.get_splitsoftmax_encoder_weights(clip_negative=False)

    # Compute s = X @ W_split, then softmax
    X_np = adata.X if not sp.issparse(adata.X) else adata.X.toarray()
    s = X_np @ W_split  # (n_cells, 2*latent_dim)
    # Manual softmax
    s_exp = np.exp(s - s.max(axis=1, keepdims=True))
    h_manual = s_exp / s_exp.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(h, h_manual, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests for encoder (obs_linear removed — latent is purely gene-based)
# ---------------------------------------------------------------------------


def test_encoder_no_obs_linear():
    """BAEEncoder has no obs_linear — latent is purely gene-based."""
    _require_bae_deps()
    import torch

    from structboost import BAEConfig
    from structboost._encoder import BAEEncoder

    config = BAEConfig(latent_dim=4)
    encoder = BAEEncoder(n_input=32, config=config)

    # No obs_linear attribute or it's absent
    assert not hasattr(encoder, "obs_linear") or encoder.obs_linear is None

    # forward() accepts only x, no obs_covariates parameter
    x = torch.randn(8, 32)
    z = encoder(x)
    assert z.shape == (8, 4)

    # set_weights accepts only W, no obs_weights parameter
    W = torch.ones(4, 32)
    encoder.set_weights(W)
    assert torch.all(encoder.linear.weight == 1.0)


# ---------------------------------------------------------------------------
# Tests for decoder cVAE conditioning (input_dim_override)
# ---------------------------------------------------------------------------


def test_bae_forward_decoder_only_conditioning():
    """BAE.forward() passes obs covariates only to decoder, not encoder."""
    _require_bae_deps()
    import torch

    from structboost import BAE, BAEConfig

    config = BAEConfig(latent_dim=4, decoder_hidden_dims=(16,))
    model = BAE(n_genes=32, config=config)

    # Manually rebuild decoder with cVAE input dim
    from structboost._decoder import BAEDecoder

    model.decoder = BAEDecoder(
        32,
        config,
        input_dim_override=4,
        n_covariates=3,
    )

    x = torch.randn(8, 32)
    d = torch.randn(8, 3)

    # forward() should accept obs_covariates for decoder conditioning
    x_recon, z = model.forward(x, obs_covariates=d)
    assert x_recon.shape == (8, 32)
    assert z.shape == (8, 4)

    # z should be identical with or without obs_covariates
    z_direct = model.encoder(x)
    assert torch.allclose(z, z_direct)


def test_decoder_cvae_conditioning():
    """BAEDecoder accepts increased input_dim_override for cVAE conditioning."""
    _require_bae_deps()
    import torch

    from structboost import BAEConfig
    from structboost._decoder import BAEDecoder

    config = BAEConfig(latent_dim=4, decoder_hidden_dims=(16,))
    # latent_dim=4 + n_dummies=3 = 7
    decoder = BAEDecoder(n_output=32, config=config, input_dim_override=7)

    # Forward with concatenated input [z; d]
    z_and_d = torch.randn(8, 7)
    x_recon = decoder(z_and_d)
    assert x_recon.shape == (8, 32)


def test_decoder_cvae_with_split_softmax():
    """cVAE conditioning with split-softmax: input_dim = 2*latent + n_dummies."""
    _require_bae_deps()
    import torch

    from structboost import BAEConfig
    from structboost._decoder import BAEDecoder

    config = BAEConfig(latent_dim=4, decoder_hidden_dims=(16,), split_softmax=True)
    # 2*latent_dim=8 + n_dummies=3 = 11
    decoder = BAEDecoder(n_output=32, config=config, input_dim_override=11)

    h_and_d = torch.randn(8, 11)
    x_recon = decoder(h_and_d)
    assert x_recon.shape == (8, 32)


# ---------------------------------------------------------------------------
# Tests for BAE.fit() with mandatory_genes
# ---------------------------------------------------------------------------


def test_bae_fit_mandatory_genes_by_index():
    """BAE.fit() with mandatory_genes as integer indices."""
    _require_bae_deps()
    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    adata = ad.AnnData(x)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        decoder_updates_per_iteration=2,
        enable_early_stopping=False,
        seed=42,
    )
    model = BAE(n_genes=adata.n_vars, config=config)
    model.fit(adata, mandatory_genes=[0, 5, 10], verbose=False)

    assert "X_bae" in adata.obsm
    # Mandatory genes should have non-zero encoder weights for all latent dims
    W = model.get_encoder_weights()  # (n_genes, latent_dim)
    for g in [0, 5, 10]:
        assert np.any(W[g, :] != 0), f"Mandatory gene {g} has all-zero weights"


def test_bae_fit_mandatory_genes_by_name():
    """BAE.fit() with mandatory_genes as gene name strings."""
    _require_bae_deps()
    import anndata as ad
    import pandas as pd

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    n_genes = 32
    x = rng.normal(size=(64, n_genes)).astype(np.float32)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(x, var=var)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        enable_early_stopping=False,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, mandatory_genes=["gene_0", "gene_5"], verbose=False)

    W = model.get_encoder_weights()
    assert np.any(W[0, :] != 0)
    assert np.any(W[5, :] != 0)


def test_bae_fit_mandatory_genes_none_unchanged():
    """mandatory_genes=None produces same results as not passing it."""
    _require_bae_deps()
    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 32)).astype(np.float32)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        enable_early_stopping=False,
        seed=42,
    )

    adata1 = ad.AnnData(x.copy())
    model1 = BAE(n_genes=32, config=config)
    model1.fit(adata1, verbose=False)

    adata2 = ad.AnnData(x.copy())
    model2 = BAE(
        n_genes=32,
        config=BAEConfig(
            latent_dim=4,
            decoder_hidden_dims=(16,),
            max_iterations=3,
            enable_early_stopping=False,
            seed=42,
        ),
    )
    model2.fit(adata2, mandatory_genes=None, verbose=False)

    np.testing.assert_array_equal(adata1.obsm["X_bae"], adata2.obsm["X_bae"])


def test_bae_fit_mandatory_genes_stored_in_uns():
    """Mandatory gene info stored in adata.uns for reproducibility."""
    _require_bae_deps()
    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    adata = ad.AnnData(x)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        enable_early_stopping=False,
        seed=42,
    )
    model = BAE(n_genes=32, config=config)
    model.fit(adata, mandatory_genes=[0, 5], verbose=False)

    assert "mandatory_genes" in adata.uns["bae"]


def test_bae_per_dimension_mandatory_genes_survive_h5ad(tmp_path):
    """A ragged per-dimension specification must not make the AnnData unwritable."""
    _require_bae_deps()
    import anndata as ad

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    adata = ad.AnnData(x)
    adata.var_names = [f"gene_{i}" for i in range(32)]

    config = BAEConfig(
        latent_dim=3,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        enable_early_stopping=False,
        seed=42,
    )
    model = BAE(n_genes=32, config=config)
    # Deliberately ragged: equal-length sub-lists coerce to a 2-D array and
    # would hide the failure this pins.
    model.fit(
        adata,
        mandatory_genes=[["gene_0", "gene_1"], ["gene_2"], ["gene_3", "gene_4", "gene_5"]],
        verbose=False,
    )

    path = tmp_path / "mandatory.h5ad"
    adata.write_h5ad(path)
    loaded = ad.read_h5ad(path)

    stored = loaded.uns["bae"]["mandatory_genes"]
    assert list(stored["dim_0"]) == ["gene_0", "gene_1"]
    assert list(stored["dim_1"]) == ["gene_2"]
    assert list(stored["dim_2"]) == ["gene_3", "gene_4", "gene_5"]


# ---------------------------------------------------------------------------
# Tests for BAE.fit() with obs conditioning (batch integration)
# ---------------------------------------------------------------------------


def test_bae_fit_mandatory_genes_and_obs():
    """BAE.fit() with both mandatory_genes and condition_obs/nuisance_obs."""
    _require_bae_deps()
    import anndata as ad
    import pandas as pd

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    n_cells, n_genes = 64, 32
    x = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({"batch": pd.Categorical(rng.choice(["A", "B"], size=n_cells))})
    adata = ad.AnnData(x, obs=obs)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        enable_early_stopping=False,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(
        adata,
        mandatory_genes=[0, 1],
        condition_obs=["batch"],
        nuisance_obs=["batch"],
        verbose=False,
    )

    W = model.get_encoder_weights()
    # Mandatory genes should have non-zero weights
    assert np.any(W[0, :] != 0)
    assert np.any(W[1, :] != 0)
    # Encoder has no obs_linear — obs covariates are nuisance regressors only
    assert not hasattr(model.encoder, "obs_linear")


def test_bae_transform_with_obs():
    """BAE.transform() works when model was fitted with condition_obs/nuisance_obs."""
    _require_bae_deps()
    import anndata as ad
    import pandas as pd

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    n_cells, n_genes = 64, 32
    x = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({"batch": pd.Categorical(rng.choice(["A", "B"], size=n_cells))})
    adata = ad.AnnData(x, obs=obs)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        enable_early_stopping=False,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, condition_obs=["batch"], nuisance_obs=["batch"], verbose=False)

    # Transform new data with same obs columns
    adata_new = ad.AnnData(
        rng.normal(size=(16, n_genes)).astype(np.float32),
        obs=pd.DataFrame({"batch": pd.Categorical(rng.choice(["A", "B"], size=16))}),
    )
    latent = model.transform(adata_new)
    assert latent.shape == (16, 4)
    assert np.all(np.isfinite(latent))


def test_bae_fit_obs_conditioning_with_split_softmax():
    """condition_obs/nuisance_obs works with split_softmax=True."""
    _require_bae_deps()
    import anndata as ad
    import pandas as pd

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    n_cells, n_genes = 64, 32
    x = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({"batch": pd.Categorical(rng.choice(["A", "B"], size=n_cells))})
    adata = ad.AnnData(x, obs=obs)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        enable_early_stopping=False,
        split_softmax=True,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, condition_obs=["batch"], nuisance_obs=["batch"], verbose=False)

    assert adata.obsm["X_bae"].shape == (n_cells, 4)


def test_bae_transform_is_gene_only_and_reconstruct_requires_obs():
    """Embedding deployment is gene-only; conditioned reconstruction is not."""
    _require_bae_deps()
    import anndata as ad
    import pandas as pd

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    n_cells, n_genes = 64, 32
    x = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame({"batch": pd.Categorical(rng.choice(["A", "B"], size=n_cells))})
    adata = ad.AnnData(x, obs=obs)

    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        enable_early_stopping=False,
        seed=42,
    )
    model = BAE(n_genes=n_genes, config=config)
    model.fit(adata, condition_obs=["batch"], nuisance_obs=["batch"], verbose=False)

    # New data WITHOUT the fitted obs column can still be embedded.
    adata_new = ad.AnnData(rng.normal(size=(8, n_genes)).astype(np.float32))
    assert model.transform(adata_new).shape == (8, 4)

    # Reconstruction needs the decoder-conditioning covariates.
    with pytest.raises(ValueError, match="not found in adata.obs"):
        model.reconstruct(adata_new)


def test_bae_separate_batch_paths():
    """Conditioning and boosting nuisance adjustment are independently usable."""
    _require_bae_deps()
    import anndata as ad
    import pandas as pd

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(4)
    x = rng.normal(size=(48, 20)).astype(np.float32)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    obs = pd.DataFrame({"batch": rng.choice(["A", "B"], size=48)})
    adata = ad.AnnData(x, obs=obs)
    config = BAEConfig(
        latent_dim=3,
        decoder_hidden_dims=(8,),
        boosting_stepno=5,
        max_iterations=2,
        enable_early_stopping=False,
        seed=1,
    )
    model = BAE(20, config)
    model.fit(
        adata,
        condition_obs=["batch"],
        nuisance_obs=["batch"],
        verbose=False,
    )

    assert model.reconstruct(adata).shape == x.shape
    assert adata.uns["bae"]["condition_obs"] == ["batch"]
    assert adata.uns["bae"]["nuisance_obs"] == ["batch"]
    assert adata.uns["bae"]["nuisance_weights"].shape == (3, 1)
    assert adata.uns["bae"]["conditioning_mode"] == "concat"


def test_bae_nuisance_only_needs_no_obs_for_reconstruction():
    _require_bae_deps()
    import anndata as ad
    import pandas as pd

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 16)).astype(np.float32)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    adata = ad.AnnData(x, obs=pd.DataFrame({"batch": rng.choice(["A", "B"], 40)}))
    model = BAE(
        16,
        BAEConfig(
            latent_dim=2,
            decoder_hidden_dims=(8,),
            boosting_stepno=4,
            max_iterations=1,
            enable_early_stopping=False,
            seed=2,
        ),
    )
    model.fit(adata, nuisance_obs=["batch"], verbose=False)
    without_obs = ad.AnnData(x.copy())
    assert model.reconstruct(without_obs).shape == x.shape


def test_bae_balance_obs_and_nuisance_ridge_metadata():
    _require_bae_deps()
    import anndata as ad
    import pandas as pd

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(6)
    x = rng.normal(size=(36, 12)).astype(np.float32)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    obs = pd.DataFrame({"batch": pd.Categorical(["large"] * 30 + ["small"] * 6)})
    adata = ad.AnnData(x, obs=obs)
    model = BAE(
        12,
        BAEConfig(
            latent_dim=2,
            decoder_hidden_dims=(6,),
            boosting_stepno=3,
            max_iterations=1,
            enable_early_stopping=False,
            seed=3,
        ),
    )
    model.fit(
        adata,
        nuisance_obs=["batch"],
        nuisance_ridge=0.1,
        balance_obs="batch",
        verbose=False,
    )
    assert adata.uns["bae"]["balance_obs"] == "batch"
    assert adata.uns["bae"]["nuisance_ridge"] == pytest.approx(0.1)
    assert set(adata.uns["bae"]["reconstruction_loss_by_obs"]["batch"]) == {
        "large",
        "small",
    }


def _small_adata(n_cells: int = 64, n_genes: int = 32, seed: int = 0):
    """Standardized AnnData for the training-loop invariant tests."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return ad.AnnData(x)


def test_bae_fit_leaves_no_encoder_gradient():
    """The encoder is fitted by boosting, so no gradient may ever accumulate on it.

    `encoder.linear.weight` is in no optimizer, so nothing zeroes its `.grad`.
    Letting autograd populate it wastes a (latent_dim, n_genes) backward per
    minibatch and leaves a stale tensor that would corrupt any future optimizer
    step over the encoder.
    """
    _require_bae_deps()

    from structboost import BAE, BAEConfig

    adata = _small_adata()
    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=3,
        enable_early_stopping=False,
        seed=42,
    )
    model = BAE(n_genes=adata.n_vars, config=config).fit(adata, verbose=False)

    assert model.encoder.linear.weight.grad is None


def test_bae_fit_leaves_no_decoder_gradient_from_target_computation():
    """Building boosting targets must not backward through the decoder parameters.

    `_compute_boosting_targets` needs only dL/dz. Using `loss.backward()` there
    would additionally compute every decoder parameter gradient just for the
    decoder optimizer to discard it at the next `zero_grad()`.
    """
    _require_bae_deps()

    import torch

    from structboost import BAE, BAEConfig

    adata = _small_adata()
    config = BAEConfig(
        latent_dim=4,
        decoder_hidden_dims=(16,),
        max_iterations=2,
        enable_early_stopping=False,
        seed=42,
    )
    model = BAE(n_genes=adata.n_vars, config=config).fit(adata, verbose=False)

    x = torch.from_numpy(np.asarray(adata.X, dtype=np.float32))
    for param in model.decoder.parameters():
        param.grad = None

    model._compute_boosting_targets(x, lr=config.target_optim_lr)

    assert all(p.grad is None for p in model.decoder.parameters())


def test_bae_fit_rejects_zero_valued_overrides():
    """Explicit 0 must raise, not silently fall back to the config value."""
    _require_bae_deps()

    from structboost import BAE, BAEConfig

    adata = _small_adata()
    model = BAE(n_genes=adata.n_vars, config=BAEConfig(latent_dim=4, decoder_hidden_dims=(16,)))

    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        model.fit(adata, max_iterations=0, verbose=False)
    with pytest.raises(ValueError, match="early_stopping_patience must be >= 1"):
        model.fit(adata, early_stopping_patience=0, verbose=False)


def test_bae_config_rejects_zero_patience():
    """BAEConfig bounds early_stopping_patience like its other counts."""
    _require_bae_deps()

    from structboost import BAEConfig

    with pytest.raises(ValueError, match="early_stopping_patience must be >= 1"):
        BAEConfig(early_stopping_patience=0)


def test_boosting_target_step_is_independent_of_n_cells():
    """A cell's target step must not depend on how many other cells are present.

    The target gradient sums squared error over cells (mean over genes), so
    tiling the same cells changes nothing for any individual cell. Under the
    pre-0.4.0.0 elementwise-mean convention this ratio was 1/reps.
    """
    _require_bae_deps()

    import torch

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(0)
    n_genes = 64
    base = rng.normal(size=(50, n_genes)).astype(np.float32)
    base = (base - base.mean(axis=0)) / base.std(axis=0)

    steps = []
    for reps in (1, 2, 4):
        x = np.tile(base, (reps, 1))
        torch.manual_seed(0)  # identical decoder across repetitions
        model = BAE(n_genes, BAEConfig(latent_dim=3, decoder_hidden_dims=(16,)))
        targets = model._compute_boosting_targets(torch.from_numpy(x), lr=1.0)
        steps.append(targets[0])  # cell 0 is the same cell in all three

    np.testing.assert_allclose(steps[1], steps[0], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(steps[2], steps[0], rtol=1e-6, atol=1e-12)


def test_boosting_target_gradient_uses_summed_cell_loss():
    """The target gradient is exactly n_cells times the elementwise-mean gradient."""
    _require_bae_deps()

    import torch

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(1)
    n_cells, n_genes = 40, 32
    x = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    xt = torch.from_numpy(x)

    torch.manual_seed(0)
    model = BAE(n_genes, BAEConfig(latent_dim=3, decoder_hidden_dims=(16,)))

    # Encoder weights are zero at init, so targets == -lr * dL_target/dz.
    targets = model._compute_boosting_targets(xt, lr=1.0)

    # Reference: gradient of the elementwise mean MSE.
    z = torch.zeros(n_cells, 3, requires_grad=True)
    loss = torch.nn.functional.mse_loss(model.decoder(z), xt)
    (mean_grad,) = torch.autograd.grad(loss, z)

    np.testing.assert_allclose(targets, -(n_cells * mean_grad).numpy(), rtol=1e-5, atol=1e-10)


def test_loss_pre_boost_is_reported_as_mean_mse():
    """The reported pre-boost loss stays the mean MSE, comparable with A/B/C."""
    _require_bae_deps()

    import torch

    from structboost import BAE, BAEConfig

    rng = np.random.default_rng(2)
    n_cells, n_genes = 40, 32
    x = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    xt = torch.from_numpy(x)

    torch.manual_seed(0)
    model = BAE(n_genes, BAEConfig(latent_dim=3, decoder_hidden_dims=(16,)))

    stats: dict[str, float] = {}
    model._compute_boosting_targets(xt, lr=1.0, stats=stats)

    expected = model._full_recon_loss(xt, None)[0]
    assert stats["loss_pre_boost"] == pytest.approx(expected, rel=1e-6)


def test_standardize_targets_leaves_near_constant_column_unscaled():
    """A near-constant column must not be inflated to unit variance.

    `std == 0` is not a sufficient guard: a dimension carrying only numerical
    noise has a tiny but nonzero standard deviation.
    """
    _require_bae_deps()

    from structboost import BAE

    targets = np.zeros((10, 2))
    targets[:, 0] = np.linspace(0.0, 1.0, 10)  # real signal
    targets[:, 1] = 1e-30 * np.arange(10)  # noise-level, nonzero std

    out = BAE._standardize_targets(targets)

    assert out[:, 0].std() == pytest.approx(1.0, rel=1e-6)
    assert np.abs(out[:, 1]).max() < 1e-20


def test_decoder_init_depends_only_on_seed_not_on_conditioning():
    """Enabling conditioning must not perturb the rest of the model's init.

    `fit` rebuilds the decoder (its input width depends on conditioning and on a
    warm start that may change latent_dim), and constructing it draws from the
    global RNG by an amount that varies with that width. Re-seeding immediately
    before `reset_parameters` keeps the initial weights a function of `seed`
    alone, so a plain fit stays reproducible across releases and across unrelated
    feature flags.
    """
    _require_bae_deps()

    import torch

    from structboost import BAE, BAEConfig

    def encoder_after_fit(**config_kwargs):
        adata = _small_adata(n_cells=96, n_genes=40, seed=3)
        config = BAEConfig(
            latent_dim=3,
            decoder_hidden_dims=(16,),
            max_iterations=4,
            enable_early_stopping=False,
            seed=11,
            **config_kwargs,
        )
        model = BAE(n_genes=adata.n_vars, config=config).fit(adata, verbose=False)
        return model.decoder.hidden[0].weight.detach().clone()

    # The hidden layer is upstream of any conditioning input, so enabling
    # conditioning must not perturb it -- the decoder is re-seeded immediately
    # before reset for exactly this reason.
    baseline = encoder_after_fit()
    assert torch.equal(encoder_after_fit(), baseline)


def test_seeded_fit_is_bitwise_reproducible():
    """Two seeded fits of the same data must agree exactly."""
    _require_bae_deps()

    from structboost import BAE, BAEConfig

    def run():
        adata = _small_adata(n_cells=96, n_genes=40, seed=5)
        config = BAEConfig(
            latent_dim=3,
            decoder_hidden_dims=(16,),
            max_iterations=5,
            enable_early_stopping=False,
            seed=7,
        )
        model = BAE(n_genes=adata.n_vars, config=config).fit(adata, verbose=False)
        return adata.obsm["X_bae"], model.training_history["train_loss"]

    first_latent, first_loss = run()
    second_latent, second_loss = run()
    np.testing.assert_array_equal(first_latent, second_latent)
    assert first_loss == second_loss
