"""Design-guided gene selection and the exact recon/design weight attribution.

Skipped automatically if the BAE dependencies (torch, anndata) are missing;
the ``allboost`` replay tests need NumPy only.
"""

from __future__ import annotations

import numpy as np
import pytest

from structboost import BAEConfig, allboost
from structboost._utils import disentangle_boosting_targets, latent_r2_per_dim


def _require_bae():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")


def _standardized(rng, n, p, dtype):
    X = rng.standard_normal((n, p))
    return ((X - X.mean(0)) / X.std(0)).astype(dtype)


# --- allboost: path replay ----------------------------------------------------


@pytest.mark.parametrize("dtype, rtol", [(np.float64, 1e-10), (np.float32, 1e-4)])
def test_followers_sum_to_their_leader(dtype, rtol):
    """Boosting is linear in its target once the path is fixed: replaying the
    path selected for ``A + B`` on ``A`` and on ``B`` gives coefficients that sum
    to the joint ones. This is the identity the BAE attribution rests on. It
    holds with the mandatory pre-step and a ridge too."""
    rng = np.random.default_rng(0)
    n, p, k = 300, 40, 3
    X = _standardized(rng, n, p, dtype)
    A = rng.standard_normal((n, k)).astype(dtype)
    B = (0.3 * rng.standard_normal((n, k))).astype(dtype)
    kw = dict(stepno=25, mandatory_features=np.array([0, 5]), mandatory_ridge=1e-3)

    beta, hist = allboost(
        X,
        np.hstack([A + B, A, B]),
        selection_from=np.tile(np.arange(k), 3),
        return_history="steps",
        **kw,
    )
    plain = allboost(X, A + B, **kw)

    np.testing.assert_array_equal(beta[:k], plain)  # leader unaffected by followers
    np.testing.assert_allclose(beta[k : 2 * k] + beta[2 * k :], beta[:k], rtol=rtol, atol=0)
    assert hist.beta_path is None
    # The applied increments account for every non-mandatory coefficient.
    free = np.setdiff1d(np.arange(p), [0, 5])
    np.testing.assert_allclose(hist.update.sum(axis=1), beta[:, free].sum(axis=1), rtol=1e-10)
    # Followers record their own would-be pick, which differs somewhere from
    # the path they were made to follow.
    assert (hist.selection[k : 2 * k] != hist.selection[:k]).any()
    assert (hist.selection >= 0).all()


def test_replay_reproduces_a_single_target_exactly():
    """A follower carrying the leader's whole target is the leader, bitwise."""
    rng = np.random.default_rng(1)
    X = _standardized(rng, 200, 30, np.float64)
    T = rng.standard_normal((200, 2))
    beta, hist = allboost(
        X, np.hstack([T, T]), selection_from=np.array([0, 1, 0, 1]), return_history=True
    )
    np.testing.assert_array_equal(beta[2:], beta[:2])
    np.testing.assert_array_equal(hist.selection[2:], hist.selection[:2])
    assert hist.beta_path.shape == (4, 20, 30)


def test_selection_from_validation():
    rng = np.random.default_rng(2)
    X = _standardized(rng, 100, 10, np.float64)
    T = rng.standard_normal((100, 2))
    with pytest.raises(ValueError, match="shape"):
        allboost(X, T, selection_from=np.array([0]))
    with pytest.raises(ValueError, match="before"):
        allboost(X, T, selection_from=np.array([1, 1]))
    with pytest.raises(ValueError, match="independent"):
        allboost(X, T, selection_from=np.array([0, 0]), independent=False)
    with pytest.raises(ValueError, match="mandatory_features differ"):
        allboost(
            X,
            T,
            selection_from=np.array([0, 0]),
            mandatory_features=[np.array([1]), np.array([2])],
        )
    with pytest.raises(ValueError, match="return_history"):
        allboost(X, T, return_history="all")


# --- helpers ------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 0.4])
def test_orthogonalization_map_keeps_components_additive(alpha):
    rng = np.random.default_rng(3)
    T = rng.standard_normal((400, 5)).astype(np.float32)
    A = (0.7 * T + 0.1 * rng.standard_normal(T.shape)).astype(np.float32)
    B = T - A
    reference = disentangle_boosting_targets(T, alpha=alpha)
    total, (a, b) = disentangle_boosting_targets(T, alpha=alpha, components=[A, B])
    np.testing.assert_allclose(total, reference, atol=1e-6)
    np.testing.assert_allclose(a + b, total, atol=1e-5)
    assert a.dtype == np.float32


def test_latent_r2_per_dim_is_the_between_group_share():
    rng = np.random.default_rng(4)
    groups = rng.integers(0, 3, size=600)
    design = np.column_stack([(groups == 1), (groups == 2)]).astype(float)
    design = (design - design.mean(0)) / design.std(0)
    Z = np.column_stack([groups * 1.0, rng.standard_normal(600)])
    r2 = latent_r2_per_dim(design, Z)
    assert r2[0] == pytest.approx(1.0)
    assert abs(r2[1]) < 0.05


def test_design_lambda_is_validated():
    with pytest.raises(ValueError, match="design_lambda"):
        BAEConfig(design_lambda=-0.1)
    with pytest.raises(ValueError, match="design_lambda"):
        BAEConfig(design_lambda=float("nan"))


# --- BAE.fit(design_key=...) --------------------------------------------------


def _planted(seed=0, n=600, p=80):
    """Standardized data with 8 condition genes and one cell-type programme."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    cond = np.arange(n) % 2
    X[:, :8] += 1.2 * (cond[:, None] - 0.5)
    X[:, 20:30] += 1.5 * ((np.arange(n) // 200)[:, None] == 1)
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"g{i}" for i in range(p)]
    adata.obs["cond"] = pd.Categorical(np.where(cond == 1, "ko", "wt"))
    adata.obs["shuffled"] = pd.Categorical(rng.permutation(adata.obs["cond"].to_numpy()))
    return adata


def _fit(adata, **kwargs):
    from structboost import BAE

    config = kwargs.pop("config", {})
    model = BAE(
        adata.n_vars,
        BAEConfig(latent_dim=3, max_iterations=25, seed=0, boosting_stepno=10, **config),
    )
    model.fit(adata, verbose=False, **kwargs)
    return model


def test_attribution_is_exact_and_belongs_to_the_restored_iteration():
    _require_bae()
    adata = _planted()
    _fit(adata, design_key="cond", config=dict(design_lambda=0.5))

    W = adata.varm["BAE_encoder_weights"]
    recon = adata.varm["BAE_encoder_weights_recon"]
    design = adata.varm["BAE_encoder_weights_design"]
    assert recon.dtype == np.float64
    # Exact to the encoder's own float32 rounding: the parts are float64 and
    # their sum *is* the float64 matrix the encoder was cast from.
    assert adata.uns["bae"]["design_dims"].tolist() == [0]
    assert adata.uns["bae"]["design_key"] == ["cond"]
    assert adata.uns["bae"]["design_lambda"] == {"cond": 0.5}

    carry = adata.varm["BAE_encoder_weights_carry"]
    np.testing.assert_array_equal((carry + recon + design).astype(np.float32), W)
    trace = adata.uns["bae"]["selection_trace"]
    assert trace["gene"].shape == (3, 10)
    assert set(trace["delta"]) == {"total", "carry", "recon", "design"}
    assert set(trace["counterfactual_gene"]) == {"no_design", "recon", "design"}
    assert set(trace["rank"]) == {"no_design", "recon", "design"}
    for key in ("fit_score", "alignment", "gain"):
        assert set(trace[key]) == {"total", "no_design", "recon", "design"}
    # The applied update is each total's own first choice, so it fits its own
    # residual best of all candidates and points along it.
    assert (trace["fit_score"]["total"] <= 1.0).all()
    assert (trace["fit_score"]["total"] > 0).all()
    assert (np.abs(trace["alignment"]["total"]) <= 1.0 + 1e-12).all()
    assert (trace["gain"]["total"] > 0).all()
    # A step the design part decided is one where the no-design target ranked the
    # applied gene below its own first choice.
    decided = trace["counterfactual_gene"]["no_design"] != trace["gene"]
    assert (trace["rank"]["no_design"][decided] > 0).all()
    assert (trace["rank"]["no_design"][~decided] == 0).all()
    np.testing.assert_allclose(
        trace["delta"]["carry"] + trace["delta"]["recon"] + trace["delta"]["design"],
        trace["delta"]["total"],
        atol=1e-12,
    )
    # The per-step increments of dimension 0 sum to its final coefficients.
    np.testing.assert_allclose(
        trace["delta"]["total"][0].sum(), W[:, 0].astype(np.float64).sum(), rtol=1e-5
    )
    # The design part of the target lives on the constrained dimension; what
    # reaches the others is the Löwdin mixing, an order of magnitude smaller.
    assert trace["target_norm"]["design"][0] > 10 * trace["target_norm"]["design"][1]

    # The stored attribution describes the *restored* encoder. Force a
    # restoration by continuing with a losing iteration: early stopping with
    # patience 1 keeps the best iteration, not the last.
    adata2 = _planted()
    model2 = _fit(
        adata2,
        design_key="cond",
        enable_early_stopping=True,
        early_stopping_patience=1,
        config=dict(design_lambda=0.5),
    )
    parts2 = model2._encoder_components
    np.testing.assert_array_equal(
        sum(parts2.values()).T.astype(np.float32), adata2.varm["BAE_encoder_weights"]
    )


def test_design_term_separates_the_condition_and_selects_its_genes():
    _require_bae()
    off = _planted()
    on = _planted()
    _fit(off, design_key="cond", config=dict(design_lambda=0.0))
    _fit(on, design_key="cond", config=dict(design_lambda=1.0))

    r2_off = off.uns["bae"]["latent_design_r2_per_dim"][0]
    r2_on = on.uns["bae"]["latent_design_r2_per_dim"][0]
    assert r2_on > r2_off + 0.3
    # With lambda=0 the design part is rounding noise, and the fit is the plain one.
    assert (
        np.abs(off.varm["BAE_encoder_weights_design"]).max()
        < 1e-6 * np.abs(off.varm["BAE_encoder_weights"]).max() + 1e-9
    )
    plain = _planted()
    _fit(plain)
    np.testing.assert_array_equal(
        plain.varm["BAE_encoder_weights"], off.varm["BAE_encoder_weights"]
    )
    assert "BAE_encoder_weights_design" not in plain.varm
    assert "selection_trace" not in plain.uns["bae"]

    selected = set(np.flatnonzero(on.varm["BAE_encoder_weights"][:, 0]))
    assert len(selected & set(range(8))) >= 6
    # The design step is a *filter*: on the genes that win, its own additive
    # share is a shrinkage, and the selection effect shows in the counterfactual.
    trace = on.uns["bae"]["selection_trace"]
    assert (trace["counterfactual_gene"]["no_design"][0] != trace["gene"][0]).any()
    assert trace["target_norm"]["design"][0] > 0.5 * trace["target_norm"]["total"][0]


def test_design_part_is_zero_off_the_constrained_dims_without_orthogonalization():
    """Under Löwdin orthogonalization the design pull legitimately reaches the
    other dimensions through the column mixing; without it, it cannot."""
    _require_bae()
    adata = _planted()
    _fit(adata, design_key={"cond": [1]}, config=dict(disentanglement="none"))
    design = adata.varm["BAE_encoder_weights_design"]
    assert np.abs(design[:, [0, 2]]).max() < 1e-6 * max(np.abs(design).max(), 1e-12) + 1e-9
    assert np.abs(design[:, 1]).max() > 0


def test_default_dims_follow_the_design_rank_and_arguments_are_validated():
    _require_bae()
    import pandas as pd

    adata = _planted()
    adata.obs["tp"] = pd.Categorical(np.arange(adata.n_obs) % 4)  # 3 encoded columns
    _fit(adata, design_key="tp")
    assert adata.uns["bae"]["design_dims"].tolist() == [0, 1, 2]
    _fit(adata, design_key="cond")
    assert adata.uns["bae"]["design_dims"].tolist() == [0]

    with pytest.raises(ValueError, match="disjoint"):
        _fit(adata, design_key={"cond": [3]})
    with pytest.raises(ValueError, match="disjoint"):
        _fit(adata, design_key={"cond": [0, 0]})
    with pytest.raises(ValueError, match="init_pca"):
        _fit(adata, design_key="cond", init_pca=True)
    with pytest.raises(ValueError, match="boosting_independent"):
        _fit(adata, design_key="cond", config=dict(boosting_independent=False))
    with pytest.raises(ValueError, match="exceeds 1"):
        _fit(adata, design_key="cond", config=dict(design_lambda=1.5))


def test_correlation_mode_gets_its_own_component():
    _require_bae()
    adata = _planted()
    _fit(adata, design_key="cond", config=dict(disentanglement="correlation"))
    keys = {k for k in adata.varm if k.startswith("BAE_encoder_weights_")}
    assert keys == {
        "BAE_encoder_weights_carry",
        "BAE_encoder_weights_recon",
        "BAE_encoder_weights_correlation",
        "BAE_encoder_weights_design",
    }
    total = sum(adata.varm[k] for k in keys)
    np.testing.assert_array_equal(total.astype(np.float32), adata.varm["BAE_encoder_weights"])
    assert set(adata.uns["bae"]["selection_trace"]["counterfactual_gene"]) == {
        "no_design",
        "recon",
        "correlation",
        "design",
    }


def test_design_key_with_batch_key_and_split_softmax():
    _require_bae()
    import pandas as pd

    adata = _planted()
    adata.obs["batch"] = pd.Categorical(np.arange(adata.n_obs) % 3)
    _fit(adata, design_key="cond", batch_key="batch", config=dict(split_softmax=True))
    assert "latent_obs_r2_per_dim" in adata.uns["bae"]
    assert "latent_design_r2_per_dim" in adata.uns["bae"]
    parts = [v for k, v in adata.varm.items() if k.startswith("BAE_encoder_weights_")]
    assert len(parts) == 3
    np.testing.assert_array_equal(sum(parts).astype(np.float32), adata.varm["BAE_encoder_weights"])


def test_transfer_models_reject_design_key():
    _require_bae()
    from structboost import BAE

    adata = _planted()
    reference = _fit(adata)
    model = BAE.from_reference(reference, adata, n_additional_dims=1)
    with pytest.raises(ValueError, match="prior encoder matrix"):
        model.fit(adata, design_key="cond", verbose=False)


def test_stability_selection_continues_the_design_guided_alternation():
    """The stability loop shares the boosting step with `fit`, so the design
    term is applied there too: continuing a design-guided fit keeps the design
    R² of the constrained dimension where the plain continuation does not."""
    _require_bae()
    on = _planted()
    model = _fit(on, design_key="cond", config=dict(design_lambda=1.0))
    frequency_on = model.stability_selection(on, n_runs=5, seed=0, verbose=False).frequency
    off = _planted()
    plain = _fit(off, design_key="cond", config=dict(design_lambda=0.0))
    frequency_off = plain.stability_selection(off, n_runs=5, seed=0, verbose=False).frequency
    # The condition genes keep being selected in dimension 0 only under guidance.
    assert frequency_on[:8, 0].mean() > frequency_off[:8, 0].mean() + 0.3


def test_design_state_survives_save_and_load(tmp_path):
    _require_bae()
    from structboost import BAE

    adata = _planted()
    model = _fit(adata, design_key="cond")
    path = model.save(tmp_path / "design.pt")
    loaded = BAE.load(path)
    assert {v: d.tolist() for v, d in loaded._design_blocks.items()} == {"cond": [0]}
    assert loaded._design_lambdas == {"cond": 1.0}
    for name, part in model._encoder_components.items():
        np.testing.assert_array_equal(loaded._encoder_components[name], part)
    np.testing.assert_array_equal(
        loaded._selection_trace["delta"]["design"], model._selection_trace["delta"]["design"]
    )
    np.testing.assert_array_equal(loaded.transform(adata.copy()), model.transform(adata.copy()))
    # A loaded model continues the guided alternation as well.
    loaded.stability_selection(adata, n_runs=1, seed=0, verbose=False)

    # Stratified form and the path log round-trip too.
    import pandas as pd

    adata.obs["stratum"] = pd.Categorical(np.where(np.arange(adata.n_obs) < 200, "a", "b"))
    model = _fit(adata, design_key="cond", design_within="stratum", track_selection_path=True)
    loaded = BAE.load(model.save(tmp_path / "within.pt"))
    assert loaded._design_within == ["stratum"]
    np.testing.assert_array_equal(loaded._selection_path["gene"], model._selection_path["gene"])
    np.testing.assert_array_equal(
        loaded._selection_trace["fit_score"]["design"],
        model._selection_trace["fit_score"]["design"],
    )
    b = adata.copy()
    loaded.stability_selection(b, n_runs=1, seed=0, verbose=False)


def test_design_readout_survives_h5ad(tmp_path):
    _require_bae()
    import anndata as ad

    adata = _planted()
    _fit(adata, design_key="cond", decompose_key="cond")
    adata.write_h5ad(tmp_path / "design.h5ad")
    back = ad.read_h5ad(tmp_path / "design.h5ad")
    assert list(back.varm["BAE_residual_variance_share"].columns) == [
        "between_cond",
        "mean",
        "within",
    ]
    np.testing.assert_array_equal(
        back.uns["bae"]["selection_trace"]["gene"], adata.uns["bae"]["selection_trace"]["gene"]
    )
    np.testing.assert_array_equal(
        back.varm["BAE_encoder_weights_design"], adata.varm["BAE_encoder_weights_design"]
    )


# --- plots --------------------------------------------------------------------


def test_attribution_panel_and_trace_plot():
    _require_bae()
    pytest.importorskip("matplotlib").use("Agg")
    from structboost import plot_latent_dimensions, plot_selection_trace

    adata = _planted()
    _fit(adata, design_key="cond")
    fig, axes = plot_latent_dimensions(
        adata, dims=[0, 1], panels=("attribution", "groups"), group_by="cond"
    )
    assert axes.shape == (2, 2)
    fig, axes = plot_selection_trace(adata, dims=[0])
    assert axes.shape == (1,)

    plain = _planted()
    _fit(plain)
    with pytest.raises(KeyError, match="design_key"):
        plot_latent_dimensions(plain, panels=("attribution",))
    with pytest.raises(KeyError, match="selection_trace"):
        plot_selection_trace(plain)


def test_design_within_keeps_the_effect_inside_strata():
    """With strata, a constrained dimension separates conditions inside each cell
    type; the cell-type main effect is removed rather than kept. Planted: the
    condition shift is the same in both strata, so the within-stratum R² should be
    high while the dimension does not separate the strata themselves."""
    _require_bae()
    import pandas as pd

    adata = _planted()
    adata.obs["stratum"] = pd.Categorical(np.where(np.arange(adata.n_obs) < 200, "a", "b"))
    _fit(adata, design_key="cond", design_within="stratum", config=dict(design_lambda=1.0))
    assert adata.uns["bae"]["design_within"] == ["stratum"]
    r2 = adata.uns["bae"]["latent_design_r2_per_dim"]
    assert r2[0] > 0.5
    z0 = adata.obsm["X_bae"][:, 0]
    strata = adata.obs["stratum"].to_numpy()
    between_strata = abs(z0[strata == "a"].mean() - z0[strata == "b"].mean())
    cond = adata.obs["cond"].to_numpy()
    between_cond = abs(z0[cond == "ko"].mean() - z0[cond == "wt"].mean())
    assert between_cond > 3 * between_strata
    with pytest.raises(ValueError, match="without a design_key"):
        _fit(adata, design_within="stratum")
    with pytest.raises(ValueError, match="not found"):
        _fit(adata, design_key="cond", design_within="nope")
    with pytest.raises(ValueError, match="different columns"):
        _fit(adata, design_key="cond", design_within="cond")


def test_track_selection_path_logs_every_iteration_without_changing_the_fit():
    _require_bae()
    plain = _planted()
    logged = _planted()
    _fit(plain)
    _fit(logged, track_selection_path=True)
    np.testing.assert_array_equal(
        plain.varm["BAE_encoder_weights"], logged.varm["BAE_encoder_weights"]
    )
    path = logged.uns["bae"]["selection_path"]
    assert path["gene"].shape == (25, 3, 10)
    assert path["update"].shape == (25, 3, 10)
    assert "selection_path" not in plain.uns["bae"]
    # With a design key the path and the trace describe the same iterations.
    guided = _planted()
    model = _fit(guided, design_key="cond", track_selection_path=True)
    trace = guided.uns["bae"]["selection_trace"]
    path = guided.uns["bae"]["selection_path"]
    assert any(np.array_equal(path["gene"][i], trace["gene"]) for i in range(path["gene"].shape[0]))
    assert model._selection_path is not None


# --- reconstruction-gradient decomposition and per-variable blocks ------------


def _planted2(seed=0, n=600, p=80):
    """Two orthogonal planted variables (condition, sex) and a cell-type programme."""
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    cond, sex, ct = np.arange(n) % 2, (np.arange(n) // 2) % 2, np.arange(n) // 200
    X[:, :8] += 1.2 * (cond[:, None] - 0.5)
    X[:, 8:14] += 1.2 * (sex[:, None] - 0.5)
    X[:, 20:30] += 1.5 * (ct[:, None] == 1)
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"g{i}" for i in range(p)]
    adata.obs["cond"] = pd.Categorical(np.where(cond == 1, "ko", "wt"))
    adata.obs["sex"] = pd.Categorical(np.where(sex == 1, "f", "m"))
    adata.obs["ct"] = pd.Categorical(ct.astype(str))
    return adata


def _parts_sum(adata):
    parts = adata.uns["bae"]["attribution_parts"]
    return sum(adata.varm[f"BAE_encoder_weights_{p}"] for p in parts)


@pytest.mark.parametrize("mode", ["orthogonal", "correlation"])
def test_decomposition_is_exact_and_leaves_the_fit_bitwise_unchanged(mode):
    """`decompose_key` only *reads* the reconstruction gradient: the encoder is
    the plain encoder bitwise, the parts add up to it, and each gene's residual
    sum of squares is split into shares that sum to one."""
    _require_bae()
    config = dict(disentanglement=mode)
    plain = _planted2()
    _fit(plain, config=config)
    W = plain.varm["BAE_encoder_weights"]

    single = _planted2()
    _fit(single, decompose_key="cond", config=config)
    np.testing.assert_array_equal(single.varm["BAE_encoder_weights"], W)
    parts = single.uns["bae"]["attribution_parts"]
    expected = {"between_cond", "mean", "within", "carry"}
    expected |= {"correlation"} if mode == "correlation" else set()
    assert set(parts) == expected
    np.testing.assert_allclose(_parts_sum(single), W, atol=1e-6)
    shares = single.varm["BAE_residual_variance_share"]
    assert list(shares.columns) == ["between_cond", "mean", "within"]
    np.testing.assert_allclose(shares.sum(axis=1), 1.0, atol=1e-5)
    assert "BAE_encoder_weights_design" not in single.varm
    assert "design_key" not in single.uns["bae"]
    assert single.uns["bae"]["decompose_key"] == ["cond"]

    multi = _planted2()
    _fit(multi, decompose_key=["cond", "sex"], decompose_within="ct", config=config)
    np.testing.assert_array_equal(multi.varm["BAE_encoder_weights"], W)
    parts = multi.uns["bae"]["attribution_parts"]
    assert {"between_cond", "between_sex", "shared", "strata", "within"} <= set(parts)
    assert "mean" not in parts
    np.testing.assert_allclose(_parts_sum(multi), W, atol=1e-6)
    np.testing.assert_allclose(multi.varm["BAE_residual_variance_share"].sum(axis=1), 1, atol=1e-5)
    assert multi.uns["bae"]["decompose_within"] == ["ct"]
    trace = multi.uns["bae"]["selection_trace"]
    assert set(trace["counterfactual_gene"]) == expected - {"mean", "carry"} | {
        "between_sex",
        "shared",
        "strata",
    }
    assert "carry" not in trace["rank"] and "carry" in trace["delta"]


def test_decomposition_counterfactuals_find_the_planted_programmes():
    """Each between part alone would pick genes of its own variable, the strata
    part the cell-type programme; two orthogonal variables share nothing."""
    _require_bae()
    adata = _planted2()
    _fit(adata, decompose_key=["cond", "sex"], decompose_within="ct")
    counter = adata.uns["bae"]["selection_trace"]["counterfactual_gene"]
    assert (counter["between_cond"] < 8).mean() > 0.8
    assert ((counter["between_sex"] >= 8) & (counter["between_sex"] < 14)).mean() > 0.8
    assert ((counter["strata"] >= 20) & (counter["strata"] < 30)).mean() > 0.8
    col = adata.varm["BAE_residual_variance_share"]
    assert col["between_cond"].iloc[:8].min() > 0.1 > col["between_cond"].iloc[8:].max()
    assert col["between_sex"].iloc[8:14].min() > 0.1 > col["between_sex"].iloc[14:].max()
    assert np.abs(col["shared"]).max() < 0.02
    # Every part scores the pick the leader made: on a step the plain fit
    # spends on a condition gene, between_cond ranks it among its eight.
    trace = adata.uns["bae"]["selection_trace"]
    cond_steps = trace["gene"] < 8
    assert cond_steps.any()
    assert (trace["rank"]["between_cond"][cond_steps] < 8).all()


def test_design_blocks_list_and_dict_forms_and_per_variable_lambda():
    _require_bae()
    from structboost._utils import latent_r2_per_dim

    adata = _planted2()
    _fit(adata, design_key=["cond", "sex"])
    uns = adata.uns["bae"]
    assert {v: d.tolist() for v, d in uns["design_blocks"].items()} == {"cond": [0], "sex": [1]}
    assert uns["design_lambda"] == {"cond": 1.0, "sex": 1.0}
    assert uns["design_dims"].tolist() == [0, 1]
    assert uns["latent_design_r2_per_dim"][0] > 0.5 and uns["latent_design_r2_per_dim"][1] > 0.5
    # Each block carries its own variable, not the other one.
    Z = adata.obsm["X_bae"]
    sex = (adata.obs["sex"] == "f").to_numpy(dtype=float)[:, None]
    cond = (adata.obs["cond"] == "ko").to_numpy(dtype=float)[:, None]
    assert latent_r2_per_dim(sex, Z)[0] < 0.1 < latent_r2_per_dim(sex, Z)[1]
    assert latent_r2_per_dim(cond, Z)[1] < 0.1 < latent_r2_per_dim(cond, Z)[0]

    _fit(adata, design_key={"cond": [2], "sex": [0]}, config=dict(design_lambda={"sex": 0.0}))
    uns = adata.uns["bae"]
    assert {v: d.tolist() for v, d in uns["design_blocks"].items()} == {"cond": [2], "sex": [0]}
    assert uns["design_lambda"] == {"cond": 1.0, "sex": 0.0}
    design = adata.varm["BAE_encoder_weights_design"]
    assert np.abs(design[:, 0]).max() == 0.0  # lambda 0: no design step in that block
    assert np.abs(design[:, 1]).max() == 0.0  # unconstrained dimension
    assert np.abs(design[:, 2]).max() > 0.0
    assert uns["latent_design_r2_per_dim"][2] > 0.5

    with pytest.raises(ValueError, match="disjoint"):
        _fit(adata, design_key={"cond": [0], "sex": [0, 1]})
    with pytest.raises(ValueError, match="disjoint"):
        _fit(adata, design_key={"cond": [0], "sex": [3]})
    with pytest.raises(ValueError, match="not in design_key"):
        _fit(adata, design_key="cond", config=dict(design_lambda={"sex": 0.5}))
    with pytest.raises(ValueError, match="'sex'"):
        _fit(adata, design_key=["cond", "sex"], config=dict(design_lambda={"sex": 1.5}))
    with pytest.raises(ValueError, match="design_lambda"):
        BAEConfig(design_lambda={"cond": -1.0})


def test_design_key_and_decompose_key_combine():
    _require_bae()
    adata = _planted2()
    _fit(adata, design_key="cond", decompose_key=["cond", "sex"])
    parts = adata.uns["bae"]["attribution_parts"]
    assert set(parts) == {
        "between_cond",
        "between_sex",
        "shared",
        "mean",
        "within",
        "carry",
        "design",
    }
    np.testing.assert_allclose(_parts_sum(adata), adata.varm["BAE_encoder_weights"], atol=1e-6)
    counter = adata.uns["bae"]["selection_trace"]["counterfactual_gene"]
    assert "no_design" in counter and "design" in counter
    with pytest.raises(ValueError, match="decompose_within"):
        _fit(adata, decompose_within="ct")
    with pytest.raises(ValueError, match="different columns"):
        _fit(adata, decompose_key="cond", decompose_within="cond")


def test_blocks_and_decomposition_survive_save_load_and_stability(tmp_path):
    _require_bae()
    from structboost import BAE

    adata = _planted2()
    model = _fit(
        adata,
        design_key={"cond": [2], "sex": [0]},
        decompose_key=["cond", "sex"],
        decompose_within="ct",
        config=dict(design_lambda={"sex": 0.5}),
    )
    loaded = BAE.load(model.save(tmp_path / "blocks.pt"))
    assert {v: d.tolist() for v, d in loaded._design_blocks.items()} == {"cond": [2], "sex": [0]}
    assert loaded._design_lambdas == {"cond": 1.0, "sex": 0.5}
    assert loaded._decompose_columns == ["cond", "sex"]
    assert loaded._decompose_within == ["ct"]
    for name, part in model._encoder_components.items():
        np.testing.assert_array_equal(loaded._encoder_components[name], part)
    np.testing.assert_array_equal(loaded.transform(adata.copy()), model.transform(adata.copy()))
    # Stability continues the block-guided alternation; the condition block is dim 2.
    result = loaded.stability_selection(adata, n_runs=3, seed=0, verbose=False)
    assert result.frequency[:8, 2].mean() > result.frequency[:8, 0].mean()


def test_trace_and_path_plots_take_a_decided_by_part():
    _require_bae()
    pytest.importorskip("matplotlib").use("Agg")
    from structboost import plot_selection_paths, plot_selection_trace

    adata = _planted2()
    _fit(adata, decompose_key=["cond", "sex"])
    fig, axes = plot_selection_trace(adata, dims=[0])  # defaults to "within"
    assert axes.shape == (1,)
    plot_selection_trace(adata, dims=[0, 1], decided_by="between_cond")
    fig, axes = plot_selection_paths(adata, dims=[0], decided_by="between_sex")
    assert axes.shape == (1,)
    with pytest.raises(ValueError, match="decided_by"):
        plot_selection_paths(adata, decided_by="design")
