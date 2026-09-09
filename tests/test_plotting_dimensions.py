"""Tests for the per-dimension readout plots and the quantities behind them.

The figures themselves are not asserted pixel by pixel -- that pins style rather
than behaviour. What is pinned here is everything the docstrings *claim*: the
variance decomposition is exact, the caller's AnnData is never mutated, the
palette tiering is what it says, and every guard fires with a message that names
the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from structboost import gene_variance_shares, palette_audit


def _require_plotting():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")


def _require_fit():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")
    _require_plotting()


def _fitted(latent_dim=3, n_iter=8, seed=0):
    """A small real fit, so the encoder is genuinely sparse and signed."""
    from structboost import BAE, BAEConfig, sim_scrnaseq_anndata

    adata = sim_scrnaseq_anndata(
        n=150, n_genes=60, stageno=4, stagep=8, stageoverlap=2, hierarchy=None, seed=seed
    )
    model = BAE(
        adata.n_vars,
        BAEConfig(latent_dim=latent_dim, boosting_stepno=10, seed=seed),
    )
    model.fit(adata, max_iterations=n_iter, verbose=False)
    adata.obs["group"] = np.where(np.arange(adata.n_obs) % 3 == 0, "a", "b")
    return adata


# --- the variance decomposition --------------------------------------------


def test_gene_variance_shares_sum_to_one():
    """The identity the gene ranking rests on: Var(s) = sum_g w_g Cov(X_g, s)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 12))
    w = rng.normal(size=12)
    shares = gene_variance_shares(X, w, X @ w)
    assert shares.sum() == pytest.approx(1.0, abs=1e-10)


def test_gene_variance_shares_ignore_genes_outside_the_score():
    """A gene with zero weight contributes exactly nothing, however it varies."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 6))
    w = np.array([1.0, -2.0, 0.0, 0.5, 0.0, 3.0])
    shares = gene_variance_shares(X, w, X @ w)
    assert shares[2] == 0.0
    assert shares[4] == 0.0
    assert shares.sum() == pytest.approx(1.0, abs=1e-10)


def test_gene_variance_shares_is_a_magnitude_not_a_direction():
    """A negative-weight gene still earns a positive share.

    `share = w * Cov(X_g, s)`, and a negative-weight gene is anti-correlated with
    the score, so the product is positive either way. Direction lives in the sign
    of the weight -- which is why the plots colour gene names by `w` and print the
    share without a sign.
    """
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 4))
    w = np.array([-1.0, -2.0, -0.5, -3.0])
    shares = gene_variance_shares(X, w, X @ w)
    assert (w < 0).all()
    assert (shares > 0).all()


def test_gene_variance_shares_handles_a_dead_dimension():
    """A dimension with no variance has no shares to compute, and must not divide."""
    X = np.random.default_rng(3).normal(size=(50, 4))
    shares = gene_variance_shares(X, np.zeros(4), np.zeros(50))
    assert np.all(shares == 0.0)


# --- palettes ---------------------------------------------------------------


def test_palette_audit_reports_the_documented_fields():
    audit = palette_audit("tab5")
    assert set(audit) == {
        "n",
        "min_de_normal",
        "min_de_cvd",
        "min_contrast",
        "n_below_contrast_floor",
    }
    assert audit["n"] == 5


def test_palette_tiers_have_the_documented_sizes():
    from structboost._plotting import _palette_for

    assert len(_palette_for(2)) == 5
    assert len(_palette_for(5)) == 5
    assert len(_palette_for(6)) == 10
    assert len(_palette_for(10)) == 10
    assert len(_palette_for(11)) == 20
    assert len(_palette_for(50)) == 20


def test_small_tier_is_a_prefix_of_the_larger_one():
    """A sixth group must not recolour the first five."""
    from structboost._plotting import _PALETTE_5, _PALETTE_10

    assert _PALETTE_10[: len(_PALETTE_5)] == _PALETTE_5


def test_safe_palette_clears_the_bars_it_claims():
    """The only tier documented as passing: >= 8 dE under CVD, >= 3.0 contrast."""
    audit = palette_audit("safe")
    assert audit["min_de_cvd"] >= 8.0
    assert audit["n_below_contrast_floor"] == 0


# --- plot_latent_dimensions -------------------------------------------------


def test_plot_latent_dimensions_shape_and_panels():
    _require_fit()
    from structboost import plot_latent_dimensions

    adata = _fitted()
    fig, axes = plot_latent_dimensions(adata, dims=[0, 2], group_by="group")
    assert axes.shape == (2, 3)
    fig.clf()


def test_groups_panel_is_dropped_without_a_grouping():
    """An ungrouped violin is the score panel rotated, so it is not drawn."""
    _require_fit()
    from structboost import plot_latent_dimensions

    adata = _fitted()
    fig, axes = plot_latent_dimensions(adata, dims=[0])
    assert axes.shape == (1, 2)
    fig.clf()


def test_weights_panel_is_available():
    _require_fit()
    from structboost import plot_latent_dimensions

    adata = _fitted()
    fig, axes = plot_latent_dimensions(
        adata, dims=[0], panels=("scores", "contributions", "weights")
    )
    assert axes.shape == (1, 3)
    fig.clf()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"panels": ("groups",)}, "no panels to draw"),
        ({"dims": [99]}, "out of range"),
        ({"panels": ("nonsense",)}, "unknown panels"),
        ({"group_by": "group", "palette": "nope"}, "unknown palette"),
        ({"group_by": "absent"}, "absent"),
    ],
)
def test_plot_latent_dimensions_guards(kwargs, message):
    _require_fit()
    from structboost import plot_latent_dimensions

    adata = _fitted()
    with pytest.raises((ValueError, KeyError), match=message):
        plot_latent_dimensions(adata, **kwargs)


def test_missing_latent_key_names_the_fix():
    _require_fit()
    from structboost import plot_latent_dimensions

    adata = _fitted()
    del adata.obsm["X_bae"]
    with pytest.raises(KeyError, match="fit the model first"):
        plot_latent_dimensions(adata, dims=[0])


def test_score_panel_draw_order_is_seeded_not_arbitrary():
    """The shuffle exists to stop one group winning every overlap; it must still be
    reproducible, or the same call gives two different figures."""
    _require_fit()
    from structboost import plot_latent_dimensions

    adata = _fitted()

    def offsets(seed):
        fig, axes = plot_latent_dimensions(
            adata, dims=[0], group_by="group", panels=("scores",), seed=seed
        )
        data = axes[0][0].collections[0].get_offsets().data.copy()
        fig.clf()
        return data

    assert np.array_equal(offsets(0), offsets(0))
    assert not np.array_equal(offsets(0), offsets(1))


def test_shuffling_does_not_move_any_point():
    """Only the paint order is permuted: the set of (rank, score) pairs is fixed,
    so the curve is identical whatever the seed."""
    _require_fit()
    from structboost import plot_latent_dimensions

    adata = _fitted()

    def sorted_points(seed):
        fig, axes = plot_latent_dimensions(
            adata, dims=[0], group_by="group", panels=("scores",), seed=seed
        )
        pts = axes[0][0].collections[0].get_offsets().data.copy()
        fig.clf()
        return pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    assert np.allclose(sorted_points(0), sorted_points(5))


# --- plot_dimension_gene_umaps ----------------------------------------------


def _with_embedding(adata):
    adata.obsm["X_umap"] = np.random.default_rng(0).normal(size=(adata.n_obs, 2))
    return adata


def test_gene_umaps_shape_includes_the_score_column():
    _require_fit()
    pytest.importorskip("scanpy")
    from structboost import plot_dimension_gene_umaps

    adata = _with_embedding(_fitted())
    fig, axes = plot_dimension_gene_umaps(adata, dims=[0, 1], n_genes=3, scale="none")
    assert axes.shape == (2, 4)
    fig.clf()


def test_gene_umaps_do_not_mutate_the_callers_anndata():
    """Panels are assembled into a throwaway AnnData; writing temp columns into the
    caller's object would leave debris if anything raised midway."""
    _require_fit()
    pytest.importorskip("scanpy")
    from structboost import plot_dimension_gene_umaps

    adata = _with_embedding(_fitted())
    before_obs = list(adata.obs.columns)
    before_obsm = sorted(adata.obsm)
    fig, _ = plot_dimension_gene_umaps(adata, dims=[0], n_genes=2, scale="none")
    fig.clf()
    assert list(adata.obs.columns) == before_obs
    assert sorted(adata.obsm) == before_obsm


def test_gene_umaps_require_an_embedding_and_say_how_to_get_one():
    _require_fit()
    pytest.importorskip("scanpy")
    from structboost import plot_dimension_gene_umaps

    adata = _fitted()
    with pytest.raises(KeyError, match="sc.pp.neighbors"):
        plot_dimension_gene_umaps(adata, dims=[0])


def test_log1p_refuses_already_scaled_expression():
    """`adata.X` is z-scored under this package's contract, so the default scale
    would otherwise colour by log1p of negative values."""
    _require_fit()
    pytest.importorskip("scanpy")
    from structboost import plot_dimension_gene_umaps

    adata = _with_embedding(_fitted())
    with pytest.raises(ValueError, match="scale='none'"):
        plot_dimension_gene_umaps(adata, dims=[0], scale="log1p")


def test_display_values_transforms():
    from structboost._plotting import _display_values

    x = np.array([0.0, 1.0, 3.0])
    assert np.allclose(_display_values(x, "none"), x)
    assert np.allclose(_display_values(x, "log1p"), np.log1p(x))
    z = _display_values(x, "zscore")
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="unknown scale"):
        _display_values(x, "bogus")


def test_zscore_of_a_constant_gene_does_not_divide_by_zero():
    from structboost._plotting import _display_values

    out = _display_values(np.full(10, 2.5), "zscore")
    assert np.all(out == 0.0)


def test_gene_labels_are_coloured_by_the_sign_of_their_weight():
    """Direction lives in the weight's sign, and the gene name's colour is where it
    is shown -- the share beside it is a magnitude and carries none.

    Regression: the shared styling used to call `tick_params(colors=...)` after the
    panels had coloured their own labels, and `colors` sets marks *and* labels, so
    every gene name came out muted.
    """
    _require_fit()
    import numpy as np

    from structboost import plot_latent_dimensions
    from structboost._plotting import _NEG, _POS

    adata = _fitted()
    weights = np.asarray(adata.varm["BAE_encoder_weights"])
    names = np.asarray(adata.var_names, dtype=str)
    fig, axes = plot_latent_dimensions(
        adata, dims=[0], panels=("scores", "contributions", "weights")
    )
    for ax in (axes[0][1], axes[0][2]):
        labels = list(ax.get_yticklabels())
        assert labels, "gene panel drew no labels"
        for text in labels:
            gene = text.get_text().split()[0]
            w = float(weights[np.flatnonzero(names == gene)[0], 0])
            assert text.get_color().upper() == (_POS if w >= 0 else _NEG).upper()
    fig.clf()


def test_non_gene_panels_keep_muted_labels():
    """Only the gene panels carry the sign encoding; a grouping is not signed."""
    _require_fit()
    from structboost import plot_latent_dimensions
    from structboost._plotting import _MUTED

    adata = _fitted()
    fig, axes = plot_latent_dimensions(adata, dims=[0], group_by="group")
    colours = {t.get_color().upper() for t in axes[0][2].get_yticklabels()}
    assert colours == {_MUTED.upper()}
    fig.clf()


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, "0 genes"), (1, "1 gene"), (2, "2 genes")],
)
def test_counts_are_inflected(n, expected):
    """Both counts these serve can legitimately be 1 -- a dimension can select a
    single gene, and the grey fold can be one group deep."""
    from structboost._plotting import _plural

    assert _plural(n, "gene") == expected


def test_grey_fold_legend_entry_is_inflected():
    """The entry naming the groups that lost their hue read '1 further groups'."""
    _require_fit()
    from structboost import plot_latent_dimensions

    adata = _fitted()
    # eight groups against a seven-hue palette leaves exactly one folded to grey
    adata.obs["many"] = [f"g{i % 8}" for i in range(adata.n_obs)]
    fig, _ = plot_latent_dimensions(adata, dims=[0], group_by="many", palette="safe", max_groups=8)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    folded = [t for t in labels if "further group" in t]
    assert folded == ["1 further group (labelled in the violins)"], labels
    fig.clf()


# --- the shares panel and the correlation heatmap ---------------------------


def test_shares_panel_draws_every_selected_gene():
    """The contribution panel shows the top handful; this one shows all of them,
    which is what says whether a dimension rests on three genes or forty."""
    _require_fit()
    import numpy as np

    from structboost import plot_latent_dimensions

    adata = _fitted()
    n_selected = int((np.asarray(adata.varm["BAE_encoder_weights"])[:, 0] != 0).sum())
    fig, axes = plot_latent_dimensions(adata, dims=[0], panels=("scores", "shares"))
    drawn = axes[0][1].collections[0].get_offsets().data
    assert drawn.shape[0] == n_selected
    fig.clf()


def test_shares_panel_is_sorted_and_respects_rank_by():
    _require_fit()
    import numpy as np

    from structboost import plot_latent_dimensions

    adata = _fitted()

    def y_of(rank_by):
        fig, axes = plot_latent_dimensions(adata, dims=[0], panels=("shares",), rank_by=rank_by)
        y = axes[0][0].collections[0].get_offsets().data[:, 1].copy()
        fig.clf()
        return y

    by_share = y_of("share")
    assert np.all(np.diff(by_share) >= -1e-12), "ordering by share must be monotone"
    # same values, different order: the y axis stays the share either way
    assert np.allclose(np.sort(by_share), np.sort(y_of("weight")))


def test_correlation_heatmap_shape_and_diagonal():
    _require_fit()
    import numpy as np

    from structboost import plot_dimension_correlation

    adata = _fitted(latent_dim=4)
    fig, ax = plot_dimension_correlation(adata)
    image = ax.images[0].get_array()
    assert image.shape == (4, 4)
    assert np.allclose(np.diag(image), 1.0)
    fig.clf()


def test_spearman_is_pearson_on_tie_averaged_ranks():
    """Ties matter: a sparse encoder can leave many cells at exactly one value."""
    import numpy as np

    from structboost._plotting import _average_ranks

    x = np.array([[3.0, 1.0], [1.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
    ranks = _average_ranks(x)
    # the two tied 1.0s share ranks 1 and 2, so both become 1.5
    assert np.allclose(ranks[:, 0], [4.0, 1.5, 1.5, 3.0])


def test_absolute_and_signed_use_different_colour_maps():
    """A signed correlation is diverging about zero; a magnitude is sequential."""
    _require_fit()
    from structboost import plot_dimension_correlation

    adata = _fitted(latent_dim=3)
    signed, ax_s = plot_dimension_correlation(adata)
    absolute, ax_a = plot_dimension_correlation(adata, absolute=True)
    assert ax_s.images[0].get_clim() == (-1.0, 1.0)
    assert ax_a.images[0].get_clim() == (0.0, 1.0)
    assert ax_a.images[0].get_cmap().name == "viridis"
    assert ax_s.images[0].get_cmap().name != "viridis"
    signed.clf()
    absolute.clf()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"method": "kendall"}, "unknown method"), ({"dims": [99]}, "out of range")],
)
def test_correlation_guards(kwargs, message):
    _require_fit()
    from structboost import plot_dimension_correlation

    adata = _fitted(latent_dim=3)
    with pytest.raises(ValueError, match=message):
        plot_dimension_correlation(adata, **kwargs)
