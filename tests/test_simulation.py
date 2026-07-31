"""Tests for the negative-binomial scRNA-seq simulation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from structboost import SimulationResult, sim_scrnaseq_data
from structboost._simulation import _marker_mask

BASE = dict(n=1000, n_genes=50, stageno=10, seed=1)


def _require_anndata():
    pytest.importorskip("anndata")


def _pure_marker(result: SimulationResult, stage: int = 0) -> int:
    """A gene that marks `stage` and no other stage."""
    mask = result.marker_mask
    others = np.delete(mask, stage, axis=0).any(axis=0)
    return int(np.flatnonzero(mask[stage] & ~others)[0])


class TestMarkerMask:
    """Exact block geometry, tested without allocating a count matrix."""

    def test_contiguous_windows_without_overlap(self):
        mask = _marker_mask(stageno=5, n_genes=50, stagep=10, stageoverlap=0)
        for k in range(5):
            expected = np.zeros(50, dtype=bool)
            expected[10 * k : 10 * k + 10] = True
            np.testing.assert_array_equal(mask[k], expected)

    def test_overlap_is_shared_between_consecutive_stages(self):
        mask = _marker_mask(stageno=5, n_genes=50, stagep=10, stageoverlap=2)
        np.testing.assert_array_equal(np.flatnonzero(mask[0] & mask[1]), [8, 9])
        assert not (mask[0] & mask[2]).any()

    def test_stride_matches_stagep_minus_overlap(self):
        mask = _marker_mask(stageno=4, n_genes=40, stagep=10, stageoverlap=3)
        starts = [int(np.flatnonzero(mask[k])[0]) for k in range(4)]
        assert starts == [0, 7, 14, 21]


class TestSimScrnaseqData:
    def test_returns_simulation_result(self):
        assert isinstance(sim_scrnaseq_data(**BASE), SimulationResult)

    def test_counts_shape_dtype_nonnegative(self):
        result = sim_scrnaseq_data(**BASE)
        assert result.counts.shape == (1000, 50)
        assert result.counts.dtype == np.int32
        assert (result.counts >= 0).all()

    def test_ground_truth_shapes(self):
        result = sim_scrnaseq_data(**BASE)
        assert result.stage_labels.shape == (1000,)
        assert result.stage_sizes.shape == (10,)
        assert result.marker_mask.shape == (10, 50)
        assert result.gene_means.shape == (50,)
        assert result.size_factors.shape == (1000,)
        assert result.batch_labels.shape == (1000,)
        assert result.marker_genes.shape == (50,)

    def test_row_major_stage_layout(self):
        """Contiguous row blocks per stage - the benchmark generator relies on this."""
        result = sim_scrnaseq_data(n=100, n_genes=50, stageno=5, seed=1)
        for k in range(5):
            assert (result.stage_labels[20 * k : 20 * k + 20] == k).all()

    def test_leftover_rows_are_unlabelled(self):
        result = sim_scrnaseq_data(n=103, n_genes=50, stageno=5, seed=1)
        assert (result.stage_labels == -1).sum() == 3
        assert (result.stage_labels[-3:] == -1).all()

    def test_balanced_stage_sizes_are_equal(self):
        assert (sim_scrnaseq_data(**BASE).stage_sizes == 100).all()

    def test_reproducibility(self):
        a = sim_scrnaseq_data(**BASE)
        b = sim_scrnaseq_data(**BASE)
        np.testing.assert_array_equal(a.counts, b.counts)
        np.testing.assert_array_equal(a.gene_means, b.gene_means)
        np.testing.assert_array_equal(a.stage_sizes, b.stage_sizes)
        np.testing.assert_array_equal(a.batch_labels, b.batch_labels)

    def test_different_seeds_differ(self):
        a = sim_scrnaseq_data(n=100, n_genes=50, stageno=5, seed=1)
        b = sim_scrnaseq_data(n=100, n_genes=50, stageno=5, seed=2)
        assert not np.array_equal(a.counts, b.counts)

    def test_params_are_h5ad_safe_scalars(self):
        params = sim_scrnaseq_data(**BASE).params
        assert all(v is not None for v in params.values())
        assert all(isinstance(v, int | float | bool | str) for v in params.values())
        assert params["design"] == "negative-binomial"

    def test_dropout_mid_is_nan_when_disabled(self):
        params = sim_scrnaseq_data(**BASE).params
        assert params["dropout"] is False
        assert np.isnan(params["dropout_mid"])

    def test_marker_genes_matches_mask(self):
        result = sim_scrnaseq_data(**BASE)
        np.testing.assert_array_equal(result.marker_genes, result.marker_mask.any(axis=0))


class TestImbalanced:
    def test_sizes_sum_to_n_and_vary(self):
        result = sim_scrnaseq_data(n=1000, n_genes=50, stageno=10, imbalanced=True, seed=42)
        assert result.stage_sizes.sum() == 1000
        assert len(set(result.stage_sizes.tolist())) > 1

    def test_reproducible(self):
        kwargs = dict(n=1000, n_genes=50, stageno=10, imbalanced=True, seed=42)
        a = sim_scrnaseq_data(**kwargs)
        b = sim_scrnaseq_data(**kwargs)
        np.testing.assert_array_equal(a.counts, b.counts)
        np.testing.assert_array_equal(a.stage_sizes, b.stage_sizes)

    def test_labels_follow_sizes(self):
        result = sim_scrnaseq_data(n=1000, n_genes=50, stageno=10, imbalanced=True, seed=42)
        counts = np.bincount(result.stage_labels[result.stage_labels >= 0], minlength=10)
        np.testing.assert_array_equal(counts, result.stage_sizes)


class TestStreamIsolation:
    """Independent sub-streams let each knob be swept without disturbing others."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lib_size_sd": 0.5},
            {"batch_effect_sd": 0.5, "n_batches": 2},
            {"ambient_frac": 0.3},
            {"dropout_mid": 1.0},
        ],
    )
    def test_gene_means_unaffected(self, kwargs):
        np.testing.assert_array_equal(
            sim_scrnaseq_data(**BASE).gene_means, sim_scrnaseq_data(**BASE, **kwargs).gene_means
        )

    def test_ambient_preserves_library_size_exactly(self):
        base = sim_scrnaseq_data(**BASE)
        amb = sim_scrnaseq_data(**BASE, ambient_frac=0.5)
        np.testing.assert_array_equal(base.counts.sum(axis=1), amb.counts.sum(axis=1))

    def test_dropout_only_removes_counts(self):
        base = sim_scrnaseq_data(**BASE)
        dropped = sim_scrnaseq_data(**BASE, dropout_mid=1.0)
        assert (dropped.counts <= base.counts).all()
        assert (dropped.counts == 0).mean() > (base.counts == 0).mean()


class TestExpressionModel:
    def test_effect_size_recovered_per_gene(self):
        """Compare the SAME gene in-stage vs out-of-stage.

        Pooling marker genes against background genes would instead measure the
        spread of the gamma-drawn gene means, not the effect size.
        """
        for effect in (2.0, 10.0, 50.0):
            result = sim_scrnaseq_data(
                n=4000, n_genes=40, stageno=4, effect_size=effect, dispersion=0.0, seed=5
            )
            j = _pure_marker(result)
            inside = result.stage_labels == 0
            ratio = result.counts[inside, j].mean() / result.counts[~inside, j].mean()
            assert 0.9 * effect < ratio < 1.1 * effect

    def test_effect_size_is_monotonic(self):
        ratios = []
        for effect in (2.0, 10.0, 50.0):
            result = sim_scrnaseq_data(
                n=4000, n_genes=40, stageno=4, effect_size=effect, dispersion=0.0, seed=5
            )
            j = _pure_marker(result)
            inside = result.stage_labels == 0
            ratios.append(result.counts[inside, j].mean() / result.counts[~inside, j].mean())
        assert ratios[0] < ratios[1] < ratios[2]

    def test_gene_mean_recovered(self):
        result = sim_scrnaseq_data(
            n=4000,
            n_genes=40,
            stageno=4,
            dispersion=0.0,
            gene_mean_shape=1e6,
            effect_size=10.0,
            seed=7,
        )
        frac = result.marker_mask.mean(axis=0)  # fraction of stages marking each gene
        expected = result.gene_means * (frac * 10.0 + (1.0 - frac))
        np.testing.assert_allclose(result.counts.mean(axis=0), expected, rtol=0.1)

    def test_dispersion_controls_fano_factor(self):
        fanos = []
        for dispersion in (0.0, 0.2, 1.0):
            result = sim_scrnaseq_data(
                n=5000, n_genes=40, stageno=4, dispersion=dispersion, seed=11
            )
            background = result.counts[:, ~result.marker_genes]
            fano = (background.var(axis=0) / np.maximum(background.mean(axis=0), 1e-9)).mean()
            expected = 1.0 + dispersion * background.mean(axis=0).mean()
            assert abs(fano - expected) < 0.15 * expected
            fanos.append(fano)
        assert fanos[0] < fanos[1] < fanos[2]

    def test_effect_size_sd_spreads_fold_changes(self):
        """Only genes marking stage 0 alone are comparable.

        Overlap genes also mark stage 1, so their out-of-stage mean is inflated
        and their ratio is far below ``effect_size`` by construction - including
        them would measure block geometry rather than the fold-change spread.
        """
        sharp = sim_scrnaseq_data(n=4000, n_genes=40, stageno=4, dispersion=0.0, seed=5)
        fuzzy = sim_scrnaseq_data(
            n=4000, n_genes=40, stageno=4, dispersion=0.0, effect_size_sd=0.5, seed=5
        )
        inside = sharp.stage_labels == 0
        others = sharp.marker_mask[1:].any(axis=0)
        pure = np.flatnonzero(sharp.marker_mask[0] & ~others)

        def ratios(result):
            return np.array(
                [result.counts[inside, j].mean() / result.counts[~inside, j].mean() for j in pure]
            )

        assert ratios(sharp).std() < 0.5
        assert ratios(fuzzy).std() > 3 * ratios(sharp).std()


class TestTechnicalEffects:
    def test_size_factors_neutral_by_default(self):
        np.testing.assert_array_equal(sim_scrnaseq_data(**BASE).size_factors, np.ones(1000))

    def test_size_factors_have_unit_mean(self):
        result = sim_scrnaseq_data(**BASE, lib_size_sd=0.5)
        assert abs(result.size_factors.mean() - 1.0) < 1e-12

    def test_library_size_variation_increases_spread(self):
        def cv(result):
            totals = result.counts.sum(axis=1)
            return totals.std() / totals.mean()

        assert cv(sim_scrnaseq_data(**BASE, lib_size_sd=0.5)) > 3 * cv(sim_scrnaseq_data(**BASE))

    def test_batches_are_round_robin_and_orthogonal_to_stage(self):
        result = sim_scrnaseq_data(**BASE, n_batches=2)
        np.testing.assert_array_equal(result.batch_labels, np.arange(1000) % 2)
        for k in range(10):
            per_batch = np.bincount(result.batch_labels[result.stage_labels == k], minlength=2)
            assert abs(int(per_batch[0]) - int(per_batch[1])) <= 1

    def test_batch_effect_separates_batches(self):
        def max_log_ratio(**kwargs):
            result = sim_scrnaseq_data(n=4000, n_genes=40, stageno=4, n_batches=2, seed=9, **kwargs)
            b0 = result.counts[result.batch_labels == 0].mean(axis=0)
            b1 = result.counts[result.batch_labels == 1].mean(axis=0)
            return np.abs(np.log(np.maximum(b0, 1e-9) / np.maximum(b1, 1e-9))).max()

        assert max_log_ratio(batch_effect_sd=0.0) < 0.25
        assert max_log_ratio(batch_effect_sd=0.5) > 0.8

    def test_batch_effect_roughly_preserves_marginal_gene_means(self):
        """Batch factors are normalized to unit *geometric* mean per gene.

        That pins the log-scale centre, so arithmetic means inflate slightly -
        averaging exp(+x) and exp(-x) gives cosh(x) > 1. The typical gene should
        therefore barely move, while a few may drift by tens of percent.
        """
        flat = sim_scrnaseq_data(n=4000, n_genes=40, stageno=4, n_batches=2, seed=9)
        shifted = sim_scrnaseq_data(
            n=4000, n_genes=40, stageno=4, n_batches=2, batch_effect_sd=0.5, seed=9
        )
        relative = np.abs(shifted.counts.mean(axis=0) / flat.counts.mean(axis=0) - 1.0)
        assert np.median(relative) < 0.1
        assert relative.max() < 0.4

    def test_ambient_dilutes_marker_signal(self):
        contrasts = []
        for frac in (0.0, 0.2, 0.5):
            result = sim_scrnaseq_data(n=2000, n_genes=40, stageno=4, ambient_frac=frac, seed=13)
            inside = result.stage_labels == 0
            marker_mean = result.counts[np.ix_(inside, result.marker_mask[0])].mean()
            background_mean = result.counts[np.ix_(inside, ~result.marker_genes)].mean()
            contrasts.append(marker_mean / background_mean)
        assert contrasts[0] > contrasts[1] > contrasts[2]

    def test_dropout_is_mean_dependent(self):
        """Low-expression genes drop out more than genes elevated in their own stage."""
        result = sim_scrnaseq_data(**BASE, dropout_mid=1.0, dropout_shape=-1.0)
        inside = result.stage_labels == 0
        background_zeros = (result.counts[:, ~result.marker_genes] == 0).mean()
        marker_zeros = (result.counts[np.ix_(inside, result.marker_mask[0])] == 0).mean()
        assert background_zeros - marker_zeros > 0.2

    def test_effects_compose(self):
        result = sim_scrnaseq_data(
            **BASE,
            lib_size_sd=0.5,
            n_batches=2,
            batch_effect_sd=0.5,
            ambient_frac=0.2,
            dropout_mid=1.0,
        )
        assert result.counts.shape == (1000, 50)
        assert (result.counts >= 0).all()
        assert result.counts.sum() > 0


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"n": 0}, "n must be >= 1"),
            ({"n_genes": 0}, "n_genes must be >= 1"),
            ({"stageno": 0}, "stageno must be >= 1"),
            ({"n": 5, "stageno": 10}, "stageno must be <= n"),
            ({"n_genes": 5, "stageno": 10}, "stagep must be >= 1"),
            ({"stagep": 100}, "stagep must be <= n_genes"),
            ({"stagep": 5, "stageoverlap": 5, "hierarchy": None}, "stageoverlap must satisfy"),
            ({"stagep": 5, "stageoverlap": -1, "hierarchy": None}, "stageoverlap must satisfy"),
            ({"stagep": 20, "stageoverlap": 0, "hierarchy": None}, "stage layout does not fit"),
            ({"imbalanced": True, "stagen": 10}, "stagen must be None"),
            ({"stagen": 0}, "stagen must be >= 1"),
            ({"stagen": 500}, r"stagen \* stageno must be <= n"),
            ({"base_mean": 0.0}, "base_mean must be > 0"),
            ({"gene_mean_shape": 0.0}, "gene_mean_shape must be > 0"),
            ({"effect_size": 0.0}, "effect_size must be > 0"),
            ({"effect_size_sd": -0.1}, "effect_size_sd must be >= 0"),
            ({"dispersion": -0.1}, "dispersion must be >= 0"),
            ({"lib_size_sd": -0.1}, "lib_size_sd must be >= 0"),
            ({"n_batches": 0}, "n_batches must satisfy"),
            ({"batch_effect_sd": -0.1}, "batch_effect_sd must be >= 0"),
            ({"ambient_frac": 1.0}, "ambient_frac must be in"),
            ({"ambient_frac": -0.1}, "ambient_frac must be in"),
            ({"dropout_mid": float("nan")}, "dropout_mid must be finite"),
            ({"dropout_shape": float("inf")}, "dropout_shape must be finite"),
        ],
    )
    def test_raises_value_error(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            sim_scrnaseq_data(**{**BASE, **kwargs})


class TestSimScrnaseqAnndata:
    def test_returns_anndata_with_expected_shape(self):
        _require_anndata()
        import anndata as ad

        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=100, n_genes=50, stageno=5, seed=1)
        assert isinstance(adata, ad.AnnData)
        assert adata.shape == (100, 50)

    def test_obs_and_var_names(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=50, n_genes=30, stageno=5, seed=1)
        assert adata.obs_names[0] == "Cell_0"
        assert adata.obs_names[-1] == "Cell_49"
        assert adata.var_names[0] == "Gene_0"
        assert adata.var_names[-1] == "Gene_29"

    def test_layers_present_with_correct_dtypes(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=100, n_genes=50, stageno=5, seed=1)
        assert adata.layers["counts"].dtype == np.int32
        assert adata.layers["lognorm"].dtype == np.float32
        assert adata.X.dtype == np.float32

    def test_counts_layer_matches_array_api(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        kwargs = dict(n=100, n_genes=50, stageno=5, seed=3)
        adata = sim_scrnaseq_anndata(**kwargs)
        np.testing.assert_array_equal(adata.layers["counts"], sim_scrnaseq_data(**kwargs).counts)

    def test_lognorm_normalizes_to_target_sum(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=100, n_genes=50, stageno=5, seed=1)
        np.testing.assert_allclose(np.expm1(adata.layers["lognorm"]).sum(axis=1), 1e4, rtol=1e-3)

    def test_standardize_true_zscores_lognorm(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=200, n_genes=50, stageno=5, seed=1)
        assert np.abs(adata.X.mean(axis=0)).max() < 1e-4
        assert np.abs(np.median(adata.X.std(axis=0)) - 1.0) < 1e-4
        assert adata.uns["simulation"]["x_content"] == "zscore"

    def test_standardize_false_leaves_lognorm_in_x(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=100, n_genes=50, stageno=5, seed=1, standardize=False)
        np.testing.assert_array_equal(adata.X, adata.layers["lognorm"])
        assert adata.uns["simulation"]["x_content"] == "lognorm"

    def test_lognorm_available_regardless_of_standardize(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        kwargs = dict(n=100, n_genes=50, stageno=5, seed=1)
        a = sim_scrnaseq_anndata(**kwargs, standardize=True)
        b = sim_scrnaseq_anndata(**kwargs, standardize=False)
        np.testing.assert_array_equal(a.layers["lognorm"], b.layers["lognorm"])

    def test_stage_labels(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=100, n_genes=50, stageno=5, seed=1)
        assert "stage" in adata.obs.columns
        assert sorted(adata.obs["stage"].unique().tolist()) == [f"Stage_{k}" for k in range(5)]
        np.testing.assert_array_equal(adata.obs["stage_id"].to_numpy(), np.repeat(np.arange(5), 20))

    def test_background_label_for_leftover_cells(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=103, n_genes=50, stageno=5, seed=1)
        assert "Background" in adata.obs["stage"].tolist()
        assert (adata.obs["stage_id"].to_numpy() == -1).sum() == 3

    def test_batch_column_present_for_single_batch(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=100, n_genes=50, stageno=5, seed=1)
        assert adata.obs["batch"].unique().tolist() == ["Batch_0"]

    def test_marker_ground_truth(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(
            n=100, n_genes=50, stageno=5, stagep=10, stageoverlap=2, hierarchy=None, seed=1
        )
        mask = adata.varm["marker_mask"]
        assert mask.shape == (50, 5)
        np.testing.assert_array_equal(adata.var["is_marker"].to_numpy(), mask.any(axis=1))
        np.testing.assert_array_equal(
            adata.var["n_marker_stages"].to_numpy(), mask.sum(axis=1).astype(np.int32)
        )
        # Genes 8 and 9 sit in the stage 0 / stage 1 overlap.
        assert adata.var["marker_stages"].iloc[8] == "Stage_0,Stage_1"
        assert adata.var["n_marker_stages"].iloc[8] == 2

    def test_obs_diagnostics(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=100, n_genes=50, stageno=5, seed=1)
        np.testing.assert_allclose(
            adata.obs["total_counts"].to_numpy(), adata.layers["counts"].sum(axis=1)
        )
        np.testing.assert_allclose(adata.obs["size_factor"].to_numpy(), 1.0)

    def test_simulation_params_stored(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(
            n=100, n_genes=50, stageno=5, effect_size=7.0, dispersion=0.3, seed=42
        )
        uns = adata.uns["simulation"]
        assert uns["n"] == 100
        assert uns["n_genes"] == 50
        assert uns["stageno"] == 5
        assert uns["effect_size"] == 7.0
        assert uns["dispersion"] == 0.3
        assert uns["seed"] == 42
        np.testing.assert_array_equal(uns["stage_sizes"], np.full(5, 20))

    def test_reproducibility(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        kwargs = dict(n=50, n_genes=30, stageno=5, seed=42)
        np.testing.assert_array_equal(
            sim_scrnaseq_anndata(**kwargs).X, sim_scrnaseq_anndata(**kwargs).X
        )

    def test_imbalanced_stage_counts(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=1000, n_genes=50, stageno=10, imbalanced=True, seed=42)
        assert len(set(adata.obs["stage"].value_counts().tolist())) > 1
        assert adata.uns["simulation"]["imbalanced"]
        assert adata.uns["simulation"]["stage_sizes"].sum() == 1000

    @pytest.mark.parametrize("imbalanced", [False, True])
    def test_h5ad_round_trip(self, tmp_path, imbalanced):
        _require_anndata()
        import anndata as ad

        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=100, n_genes=50, stageno=5, imbalanced=imbalanced, seed=1)
        path = tmp_path / "sim.h5ad"
        adata.write_h5ad(path)
        loaded = ad.read_h5ad(path)

        assert sorted(loaded.uns["simulation"]) == sorted(adata.uns["simulation"])
        np.testing.assert_array_equal(
            loaded.uns["simulation"]["stage_sizes"], adata.uns["simulation"]["stage_sizes"]
        )
        assert bool(loaded.uns["simulation"]["imbalanced"]) == imbalanced
        assert np.isnan(loaded.uns["simulation"]["dropout_mid"])
        np.testing.assert_array_equal(loaded.layers["counts"], adata.layers["counts"])

    def test_target_sum_must_be_positive(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        with pytest.raises(ValueError, match="target_sum must be > 0"):
            sim_scrnaseq_anndata(n=100, n_genes=50, stageno=5, target_sum=0.0)


class TestHierarchicalStructure:
    """Nested cell-type taxonomies, on by default."""

    def test_default_is_hierarchical(self):
        from structboost import sim_scrnaseq_data

        res = sim_scrnaseq_data(n=200, n_genes=300, stageno=10, seed=0)
        assert res.hierarchy == (2, 5)
        assert res.leaf_paths.shape == (10, 2)
        # Every level is represented among the marker genes.
        assert set(np.unique(res.gene_level)) == {-1, 0, 1}

    def test_hierarchy_none_restores_flat_layout(self):
        from structboost import sim_scrnaseq_data

        res = sim_scrnaseq_data(n=200, n_genes=300, stageno=10, hierarchy=None, seed=0)
        assert res.hierarchy is None
        # Flat markers are all one level; only noise genes are -1.
        assert set(np.unique(res.gene_level)) <= {-1, 0}

    def test_leaves_share_ancestor_markers(self):
        """Siblings share their parent's block; cousins do not."""
        from structboost._simulation import _hierarchical_marker_mask

        mask, level, paths = _hierarchical_marker_mask((2, 5), 200, (20, 8))
        class_block = np.flatnonzero(level == 0)[:20]  # first class's markers
        siblings = [k for k in range(10) if paths[k, 0] == 0]
        cousins = [k for k in range(10) if paths[k, 0] == 1]
        assert mask[np.ix_(siblings, class_block)].all()
        assert not mask[np.ix_(cousins, class_block)].any()

    def test_marker_blocks_are_disjoint_across_the_tree(self):
        """Each gene marks exactly one node, so its level is unambiguous."""
        from structboost._simulation import _hierarchical_marker_mask

        _, level, _ = _hierarchical_marker_mask((2, 5), 200, (20, 8))
        assert (level >= 0).sum() == 2 * 20 + 10 * 8

    def test_default_hierarchy_adapts_to_stageno(self):
        from structboost._simulation import _default_hierarchy

        for stageno in (4, 6, 10, 12, 20):
            hierarchy = _default_hierarchy(stageno)
            assert int(np.prod(hierarchy)) == stageno
        # Primes have no non-trivial factorisation and stay single-level.
        assert _default_hierarchy(7) == (7,)

    def test_marker_budget_is_respected(self):
        """A tight gene budget must shift markers to broader levels, not overflow."""
        from structboost._simulation import _resolve_markers_per_level

        blocks = _resolve_markers_per_level((2, 5), n_genes=50, stagep=20)
        assert sum(blocks) == 20  # each population still carries `stagep` markers
        assert 2 * blocks[0] + 10 * blocks[1] <= 50

    def test_impossible_budget_raises(self):
        from structboost._simulation import _resolve_markers_per_level

        with pytest.raises(ValueError, match="does not fit"):
            _resolve_markers_per_level((2, 5), n_genes=5, stagep=4)

    def test_conflicting_stageno_and_hierarchy_raise(self):
        from structboost import sim_scrnaseq_data

        with pytest.raises(ValueError, match="Set one or the other"):
            sim_scrnaseq_data(n=100, n_genes=200, stageno=7, hierarchy=(2, 5), seed=0)

    def test_stageoverlap_rejected_with_hierarchy(self):
        """The two overlap mechanisms are mutually exclusive, and silently ignoring
        one would misrepresent the generated structure."""
        from structboost import sim_scrnaseq_data

        with pytest.raises(ValueError, match="stageoverlap is not used"):
            sim_scrnaseq_data(n=100, n_genes=200, stageno=10, stageoverlap=2, seed=0)

    def test_anndata_level_labels_are_nested_and_distinct(self):
        _require_anndata()
        from structboost import sim_scrnaseq_anndata

        adata = sim_scrnaseq_anndata(n=400, n_genes=300, stageno=10, seed=0)
        assert adata.obs["level_0"].nunique() == 2
        # Full ancestry in the label: within-parent indices alone would collide.
        assert adata.obs["level_1"].nunique() == 10
        nested = adata.obs.groupby("level_1", observed=True)["level_0"].nunique()
        assert (nested == 1).all()
        assert "marker_level" in adata.var
