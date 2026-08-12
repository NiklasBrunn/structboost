"""Contracts for the covariance-cache strategy and the boosting dtype rules.

These pin the *mechanism* behind the speed of the boosting loop rather than a
wall-clock number: how often an invariant quantity is recomputed, whether the
design matrix is silently promoted, and which cache strategy a fit resolved to.
Timing assertions would be flaky in CI; call counts and dtypes are exact.
"""

import numpy as np
import pytest

from structboost import BAEConfig, allboost, compute_covariance_cache
from structboost._boosting import column_norms_sq
from structboost._utils import resolve_precompute_covcache


def _require_bae():
    pytest.importorskip("torch")
    pytest.importorskip("anndata")


@pytest.fixture
def f32_data():
    """float32 predictors and targets, the dtypes ``BAE.fit`` actually uses."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((300, 60)).astype(np.float32)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = rng.standard_normal((300, 4)).astype(np.float32)
    return x, y


class TestCovarianceCacheEquivalence:
    def test_lazy_and_precomputed_agree_in_float32(self, f32_data):
        """The float64 fixture elsewhere hides this: at float32 the two cache
        strategies differ in the last bits, so the contract is that they select
        the same genes, not that they produce identical coefficients."""
        x, y = f32_data
        lazy = allboost(x, y, stepno=15)
        pre = allboost(x, y, covcache=compute_covariance_cache(x), stepno=15)

        np.testing.assert_array_equal(np.abs(lazy) > 0, np.abs(pre) > 0)
        np.testing.assert_allclose(lazy, pre, rtol=1e-5, atol=1e-7)

    def test_precomputed_cache_is_column_contiguous(self):
        """allboost only ever reads ``covcache[:, j]``; a C-ordered Gram makes
        that a strided read touching one cache line per element."""
        rng = np.random.default_rng(0)
        cache = compute_covariance_cache(rng.standard_normal((80, 25)))
        assert cache.flags.f_contiguous
        assert cache[:, 3].flags.contiguous


class TestColumnNormsHoist:
    def test_hoisted_norms_reproduce_the_internal_value(self, f32_data):
        """Passing ``col_norms_sq`` must be indistinguishable from letting
        allboost compute it -- that is what makes hoisting it out of a training
        loop safe."""
        x, y = f32_data
        internal = allboost(x, y, stepno=12)
        hoisted = allboost(x, y, stepno=12, col_norms_sq=column_norms_sq(x))
        np.testing.assert_array_equal(internal, hoisted)

    def test_wrong_shape_rejected(self, f32_data):
        x, y = f32_data
        with pytest.raises(ValueError, match="col_norms_sq must have shape"):
            allboost(x, y, stepno=5, col_norms_sq=np.ones(3))

    def test_computed_once_per_fit_not_once_per_iteration(self):
        _require_bae()
        import anndata as ad

        import structboost._model as model_module
        from structboost import BAE

        rng = np.random.default_rng(0)
        adata = ad.AnnData(rng.standard_normal((200, 40), dtype=np.float32))
        calls = []
        original = model_module.column_norms_sq
        model_module.column_norms_sq = lambda m: (calls.append(1), original(m))[1]
        try:
            BAE(40, BAEConfig(latent_dim=2, seed=0, max_iterations=6)).fit(
                adata, verbose=False, enable_early_stopping=False
            )
        finally:
            model_module.column_norms_sq = original
        assert sum(calls) == 1, f"recomputed {sum(calls)}x across 6 iterations"


class TestMandatoryBlockHoist:
    def test_block_built_once_per_target(self):
        """Nothing in the mandatory block depends on the boosting step, so it
        must be built ``n_targets`` times, not ``n_targets * stepno``."""
        import structboost._boosting as boosting

        rng = np.random.default_rng(1)
        x = rng.standard_normal((200, 30))
        y = rng.standard_normal((200, 3))
        built = []
        original = boosting._MandatoryBlock

        class Counting(original):
            def __init__(self, *args, **kwargs):
                built.append(1)
                super().__init__(*args, **kwargs)

        boosting._MandatoryBlock = Counting
        try:
            allboost(x, y, stepno=9, mandatory_features=np.array([0, 4], dtype=np.intp))
        finally:
            boosting._MandatoryBlock = original
        assert sum(built) == 3, f"built {sum(built)}x for 3 targets x 9 steps"


class TestTargetDtypeAlignment:
    def test_float64_target_does_not_promote_the_design_matrix(self, f32_data):
        """A float64 target against float32 predictors used to make numpy promote
        *sourcemat*, materializing a full float64 copy per latent dimension. The
        target is aligned to the predictors instead, so both give one answer."""
        x, y = f32_data
        from_f32 = allboost(x, y, stepno=12)
        from_f64 = allboost(x, y.astype(np.float64), stepno=12)
        np.testing.assert_array_equal(from_f32, from_f64)

    def test_integer_targets_still_accepted(self, f32_data):
        x, _ = f32_data
        y = np.arange(x.shape[0] * 2, dtype=np.int64).reshape(-1, 2) % 5
        assert np.isfinite(allboost(x, y, stepno=4)).all()


class TestPrecomputeResolver:
    def test_auto_precomputes_when_the_matrix_is_small(self):
        assert resolve_precompute_covcache("auto", 500) is True

    def test_auto_declines_when_the_matrix_cannot_fit(self):
        # 8 * p^2 is well beyond any machine's memory at this width.
        assert resolve_precompute_covcache("auto", 400_000) is False

    @pytest.mark.parametrize("setting", [True, False])
    def test_explicit_setting_always_wins(self, setting):
        assert resolve_precompute_covcache(setting, 400_000) is setting
        assert resolve_precompute_covcache(setting, 10) is setting

    def test_invalid_setting_rejected(self):
        with pytest.raises(ValueError, match="must be True, False or 'auto'"):
            resolve_precompute_covcache("sometimes", 10)

    def test_config_validates_the_field(self):
        with pytest.raises(ValueError, match="must be True, False or 'auto'"):
            BAEConfig(boosting_precompute_covcache="yes")

    def test_default_is_auto(self):
        assert BAEConfig().boosting_precompute_covcache == "auto"


class TestFitRecordsResolvedStrategy:
    @pytest.mark.parametrize("setting,expected", [("auto", True), (True, True), (False, False)])
    def test_uns_records_the_resolved_decision(self, setting, expected):
        """ "auto" is the default, so without recording it a run does not say
        which strategy it actually used."""
        _require_bae()
        import anndata as ad

        from structboost import BAE

        rng = np.random.default_rng(0)
        adata = ad.AnnData(rng.standard_normal((150, 30), dtype=np.float32))
        BAE(
            30,
            BAEConfig(latent_dim=2, seed=0, max_iterations=3, boosting_precompute_covcache=setting),
        ).fit(adata, verbose=False, enable_early_stopping=False)
        assert adata.uns["bae"]["boosting_precompute_covcache"] is expected


class TestTransferSkipsUnusedCache:
    def test_frozen_transfer_without_new_dims_builds_no_gram(self):
        """A frozen transfer with no added dimensions never calls allboost, so
        building an 8*p^2 covariance matrix for it is pure waste."""
        _require_bae()
        import anndata as ad

        import structboost._utils as utils
        from structboost import BAE

        rng = np.random.default_rng(0)
        adata = ad.AnnData(rng.standard_normal((150, 30), dtype=np.float32))
        adata.var_names = [f"g{i}" for i in range(30)]
        cfg = dict(latent_dim=2, seed=0, max_iterations=3)
        reference = BAE(30, BAEConfig(**cfg)).fit(
            adata.copy(), verbose=False, enable_early_stopping=False
        )

        built = []
        original = utils.compute_covariance_cache
        utils.compute_covariance_cache = lambda *a, **k: (built.append(1), original(*a, **k))[1]
        try:
            target = adata.copy()
            BAE.from_reference(
                reference,
                target,
                n_additional_dims=0,
                prior_mode="frozen",
                config=BAEConfig(**cfg),
            ).fit(target, verbose=False, enable_early_stopping=False)
            assert sum(built) == 0

            target2 = adata.copy()
            BAE.from_reference(
                reference,
                target2,
                n_additional_dims=2,
                prior_mode="frozen",
                config=BAEConfig(**cfg),
            ).fit(target2, verbose=False, enable_early_stopping=False)
            assert sum(built) >= 1
        finally:
            utils.compute_covariance_cache = original


class TestStabilityPathsMatchFit:
    def test_iteration_mode_runs_the_same_dtype_as_fit(self):
        """The three entry points to the same boosting problem used to run at two
        different precisions, with the stability paths holding an extra float64
        copy of the whole expression matrix."""
        _require_bae()
        import anndata as ad

        import structboost._model as model_module
        from structboost import BAE

        rng = np.random.default_rng(0)
        adata = ad.AnnData(rng.standard_normal((200, 30), dtype=np.float32))
        model = BAE(30, BAEConfig(latent_dim=2, seed=0, max_iterations=4)).fit(
            adata, verbose=False, enable_early_stopping=False
        )

        seen = {}
        original = model_module.allboost

        def probe(sourcemat, targetmat, **kwargs):
            seen.setdefault("source", sourcemat.dtype)
            seen.setdefault("target", targetmat.dtype)
            return original(sourcemat, targetmat, **kwargs)

        model_module.allboost = probe
        try:
            model.stability_selection(adata, n_runs=2, seed=0, verbose=False)
        finally:
            model_module.allboost = original
        assert seen["source"] == np.float32
        assert seen["target"] == np.float32
