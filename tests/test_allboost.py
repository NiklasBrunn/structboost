"""Tests for allboost componentwise L2 boosting function."""

import numpy as np
import pytest

from structboost import allboost, compute_covariance_cache


@pytest.fixture
def synthetic_data():
    """Generate small synthetic regression data."""
    rng = np.random.default_rng(42)
    n_samples, n_features, n_targets = 100, 20, 5
    x = rng.standard_normal((n_samples, n_features))
    x = (x - x.mean(axis=0)) / x.std(axis=0)  # Standardize
    y = rng.standard_normal((n_samples, n_targets))
    return x, y


class TestAllboostShape:
    def test_allboost_shape(self, synthetic_data):
        x, y = synthetic_data
        beta = allboost(x, y, stepno=5)
        assert beta.shape == (y.shape[1], x.shape[1])

    def test_allboost_single_target(self, synthetic_data):
        x, _ = synthetic_data
        y_single = np.random.default_rng(0).standard_normal((x.shape[0], 1))
        beta = allboost(x, y_single, stepno=5)
        assert beta.shape == (1, x.shape[1])

    def test_allboost_single_feature(self):
        rng = np.random.default_rng(99)
        x = rng.standard_normal((50, 1))
        x = (x - x.mean()) / x.std()
        y = rng.standard_normal((50, 3))
        beta = allboost(x, y, stepno=3)
        assert beta.shape == (3, 1)


class TestAllboostDtype:
    def test_allboost_dtype_float64(self, synthetic_data):
        x, y = synthetic_data
        beta = allboost(x, y, stepno=5)
        assert beta.dtype == np.float64


class TestAllboostFinite:
    def test_allboost_finite(self, synthetic_data):
        x, y = synthetic_data
        beta = allboost(x, y, stepno=10)
        assert np.all(np.isfinite(beta))


class TestAllboostEdgeCases:
    def test_allboost_single_step(self, synthetic_data):
        x, y = synthetic_data
        beta = allboost(x, y, stepno=1)
        assert beta.shape == (y.shape[1], x.shape[1])
        assert np.all(np.isfinite(beta))

    def test_allboost_many_steps(self, synthetic_data):
        x, y = synthetic_data
        beta = allboost(x, y, stepno=100)
        assert beta.shape == (y.shape[1], x.shape[1])
        assert np.all(np.isfinite(beta))


class TestAllboostReproducibility:
    def test_allboost_reproducibility(self, synthetic_data):
        x, y = synthetic_data
        beta1 = allboost(x, y, stepno=10, nu=0.1, csf=0.9)
        beta2 = allboost(x, y, stepno=10, nu=0.1, csf=0.9)
        np.testing.assert_array_equal(beta1, beta2)


class TestAllboostSparsity:
    def test_allboost_sparsity(self, synthetic_data):
        x, y = synthetic_data
        beta = allboost(x, y, stepno=3)
        n_zero = np.sum(beta == 0)
        n_total = beta.size
        assert n_zero / n_total > 0.5, "Expected majority zero coefficients with few steps"


class TestAllboostParameters:
    def test_allboost_high_nu(self, synthetic_data):
        x, y = synthetic_data
        beta = allboost(x, y, stepno=5, nu=0.5, csf=0.9)
        assert beta.shape == (y.shape[1], x.shape[1])
        assert np.all(np.isfinite(beta))

    def test_allboost_low_csf(self, synthetic_data):
        x, y = synthetic_data
        beta = allboost(x, y, stepno=5, nu=0.1, csf=0.5)
        assert beta.shape == (y.shape[1], x.shape[1])
        assert np.all(np.isfinite(beta))

    def test_allboost_extreme_params(self, synthetic_data):
        x, y = synthetic_data
        beta = allboost(x, y, stepno=2, nu=0.9, csf=0.1)
        assert np.all(np.isfinite(beta))


class TestAllboostStandardization:
    def test_allboost_with_standardized_targets(self, synthetic_data):
        """Test allboost works correctly with pre-standardized targets."""
        x, y = synthetic_data
        # Pre-standardize targets (as caller should do)
        y_std = (y - y.mean(axis=0)) / y.std(axis=0)
        beta = allboost(x, y_std, stepno=10)
        assert beta.shape == (y.shape[1], x.shape[1])
        assert np.all(np.isfinite(beta))

    def test_allboost_with_non_standardized_targets(self, synthetic_data):
        """Test allboost works (but suboptimally) with non-standardized targets."""
        x, _ = synthetic_data
        # Non-standardized targets with different scales
        rng = np.random.default_rng(123)
        y_non_std = rng.standard_normal((x.shape[0], 3))
        y_non_std[:, 0] *= 10  # Scale first target
        y_non_std[:, 1] += 100  # Shift second target
        beta = allboost(x, y_non_std, stepno=10)
        assert beta.shape == (3, x.shape[1])
        assert np.all(np.isfinite(beta))


class TestCovarianceCache:
    def test_compute_covariance_cache_shape(self, synthetic_data):
        x, _ = synthetic_data
        cov = compute_covariance_cache(x)
        assert cov.shape == (x.shape[1], x.shape[1])
        assert cov.dtype == np.float64

    def test_compute_covariance_cache_values(self, synthetic_data):
        x, _ = synthetic_data
        cov = compute_covariance_cache(x)
        expected = x.T @ x
        np.testing.assert_allclose(cov, expected)

    def test_compute_covariance_cache_out(self, synthetic_data):
        x, _ = synthetic_data
        out = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
        result = compute_covariance_cache(x, out=out)
        assert result is out
        expected = x.T @ x
        np.testing.assert_allclose(out, expected)

    def test_allboost_with_covcache(self, synthetic_data):
        x, y = synthetic_data
        cov = compute_covariance_cache(x)
        beta_with_cache = allboost(x, y, covcache=cov, stepno=10)
        beta_without_cache = allboost(x, y, stepno=10)
        # Allow tiny floating-point differences from lazy vs. pre-computed cache
        np.testing.assert_allclose(beta_with_cache, beta_without_cache, rtol=1e-14)

    def test_lazy_cache_stores_only_requested_columns(self, synthetic_data):
        x, y = synthetic_data
        stepno = 3
        _, cache = allboost(x, y, stepno=stepno, return_covcache=True)

        assert isinstance(cache, dict)
        assert len(cache) <= y.shape[1] * stepno
        assert all(column.shape == (x.shape[1],) for column in cache.values())
        assert sum(column.nbytes for column in cache.values()) < x.shape[1] ** 2 * 8

    def test_lazy_cache_is_reusable(self, synthetic_data):
        x, y = synthetic_data
        beta_first, cache = allboost(x, y, stepno=10, return_covcache=True)
        n_cached = len(cache)
        beta_second = allboost(x, y, covcache=cache, stepno=10)

        np.testing.assert_array_equal(beta_first, beta_second)
        assert len(cache) == n_cached

    def test_lazy_cache_with_mandatory_features_matches_full_cache(self, synthetic_data):
        x, y = synthetic_data
        mandatory = np.array([0, 3], dtype=np.intp)
        beta_lazy = allboost(x, y, stepno=10, mandatory_features=mandatory)
        beta_full = allboost(
            x,
            y,
            covcache=compute_covariance_cache(x),
            stepno=10,
            mandatory_features=mandatory,
        )

        np.testing.assert_allclose(beta_lazy, beta_full, rtol=1e-14)

    def test_allboost_covcache_wrong_shape(self, synthetic_data):
        x, y = synthetic_data
        wrong_cov = np.zeros((5, 5), dtype=np.float64)
        with pytest.raises(ValueError, match="covcache must have shape"):
            allboost(x, y, covcache=wrong_cov, stepno=5)


class TestAllboostMandatory:
    """Tests for mandatory_features in allboost (Binder 2008 Approach B)."""

    def test_mandatory_none_unchanged(self, synthetic_data):
        """mandatory_features=None produces identical results to no argument."""
        x, y = synthetic_data
        beta_default = allboost(x, y, stepno=10)
        beta_none = allboost(x, y, stepno=10, mandatory_features=None)
        np.testing.assert_array_equal(beta_default, beta_none)

    def test_mandatory_global_shape(self, synthetic_data):
        """Global mandatory features: betamat shape unchanged."""
        x, y = synthetic_data
        mand = np.array([0, 5], dtype=np.intp)
        beta = allboost(x, y, stepno=10, mandatory_features=mand)
        assert beta.shape == (y.shape[1], x.shape[1])
        assert np.all(np.isfinite(beta))

    def test_mandatory_features_always_nonzero(self, synthetic_data):
        """Mandatory features must have non-zero coefficients for every target."""
        x, y = synthetic_data
        mand = np.array([2, 7], dtype=np.intp)
        beta = allboost(x, y, stepno=10, mandatory_features=mand)
        for j in mand:
            assert np.all(beta[:, j] != 0), f"Mandatory feature {j} has zero coeff"

    def test_mandatory_excluded_from_selection(self, synthetic_data):
        """Mandatory features must not appear in selection history."""
        x, y = synthetic_data
        mand = np.array([0, 1], dtype=np.intp)
        beta, hist = allboost(x, y, stepno=10, mandatory_features=mand, return_history=True)
        # No selection step should have picked a mandatory feature
        for j in mand:
            assert not np.any(hist.selection == j), (
                f"Mandatory feature {j} found in selection history"
            )
        assert beta.shape == (y.shape[1], x.shape[1])

    def test_mandatory_single_feature_ols(self):
        """Single mandatory feature gets its full univariate OLS coefficient.

        With one mandatory feature and stepno=1, the mandatory pre-step
        computes the multivariate OLS (which reduces to univariate for a
        single feature).
        """
        rng = np.random.default_rng(42)
        n, p = 100, 10
        x = rng.standard_normal((n, p))
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        y = rng.standard_normal((n, 1))

        mand = np.array([3], dtype=np.intp)
        beta = allboost(x, y, stepno=1, mandatory_features=mand)

        # Expected: full OLS coeff for feature 3 against y
        col_norms_sq = (x**2).sum(axis=0)
        expected_ols = (x[:, 3] @ y[:, 0]) / col_norms_sq[3]
        np.testing.assert_allclose(beta[0, 3], expected_ols, rtol=1e-10)

    def test_mandatory_joint_multivariate_ols(self):
        """Multiple mandatory features get the joint multivariate OLS solution.

        Binder 2008 Approach B: the pre-step solves
            γ_mand = (X_mand' X_mand)^{-1} X_mand' r
        jointly for all mandatory features, accounting for their mutual
        correlations. This test uses correlated mandatory features to verify
        that the implementation does a joint solve, not sequential univariate.
        """
        rng = np.random.default_rng(42)
        n, p = 200, 10
        x = rng.standard_normal((n, p))
        # Introduce correlation between features 2 and 3
        x[:, 3] = 0.8 * x[:, 2] + 0.2 * rng.standard_normal(n)
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        y = rng.standard_normal((n, 1))

        mand = np.array([2, 3], dtype=np.intp)
        beta = allboost(x, y, stepno=1, mandatory_features=mand)

        # Expected: joint multivariate OLS for mandatory features
        x_mand = x[:, mand]
        gamma_expected = np.linalg.solve(x_mand.T @ x_mand, x_mand.T @ y[:, 0])
        np.testing.assert_allclose(beta[0, 2], gamma_expected[0], rtol=1e-10)
        np.testing.assert_allclose(beta[0, 3], gamma_expected[1], rtol=1e-10)

    def test_mandatory_ridge_is_selective_and_matches_closed_form(self):
        """Per-feature ridge stabilizes only the requested mandatory column."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal((100, 5))
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        y = rng.standard_normal((100, 1))
        mand = np.array([0, 1], dtype=np.intp)
        ridge = np.zeros(5)
        ridge[1] = 0.5

        beta = allboost(
            x,
            y,
            stepno=1,
            mandatory_features=mand,
            mandatory_ridge=ridge,
        )

        gram = x[:, mand].T @ x[:, mand]
        penalty = np.diag(ridge[mand] * (x[:, mand] ** 2).sum(axis=0))
        expected = np.linalg.solve(gram + penalty, x[:, mand].T @ y[:, 0])
        np.testing.assert_allclose(beta[0, mand], expected, rtol=1e-10)

    @pytest.mark.parametrize("ridge", [-1.0, np.array([0.0, np.nan])])
    def test_mandatory_ridge_validation(self, synthetic_data, ridge):
        x, y = synthetic_data
        with pytest.raises(ValueError, match="mandatory_ridge"):
            allboost(
                x,
                y,
                mandatory_features=np.array([0], dtype=np.intp),
                mandatory_ridge=ridge,
            )

    def test_mandatory_per_target(self, synthetic_data):
        """Per-target mandatory features: different sets per target."""
        x, y = synthetic_data
        k = y.shape[1]
        # First target: features [0, 1], rest: feature [2]
        mand = [np.array([0, 1], dtype=np.intp)] + [
            np.array([2], dtype=np.intp) for _ in range(k - 1)
        ]
        beta = allboost(x, y, stepno=10, mandatory_features=mand)
        assert beta.shape == (k, x.shape[1])
        # Feature 0 mandatory only for target 0
        assert beta[0, 0] != 0
        # Feature 2 mandatory for targets 1..k-1
        for t in range(1, k):
            assert beta[t, 2] != 0

    def test_mandatory_empty_per_target(self, synthetic_data):
        """Empty per-target list means no mandatory features for that target."""
        x, y = synthetic_data
        k = y.shape[1]
        mand = [np.array([], dtype=np.intp) for _ in range(k)]
        beta_mand = allboost(x, y, stepno=10, mandatory_features=mand)
        beta_none = allboost(x, y, stepno=10, mandatory_features=None)
        np.testing.assert_array_equal(beta_mand, beta_none)

    def test_mandatory_invalid_index_raises(self, synthetic_data):
        """Out-of-range index raises ValueError."""
        x, y = synthetic_data
        with pytest.raises(ValueError, match="out of range"):
            allboost(x, y, stepno=5, mandatory_features=np.array([999], dtype=np.intp))

    def test_mandatory_duplicate_raises(self, synthetic_data):
        """Duplicate indices raise ValueError."""
        x, y = synthetic_data
        with pytest.raises(ValueError, match="duplicate"):
            allboost(x, y, stepno=5, mandatory_features=np.array([0, 0], dtype=np.intp))

    def test_mandatory_wrong_list_length_raises(self, synthetic_data):
        """Per-target list with wrong length raises ValueError."""
        x, y = synthetic_data
        with pytest.raises(ValueError, match="length"):
            allboost(x, y, stepno=5, mandatory_features=[np.array([0], dtype=np.intp)])

    def test_mandatory_with_covcache(self, synthetic_data):
        """Mandatory features work with pre-computed covariance cache."""
        x, y = synthetic_data
        cov = compute_covariance_cache(x)
        mand = np.array([0, 3], dtype=np.intp)
        beta_with = allboost(x, y, stepno=10, mandatory_features=mand, covcache=cov)
        beta_without = allboost(x, y, stepno=10, mandatory_features=mand)
        np.testing.assert_allclose(beta_with, beta_without, rtol=1e-14)

    def test_mandatory_with_independent_false(self, synthetic_data):
        """Mandatory features work with independent=False."""
        x, y = synthetic_data
        mand = np.array([0, 1], dtype=np.intp)
        beta = allboost(x, y, stepno=10, mandatory_features=mand, independent=False)
        assert beta.shape == (y.shape[1], x.shape[1])
        assert np.all(np.isfinite(beta))
        for j in mand:
            assert np.all(beta[:, j] != 0)


class TestAllboostBackwardCompat:
    """Verify mandatory_features=None is bit-identical to old behavior."""

    def test_standard_mode_unchanged(self, synthetic_data):
        x, y = synthetic_data
        beta_old = allboost(x, y, stepno=20, nu=0.1, csf=0.9)
        beta_new = allboost(x, y, stepno=20, nu=0.1, csf=0.9, mandatory_features=None)
        np.testing.assert_array_equal(beta_old, beta_new)

    def test_independent_false_unchanged(self, synthetic_data):
        x, y = synthetic_data
        beta_old = allboost(x, y, stepno=10, independent=False)
        beta_new = allboost(x, y, stepno=10, independent=False, mandatory_features=None)
        np.testing.assert_array_equal(beta_old, beta_new)


class TestAllboostSelectionCriterion:
    """Selection uses the penalized variance-reduction (score) criterion
    ``(x_j'r)^2 / (||x_j||^2 + penvec_j)``, changed in v0.1.21.0 from the squared
    shrunken coefficient ``((x_j'r) / (||x_j||^2 + penvec_j))^2``. See CHANGELOG
    and Hofner et al. (2011)."""

    @staticmethod
    def _diverging_data():
        # Two orthogonal features with unequal column norms, chosen so the score
        # and the old squared-coefficient criteria select DIFFERENT features on
        # the first step (initial residual r = y):
        #   feature 0: ||x||^2 = 4, x'y = sqrt(8)  -> s^2/m = 2,   s^2/m^2 = 0.5
        #   feature 1: ||x||^2 = 1, x'y = 1         -> s^2/m = 1,   s^2/m^2 = 1.0
        # score picks feature 0; squared-coefficient picks feature 1.
        x = np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        y = np.array([[np.sqrt(2.0)], [1.0], [0.0], [0.0]], dtype=np.float64)
        return x, y

    def test_step1_uses_score_criterion(self):
        x, y = self._diverging_data()
        s = (x * y[:, :1]).sum(axis=0)  # x_j' y
        m = (x**2).sum(axis=0)  # ||x_j||^2
        score_argmax = int(np.argmax(s**2 / m))
        coef_argmax = int(np.argmax(s**2 / m**2))
        # Sanity: the two criteria genuinely disagree on this data.
        assert score_argmax == 0
        assert coef_argmax == 1

        _, hist = allboost(x, y, stepno=1, nu=0.1, return_history=True)
        assert int(hist.selection[0, 0]) == score_argmax

    def test_equal_norms_criteria_agree(self):
        # With equal column norms (standardized predictors) the score and the old
        # squared-coefficient criteria rank features identically.
        rng = np.random.default_rng(0)
        x = rng.standard_normal((80, 6))
        x = (x - x.mean(0)) / x.std(0)  # unit column norms
        y = (x[:, 2] - x[:, 4])[:, None]
        s = (x * y[:, :1]).sum(axis=0)
        m = (x**2).sum(axis=0)
        assert int(np.argmax(s**2 / m)) == int(np.argmax(s**2 / m**2))
        _, hist = allboost(x, y, stepno=1, return_history=True)
        assert int(hist.selection[0, 0]) == int(np.argmax(s**2 / m))


# --- Mandatory-feature correctness and input validation ---


def test_all_mandatory_terminates_without_spurious_selection():
    """With every predictor mandatory there is no eligible candidate to select.

    Continuing would take argmax over an all -inf criterion, land on an arbitrary
    mandatory index and record it in `selection` as though it had been chosen.
    """
    rng = np.random.default_rng(0)
    n, p = 120, 5
    x = rng.standard_normal((n, p))
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = (x[:, :3] @ np.array([2.0, -1.0, 0.5]))[:, None]

    all_mandatory = np.arange(p, dtype=np.intp)
    beta, hist = allboost(x, y, stepno=5, mandatory_features=all_mandatory, return_history=True)

    # -1 is the "no step taken" sentinel; nothing may be reported as selected.
    assert (hist.selection[0] == -1).all()
    # The joint mandatory update is exact OLS, so the fit must match it.
    ols = np.linalg.lstsq(x, y[:, 0], rcond=None)[0]
    np.testing.assert_allclose(beta[0], ols, atol=1e-10)
    # The coefficient path must not be left as zeros after early termination.
    np.testing.assert_allclose(hist.beta_path[0, -1], beta[0], atol=1e-10)


def test_mandatory_features_rejects_boolean_mask():
    """A mask silently becomes the indices 0/1 — the wrong features entirely."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 4))
    y = rng.standard_normal((60, 1))
    with pytest.raises(ValueError, match="not a boolean mask"):
        allboost(x, y, stepno=3, mandatory_features=np.array([False, True, False, False]))


def test_mandatory_features_rejects_float_indices():
    """1.9 would be truncated to 1, silently selecting a different feature."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 4))
    y = rng.standard_normal((60, 1))
    with pytest.raises(ValueError, match="integer dtype"):
        allboost(x, y, stepno=3, mandatory_features=np.array([1.9, 2.9]))


def test_mandatory_features_accepts_python_ints_and_empty():
    """Ordinary integer input must keep working, including an empty selection."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 4))
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = rng.standard_normal((60, 1))
    assert allboost(x, y, stepno=3, mandatory_features=np.array([0, 2])).shape == (1, 4)
    assert allboost(x, y, stepno=3, mandatory_features=np.array([], dtype=np.intp)).shape == (1, 4)


def test_stepno_below_one_raises():
    """`stepno=0` used to return an all-zero model that looked like a real fit."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3))
    y = rng.standard_normal((40, 1))
    for bad in (0, -5):
        with pytest.raises(ValueError, match="stepno must be >= 1"):
            allboost(x, y, stepno=bad)


def test_singular_mandatory_block_raises_informative_error():
    """Exact collinearity must explain what failed and how to fix it."""
    rng = np.random.default_rng(0)
    n = 80
    x = rng.standard_normal((n, 5))
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    x[:, 4] = x[:, 3]  # duplicate column inside the mandatory block
    y = rng.standard_normal((n, 1))

    with pytest.raises(ValueError, match="mandatory feature block is singular"):
        allboost(x, y, stepno=3, mandatory_features=np.array([3, 4], dtype=np.intp))

    # An explicit ridge is the documented remedy and must work.
    beta = allboost(
        x, y, stepno=3, mandatory_features=np.array([3, 4], dtype=np.intp), mandatory_ridge=1e-3
    )
    assert np.isfinite(beta).all()


def test_mandatory_feature_may_end_with_a_zero_coefficient():
    """ "Mandatory" constrains the specification, not the fitted support.

    Documented explicitly because the previous wording promised the opposite.
    """
    rng = np.random.default_rng(3)
    n = 200
    x = rng.standard_normal((n, 4))
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    # Target exactly orthogonal to the mandatory column, so its OLS update is 0.
    signal = x[:, 0]
    mandatory_col = x[:, 3] - x[:, 3] @ signal / (signal @ signal) * signal
    x[:, 3] = mandatory_col
    y = signal[:, None]

    beta = allboost(x, y, stepno=1, mandatory_features=np.array([3], dtype=np.intp))
    assert abs(beta[0, 3]) < 1e-10
