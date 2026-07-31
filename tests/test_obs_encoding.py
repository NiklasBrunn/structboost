"""Tests for obs covariate encoding utilities."""

import numpy as np
import pytest


def _require_anndata():
    pytest.importorskip("anndata")


class TestEncodObsCovariates:
    def test_categorical_dummy_encoding(self):
        _require_anndata()
        import anndata as ad
        import pandas as pd

        from structboost._utils import encode_obs_covariates

        obs = pd.DataFrame({"batch": pd.Categorical(["A", "B", "A", "C", "B"])})
        adata = ad.AnnData(np.zeros((5, 3)), obs=obs)

        result = encode_obs_covariates(adata, ["batch"])

        # k=3 categories -> k-1=2 dummy columns
        assert result.encoded.shape == (5, 2)
        # Should be standardized (mean~0, std~1)
        np.testing.assert_allclose(result.encoded.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.encoded.std(axis=0), 1.0, atol=1e-10)
        # Metadata for reconstruction
        assert "batch" in result.column_info
        assert result.column_info["batch"]["type"] == "categorical"
        assert len(result.column_info["batch"]["categories"]) == 3

    def test_numeric_covariate(self):
        _require_anndata()
        import anndata as ad
        import pandas as pd

        from structboost._utils import encode_obs_covariates

        obs = pd.DataFrame({"total_counts": [100.0, 200.0, 300.0, 400.0]})
        adata = ad.AnnData(np.zeros((4, 3)), obs=obs)

        result = encode_obs_covariates(adata, ["total_counts"])

        assert result.encoded.shape == (4, 1)
        np.testing.assert_allclose(result.encoded.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(result.encoded.std(axis=0), 1.0, atol=1e-10)

    def test_mixed_covariates(self):
        _require_anndata()
        import anndata as ad
        import pandas as pd

        from structboost._utils import encode_obs_covariates

        obs = pd.DataFrame(
            {
                "batch": pd.Categorical(["A", "B", "A", "B"]),
                "total_counts": [100.0, 200.0, 300.0, 400.0],
            }
        )
        adata = ad.AnnData(np.zeros((4, 3)), obs=obs)

        result = encode_obs_covariates(adata, ["batch", "total_counts"])

        # batch: 2 categories -> 1 dummy, total_counts: 1 numeric -> 2 columns total
        assert result.encoded.shape == (4, 2)

    def test_missing_obs_column_raises(self):
        _require_anndata()
        import anndata as ad

        from structboost._utils import encode_obs_covariates

        adata = ad.AnnData(np.zeros((4, 3)))

        with pytest.raises(ValueError, match="not found in adata.obs"):
            encode_obs_covariates(adata, ["nonexistent"])

    def test_transform_with_stored_encoding(self):
        """Reconstruct dummies from new adata using stored encoding params."""
        _require_anndata()
        import anndata as ad
        import pandas as pd

        from structboost._utils import encode_obs_covariates, transform_obs_covariates

        obs = pd.DataFrame({"batch": pd.Categorical(["A", "B", "C", "A"])})
        adata_train = ad.AnnData(np.zeros((4, 3)), obs=obs)
        result = encode_obs_covariates(adata_train, ["batch"])

        # New data with same categories
        obs_new = pd.DataFrame({"batch": pd.Categorical(["B", "C"])})
        adata_new = ad.AnnData(np.zeros((2, 3)), obs=obs_new)
        encoded_new = transform_obs_covariates(adata_new, result)

        assert encoded_new.shape == (2, 2)
        # Standardized with TRAINING mean/std
        assert np.all(np.isfinite(encoded_new))

    @pytest.mark.parametrize(
        "values",
        [
            ["A", "B", "A"],
            [True, False, True],
        ],
    )
    def test_string_and_bool_are_categorical(self, values):
        _require_anndata()
        import anndata as ad
        import pandas as pd

        from structboost._utils import encode_obs_covariates

        adata = ad.AnnData(
            np.zeros((3, 2)),
            obs=pd.DataFrame({"batch": values}),
        )
        result = encode_obs_covariates(adata, ["batch"])
        assert result.encoded.shape == (3, 1)
        assert result.column_info["batch"]["type"] == "categorical"

    @pytest.mark.parametrize("values", [[1.0, 1.0, 1.0], ["A", "A", "A"]])
    def test_constant_column_raises(self, values):
        _require_anndata()
        import anndata as ad
        import pandas as pd

        from structboost._utils import encode_obs_covariates

        adata = ad.AnnData(np.zeros((3, 2)), obs=pd.DataFrame({"batch": values}))
        with pytest.raises(ValueError, match="constant"):
            encode_obs_covariates(adata, ["batch"])

    def test_unseen_category_raises(self):
        _require_anndata()
        import anndata as ad
        import pandas as pd

        from structboost._utils import encode_obs_covariates, transform_obs_covariates

        train = ad.AnnData(
            np.zeros((3, 2)),
            obs=pd.DataFrame({"batch": ["A", "B", "A"]}),
        )
        encoding = encode_obs_covariates(train, ["batch"])
        new = ad.AnnData(
            np.zeros((1, 2)),
            obs=pd.DataFrame({"batch": ["C"]}),
        )
        with pytest.raises(ValueError, match="not seen during fitting"):
            transform_obs_covariates(new, encoding)

    def test_rank_deficient_design_raises(self):
        _require_anndata()
        import anndata as ad
        import pandas as pd

        from structboost._utils import encode_obs_covariates

        adata = ad.AnnData(
            np.zeros((4, 2)),
            obs=pd.DataFrame(
                {
                    "batch": ["A", "B", "A", "B"],
                    "duplicate": ["A", "B", "A", "B"],
                }
            ),
        )
        with pytest.raises(ValueError, match="rank-deficient"):
            encode_obs_covariates(adata, ["batch", "duplicate"])
