"""Tests for resolving mandatory gene names to indices."""

import numpy as np
import pytest


def _require_anndata():
    pytest.importorskip("anndata")


class TestResolveMandatoryGenes:
    def test_none_returns_none(self):
        _require_anndata()
        import anndata as ad

        from structboost._utils import resolve_mandatory_genes

        adata = ad.AnnData(np.zeros((4, 5)))
        assert resolve_mandatory_genes(None, adata) is None

    def test_int_indices_passthrough(self):
        _require_anndata()
        import anndata as ad

        from structboost._utils import resolve_mandatory_genes

        adata = ad.AnnData(np.zeros((4, 5)))
        mand = [0, 3]
        result = resolve_mandatory_genes(mand, adata)
        np.testing.assert_array_equal(result, np.array([0, 3], dtype=np.intp))

    def test_string_names_resolved(self):
        _require_anndata()
        import anndata as ad
        import pandas as pd

        from structboost._utils import resolve_mandatory_genes

        var = pd.DataFrame(index=["gene_A", "gene_B", "gene_C", "gene_D", "gene_E"])
        adata = ad.AnnData(np.zeros((4, 5)), var=var)
        result = resolve_mandatory_genes(["gene_B", "gene_D"], adata)
        np.testing.assert_array_equal(result, np.array([1, 3], dtype=np.intp))

    def test_per_target_lists(self):
        _require_anndata()
        import anndata as ad
        import pandas as pd

        from structboost._utils import resolve_mandatory_genes

        var = pd.DataFrame(index=["g0", "g1", "g2", "g3", "g4"])
        adata = ad.AnnData(np.zeros((4, 5)), var=var)
        mand = [["g0", "g1"], [], ["g4"]]
        result = resolve_mandatory_genes(mand, adata)
        assert isinstance(result, list) and len(result) == 3
        np.testing.assert_array_equal(result[0], np.array([0, 1], dtype=np.intp))
        np.testing.assert_array_equal(result[1], np.array([], dtype=np.intp))
        np.testing.assert_array_equal(result[2], np.array([4], dtype=np.intp))

    def test_unknown_gene_name_raises(self):
        _require_anndata()
        import anndata as ad
        import pandas as pd

        from structboost._utils import resolve_mandatory_genes

        var = pd.DataFrame(index=["gene_A", "gene_B"])
        adata = ad.AnnData(np.zeros((4, 2)), var=var)
        with pytest.raises(ValueError, match="not found"):
            resolve_mandatory_genes(["gene_X"], adata)

    def test_numpy_array_passthrough(self):
        _require_anndata()
        import anndata as ad

        from structboost._utils import resolve_mandatory_genes

        adata = ad.AnnData(np.zeros((4, 5)))
        mand = np.array([0, 2, 4], dtype=np.intp)
        result = resolve_mandatory_genes(mand, adata)
        np.testing.assert_array_equal(result, mand)
