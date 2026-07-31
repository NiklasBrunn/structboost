"""Tests for the annotation helper module.

Skipped automatically if anndata is missing.
"""

from __future__ import annotations

import numpy as np
import pytest


def _require_anndata():
    pytest.importorskip("anndata")


def test_gene_ranking_construction():
    """GeneRanking dataclass holds gene names and weights."""
    from structboost._annotation import GeneRanking

    gr = GeneRanking(gene_names=["A", "B"], weights=[0.5, 0.3])
    assert gr.gene_names == ["A", "B"]
    assert gr.weights == [0.5, 0.3]


def test_dimension_gene_ranking_construction():
    """DimensionGeneRanking holds pos/neg gene rankings for a dimension."""
    from structboost._annotation import DimensionGeneRanking, GeneRanking

    pos = GeneRanking(gene_names=["A"], weights=[0.5])
    neg = GeneRanking(gene_names=["B"], weights=[-0.3])
    dgr = DimensionGeneRanking(
        dimension=0, n_selected_genes=2, positive_genes=pos, negative_genes=neg
    )
    assert dgr.dimension == 0
    assert dgr.n_selected_genes == 2


def test_dimension_annotation_construction():
    """DimensionAnnotation holds annotation results."""
    from structboost._annotation import DimensionAnnotation

    da = DimensionAnnotation(
        dimension=0,
        positive_annotation="T cells",
        negative_annotation="B cells",
        overall_annotation="T vs B cell axis",
        database_references=["https://genecards.org"],
    )
    assert da.dimension == 0
    assert da.overall_annotation == "T vs B cell axis"


def test_extract_gene_rankings_bae(tmp_path):
    """Extract ranked gene lists from a fitted BAE model saved to h5ad."""
    _require_anndata()
    import anndata as ad

    from structboost._annotation import extract_gene_rankings

    # Create a mock AnnData with known encoder weights
    rng = np.random.default_rng(42)
    n_genes, latent_dim = 20, 3
    adata = ad.AnnData(rng.normal(size=(10, n_genes)).astype(np.float32))
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Manually set encoder weights: dim 0 has 5 positive, 3 negative nonzero
    W = np.zeros((n_genes, latent_dim), dtype=np.float32)
    W[0, 0] = 0.9  # gene_0 positive in dim 0
    W[1, 0] = 0.7
    W[2, 0] = 0.5
    W[3, 0] = 0.3
    W[4, 0] = 0.1
    W[10, 0] = -0.8  # gene_10 negative in dim 0
    W[11, 0] = -0.6
    W[12, 0] = -0.4
    adata.varm["BAE_encoder_weights"] = W

    h5ad_path = tmp_path / "test.h5ad"
    adata.write_h5ad(h5ad_path)

    rankings = extract_gene_rankings(h5ad_path, top_k=50)

    assert len(rankings) == latent_dim

    # Check dimension 0
    r0 = rankings[0]
    assert r0.dimension == 0
    assert r0.n_selected_genes == 8  # 5 pos + 3 neg
    assert r0.positive_genes.gene_names == [
        "gene_0",
        "gene_1",
        "gene_2",
        "gene_3",
        "gene_4",
    ]
    assert r0.positive_genes.weights == pytest.approx([0.9, 0.7, 0.5, 0.3, 0.1])
    assert r0.negative_genes.gene_names == ["gene_10", "gene_11", "gene_12"]
    assert r0.negative_genes.weights == pytest.approx([-0.8, -0.6, -0.4])

    # Dimension 1 and 2 have no nonzero genes
    assert rankings[1].n_selected_genes == 0
    assert rankings[1].positive_genes.gene_names == []
    assert rankings[1].negative_genes.gene_names == []


def test_extract_subset_dimensions(tmp_path):
    """Only requested dimensions are returned."""
    _require_anndata()
    import anndata as ad

    from structboost._annotation import extract_gene_rankings

    rng = np.random.default_rng(0)
    n_genes, latent_dim = 10, 5
    adata = ad.AnnData(rng.normal(size=(5, n_genes)).astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(n_genes)]

    W = rng.normal(size=(n_genes, latent_dim)).astype(np.float32)
    adata.varm["BAE_encoder_weights"] = W

    h5ad_path = tmp_path / "sub.h5ad"
    adata.write_h5ad(h5ad_path)

    rankings = extract_gene_rankings(h5ad_path, dimensions=[0, 2])
    assert len(rankings) == 2
    assert rankings[0].dimension == 0
    assert rankings[1].dimension == 2


def test_top_k_clamping(tmp_path):
    """top_k clamps to min(top_k, n_available) per sign group."""
    _require_anndata()
    import anndata as ad

    from structboost._annotation import extract_gene_rankings

    n_genes, latent_dim = 10, 1
    adata = ad.AnnData(np.zeros((5, n_genes), dtype=np.float32))
    adata.var_names = [f"g{i}" for i in range(n_genes)]

    # Only 2 positive genes in dim 0
    W = np.zeros((n_genes, latent_dim), dtype=np.float32)
    W[0, 0] = 0.5
    W[1, 0] = 0.3
    adata.varm["BAE_encoder_weights"] = W

    h5ad_path = tmp_path / "clamp.h5ad"
    adata.write_h5ad(h5ad_path)

    rankings = extract_gene_rankings(h5ad_path, top_k=50)
    # Should return only 2 positive genes (not 50)
    assert len(rankings[0].positive_genes.gene_names) == 2
    assert len(rankings[0].negative_genes.gene_names) == 0


def test_extract_missing_key_raises(tmp_path):
    """KeyError when model_key not in adata.varm."""
    _require_anndata()
    import anndata as ad

    from structboost._annotation import extract_gene_rankings

    adata = ad.AnnData(np.zeros((5, 10), dtype=np.float32))
    h5ad_path = tmp_path / "nokey.h5ad"
    adata.write_h5ad(h5ad_path)

    with pytest.raises(KeyError, match="BAE_encoder_weights"):
        extract_gene_rankings(h5ad_path)


def test_extract_missing_file_raises():
    """FileNotFoundError for nonexistent path."""
    from structboost._annotation import extract_gene_rankings

    with pytest.raises(FileNotFoundError):
        extract_gene_rankings("/nonexistent/path.h5ad")


def test_detect_model_keys_bae(tmp_path):
    """detect_model_keys finds BAE_encoder_weights."""
    _require_anndata()
    import anndata as ad

    from structboost._annotation import detect_model_keys

    adata = ad.AnnData(np.zeros((5, 10), dtype=np.float32))
    adata.varm["BAE_encoder_weights"] = np.zeros((10, 3), dtype=np.float32)
    h5ad_path = tmp_path / "bae.h5ad"
    adata.write_h5ad(h5ad_path)

    keys = detect_model_keys(h5ad_path)
    assert keys == ["BAE_encoder_weights"]


def test_detect_model_keys_none(tmp_path):
    """detect_model_keys returns empty list when no keys found."""
    _require_anndata()
    import anndata as ad

    from structboost._annotation import detect_model_keys

    adata = ad.AnnData(np.zeros((5, 10), dtype=np.float32))
    h5ad_path = tmp_path / "empty.h5ad"
    adata.write_h5ad(h5ad_path)

    keys = detect_model_keys(h5ad_path)
    assert keys == []


def test_extract_metadata_present(tmp_path):
    """extract_metadata reads organism and tissue from adata.uns."""
    _require_anndata()
    import anndata as ad

    from structboost._annotation import extract_metadata

    adata = ad.AnnData(np.zeros((5, 10), dtype=np.float32))
    adata.uns["organism"] = "human"
    adata.uns["tissue"] = "PBMC"
    h5ad_path = tmp_path / "meta.h5ad"
    adata.write_h5ad(h5ad_path)

    meta = extract_metadata(h5ad_path)
    assert meta["organism"] == "human"
    assert meta["tissue"] == "PBMC"


def test_extract_metadata_absent(tmp_path):
    """extract_metadata returns None values when metadata missing."""
    _require_anndata()
    import anndata as ad

    from structboost._annotation import extract_metadata

    adata = ad.AnnData(np.zeros((5, 10), dtype=np.float32))
    h5ad_path = tmp_path / "nometa.h5ad"
    adata.write_h5ad(h5ad_path)

    meta = extract_metadata(h5ad_path)
    assert meta["organism"] is None
    assert meta["tissue"] is None


def test_write_annotations_to_h5ad(tmp_path):
    """Write annotations to h5ad and verify they persist on reload."""
    _require_anndata()
    import anndata as ad

    from structboost._annotation import DimensionAnnotation, write_annotations_to_h5ad

    adata = ad.AnnData(np.zeros((5, 10), dtype=np.float32))
    adata.varm["BAE_encoder_weights"] = np.zeros((10, 2), dtype=np.float32)
    h5ad_path = tmp_path / "write.h5ad"
    adata.write_h5ad(h5ad_path)

    annotations = [
        DimensionAnnotation(
            dimension=0,
            positive_annotation="T cells (CD3D, CD8A)",
            negative_annotation="B cells (MS4A1, CD79A)",
            overall_annotation="T vs B cell differentiation",
            database_references=["https://panglaodb.se"],
        ),
        DimensionAnnotation(
            dimension=1,
            positive_annotation="Cell cycle (MKI67, TOP2A)",
            negative_annotation="Quiescent cells",
            overall_annotation="Proliferation axis",
            database_references=[],
        ),
    ]

    write_annotations_to_h5ad(h5ad_path, annotations)

    # Reload and verify
    adata2 = ad.read_h5ad(h5ad_path)
    assert "bae_dimension_annotations" in adata2.uns
    stored = adata2.uns["bae_dimension_annotations"]
    assert "0" in stored  # h5ad stores dict keys as strings
    assert stored["0"]["overall_annotation"] == "T vs B cell differentiation"
    assert stored["1"]["positive_annotation"] == "Cell cycle (MKI67, TOP2A)"


def test_public_api_imports():
    """New annotation symbols are importable from structboost."""
    from structboost import (
        DimensionAnnotation,
        DimensionGeneRanking,
        GeneRanking,
        extract_gene_rankings,
        write_annotations_to_h5ad,
    )

    assert callable(extract_gene_rankings)
    assert callable(write_annotations_to_h5ad)
    assert GeneRanking is not None
    assert DimensionGeneRanking is not None
    assert DimensionAnnotation is not None
