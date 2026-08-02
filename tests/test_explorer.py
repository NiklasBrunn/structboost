"""Tests for the interactive HTML explorer module.

Skipped automatically if anndata is missing.
"""

from __future__ import annotations

import numpy as np
import pytest


def _require_anndata():
    pytest.importorskip("anndata")


def _make_bae_adata(n_cells=30, n_genes=20, latent_dim=3, rng_seed=42):
    """Create a mock AnnData with BAE results for testing."""
    import anndata as ad

    rng = np.random.default_rng(rng_seed)
    X = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Encoder weights with known structure
    W = np.zeros((n_genes, latent_dim), dtype=np.float32)
    W[0, 0] = 0.9
    W[1, 0] = 0.7
    W[2, 0] = 0.5
    W[10, 0] = -0.8
    W[11, 0] = -0.6
    W[3, 1] = 0.4
    W[4, 1] = -0.3
    adata.varm["BAE_encoder_weights"] = W

    # Latent embedding
    adata.obsm["X_bae"] = rng.normal(size=(n_cells, latent_dim)).astype(np.float32)

    # UMAP coords
    adata.obsm["X_umap"] = rng.normal(size=(n_cells, 2)).astype(np.float32)

    # A categorical obs column
    adata.obs["leiden"] = [str(i % 3) for i in range(n_cells)]
    adata.obs["leiden"] = adata.obs["leiden"].astype("category")

    return adata


def test_build_payload_basic():
    """Payload contains expected top-level keys and structure."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload

    adata = _make_bae_adata()
    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    assert "umap" in payload
    assert "spatial" in payload
    assert payload["spatial"] is None
    assert "latent" in payload
    assert "dimensions" in payload
    assert "expression" in payload
    assert "obs" in payload
    assert "gene_names" in payload

    assert len(payload["umap"]) == 30
    assert len(payload["latent"]) == 30
    assert len(payload["dimensions"]) == 3


def test_build_payload_dimension_genes():
    """Dimensions contain correct top-k positive and negative genes."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload

    adata = _make_bae_adata()
    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    dim0 = payload["dimensions"][0]
    assert dim0["index"] == 0
    pos_names = [g["name"] for g in dim0["positive_genes"]]
    neg_names = [g["name"] for g in dim0["negative_genes"]]
    assert pos_names == ["gene_0", "gene_1", "gene_2"]
    assert neg_names == ["gene_10", "gene_11"]


def test_build_payload_expression_keys():
    """Expression dict contains only genes that appear in dimensions."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload

    adata = _make_bae_adata()
    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    # gene_names should be the union of all top-k genes across dimensions
    assert "gene_0" in payload["gene_names"]
    assert "gene_10" in payload["gene_names"]
    # Expression should have an "X" layer
    assert "X" in payload["expression"]
    assert "gene_0" in payload["expression"]["X"]
    assert len(payload["expression"]["X"]["gene_0"]) == 30


def test_build_payload_multi_layer():
    """Multiple layers are included in expression dict."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload

    adata = _make_bae_adata()
    rng = np.random.default_rng(99)
    adata.layers["counts"] = rng.poisson(5, size=adata.shape).astype(np.float32)

    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    # Exact key set, not membership: anndata >= 0.13 exposes `.X` as
    # `layers[None]`, so taking the mapping's keys verbatim adds a `None` entry
    # holding a second copy of `.X`. Membership assertions pass right through it.
    assert set(payload["expression"]) == {"X", "counts"}
    assert None not in payload["expression"]


def test_build_payload_sparse_layer():
    """Sparse expression layers are converted to numeric vectors."""
    _require_anndata()
    from scipy import sparse

    from structboost._explorer import _build_explorer_payload

    adata = _make_bae_adata()
    rng = np.random.default_rng(123)
    dense_counts = rng.poisson(3, size=adata.shape).astype(np.float32)
    adata.layers["counts_sparse"] = sparse.csr_matrix(dense_counts)

    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    assert "counts_sparse" in payload["expression"]
    first_gene = payload["gene_names"][0]
    values = payload["expression"]["counts_sparse"][first_gene]
    assert len(values) == adata.n_obs
    assert all(isinstance(v, float) for v in values)


def test_build_payload_spatial():
    """Spatial coords included when spatial_key is present."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload

    adata = _make_bae_adata()
    rng = np.random.default_rng(7)
    adata.obsm["spatial"] = rng.normal(size=(30, 2)).astype(np.float32)

    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key="spatial",
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    assert payload["spatial"] is not None
    assert len(payload["spatial"]) == 30


def test_build_payload_annotations():
    """Annotations from adata.uns are included in dimension data."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload

    adata = _make_bae_adata()
    adata.uns["bae_dimension_annotations"] = {
        "0": {
            "positive_annotation": "T cells",
            "negative_annotation": "B cells",
            "overall_annotation": "T/B axis",
            "database_references": [],
        }
    }

    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key="bae_dimension_annotations",
    )

    dim0 = payload["dimensions"][0]
    assert dim0["annotation"] is not None
    assert dim0["annotation"]["overall"] == "T/B axis"

    dim1 = payload["dimensions"][1]
    assert dim1["annotation"] is None


def test_build_payload_obs_keys():
    """Categorical obs columns are included."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload

    adata = _make_bae_adata()
    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=["leiden"],
        top_k=5,
        annotations_key=None,
    )

    assert "leiden" in payload["obs"]
    assert len(payload["obs"]["leiden"]) == 30


def test_build_payload_subsampled():
    """Payload is subsampled when n_cells > max_cells."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload, _subsample_adata

    adata = _make_bae_adata(n_cells=50)

    with pytest.warns(UserWarning, match="Subsampling"):
        sub = _subsample_adata(adata, max_cells=20, seed=42)

    assert sub.n_obs == 20

    payload = _build_explorer_payload(
        sub,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )
    assert len(payload["umap"]) == 20


def test_subsample_noop_when_under_threshold():
    """No subsampling when n_cells <= max_cells."""
    _require_anndata()
    from structboost._explorer import _subsample_adata

    adata = _make_bae_adata(n_cells=30)
    sub = _subsample_adata(adata, max_cells=50, seed=42)
    assert sub.n_obs == 30


def test_render_html_contains_plotly():
    """Rendered HTML contains Plotly CDN and data div."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload, _render_html

    adata = _make_bae_adata()
    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    html = _render_html(payload, title="Test Explorer")

    assert "plotly" in html.lower()
    assert "Test Explorer" in html
    assert "umap-plot" in html
    assert "coeff-plot" in html


def test_render_html_has_colorscale_controls_and_defaults():
    """HTML includes requested colorscale controls and default blue-orange."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload, _render_html

    adata = _make_bae_adata()
    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    html = _render_html(payload, title="Colorscale Explorer")

    assert 'id="colorscale-select"' in html
    assert 'value="blueorange" selected' in html
    assert 'value="redblue"' in html
    assert 'value="viridis"' in html


def test_render_html_uses_active_colorscale_for_scatter_and_coeff():
    """UMAP and coefficient plot both use the selected continuous colorscale."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload, _render_html

    adata = _make_bae_adata()
    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    html = _render_html(payload, title="Colorscale Sync Explorer")

    assert "const COLOR_SCALES =" in html
    assert "trace.marker.colorscale = getActiveColorscale();" in html
    assert "colorscale: getActiveColorscale()," in html


def test_render_html_no_grid_and_dot_size_slider():
    """UMAP disables grid and exposes smooth point-size control."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload, _render_html

    adata = _make_bae_adata()
    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    html = _render_html(payload, title="Point Size Explorer")

    assert 'id="point-size"' in html
    assert 'step="0.5"' in html
    assert 'pointSizeInput.addEventListener("input"' in html
    assert "xaxis: { zeroline: false, showgrid: false }" in html


def test_render_html_has_obs_and_layer_switch_controls():
    """HTML includes explicit selectors for obs columns and expression layers."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload, _render_html

    adata = _make_bae_adata()
    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    html = _render_html(payload, title="Obs Layer Controls Explorer")

    assert 'id="obs-select"' in html
    assert 'id="layer-select"' in html
    assert 'o.value = "obs";' in html
    assert 'obsSel.addEventListener("change"' in html
    assert "function updateControlStates()" in html


def test_render_html_defaults_when_no_obs_or_layers():
    """HTML handles datasets with no selectable obs columns or layers."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload, _render_html

    adata = _make_bae_adata()
    # Force no obs options and no expression layer options.
    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=[],
        obs_keys=[],
        top_k=5,
        annotations_key=None,
    )

    html = _render_html(payload, title="No Obs No Layer Explorer")

    assert "No obs columns available" in html
    assert "No layers available" in html
    assert "HAS_OBS_OPTIONS" in html
    assert "HAS_LAYER_OPTIONS" in html


def test_render_html_syncs_top_gene_colors_with_selected_colorscale():
    """Top-gene list uses colors that follow the active colorscale choice."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload, _render_html

    adata = _make_bae_adata()
    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key=None,
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    html = _render_html(payload, title="Gene List Colors Explorer")

    assert ".gene-pos { color: var(--pos-color);" in html
    assert ".gene-neg { color: var(--neg-color);" in html
    assert 'document.documentElement.style.setProperty("--pos-color", palette.pos);' in html
    assert 'document.documentElement.style.setProperty("--neg-color", palette.neg);' in html


def test_render_html_spatial_mode():
    """Rendered HTML includes tissue plot div when spatial data present."""
    _require_anndata()
    from structboost._explorer import _build_explorer_payload, _render_html

    adata = _make_bae_adata()
    rng = np.random.default_rng(7)
    adata.obsm["spatial"] = rng.normal(size=(30, 2)).astype(np.float32)

    payload = _build_explorer_payload(
        adata,
        model_key="BAE_encoder_weights",
        embedding_key="X_umap",
        spatial_key="spatial",
        latent_key="X_bae",
        layers=None,
        obs_keys=None,
        top_k=5,
        annotations_key=None,
    )

    html = _render_html(payload, title="Spatial Explorer")
    assert "tissue-plot" in html
    assert 'id="point-size"' in html
    assert 'makeScatter("tissue-plot"' in html


def test_export_html_basic(tmp_path):
    """export_interactive_html writes a valid HTML file."""
    _require_anndata()
    from structboost._explorer import export_interactive_html

    adata = _make_bae_adata()
    out = tmp_path / "explorer.html"

    result = export_interactive_html(adata, out)

    assert result == out
    assert out.exists()
    content = out.read_text()
    assert "plotly" in content.lower()
    assert "umap-plot" in content


def test_export_html_missing_embedding_raises():
    """KeyError when embedding_key is not in adata.obsm."""
    _require_anndata()
    from structboost._explorer import export_interactive_html

    adata = _make_bae_adata()
    del adata.obsm["X_umap"]

    with pytest.raises(KeyError, match="X_umap"):
        export_interactive_html(adata, "/tmp/nope.html")


def test_export_html_missing_model_key_raises():
    """KeyError when model_key is not in adata.varm."""
    _require_anndata()
    from structboost._explorer import export_interactive_html

    adata = _make_bae_adata()
    del adata.varm["BAE_encoder_weights"]

    with pytest.raises(KeyError, match="BAE_encoder_weights"):
        export_interactive_html(adata, "/tmp/nope.html")


def test_export_html_auto_detects_latent_key(tmp_path):
    """latent_key is auto-detected from model_key."""
    _require_anndata()
    from structboost._explorer import export_interactive_html

    adata = _make_bae_adata()
    out = tmp_path / "auto.html"

    # latent_key=None should auto-detect "X_bae"
    result = export_interactive_html(adata, out, latent_key=None)
    assert result == out
    assert out.exists()


def test_export_html_auto_detects_annotations(tmp_path):
    """Annotations are auto-detected from adata.uns."""
    _require_anndata()
    from structboost._explorer import export_interactive_html

    adata = _make_bae_adata()
    adata.uns["bae_dimension_annotations"] = {
        "0": {
            "positive_annotation": "T cells",
            "negative_annotation": "B cells",
            "overall_annotation": "T/B axis",
            "database_references": [],
        }
    }
    out = tmp_path / "ann.html"

    export_interactive_html(adata, out)
    content = out.read_text()
    assert "T/B axis" in content


def test_export_html_with_subsampling(tmp_path):
    """Subsampling is applied when n_cells > max_cells."""
    _require_anndata()
    from structboost._explorer import export_interactive_html

    adata = _make_bae_adata(n_cells=50)
    out = tmp_path / "sub.html"

    with pytest.warns(UserWarning, match="Subsampling"):
        export_interactive_html(adata, out, max_cells=20)

    assert out.exists()


def test_public_api_import():
    """export_interactive_html is importable from structboost."""
    from structboost import export_interactive_html

    assert callable(export_interactive_html)
