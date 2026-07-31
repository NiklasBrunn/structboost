# Interpreting latent dimensions

A latent dimension is a signed, weighted gene list. These are the tools for
reading it.

## Ranked gene lists

```python
from structboost import extract_gene_rankings

rankings = extract_gene_rankings("fitted.h5ad", top_k=50)

for dim in rankings:
    print(dim.dimension, dim.n_selected_genes)
    print("  up:  ", [g.gene for g in dim.positive_genes.genes[:10]])
    print("  down:", [g.gene for g in dim.negative_genes.genes[:10]])
```

Positive and negative loadings are kept separate on purpose, a dimension usually
contrasts two programs, and collapsing the sign loses that.

:::{note}
This reads a `.h5ad` **from a path**, not an in-memory AnnData. Write your fitted
object out first.
:::

## Storing functional annotations

Once you (or a language model, or a colleague) have interpreted the dimensions,
store the result next to the model:

```python
from structboost import DimensionAnnotation, write_annotations_to_h5ad

annotations = [
    DimensionAnnotation(
        dimension=0,
        positive_annotation="Cytotoxic effector program (GZMB, PRF1, NKG7)",
        negative_annotation="Naive T-cell program (CCR7, SELL, TCF7)",
        overall_annotation="Effector-naive axis",
        database_references=["https://www.gsea-msigdb.org/..."],
    ),
]
write_annotations_to_h5ad("fitted.h5ad", annotations)
```

They land in `adata.uns["bae_dimension_annotations"]`, keyed by dimension index
as a string, and the interactive explorer picks them up automatically.

:::{warning}
`write_annotations_to_h5ad` reads the file, mutates it, and **writes back over the
same path**. Keep a copy if that matters.
:::

## Interactive explorer

A single self-contained HTML file with no server and no dependencies at view time,
that you can send to a collaborator:

```python
from structboost import export_interactive_html

export_interactive_html(adata, "explorer.html", title="PBMC BAE")
```

It gives you a UMAP scatter colored by latent dimension, gene expression or obs
column, an encoder-coefficient bar chart per dimension, gene-expression overlays
from any layer, a spatial panel when `obsm["spatial"]` exists, and the dimension
annotations from above.

Requirements worth knowing before it raises:

- `embedding_key` (default `"X_umap"`) must be **precomputed**. This does not
  run UMAP for you. Same for `model_key` and `latent_key`.
- `spatial_key` is *silently omitted* if absent, rather than raising.
- Above `max_cells` (default 50,000) it randomly subsamples, with a warning.

## Diagnostic plots

```python
from structboost import plot_training_diagnostics, plot_top_boosting_coefficients

plot_training_diagnostics(model)         # also accepts a TrainingReport or AnnData

plot_top_boosting_coefficients(
    adata.varm["BAE_encoder_weights"].T,  # note the transpose
    gene_names=adata.var_names,
    cluster_labels=[f"dim {i}" for i in range(model.config.latent_dim)],
    top_m=10,
)
```

`plot_training_diagnostics` needs a fit with `diagnostics=True`. It raises with a
clear message otherwise. See {doc}`../concepts/reading-quality` for what the six
panels mean.

`plot_boosting_coefficient_paths` is an `allboost` tool rather than a BAE one. It
plots coefficient trajectories from `allboost(..., return_history=True)`. See
{doc}`allboost`.

These need the `[plot]` extra.

## A sanity check worth doing

Before interpreting anything, confirm the dimension is stable. A gene list from a
single fit is one arbitrary position on the optimizer's plateau. See
{doc}`gene-selection`. Interpreting an unstable dimension is interpreting noise
with a confident vocabulary.

```python
res = model.stability_selection(adata, mode="iteration")
stable = [adata.var_names[res.stable_support[:, j]] for j in range(res.frequency.shape[1])]
```

## Related

- {func}`~structboost.extract_gene_rankings`, {func}`~structboost.write_annotations_to_h5ad`
- {class}`~structboost.DimensionAnnotation`, {class}`~structboost.GeneRanking`
- {func}`~structboost.export_interactive_html`
