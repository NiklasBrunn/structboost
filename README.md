# structboost

> **Note that this package is under active development.** The API is still
> moving, and a minor version bump may break it.

**Structured representation learning for single-cell data. A latent space you can
read gene by gene.**

The **Boosting Autoencoder (BAE)** pairs a linear encoder fitted by componentwise
L2 boosting with an MLP decoder trained by gradient descent. Each training
iteration takes a gradient step on the latent code itself and hands the result to
the boosting fit as a regression target, so the encoder is fitted against the
gradient-updated latent code rather than by backpropagation. The target is that
updated code rather than the gradient alone because the encoder is rebuilt from
zero every iteration: it has to reproduce where the code should be, not the
correction to where it already is. Componentwise boosting adds one gene at a time
and shrinks each step, which keeps the encoder weights sparse by construction
rather than by a post-hoc threshold.

Each latent dimension is therefore a short, signed gene list, and `X_bae` is
exactly `X @ varm["BAE_encoder_weights"]`.

The package also ships `allboost`, the componentwise boosting routine on its own,
for sparse supervised problems with no autoencoder involved.

## Relation to the original method

This is a scanpy-compatible Python re-implementation of the Boosting Autoencoder
introduced in [Hackenberg et al. (2025)](https://doi.org/10.1038/s42003-025-07872-9),
where the method and its componentwise boosting core were developed in Julia.

**Some methodological components differ from the original proposal.** Defaults
and several parts of the training procedure were re-derived here against
simulated data with known ground truth, and the measurements behind each are
recorded in the [user guide](https://niklasbrunn.github.io/structboost/guide/index.html)
next to the setting they justify. Results from this implementation should
therefore not be assumed identical to the original paper's.

📖 **[Documentation](https://niklasbrunn.github.io/structboost)** · 
[User guide](https://niklasbrunn.github.io/structboost/guide/index.html) · 
[API reference](https://niklasbrunn.github.io/structboost/api/index.html) · 
[Changelog](CHANGELOG.md)

## Installation

Requires Python 3.10 or newer. The core depends on NumPy alone, and everything
heavier is opt-in.

```bash
pip install "structboost[bae,plot]"     # the BAE
pip install structboost                 # allboost only, NumPy-only
```

| Extra | Brings in | Needed for |
| --- | --- | --- |
| *(none)* | numpy | `allboost`, `stability_selection` |
| `bae` | torch, anndata, scipy, pandas, tqdm | `BAE`, the simulator, everything AnnData |
| `plot` | matplotlib | the `plot_*` functions |
| `io` | pyarrow | Parquet encoder-weight files |

See [Installation](https://niklasbrunn.github.io/structboost/installation.html)
for the from-source and development setups.

## Quickstart

`adata.X` must be z-scored, which `sc.pp.scale` gives you.

```python
from structboost import BAE, BAEConfig

model = BAE(adata.n_vars, BAEConfig(latent_dim=10))
model.fit(adata)

adata.obsm["X_bae"]                 # (n_cells, 10) latent space
adata.varm["BAE_encoder_weights"]   # (n_genes, 10), sparse
```

To integrate over a batch or any other unwanted covariate, name the obs column:

```python
model.fit(adata, batch_key="batch")          # or ["batch", "donor"]

adata.uns["bae"]["latent_obs_r2_per_dim"]    # near zero means it worked
```

By default this both conditions the decoder on the covariate and adds it to the
boosting design as a mandatory regressor, so a batch-correlated gene is not
selected *because of* the batch. The covariate is never an encoder input, so
`transform` stays gene-only and needs no batch labels on new data.

A single fit gives one gene list, and that list is **not reproducible**: in a
high-dimensional feature space with strongly correlated genes the encoder support
is not identifiable, so many different sparse gene sets reconstruct the data about
equally well and a fit returns one of them. See
[gene selection](https://niklasbrunn.github.io/structboost/guide/tasks/gene-selection.html)
before trusting a single list.

Componentwise L2 boosting on its own, no autoencoder involved:

```python
from structboost import allboost

betamat = allboost(sourcemat, targetmat_std, stepno=20, nu=0.1)
# (n_targets, n_features), sparse
```

## What else it does

Each of these has a guide page.

| | |
| --- | --- |
| [Batch integration](https://niklasbrunn.github.io/structboost/guide/tasks/batch-integration.html) | `batch_key` names the covariate and `batch_integration_mode` chooses whether it conditions the decoder, protects gene selection, or both. `transform` stays gene-only and needs no batch labels. |
| [Mandatory features](https://niklasbrunn.github.io/structboost/guide/tasks/gene-selection.html) | `mandatory_genes` puts known markers in boosting's unpenalized adjustment block, so they are never subject to competitive selection. Flat or per latent dimension. It forces them into the *specification*, not into the fitted support. |
| [Reading a fit](https://niklasbrunn.github.io/structboost/guide/tasks/reading-dimensions.html) | `plot_latent_dimensions` draws one row per latent dimension — the sorted score curve, the per-gene contributions, and the scores split by any grouping — plus a UMAP grid and a dimension-correlation heatmap. Genes are ranked by their exact share of a dimension's variance, not by coefficient size. |
| [Persistence](https://niklasbrunn.github.io/structboost/guide/tasks/persistence.html) | `save` and `load` a fitted model as one checkpoint, readable with `weights_only=True`. |
| [Interpretation](https://niklasbrunn.github.io/structboost/guide/tasks/interpreting.html) | Ranked gene lists per dimension, stored functional annotations, and a self-contained interactive HTML explorer. |
| [Simulation](https://niklasbrunn.github.io/structboost/guide/tasks/simulating.html) | Negative-binomial counts with planted gene programs and a cell-type hierarchy, so marker recovery can be scored against ground truth. |

## Exploratory features

The whole package is pre-1.0, but these four are **under active development** and
less settled than the rest. They work, and each is documented with what is known
about it — but their behaviour, defaults and APIs are more likely to change, and
results from them warrant more scepticism than the core fit does.

| Feature | Why it is still exploratory |
| --- | --- |
| **Stability selection** (`BAE.stability_selection`) | Provides **no formal error control** — `expected_false_positives` is deliberately `NaN`, because training iterations are neither independent nor exchangeable. Per-dimension frequencies are only meaningful when dimensions keep their identity, which `dim_match_quality` reports and does not guarantee. |
| **Disentanglement** (`disentanglement=`) | On by default since 0.5.0, and both methods are provisional. Decorrelation is an extra constraint that real gene programs do not satisfy, so it costs biological structure — measured at marker-recovery F1 0.98 to 0.88 on simulated data. `disentanglement_alpha` softens it; `"none"` turns it off. |
| **Starting from an existing representation** (`init_pca`, `init_obsm`) | The warm start is applied once, on the first iteration, and silently overrides `latent_dim` if the supplied representation is a different width. It also depends on the decoder having enough steps in that first iteration to follow it — on few cells at the default `batch_size` the effect reverses. |
| **Starting from a prior encoder matrix** (`BAE.from_reference`) | Transfer works, but the two latent blocks land on **incomparable scales** — measured at a 232× gap in per-dimension standard deviation — so anything Euclidean must be handed `obsm["X_bae_scaled"]` rather than `X_bae`. `novel_variance_share` is not evidence of novel biology on the fitting data. |

## Citation

If you use the **BAE**:

> Hackenberg, M., Brunn, N., Vogel, T. et al. *Infusing structural assumptions
> into dimensionality reduction for single-cell RNA sequencing data to identify
> small gene sets.* Commun Biol 8, 414 (2025).
> <https://doi.org/10.1038/s42003-025-07872-9>

If you use **allboost**:

> Binder, H., Schumacher, M. *Incorporating pathway information into boosting
> estimation of high-dimensional risk prediction models.* BMC Bioinformatics 10,
> 18 (2009). <https://doi.org/10.1186/1471-2105-10-18>

## Development note

[Claude Code](https://claude.com/claude-code) (Anthropic) was used in building
this package, to support implementation, to write tests, and to write the
documentation. Individual commits record it as a co-author.

The methods, the design decisions, and the scientific claims are the authors'.
Everything committed was reviewed, and the behavioural claims in the docstrings
and the user guide are backed by the test suite or by the measurements cited
alongside them.

## Open points

Known gaps and planned work.

- [ ] **Revise the early-stopping criterion.** Early stopping is off by default
  because the training-loss rule is a convergence check being used as a quality
  check, and it stops well before gene selection has settled. No replacement has
  been found yet.
- [ ] **Stability selection.** Iteration mode carries no formal error control, and
  its per-dimension frequencies are only interpretable when `dim_match_quality` is
  high. A scheme with a defensible bound under a model fitted on the same cells is
  still open.
- [ ] **Multimodal architecture** — reconstruction-based, with a shared latent
  space across modalities, for paired single-cell data.
- [ ] **Contrastive objective**, as an alternative or addition to the
  reconstruction target the boosting step is currently fitted against.
- [ ] **Stochastic gradient boosting** ([Friedman
  2002](https://doi.org/10.1016/S0167-9473(01)00065-2)): subsample the cells at
  each boosting step, which is both a regularizer and a route to cheaper
  iterations on large datasets.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the versioning
policy, and what a pull request needs. Planned work is tracked in the
[issue tracker](https://github.com/NiklasBrunn/structboost/issues).

Licensed under the [MIT License](LICENSE).
