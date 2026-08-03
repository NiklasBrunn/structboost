# structboost

> **Note that this package is under active development.** The API is still
> moving, and a minor version bump may break it.

**Structured representation learning for single-cell data. A latent space you can
read gene by gene.**

The **Boosting Autoencoder (BAE)** pairs a linear encoder fitted by componentwise
L2 boosting with an MLP decoder trained by gradient descent. Each training
iteration takes a gradient step on the latent code itself and hands the result to
the boosting fit as a regression target, so the encoder is fitted against the
negative gradient of the reconstruction loss rather than by backpropagation.
Componentwise boosting adds one gene at a time and shrinks each step, which keeps
the encoder weights sparse by construction rather than by a post-hoc threshold.

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

A single fit gives one gene list, and that list is **not reproducible**. In a
high-dimensional feature space with strongly correlated genes the encoder support
is not identifiable: many different sparse gene sets reconstruct the data about
equally well, and a fit returns one of them.

```python
res = model.stability_selection(adata, mode="iteration")
genes = [adata.var_names[res.stable_support[:, j]] for j in range(res.frequency.shape[1])]
```

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
| [Transfer](https://niklasbrunn.github.io/structboost/guide/tasks/transfer.html) | Carry a trained encoder matrix onto a new dataset with `from_reference`, aligned by gene name, with the prior programs frozen. |
| [Persistence](https://niklasbrunn.github.io/structboost/guide/tasks/persistence.html) | `save` and `load` a fitted model as one checkpoint, readable with `weights_only=True`. |
| [Interpretation](https://niklasbrunn.github.io/structboost/guide/tasks/interpreting.html) | Ranked gene lists per dimension, stored functional annotations, and a self-contained interactive HTML explorer. |
| [Simulation](https://niklasbrunn.github.io/structboost/guide/tasks/simulating.html) | Negative-binomial counts with planted gene programs and a cell-type hierarchy, so marker recovery can be scored against ground truth. |

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the versioning
policy, and what a pull request needs. Planned work is tracked in the
[issue tracker](https://github.com/NiklasBrunn/structboost/issues).

Licensed under the [MIT License](LICENSE).
