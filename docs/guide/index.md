# User guide

Everything the BAE can do, grouped by the question it answers.

## Concepts

Read these once. They explain choices that otherwise look arbitrary.

- **{doc}`concepts/how-it-works`**: the boosting/SGD alternation, why the
  encoder is reset to zero every iteration, and where sparsity actually comes
  from.
- **{doc}`concepts/standardization`**: why the input must be z-scored, and the
  second job that contract quietly does.
- **{doc}`concepts/anndata-contract`**: every key `fit` writes into your
  AnnData, and which two metrics are routinely confused with each other.
- **{doc}`concepts/reading-quality`**: why a reconstruction MSE means nothing on
  its own, and how to decide between "more dimensions" and "more iterations".

## Capabilities

### Fitting a model

The core loop: an AnnData in, a sparse encoder and a latent space out.
{doc}`tasks/fitting` covers the minimal path, the config fields worth touching
(and the several that benchmarks say to leave alone), early stopping, and
warm starts from an existing embedding or a PCA.

→ {class}`~structboost.BAE`, {class}`~structboost.BAEConfig`

### Getting a gene list you can trust

A single fit reports one gene list, and that list is **not reproducible**: the
reconstruction loss converges long before the encoder support does. Two stability
modes quantify different sources of that instability, and `mandatory_genes`
forces known markers into the model specification.

{doc}`tasks/gene-selection` also covers the part most easily misread, 
`mandatory_genes` forces genes into the *specification*, not into the fitted
support, and neither stability mode gives you usable error control.

→ {meth}`BAE.stability_selection <structboost.BAE.stability_selection>`,
{class}`~structboost.StabilitySelectionResult`,
{meth}`BAE.apply_encoder <structboost.BAE.apply_encoder>`

### Handling batches and unwanted covariates

Name the covariate once with `batch_key`, and `batch_integration_mode` decides
which of two mechanisms act on it: giving it to the decoder, regressing it out
inside the boosting fit so it cannot confound gene selection, or both, which is
the default. A separate `balance_obs` reweights groups, with an important limit
on how far that reweighting reaches.

`transform` stays gene-only and needs no batch labels. That is what makes the
encoder deployable on data with no covariate annotation.

→ {doc}`tasks/batch-integration`

### Carrying a model to a new dataset

The transferable product of a fit is its encoder matrix: *k* sparse gene
programs. {meth}`~structboost.BAE.from_reference` aligns such a matrix to a new
gene panel by gene name, freezes it, and adds dimensions for the structure the
prior cannot explain.

→ {doc}`tasks/transfer`, {func}`~structboost.read_encoder_weights`,
{func}`~structboost.write_encoder_weights`

### Working out what a dimension means

Ranked gene lists per dimension, storage for functional annotations, an
interactive HTML explorer, and diagnostic plots.

→ {doc}`tasks/interpreting`, {func}`~structboost.extract_gene_rankings`,
{func}`~structboost.export_interactive_html`

### Saving a fitted model

One `.pt` checkpoint holding the encoder, decoder, config, covariate encodings
and diagnostics, and deliberately not holding the optimizer state or any
cell-level training data.

→ {doc}`tasks/persistence`, {meth}`BAE.save <structboost.BAE.save>`,
{meth}`BAE.load <structboost.BAE.load>`

### Testing a method against known truth

A negative-binomial simulator with planted gene programs, a cell-type hierarchy,
and optional batch, ambient-RNA and dropout effects, so you can score marker
recovery against ground truth.

→ {doc}`tasks/simulating`, {func}`~structboost.sim_scrnaseq_anndata`

### Sparse supervised boosting on its own

`allboost` without the autoencoder: a sparse coefficient matrix mapping features
to targets. Regressing cluster indicators on genes gives per-cluster marker
signatures directly.

→ {doc}`tasks/allboost`, {func}`~structboost.allboost`

## Caveats worth reading before you trust a result

Each is explained where it belongs, and each has bitten someone:

```{list-table}
:header-rows: 1
:widths: 40 60

* - Caveat
  - Where
* - `mandatory_genes` does not guarantee a non-zero weight
  - {doc}`tasks/gene-selection`
* - `balance_obs` does not balance *gene selection*, only targets and losses
  - {doc}`tasks/batch-integration`
* - Subsample-mode error bounds are violated by ~10× on ground-truth data
  - {doc}`tasks/gene-selection`
* - `novel_variance_share` on training data is not evidence of novel biology
  - {doc}`tasks/transfer`
* - Installing a stability-selected encoder on a frozen transfer can silently zero the prior
  - {doc}`tasks/transfer`
* - After a transfer, hand `X_bae_scaled`, not `X_bae`, to any Euclidean tool
  - {doc}`tasks/transfer`
* - A mismatched gene panel warns. It does not raise
  - {doc}`concepts/anndata-contract`
* - A bare reconstruction MSE is uninterpretable
  - {doc}`concepts/reading-quality`
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Concepts

concepts/how-it-works
concepts/standardization
concepts/anndata-contract
concepts/reading-quality
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Tasks

tasks/fitting
tasks/gene-selection
tasks/batch-integration
tasks/transfer
tasks/interpreting
tasks/persistence
tasks/simulating
tasks/allboost
```
