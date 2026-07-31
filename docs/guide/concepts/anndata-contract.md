# What a fit writes into your AnnData

`fit` mutates `adata` in place and returns the model. Everything below is written
by {meth}`~structboost.BAE.fit` unless noted.

## `obsm`

```{list-table}
:header-rows: 1
:widths: 30 20 50

* - Key
  - Shape
  - Meaning
* - `X_bae`
  - `(n_cells, latent_dim)`
  - The latent space. Exactly `X @ varm["BAE_encoder_weights"]`.
* - `X_bae_scaled`
  - `(n_cells, latent_dim)`
  - **Transfer models only.** Per-dimension standardized latent. see {doc}`../tasks/transfer`.
* - `X_bae_splitsoftmax`
  - `(n_cells, 2 * latent_dim)`
  - Only when you call {meth}`~structboost.BAE.transform_splitsoftmax`.
```

`transform` also writes `X_bae` (and `X_bae_scaled` for transfers), so calling it
on a query dataset annotates that dataset in place.

## `varm`

```{list-table}
:header-rows: 1
:widths: 40 25 35

* - Key
  - Shape
  - Written by
* - `BAE_encoder_weights`
  - `(n_genes, latent_dim)`
  - `fit`, `apply_encoder`
* - `BAE_iteration_frequency`
  - `(n_genes, latent_dim)`
  - `stability_selection(mode="iteration")`
* - `BAE_selection_frequency`
  - `(n_genes, latent_dim)`
  - `stability_selection(mode="subsample")`
* - `bae_program_weights`
  - `(n_genes, 2 * latent_dim)`
  - `transform_splitsoftmax`
```

Note the encoder matrix is stored transposed relative to the internal
`(latent_dim, n_genes)` layout, so it lines up with `adata.var`.

## `uns["bae"]`

Always present after a fit:

`latent_dim`, `is_fitted`, `training_history`, `latent_init`, `disentanglement`.

Present when the corresponding feature was used:

```{list-table}
:header-rows: 1
:widths: 35 65

* - Key
  - Written when
* - `layer`
  - the fit read a layer rather than `adata.X`
* - `disentanglement_lambda` / `disentanglement_standardize`
  - the matching disentanglement method was selected
* - `training_report`
  - `diagnostics=True`
* - `mandatory_genes`
  - `mandatory_genes` was passed
* - `batch_key`, `batch_columns`
  - `batch_key` was passed
* - `batch_weights`, `nuisance_ridge`
  - the mode included `"encoder"`
* - `balance_obs`
  - `balance_obs` was passed
* - `latent_obs_r2_per_dim`, `variance_explained`, `reconstruction_loss_by_obs`
  - any covariate argument was passed
* - `stability_selection`
  - a stability run was performed
* - `encoder_source`
  - `apply_encoder` installed an aggregated encoder
```

:::{note}
`layer` is absent, not `None`, when the fit read `adata.X`. AnnData's `uns`
cannot hold `None`, and no key is the honest encoding of "the default".
:::

:::{warning}
`variance_explained` is only written when a covariate argument was used. A plain
`fit(adata)` does not produce it. Compute it yourself against
{func}`~structboost.linear_ceiling` instead. See {doc}`reading-quality`.
:::

## The two metrics people confuse

Both appear only for conditioned fits, and they answer different questions.

`latent_obs_r2_per_dim`
: **An integration metric.** The fraction of each latent dimension's variance
  explained by the conditioned obs columns. Values near zero are the goal. A
  single high entry is a residual covariate axis worth inspecting.

`reconstruction_loss_by_obs`
: **A fairness metric.** Reconstruction MSE per group. It says whether the model
  fits all groups comparably, *not* whether it integrated them.

A group can reconstruct poorly in a perfectly integrated model, and a model can
reconstruct every group equally well while still routing batch through the
latent. Check the one that matches your question.

Per-group losses are skipped, with a warning, for any column with more than 50
levels. Beyond that the breakdown is per-cell noise rather than a summary.

## Gene panel identity

The encoder maps gene columns **by position**. Applying a model to a matrix with
the same width but a different gene order produces plausible, wrong numbers.

`transform` and `reconstruct` compare `adata.var_names` against the names
recorded at fit time and emit a `UserWarning` naming the first differing
positions. **It is a warning, not an error**, and the call proceeds. To move a model
onto a genuinely different panel, use
{meth}`~structboost.BAE.from_reference`, which aligns by gene name.

## Round-tripping to disk

Everything above survives `adata.write_h5ad(...)`. To persist the *model*, meaning the
decoder, the config and the covariate encodings, see {doc}`../tasks/persistence`.
the AnnData keys alone are not enough to reproduce `reconstruct`.
