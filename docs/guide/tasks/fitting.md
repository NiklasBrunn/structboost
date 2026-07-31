# Fitting a model

## The minimal path

```python
from structboost import BAE, BAEConfig

config = BAEConfig(latent_dim=10)
model = BAE(adata.n_vars, config)
model.fit(adata)                    # adata.X must be z-scored

latent = adata.obsm["X_bae"]
weights = adata.varm["BAE_encoder_weights"]
```

`fit` returns the model, so it chains. `fit_transform(adata)` is exactly
`fit(...)` followed by returning `adata.obsm["X_bae"]`, with no extra behaviour, in
case you were looking for some.

Reading from a layer instead of `adata.X`:

```python
model.fit(adata, layer="scaled")
```

The layer is remembered for every later call. See {doc}`../concepts/anndata-contract`.

## Configuration

{class}`~structboost.BAEConfig` has 28 fields. Most have measured defaults you
should leave alone. These are the ones worth thinking about.

### Architecture

| Field | Default | Notes |
| --- | --- | --- |
| `latent_dim` | `10` | Number of gene programs to learn. |
| `decoder_hidden_dims` | `(64,)` | One hidden layer. See below. |
| `decoder_activation` | `"tanh"` | Most stable. `"elu"` matched it. |
| `split_softmax` | `False` | Compositional latent, for soft clustering. |

:::{warning}
**Do not narrow the decoder toward the output.** A funnel such as `(128, 64)`
dropped marker-recovery F1 to 0.80 in benchmarks, against 0.99–1.00 for `(64,)`.
A narrow layer immediately before the gene output distorts the reconstruction
gradient that the boosting target is built from, and that gradient is what the
encoder is fitted against, so the damage lands on gene selection specifically. If
you need capacity, **widen**: `(64, 128)`.

`decoder_use_batch_norm=True` is likewise measured as harmful (F1 0.59–0.73): the
target is computed with the decoder in eval mode and the update applied in train
mode, so batch norm makes the target inconsistent with the decoder that produced
it.

A linear decoder (`decoder_hidden_dims=()`) is a legitimate faster, more
interpretable choice, at a cost: F1 down to 0.83 and 65–73% of the linear ceiling
on harder data.

These are simulation results. Confirm on your data before assuming `(64,)` has
enough capacity for genuinely nonlinear structure.
:::

### Boosting: where sparsity comes from

| Field | Default | Notes |
| --- | --- | --- |
| `boosting_stepno` | `50` | **The sparsity dial.** Caps genes per dimension per iteration. |
| `boosting_nu` | `0.1` | Boosting learning rate. |
| `boosting_csf` | `0.9` | `<1` promotes diversity, `>1` reinforces selected features. |
| `boosting_independent` | `True` | Reset boosting state per latent dimension. |
| `nuisance_ridge` | `0.0` | Stabilizes near-collinear batch covariates. See {doc}`batch-integration`. |

These defaults replaced an earlier `0.3` / `100` pairing, which over-selected
badly: marker-recovery F1 of 0.72/0.97/0.57 across three
simulated scenarios, against 0.98/1.00/0.98 for the current defaults.

`boosting_precompute_covcache=True` builds the full *p*×*p* Gram matrix up front, 
8·*p*² bytes, or 3.2 GB at 20,000 genes. The default lazy cache computes a column
the first time its gene is selected, so memory scales with the number of
*distinct selected* genes instead. Precompute only when *p* is small.

### Disentanglement

Latent dimensions can end up redundant. `disentanglement="correlation"` adds a
soft squared-correlation penalty to the target objective, plus a variance barrier
so that low correlation cannot be achieved by collapsing dimensions. Unlike a
covariance penalty, it cannot be reduced merely by shrinking every dimension and
letting the decoder compensate with larger weights:

```python
BAEConfig(latent_dim=10, disentanglement="correlation", disentanglement_lambda=1e-4)
```

`1e-4` is a conservative starting point from simulations, not a universal
optimum. A useful sweep: `0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3`.

`"leave_one_out"` is the earlier experimental method, retained but not
recommended. It residualizes each target against the others and does not
mathematically produce orthogonal residuals.

### `standardize_targets` is a modelling choice

Off by default, and worth understanding before switching on. For a fixed decoder
it does **not** change which genes get selected. Rescaling a target column by
*c* scales the selection criterion by *c*², leaving every argmax untouched. What
it changes is the encoder magnitude, and therefore every later decoder update and
every later target.

It equalizes target variance across strong and weak dimensions, which constrains
the latent scale and can stabilize split-softmax. The cost: it promotes
near-empty dimensions, discards the scale of a warm start, and on simulated data
raises selection precision while roughly halving recall.

### Optimization and stopping

| Field | Default |
| --- | --- |
| `target_optim_lr` | `1.0` |
| `decoder_lr` | `1e-3` |
| `decoder_updates_per_iteration` | `10` |
| `max_iterations` | `1000` |
| `enable_early_stopping` | `True` |
| `early_stopping_patience` | `50` |
| `batch_size` | `512` |

`fit` always restores the best checkpoint's encoder, decoder and nuisance weights
at the end, whether or not early stopping triggered.

The objective early stopping watches is **not always the training loss**: with
`disentanglement="correlation"` it includes the weighted penalty, so that
restoring the best-reconstruction checkpoint cannot silently undo the constraint.

Per-call overrides beat the config, and are compared against `None` rather than
truthiness, so an explicit `0` overrides instead of falling back:

```python
model.fit(adata, max_iterations=50, seed=0, verbose=False)
```

## Warm starts

Initialize the latent code from an existing embedding or a PCA instead of from
zero:

```python
model.fit(adata, init_pca=True)
model.fit(adata, init_obsm="X_pca")
model.fit(adata, init_pca=True, init_pretrain_epochs=20)
```

`init_obsm` and `init_pca` are mutually exclusive.

:::{warning}
**The warm start applies once**, on the first iteration only. It sets the
boosting targets. The encoder learns that representation, and the decoder is then
trained against the encoder's output.

**It can change `latent_dim` silently-ish.** If the supplied representation has a
different number of columns than `config.latent_dim`, the representation wins:
`latent_dim` is overwritten for this fit, the encoder is rebuilt, and a
`UserWarning` is emitted.

`init_pretrain_epochs` pre-fits the decoder against the fixed warm-start latent.
Measured trade-off: it lowers the *initial* loss substantially but does not
improve the converged loss, and large values noticeably reduce gene-selection
precision, because the decoder is tuned to a latent code the sparse encoder
cannot exactly reproduce. 20 is conservative.

None of these can be combined with a transfer model. See {doc}`transfer`.
:::

## Reproducibility

```python
BAEConfig(latent_dim=10, seed=42)
model.fit(adata, seed=42)          # per-call override
```

A seeded fit is reproducible and **does not disturb your own RNG state**: the global torch stream is restored when `fit` returns, and NumPy's
global generator is never touched at all. Constructing a `BAE` still advances the
torch generator, as constructing any `nn.Module` does.

Two unseeded fits still differ from each other, deliberately.

## Related

- {doc}`../concepts/how-it-works`: what the loop is actually doing
- {doc}`../concepts/reading-quality`: is the result any good?
- {doc}`gene-selection`: is the gene list reproducible?
