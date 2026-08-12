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

{class}`~structboost.BAEConfig` has 24 fields. Most have measured defaults you
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
| `boosting_nu` | `0.1` | Boosting learning rate. See below before raising it. |
| `boosting_csf` | `0.9` | `<1` promotes diversity, `>1` reinforces selected features. |
| `boosting_independent` | `True` | Reset boosting state per latent dimension. |
| `nuisance_ridge` | `0.0` | Stabilizes near-collinear batch covariates. See {doc}`batch-integration`. |

These defaults replaced an earlier `0.3` / `100` pairing, which over-selected
badly: marker-recovery F1 of 0.72/0.97/0.57 across three
simulated scenarios, against 0.98/1.00/0.98 for the current defaults.

`nu` and `stepno` are not independent. The effective ridge degrees of freedom of
a boosting step *is* `nu`, so a fit's per-iteration complexity budget is roughly
`stepno * nu`: 5 for the current defaults, 30 for the rejected pairing.

:::{warning}
**`boosting_nu=0.3` at the default `stepno=50` is not safe under the current
stopping rule.** Measured on the same three scenarios, it reaches a marker-recovery
F1 of 0.975–0.987 within 18–57 iterations — as good as `0.1`, three to four times
faster — and then *degrades* to 0.57–0.78 if training continues, while the
training MSE keeps falling. Early stopping watches that MSE, so nothing detects it,
and on two of three seeds the fit ran to `max_iterations`.

On low-signal data the picture reverses: on the `hard` scenario (weak effects,
dropout, ambient contamination) `0.3` reaches F1 0.787 against `0.1`'s 0.406, and
needs ~560 iterations to get there.

So the right value depends on the regime, and the stopping rule cannot currently
tell them apart. Until it can, `0.1` is the safe default. If you raise it, watch
`n_iterations` against `max_iterations` and `variance_explained` against
{func}`~structboost.linear_ceiling` — a fit that exhausts its iteration budget
with variance explained *above* the linear ceiling is in the degrading regime.
:::

`boosting_precompute_covcache` decides how the predictor Gram matrix is cached.
The default `"auto"` builds it in full up front whenever 8·*p*² bytes fit a
conservative share of system memory, and falls back to a lazy column cache when
they do not. `True` and `False` are honoured exactly, and the resolved decision
is recorded in `adata.uns["bae"]["boosting_precompute_covcache"]`.

:::{note}
This reads like a memory setting and is really a **speed** setting. The lazy
cache computes a column the first time its gene is selected, so its memory
scales with the number of *distinct selected* genes rather than with *p*² — it
is strictly cheaper in memory. But building all *p* columns at once is one
compute-bound matrix product running near hardware peak, while fetching them one
at a time is a sequence of memory-bound matrix-vector products. Precomputing
wins from roughly *p*/59 distinct selected genes onwards, and a fit passes that
in its first iteration, where up to `boosting_stepno × latent_dim` genes can
enter. Measured end-to-end: **1.6–1.9× faster** at *p* = 2,000–8,000.

The cost is real memory: 8·*p*² is 32 MB at *p* = 2,000, 800 MB at 10,000 and
3.2 GB at 20,000, transiently doubled while it is built. That is what `"auto"`
guards. Set `False` explicitly on a memory-constrained machine.
:::

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
mathematically produce orthogonal residuals. Its projection also carries no
intercept, so it assumes near-centered targets — true of `z*` in practice, since
`z = X @ W` on z-scored `X` is centered, but nothing enforces it.

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

**Leave `target_optim_lr` at 1.0.** It is exposed because it is a real
coefficient of the method, not because it is a knob: it scales the entire
coupling between the two optimizers, and the quantity it scales has no natural
unit. `∂L/∂z` tracks the decoder's magnitude, which grows by orders of magnitude
over a fit, so a value that suits one training stage will not suit the next. The
alternation is what adapts; this stays fixed.

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
