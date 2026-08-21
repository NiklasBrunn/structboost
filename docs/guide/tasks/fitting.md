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

:::{admonition} Exploratory
:class: caution
Under active development. Both methods are provisional, and the measurements
below come from three datasets at three seeds each.
:::

Latent dimensions can end up redundant. **Since 0.5.0 the default is
`disentanglement="orthogonal"`**, which replaces the boosting targets with the
nearest mutually orthogonal set of the same column norms — symmetric Löwdin
orthogonalization, computed from the thin SVD as `U @ Vt`.

| Field | Default | Notes |
| --- | --- | --- |
| `disentanglement` | `"orthogonal"` | `"none"` and `"correlation"` also available. |
| `disentanglement_alpha` | `1.0` | Strength of the orthogonality constraint, in `[0, 1]`. |
| `disentanglement_lambda` | `1e-2` | Correlation-penalty strength; **larger decorrelates more**. |

**Softening it.** `disentanglement_alpha` interpolates:
`(1 - alpha) * targets + alpha * orthogonalized`. At `1.0` the targets are exactly
orthogonal, at `0.0` untouched — so **decorrelation can be softened by choosing an
alpha strictly between 0 and 1**. The response is monotone and the scale is
bounded, so no calibration sweep is needed to find a usable range.

```python
BAEConfig(latent_dim=10, disentanglement_alpha=0.5)   # half-strength
BAEConfig(latent_dim=10, disentanglement="none")      # off
```

**`"correlation"` is the alternative**, adding a soft squared-correlation penalty
to the target objective instead of transforming the targets, plus a variance
barrier so low correlation cannot be achieved by collapsing dimensions. It
**requires manual tuning**, and **larger `disentanglement_lambda` gives stronger
decorrelation**.

:::{warning}
The old `disentanglement_lambda=1e-4` default was measured to be **inert** — on
simulated data, Tasic mouse cortex and human pancreas it moved the mean absolute
latent correlation by less than its own seed-to-seed noise, and so did `1e-3`. On
pancreas, mean `|corr|` ran 0.103 (off), 0.100 (`1e-4`), 0.112 (`1e-3`), 0.080
(`1e-2`), 0.061 (`1e-1`). The default is now `1e-2`; sweep upward from there
(`1e-2, 3e-2, 1e-1, 3e-1`) rather than downward.
:::

**Expect either method to cost biological structure.** Decorrelation is an extra
constraint on the latent space, and real gene programs are not orthogonal —
overlapping pathways and shared markers are the norm — so it trades fidelity to
that structure, and the interpretability resting on it, for a less redundant
representation. Measured against ground truth on simulated data, orthogonalization
moved marker-recovery F1 from 0.98 to 0.88; on Tasic and pancreas it lowered the
share of latent variance the annotated cell type explains. `"none"` is a
reasonable choice, and `disentanglement_alpha` exists so the trade can be made
partially.

### Optimization and stopping

| Field | Default |
| --- | --- |
| `target_optim_lr` | `1.0` |
| `decoder_lr` | `1e-3` |
| `max_iterations` | `1000` |
| `enable_early_stopping` | `False` |
| `early_stopping_patience` | `50` |
| `batch_size` | `512` |

### `batch_size` sets the decoder budget

Each iteration gives the decoder **one shuffled pass** over the cells:
`ceil(n_cells / batch_size)` AdamW steps, with every cell contributing to exactly
one. There is no separate step-count setting — `decoder_updates_per_iteration`
was removed in 0.5.0.

The reason is that the boosting half is full-batch at every dataset size: `z*` is
computed on all cells and the encoder is re-solved on all cells, every iteration.
A fixed step count made the decoder's share shrink as data grew — at the old
`10 × 512` that was every cell below 5,120, 31% at 16,000 and 5% at 100,000.

:::{warning}
Two consequences worth planning for.

**Fit time now grows with cell count**, because `max_iterations` does not decay to
compensate. Raise `batch_size` to buy it back, at the price of fewer, noisier
steps.

**On small datasets you may want a smaller `batch_size` than you think.** At the
default 512, anything under ~1,000 cells gets one or two decoder steps per
iteration, and effects that need the decoder to move within an iteration weaken
or reverse. Measured on 300 cells, a PCA warm start *raised* the first
iteration's loss at one step per iteration (1.083 against 1.023 for a zero start)
and only paid off once the pass held enough steps (0.920 against 1.000 at
`batch_size=8`). If a fit on few cells looks sluggish, lower `batch_size` before
reaching for anything else.
:::

**Leave `target_optim_lr` at 1.0.** It is exposed because it is a real
coefficient of the method, not because it is a knob: it scales the entire
coupling between the two optimizers, and the quantity it scales has no natural
unit. `∂L/∂z` tracks the decoder's magnitude, which grows by orders of magnitude
over a fit, so a value that suits one training stage will not suit the next. The
alternation is what adapts; this stays fixed.

### Early stopping is off by default

Changed in 0.4.0. The criterion is a *convergence* check being used as a *quality*
check, and it stops far too early. There is no validation split, so the training
loss cannot see a model that is starting to memorize.

Measured against ground truth, patience-50 stopped at iteration 90 on a dataset
whose marker recovery peaked at 259, and at 196 on a simulated scenario peaking at
560 — returning F1 0.502 against 0.787. Across three real datasets the best
iteration ranged from **154 to 1975**, always past where patience fires.

:::{warning}
**No stopping rule replaced it, because none was found that works.** Latent
stability, encoder-support overlap and held-out reconstruction were each measured
as candidates. The representation settles long before gene selection does — on one
dataset consecutive latent codes were rank-identical while the gene set still
turned over 65% cumulatively — and no observable signal tracks the quality peak.
The best iteration varies by an order of magnitude between datasets, so no fixed
budget works either.

Run the iteration budget and let {meth}`~structboost.BAE.stability_selection`
absorb the variance. That is what it is for.
:::

`max_iterations=1000` is a defensible middle of the measured 154–1975 range, not an
optimum. Nothing better is available.

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

:::{admonition} Exploratory
:class: caution
Under active development. Starting from an existing representation works, but it
is applied only once, it can silently change `latent_dim`, and its benefit is
conditional on the decoder having enough steps in the first iteration to follow
it — see the warning below.
:::

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
