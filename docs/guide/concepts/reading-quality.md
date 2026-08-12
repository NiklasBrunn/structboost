# Reading reconstruction quality

A reconstruction MSE means nothing on its own. This page is about what to compare
it against.

## The zero baseline

On the z-scored input BAE expects, **predicting zero everywhere scores an MSE of
exactly 1.0**. So:

| MSE | Variance explained |
| --- | --- |
| 1.00 | 0%, no better than predicting the mean |
| 0.85 | 15% |
| 0.50 | 50% |

An MSE of 0.85 reads like "nearly perfect" and is nothing of the kind. Use
`variance_explained` rather than the raw loss:

```python
adata.uns["bae"]["variance_explained"]
```

## The achievable ceiling

Even 100% is the wrong target. Most per-gene variance in scRNA-seq is dropout and
sampling noise, so no low-dimensional model can reach it.
{func}`~structboost.linear_ceiling` gives the fraction of total variance that the
best possible *linear* model of the same width captures:

```python
from structboost import linear_ceiling

ceiling = linear_ceiling(adata, n_components=model.config.latent_dim)
achieved = adata.uns["bae"]["variance_explained"]

print(f"{achieved / ceiling:.0%} of the achievable maximum")
```

:::{important}
Pass the same layer the model was fitted on: `adata.uns["bae"]["layer"]`, or
omit it when the fit read `adata.X`. A ceiling computed on a different matrix is
not comparable with the model's reconstruction.
:::

### What the ratio tells you to do

This is the decision the ceiling exists to support:

**Near the ceiling** (say above 90%)
: The model is doing as well as its latent budget allows. More iterations will
  not help. **Give it more dimensions.**

**Far below the ceiling**
: The model is underfitting its own capacity. More dimensions will not help.
  **Raise `boosting_stepno` or `max_iterations`.**

Without the ceiling these two situations look identical. Both are "the MSE is
0.7", and the natural response of adding more training is right in exactly one
of them.

For reference, on simulated data a healthy fit lands at roughly 91–95% of the
linear ceiling with the default `(64,)` decoder.

## Is it converged?

The reconstruction loss converging does **not** mean the model has converged.
Measured on real data, the loss plateaus at ~95% of the achievable ceiling while
consecutive iterations still share only about a third of their selected genes.

If you care about the gene list, and if you did not, you would not be using this
method, then the loss curve is the wrong thing to watch. Fit with `diagnostics=True`
and read `weight_change_rel` and `support_jaccard` from the
{class}`~structboost.TrainingReport`:

```python
model.fit(adata, diagnostics=True)

report = model.training_report
report.weight_change_rel   # primary convergence signal, should approach 0
report.support_jaccard     # has the *gene set* settled?
```

```python
from structboost import plot_training_diagnostics

plot_training_diagnostics(model)   # six panels, also accepts the AnnData
```

Diagnostics cost extra full-data passes per iteration, which is why they are off
by default. They are strictly read-only: enabling them does not change the fitted
model.

### Which half is doing the work

The report separates the loss into the two halves of the alternation, measured at
three points per iteration:

- **A** `loss_pre_boost`: before anything changes
- **B** `loss_post_boost`: after the encoder is refit
- **C** `loss_post_decoder`: after the decoder steps

with `encoder_delta = B - A` and `decoder_delta = C - B`. A single loss curve
cannot tell you whether the boosting step is helping or whether the decoder is
merely repairing the damage it does. These two can.

Also worth watching: `target_grad_norm`. It is the signal the encoder is fitted
against. If it collapses toward zero, the boosting targets carry no information
and nothing further will be learned.

## Related

- {doc}`../tasks/gene-selection`: how reproducible is that gene list?
- {func}`~structboost.linear_ceiling`, {class}`~structboost.TrainingReport`
