# Transferring a model to a new dataset

:::{admonition} Exploratory
:class: caution
Under active development. Alignment, freezing and the diagnostics all work, but
the **latent-scale problem** below is a live limitation rather than a settled
design: the prior and novel blocks land on incomparable scales, so every
Euclidean consumer has to be handed `obsm["X_bae_scaled"]` instead of `X_bae`.
:::

The transferable product of a BAE fit is its **encoder matrix**: *k* sparse gene
programs. Not the decoder, not the latent space, but the gene programs.
{meth}`~structboost.BAE.from_reference` aligns such a matrix to a new gene panel
by gene name, freezes it, and adds dimensions for whatever the prior cannot
explain.

## End to end

```python
from structboost import BAE, write_encoder_weights

# 1. Export the reference (or pass the fitted model / its AnnData directly)
write_encoder_weights(
    reference_model.get_encoder_weights(),
    "reference.parquet",
    gene_ids=reference_adata.var["gene_ids"],
    gene_symbols=reference_adata.var_names,
    metadata={"species": "mus_musculus", "annotation_release": "GRCm39.110"},
)

# 2. Build the transfer model against the target panel
model = BAE.from_reference("reference.parquet", target, n_additional_dims=5)

# 3. Fit, settling the decoder against the prior first
model.fit(target, decoder_warmup_epochs=20)

# 4. Read what the new dimensions found
target.uns["bae_transfer"]["novel_variance_share_per_dim"]
model.novel_weights

# 5. Validate on held-out cells
model.transfer_diagnostics(heldout)
```

`from_reference` accepts a fitted {class}`~structboost.BAE`, an AnnData carrying
`varm["BAE_encoder_weights"]`, a gene-indexed DataFrame, or a `.parquet` / `.csv`
path. It **rejects a bare array**. Without gene identifiers, aligning it to
another panel would be guesswork.

The target AnnData is required at construction, not at fit, because the encoder's
shape depends on the gene panel and because a coverage failure should surface
before any training happens.

## Gene alignment

By default every identifier set present on both sides is tried and the one
matching the most genes wins, so a reference keyed on Ensembl accessions aligns
to a symbol-indexed dataset without intervention. The choice is recorded in
`uns["bae_transfer"]["join_key"]`. Set `join_on=` explicitly when a dataset
carries several identifier columns and the automatic choice must not be trusted.

`min_coverage=0.5` is the minimum share of a prior dimension's absolute weight
mass that must be present in the target. Below it, the transfer **raises** rather
than silently returning an attenuated program.

## Frozen or anchored

`prior_mode="frozen"` (default)
: The prior columns are never boosted. They stay **bitwise equal** to the
  reference matrix, so the transferred programs are literally the reference's.

`prior_mode="anchored"`
: The prior columns are boosted too, but each iteration restarts from the fixed
  original matrix rather than the previous iteration's result. This is boosting from an
  offset model in the sense of Bühlmann & Hothorn (2007). Deviation is bounded
  by `boosting_stepno` and never accumulates.

There is no third option. Unanchored re-fitting is not offered because a prior
used merely as a starting point is *forgotten* within a few hundred iterations, 
the encoder support random-walks, and two runs end no more similar than two
unrelated ones.

## Sizing `n_additional_dims`

The default of 5 is a pragmatic starting point, not something derived from your
data. How much residual structure a dataset holds is not knowable in advance.

Erring high is the cheaper mistake under `prior_mode="frozen"`: the transferred
programs stay bitwise fixed no matter how many dimensions you add. Too small a
value silently misses structure. `0` is valid and means decoder-only adaptation.

Afterwards, read `novel_variance_share_per_dim` and the novel dimensions'
stability, a dimension contributing almost nothing is surplus.

## Hyperparameters are not inherited

`latent_dim` is the only setting taken from the reference, derived as the prior's
dimensions plus `n_additional_dims`. Everything else comes from
{class}`~structboost.BAEConfig` defaults unless you pass `config=` yourself, so a
reference tuned to `boosting_stepno=50, max_iterations=200` transfers at `50` and
`1000`, and a reference fitted with `seed=42` transfers unseeded.

```python
# Inherits nothing but latent_dim, and says so.
model = BAE.from_reference(reference_model, target, n_additional_dims=5)

# Carries the settings over.
model = BAE.from_reference(
    reference_model,
    target,
    n_additional_dims=5,
    config=BAEConfig(latent_dim=15, boosting_stepno=50, max_iterations=200, seed=42),
)
```

The first form warns and names the fields that differed. It warns only for a
model reference, because a Parquet or CSV prior carries no configuration to
compare against, which is also why the settings are not inherited automatically:
the same prior would otherwise behave differently depending on whether you passed
the model or its exported weights.

## `decoder_warmup_epochs`

Trains the decoder against the frozen prior programs *before* boosting starts.
This is what makes the added dimensions genuinely **residual**: until the decoder
has converged against the prior, `∂L/∂z` still carries signal those programs could
explain, and the new dimensions would simply re-learn it.

The warm-up runs outside the training loop, deliberately. Inside it, it would win
checkpoint selection, and `fit` would restore an encoder whose novel columns are
all zero, returning the prior unchanged, with no error, reading as "no novel
structure found". Its loss is recorded separately as
`training_history["warmup_loss"]` so `train_loss` stays comparable with an
ordinary fit.

## What a transfer fit rejects

Three hard errors, each for a reason:

`split_softmax`
: applies one softmax across all 2·`latent_dim` entries, so adding dimensions
  dilutes every prior entry by a data-dependent amount. A frozen matrix would no
  longer mean the prior programs act unchanged.

`init_pca` / `init_obsm`
: they override the latent state on the first boosting iteration, which is
  exactly where the prior programs define the target. Use `decoder_warmup_epochs`
  instead.

`decoder_warmup_epochs` on a non-transfer model
: there is no prior to warm against. Use `init_pretrain_epochs`.

## Reading the diagnostics

```python
target.uns["bae_transfer"]
```

Carries the alignment provenance (`join_key`, `n_matched_genes`,
`prior_coverage`), support sizes split into prior and novel blocks, and the
variance accounting.

:::{danger}
**`novel_variance_share` on the fitting data is not evidence of novel biology.**
It is the rise in reconstruction MSE when the added dimensions are zeroed, and
*k* free dimensions always reduce in-sample error, so the number is positive
even when the target contains nothing the prior missed. Measured on gene-shuffled
data with no recoverable structure at all, it still reads +0.002 to +0.008 as *k*
grows from 1 to 4. It also grows with *k* under the null, so raw values are not
comparable across different `n_additional_dims`.

Use {meth}`~structboost.BAE.transfer_diagnostics` on **held-out cells**. Capacity
that merely fits noise does not generalize: a novel dimension carrying real
structure keeps its share out of sample, one that does not collapses. This needs
nothing but the model and your own data. It needs no access to the reference dataset and no
marker ground truth, which matters, because a prior encoder matrix is often all
that gets shared.
:::

`novel_variance_share_per_dim` entries do not sum to the aggregate: the dimensions
are not orthogonal, so structure carried by several is counted in each.

Support sizes are reported separately for prior and novel blocks because the
union is dominated by the prior, a sparsity check against the combined count
would be measuring the reference matrix, not this fit.

## Use `X_bae_scaled` for anything Euclidean

:::{warning}
A transfer's two blocks are on incomparable scales. Hand-authored prior
coefficients are written on a human scale, while boosted ones land wherever the
gradient targets put them. Measured on one transfer, prior dimensions had a
median latent SD of 9.98 against 0.043 for the novel ones, a **232× gap**, and
because distance is squared, the novel block contributed ~0.00% of the total.

Every Euclidean consumer inherits that: neighbour graphs, Leiden, kNN, kBET,
LISI. Hand them `obsm["X_bae_scaled"]`.
:::

`X_bae` is deliberately left as the raw projection, because that is what makes
the encoder auditable and the frozen-prior guarantee checkable. The centering
cannot be folded back into the weights anyway, since the encoder has no bias term.

The scaling statistics are estimated on the fitting data and **reused** by
`transform`, never re-estimated: new cells must go through the same map, and a
small query set would estimate its own moments badly.

## Encoder weight files

{func}`~structboost.write_encoder_weights` /
{func}`~structboost.read_encoder_weights` are the file form of a prior.

**Parquet is the recommended format.** Beyond preserving dtypes and the index, it
is immune to the spreadsheet round-trip that silently rewrites gene symbols such
as `SEPT2`, `MARCH1` and `DEC1` as dates. CSV/TSV is accepted on read for interoperability.

At least one identifier column is required: an encoder matrix without gene
identifiers cannot be aligned to another dataset, which is the only reason to
write one out. `gene_id` (Ensembl) is the default join key. `gene_symbol` is
carried but never authoritative, since symbols are revised between releases and
collide through aliases.

Add the species and the **annotation release** to `metadata`: without the release,
Ensembl accessions are only probably joinable to another dataset.

## Related

- {meth}`~structboost.BAE.from_reference`, {meth}`~structboost.BAE.transfer_diagnostics`
- {meth}`~structboost.BAE.apply_encoder`, and its `preserve_prior` footgun, in
  {doc}`gene-selection`
