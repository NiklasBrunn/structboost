# Reading a fitted model, dimension by dimension

```python
from structboost import plot_latent_dimensions, plot_dimension_correlation

plot_latent_dimensions(adata, group_by="cell_type")   # one row per dimension
plot_dimension_correlation(adata)                     # are any two the same thing?
```

Both read only what `fit` writes — `obsm["X_bae"]` and
`varm["BAE_encoder_weights"]` — so they work on a saved `.h5ad` with no model
object, and on `MultiBAE` output, which writes the same keys per modality.

## The panels

{func}`~structboost.plot_latent_dimensions` draws one row per dimension, from
`panels=`:

| Panel | Shows | Answers |
| --- | --- | --- |
| `"scores"` | every cell's score, sorted | is this a gradient or a subgroup? |
| `"contributions"` | the top genes' per-cell contributions | which genes build it, and at which end? |
| `"shares"` | *every* selected gene's share, sorted | three genes or forty? |
| `"weights"` | the raw coefficients | (available, not default) |
| `"groups"` | scores split by `group_by` | which cells sit at which end? |

The default is `("scores", "contributions", "groups")`; `"groups"` is dropped
when `group_by` is `None`, because an ungrouped violin is the score panel turned
on its side.

The **score curve's shape is the first thing to read.** A smooth ramp is a graded
axis across the population; a flat plateau with a hook is a subgroup axis. The
two are summarised differently and the panel reports the top and bottom decile
mass separately, because most dimensions are one-sided.

{func}`~structboost.plot_dimension_gene_umaps` draws a UMAP grid — dimensions by
top genes — through `scanpy.pl.umap`, and needs the `scanpy` extra. Reading
across a row shows whether a dimension's genes light up *the same* cells. No
other panel can: the score curve and the contribution violins have both already
summed over cells.

## How genes are ranked

By **share of the dimension's variance**, not by `|weight|`. From
`Var(s) = Cov(s, s)` with `s = X @ w`:

```
share_g = w_g · Cov(X_g, s) / Var(s)          and   sum_g share_g = 1
```

An exact decomposition, needing no orthogonality assumption, and splitting credit
between correlated genes rather than letting both claim it.
{func}`~structboost.gene_variance_shares` computes it directly.

`|weight|` is the wrong ranking because it ignores how much a gene actually
varies. Measured on 59,261 PBMCs the two agree on 8.6 of 10 genes per dimension,
and where they differ `|weight|` promotes genes that move no cells — one gene
ranked #4 by weight and #35 of 38 by share, being near-absent in that tissue.
`rank_by="weight"` is available when the coefficient itself is the question.

**The share is a magnitude.** A negative-weight gene is anti-correlated with the
score, so `w_g · Cov(X_g, s)` is positive either way; about 1% of genes come out
negative, which means genuine suppression. Direction lives in the sign of the
weight, and is shown by the **colour of the gene name** — orange positive, blue
negative. So a negative-weight gene outranks a positive one whenever it accounts
for more variance, and the two interleave freely.

## Colour

Blue and orange mean negative and positive everywhere: cells below and above
zero in the split violins, and weight signs in the gene names. Sequential maps
are for magnitudes, diverging-about-zero for signed quantities — so the score
column of a UMAP grid is diverging, and its gene columns are sequential unless
`scale="zscore"` makes them signed too.

Group colours follow the group count: five of `tab10`, then `tab10`, then
scanpy's `default_20`. Groups past the palette are drawn grey and keep their
labelled violin, so only the hue is lost.
{func}`~structboost.palette_audit` reports any palette's measured separation and
contrast. Only the smallest tier clears every bar — `tab10` pairs red with green,
the colour-blind confusion axis — so past five groups, colour is a supporting
encoding and the violin labels carry identity.

## Are two dimensions the same thing?

{func}`~structboost.plot_dimension_correlation` is the only view that asks this;
every other panel looks at one dimension at a time.

```python
plot_dimension_correlation(adata, method="spearman")   # or "pearson", the default
plot_dimension_correlation(adata, absolute=True)       # |r|, sequential colours
```

Prefer Spearman when checking a fit that enforced orthogonality: the constraint
is *linear*, so a monotone but non-linear relationship survives it. On one
10-dimension fit Pearson reported a largest off-diagonal `|r|` of 0.12 while
Spearman reported 0.31.

## What these plots are not

Descriptive, not inferential. No p-values, no error control; every number is
either an exact decomposition or a summary statistic. A high share means a gene
accounts for much of a dimension *in this fit* — and a single fit's support is
not reproducible, so see {doc}`gene-selection` before trusting one gene list.
