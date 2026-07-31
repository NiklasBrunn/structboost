# Simulating data with known truth

To score a method you need to know the answer. The simulator generates
negative-binomial UMI counts with planted gene programs and hands back the ground
truth alongside. The count model is gamma-Poisson, with gamma-distributed gene
means and optional log-normal library sizes, multiplicative batch effects and
mean-dependent logistic dropout.

```python
from structboost import sim_scrnaseq_anndata

adata = sim_scrnaseq_anndata(n=1000, n_genes=500, stageno=10, seed=1)
```

`adata.X` is z-scored log1p, ready for {class}`~structboost.BAE` with no further
preprocessing.

## The ground truth you get back

```python
adata.obs["stage"]          # population label per cell
adata.obs["stage_id"]       # integer, -1 for background cells
adata.obs["level_0"]...   # one column per hierarchy depth
adata.var["is_marker"]      # marker of any population?
adata.var["marker_level"]   # which taxonomy level this gene marks
adata.varm["marker_mask"]   # (n_genes, stageno) per-population markers
adata.uns["simulation"]     # every parameter used
```

Which makes marker-recovery scoring a two-liner:

```python
selected = (adata.varm["BAE_encoder_weights"] != 0).any(axis=1)
truth = adata.var["is_marker"].to_numpy()

precision = (selected & truth).sum() / selected.sum()
recall = (selected & truth).sum() / truth.sum()
```

Counts and normalized values are always kept as layers, so `adata.X` can be
swapped without re-simulating:

```python
adata.layers["counts"]    # raw integer UMIs
adata.layers["lognorm"]   # library-size-normalized log1p
```

## Two orthogonal difficulty knobs

`effect_size` (default `10.0`)
: **Signal strength.** Lower values make populations harder to separate.

`dispersion` (default `0.2`)
: **Noise.** `Var[c] = mu + dispersion * mu²`. Setting `0.0` gives exact Poisson
  counts.

They are independent, so you can sweep a signal-to-noise grid cleanly.

## Hierarchical structure

By default the populations form a **taxonomy**, not a trajectory. Real cell types
nest, a marker like *Vip* labels a whole subclass while another distinguishes
one type within it, and every leaf expresses the union of its ancestors' marker
blocks.

```python
adata = sim_scrnaseq_anndata(n=2000, n_genes=800, hierarchy=(2, 5), seed=0)
# 2 classes x 5 types = 10 leaf populations
# adata.obs["level_0"], adata.obs["level_1"]
```

Per-level labels mean you can score subgroup detection at every resolution the
taxonomy defines, not just at the leaves. Labels carry the full ancestry path,
because a within-parent index alone would collide across parents and silently
merge distinct subgroups.

Pass `hierarchy=None` for the older flat sliding-window layout, where consecutive
populations share `stageoverlap` markers, a trajectory model rather than a
taxonomy. `stageoverlap` is rejected when `hierarchy` is set, since overlap then
comes from shared ancestor markers instead.

## Technical effects

All neutral by default, so a bare call gives clean, well-separated blocks. Turn
them on one at a time to test robustness:

| Argument | Effect |
| --- | --- |
| `lib_size_sd` | log-normal library size factors |
| `n_batches`, `batch_effect_sd` | multiplicative batch effects |
| `ambient_frac` | pooled ambient "soup" contamination |
| `dropout_mid`, `dropout_shape` | mean-dependent logistic dropout |
| `imbalanced=True` | population sizes drawn from a Dirichlet |

Batch is assigned round-robin, so it is **orthogonal to population**, which is
what makes it a fair test of {doc}`batch-integration`. Batch shifts are
normalized to unit geometric mean per gene, so enabling them does not move
marginal gene means. Ambient contamination preserves library size exactly.

## Reproducibility

Independent random sub-streams are spawned per component, so changing one
parameter does not perturb another's draws, a batch sweep does not also reshuffle
your counts.

:::{note}
Counts are generated in fixed-size row chunks, and that chunk size is part of the
reproducibility contract: changing it changes the random stream for a fixed seed,
even though the distribution is unaffected.
:::

## Without AnnData

{func}`~structboost.sim_scrnaseq_data` returns a
{class}`~structboost.SimulationResult` with the raw arrays (`counts`,
`stage_labels`, `marker_mask`, `gene_means`, `size_factors`, `batch_labels`) and
needs no `anndata` install.

## Related

- {func}`~structboost.sim_scrnaseq_anndata`, {func}`~structboost.sim_scrnaseq_data`
- {class}`~structboost.SimulationResult`
