---
sd_hide_title: true
---

# structboost

```{div} sd-text-center sd-fs-2 sd-font-weight-bold
structboost
```

```{div} sd-text-center sd-fs-5 sd-text-secondary
Structured representation learning for single-cell data. A latent space you
can read gene by gene.
```

---

The **Boosting Autoencoder (BAE)** pairs a *linear encoder fitted by componentwise
L2 boosting* with an MLP decoder trained by gradient descent. Each training
iteration takes a gradient step on the latent code itself and hands the result to
the boosting fit as a regression target, so the encoder is fitted against the
negative gradient of the reconstruction loss rather than by backpropagation.
Componentwise boosting adds one gene at a time and shrinks each step, which keeps
the encoder weights sparse by construction rather than by a post-hoc threshold.

Each latent dimension is therefore a short, signed gene list, and `X_bae` is
exactly `X @ varm["BAE_encoder_weights"]`.

The package also ships {func}`~structboost.allboost`, the componentwise boosting
routine on its own, for sparse supervised problems with no autoencoder involved.

```{code-block} python
:caption: From an AnnData to an interpretable latent space

from structboost import BAE, BAEConfig

model = BAE(adata.n_vars, BAEConfig(latent_dim=10))
model.fit(adata)                            # adata.X must be z-scored

adata.obsm["X_bae"]                         # (n_cells, 10)
adata.varm["BAE_encoder_weights"]           # (n_genes, 10), sparse
```

```{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

The core install is NumPy-only. The BAE needs the `[bae]` extra.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` User guide
:link: guide/index
:link-type: doc

What each capability is for, when it does *not* work, and the caveats worth
knowing before you trust a gene list.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API reference
:link: api/index
:link-type: doc

Every public symbol, grouped by the task it belongs to.
:::

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Release notes
:link: changelog
:link-type: doc

What changed, and the measurements behind each change.
:::
```

## When this is the right tool

BAE is built for the case where **you have to be able to say which genes produced
a latent axis**, such as marker discovery, gene-program interpretation, or
handing a representation to a collaborator who will check it against biology.

It is not built to win at reconstruction. A dense autoencoder might reconstruct
better. The point here is that its latent dimensions cannot be read.

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

```{toctree}
:hidden:
:maxdepth: 2

installation
guide/index
api/index
changelog
contributing
```
