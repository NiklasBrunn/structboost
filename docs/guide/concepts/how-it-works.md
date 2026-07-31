# How the BAE works

A BAE is an autoencoder whose two halves are fitted by two different optimizers.

**Encoder**, one linear layer, no bias: `z = W @ x`. Its weights are never
touched by gradient descent. They are produced by componentwise L2 boosting
[^hackenberg2025].

**Decoder**, an MLP, trained by AdamW in the ordinary way.

That asymmetry is the whole design. Boosting adds one feature at a time, so the
encoder is sparse *because of how it was fitted*, not because a threshold was
applied afterwards.

## The training loop

Each iteration runs these six steps, in this order:

1. **Compute boosting targets.** Take one gradient step on the latent code
   itself: `z* = z - lr * ∂L/∂z`. This is functional gradient descent. `z*` is
   where the latent code *should* move to reduce reconstruction error.
2. *(Optional)* residualize the targets against each other
   (`disentanglement="leave_one_out"`).
3. *(Optional)* standardize the target columns (`standardize_targets`).
4. **Reset the encoder weights to zero.**
5. **Fit the encoder with {func}`~structboost.allboost`**, regressing `X` onto
   `z*`. This is the step that selects genes.
6. **Update the decoder** with several minibatch AdamW steps.

Then check early stopping, and repeat.

### Why the encoder is reset every iteration

Step 4 looks wasteful and is load-bearing. Boosting's sparsity guarantee comes
from starting at zero and taking `boosting_stepno` steps: at most that many
distinct genes can enter. Carrying weights over from the previous iteration would
let the support accumulate without bound, and after a few hundred iterations the
encoder would be dense, which is precisely the property the method exists to
avoid.

The consequence: **`boosting_stepno` is the sparsity dial.** It caps how many
genes a dimension can use, per iteration, from scratch.

### Why the target is `z*` and not `z`

The encoder is fitted against the *gradient-updated* code, not the current one.
This matters later, when stability selection reuses the same targets: the codes
`z` are already a sparse linear function of the selected genes, so predicting
them back is nearly circular. `z*` carries the decoder's full reconstruction
gradient, including the part the current encoder is missing.

### Two loss reductions, deliberately

The target objective sums squared error over cells and averages over genes. Every
*reported* loss, and the decoder update, use the plain elementwise mean.

They differ by a constant factor of `n_cells`, which is exactly the point. Under
the elementwise mean, each cell's share of the gradient shrinks as the dataset
grows, so the same `target_optim_lr` would mean something different at every
dataset size. Do not "simplify" these into one reduction.

:::{note}
An earlier formulation took the target gradient from the elementwise mean, making
it a factor `n_cells` smaller. If you are porting a `target_optim_lr` value from
such a setup, divide it by the number of cells.
:::

## What comes out

Because the encoder is a single bias-free linear map, the latent space is exactly
a matrix product:

```python
adata.obsm["X_bae"] == adata.X @ adata.varm["BAE_encoder_weights"]
```

That identity is worth stating because it is what makes the model auditable. A
latent dimension is a signed, weighted gene list. You can read it, hand it to a
biologist, and check it. It is also what makes the frozen-prior guarantee in
{doc}`../tasks/transfer` checkable end to end.

## Split-softmax (optional)

With `split_softmax=True`, each latent dimension `z_i` is paired with `-z_i` and
the 2*d* values are softmaxed onto the simplex:

```
h = softmax(z_1, -z_1..., z_d, -z_d)
```

The decoder then receives a compositional 2*d*-dimensional input, which turns the
latent space into a soft clustering of cells into 2*d* groups. The encoder is
unchanged, and `X_bae` is still the raw linear projection. The construction follows
the supplementary material of Brunn et al. [^brunn2025].

## References

[^hackenberg2025]: Hackenberg, M., Brunn, N., Vogel, T. et al. (2025).
    *Infusing structural assumptions into dimensionality reduction for
    single-cell RNA sequencing data to identify small gene sets.*
    Communications Biology 8, 414. <https://doi.org/10.1038/s42003-025-07872-9>. Provides the Boosting Autoencoder itself.

[^brunn2025]: Brunn, N. et al. (2025). Bioinformatics Advances 5(1), vbaf230.
    <https://doi.org/10.1093/bioadv/vbaf230>. Provides the split-softmax construction.

## Where to go next

- {doc}`standardization`: the input contract this all assumes.
- {doc}`../tasks/fitting`: the knobs, and which ones benchmarks say to leave alone.
- {doc}`reading-quality`: how to tell whether the result is any good.
