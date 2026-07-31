# Saving and loading a model

```python
model.fit(adata)
model.save("bae_model.pt")

# later, or elsewhere
model = BAE.load("bae_model.pt")
model.transform(query)
model.reconstruct(query)      # needs the fitted condition_obs columns
```

`save` returns the path it wrote and creates parent directories. `.pt` is the
conventional suffix.

## What a checkpoint holds

Everything needed to reproduce {meth}`~structboost.BAE.transform` and
{meth}`~structboost.BAE.reconstruct`, to use the model as a
{meth}`~structboost.BAE.from_reference` prior, and to read the training
diagnostics back:

encoder and decoder weights · the full config · the fitted layer · gene names ·
covariate encoding *parameters* · nuisance weights · transfer state (prior
weights, provenance, latent scaling) · `training_history` and `training_report`

## What it deliberately does not hold

**The decoder's optimizer state.** A loaded model is deployable but does not
resume a training run mid-flight. A fresh `fit` works regardless. It rebuilds
the decoder and optimizer every time anyway.

**The training-set covariate design matrix.** Only the encoding *parameters*
needed to encode new data are kept, so a shipped model file carries no cell-level
training data and its size does not grow with your training set. It is restored
as a zero-row array, so any accidental use fails on shape rather than quietly
contributing wrong numbers.

## Loading cannot execute code from the file

The payload contains only tensors and plain Python values, with no pickled objects,
NumPy arrays included, which is what lets `load` read it with
`torch.load(..., weights_only=True)`.

That is a real property, not a hope: a test pins it, so it cannot be quietly lost
by a later change. Loading a `.pt` file from an untrusted source is normally an
arbitrary-code-execution risk. Here it is not.

## Devices

```python
BAE.load("bae_model.pt", device="cpu")
```

Without `device=`, the checkpoint's own device is used when available. If it
names a device the machine does not have, the model loads onto CPU **with a
warning**, never a silent relocation.

## Errors you may hit

`RuntimeError` on save
: The model is not fitted. To persist the prior encoder matrix of an unfitted
  `from_reference` model, use {func}`~structboost.write_encoder_weights`.

`ValueError` on save, naming an obs column
: That column's categorical levels cannot be persisted. Datetimes, for example.
  Convert it to string, integer or boolean before fitting. This fails loudly at
  save rather than producing a wrong reconstruction months later, which is what
  a silently re-typed level list would do.

`ValueError` on load, "not a structboost BAE checkpoint"
: The file is a `.pt` from somewhere else. A refusal is a better diagnostic than
  a `KeyError` deep inside the restore.

`ValueError` on load, "Upgrade structboost"
: Written by a newer checkpoint format than this install understands.

## Checkpoint format versions

The format carries an integer version, bumped when the payload changes in a way
older installs would misread.

This release writes format 2. A reader refuses any format newer than it
understands rather than guessing, which matters because ignoring an unknown key
can mean silently reading the wrong matrix, a wrong answer rather than a missing
one.

## Just the gene programs

If what you want to share is the *encoder* rather than the model, which is the
transferable artifact, use {func}`~structboost.write_encoder_weights` and see
{doc}`transfer`. That form is human-readable, carries gene identifiers, and does
not require the recipient to trust a binary.

## Related

- {meth}`~structboost.BAE.save`, {meth}`~structboost.BAE.load`
- {doc}`transfer`
