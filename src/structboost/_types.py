"""Type definitions and configuration for BAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

# Type aliases
if TYPE_CHECKING:  # pragma: no cover
    import torch

    ArrayLike = npt.NDArray[np.floating] | torch.Tensor
    Device = torch.device | str
else:
    ArrayLike = npt.NDArray[np.floating] | Any
    Device = str | Any


@dataclass
class BAEConfig:
    """Configuration for Boosting Autoencoder.

    The revised BAE architecture uses a linear encoder (optimized via boosting)
    and an MLP decoder (optimized via SGD).

    Attributes
    ----------
    latent_dim
        Number of latent dimensions.
    decoder_hidden_dims
        Hidden layer sizes for the decoder MLP, in forward order from the latent
        code to the gene output. Default ``(64,)`` — a single hidden layer.

        On simulated data with planted gene programs, one hidden layer of 64 units
        recovered markers as well as or better than ``(128,)`` (marker-recovery F1
        0.99–1.00 across four scenarios spanning 500–3000 genes), reconstructed to
        ~91–95% of the linear ceiling, and trained faster. A *linear* decoder
        (``()``) is a valid faster, more interpretable choice but loses recall on
        harder data (F1 down to 0.83, reconstruction 65–73% of ceiling). Deeper or
        wider decoders did not help and often hurt marker recovery: a funnel such
        as ``(128, 64)`` dropped to F1 0.80 because a narrow layer immediately
        before the gene output distorts the reconstruction gradient that the
        boosting target is built from — a BAE-specific effect, since that gradient
        is what the encoder is fitted against. If more capacity is needed, widen
        (``(64, 128)``) rather than narrowing toward the output. These are
        simulation results; confirm on real data before assuming ``(64,)`` has
        enough capacity for genuinely nonlinear expression structure.
    decoder_activation
        Activation function for decoder hidden layers. ``"tanh"`` (default) is the
        most stable; ``"elu"`` matched it; ``"relu"`` was unstable at larger widths
        (marker-recovery F1 collapsing to 0.80 at 256 units).
    split_softmax
        If True, apply split-softmax transformation between encoder and decoder.
        Each latent dimension z_i is paired with −z_i (interleaved) and softmax-
        normalized onto the 2d-dimensional simplex:
        σ_split(z) = softmax((z_1, −z_1, ..., z_d, −z_d)).
        The decoder then receives a 2d-dimensional compositional input,
        enabling soft clustering of cells into 2d groups.
        Default is False (standard unconstrained latent space).
    boosting_stepno
        Number of boosting iterations per training iteration.
    boosting_nu
        Boosting learning rate.
    boosting_csf
        Cumulative shrinkage factor for boosting.
    boosting_independent
        If True, reset boosting state for each latent dimension (recommended).
    nuisance_ridge
        Ridge strength for the batch covariates when batch_integration_mode
        includes "encoder", expressed relative to each encoded column's
        squared norm. Mandatory *gene* coefficients are never penalized. Default
        0 preserves exact least squares.

        This is a numerical-stability knob rather than a modelling one. Boosting
        refits the mandatory block jointly at every step, and that block is the
        mandatory genes together with the batch columns. When two of those are
        collinear the solve has no unique answer and raises; when they are merely
        close to collinear it returns a large, sign-unstable answer and raises
        nothing. Reaching for a small value here, 1e-3 say, is the remedy in
        the second case. It is never applied automatically, because ridge changes
        the estimates and doing so silently would fit a different model than the
        one asked for.
    prior_mode
        How the prior encoder weight matrix is treated when the model was built
        by :meth:`BAE.from_reference`; ignored otherwise. ``"frozen"`` (default)
        never boosts the prior columns, so they remain bitwise equal to the
        reference matrix and the transferred gene programs are literally the
        reference's. ``"anchored"`` boosts them too, but restarts each iteration
        from the fixed original matrix rather than from the previous iteration's
        result — boosting from an offset model, so deviation is bounded by
        ``boosting_stepno`` and never accumulates. Unanchored re-fitting is not
        offered: a prior used merely as a starting point is forgotten within a
        few hundred iterations, since the encoder support random-walks and two
        runs end no more similar than two unrelated ones.
    boosting_precompute_covcache
        Whether to compute the full p×p covariance matrix before training.
        ``"auto"`` (default) does so whenever the 8·p² byte matrix fits a
        conservative share of system memory; ``True`` and ``False`` are honoured
        exactly. The resolved decision is recorded in
        ``adata.uns["bae"]["boosting_precompute_covcache"]``.

        This is a *speed* setting first and a memory setting second, which is not
        obvious. The alternative is a dict-backed column cache: a covariance
        column is computed the first time its feature is selected and reused
        across targets and iterations, so memory scales with the number of
        *distinct selected* features (8·p bytes per column) rather than with p².
        That sounds strictly cheaper, and in memory it is — but building all p
        columns at once is a single compute-bound matrix product running near
        hardware peak, while fetching them one at a time is a sequence of
        memory-bound matrix-vector products. Measured on real data, precomputing
        wins from roughly ``p/59`` distinct selected features onwards, and a fit
        passes that within its first iteration, where up to
        ``boosting_stepno × latent_dim`` distinct features can enter. End-to-end
        it measured 1.7–2.3× faster at p=3,000–10,000.

        The cost is memory, and it is real: 8·p² is 32 MB at p=2,000, 800 MB at
        p=10,000 and 3.2 GB at p=20,000, transiently doubled while the matrix is
        built. That is what ``"auto"`` guards against; set ``False`` explicitly on
        a memory-constrained machine.
    disentanglement
        Latent-dimension disentanglement method. ``"none"`` (default) applies no
        constraint. ``"correlation"`` adds a soft, differentiable squared-
        correlation penalty to the functional-gradient target objective and is the
        recommended method. ``"leave_one_out"`` retains the earlier experimental
        target residualization method, in which each target column is regressed on
        all other original target columns. Leave-one-out reduces some redundancies
        but does not mathematically produce mutually orthogonal residuals.
    disentanglement_lambda
        Strength of the soft correlation penalty. The penalty is normalized over
        latent-dimension pairs and scaled with the number of cells so that this
        value has the same meaning at different dataset sizes. Default ``1e-4`` is
        a conservative starting point from simulations; it is not universally
        optimal. A useful tuning grid is ``0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3``.
        Only used when ``disentanglement="correlation"``.
    target_optim_lr
        Step size for computing boosting targets via gradient descent on z.
        Targets are computed as z* = z - lr * ∂L_target/∂z (single gradient step),
        where L_target sums the squared error over cells and averages it over
        genes. That convention — rather than the elementwise mean used for the
        decoder update and every reported loss — makes a cell's target step
        independent of how many cells the dataset contains, so a given
        ``target_optim_lr`` means the same thing at every dataset size.

        **Leave this at 1.0.** It is exposed because it is a real coefficient of
        the method, not because it is a tuning knob: it scales the whole coupling
        between the two optimizers, and the quantity it scales has no natural
        unit — ``∂L/∂z`` tracks the decoder's magnitude, which grows by orders of
        magnitude over a fit. A value that suits one training stage will not suit
        the next, so the alternation is what adapts, and this stays fixed.

        .. note::
           An earlier formulation took the target gradient from the elementwise
           mean MSE, making it a factor ``n_cells`` smaller. A value ported from
           such a setup must be divided by the number of cells.
    decoder_lr
        Learning rate for decoder SGD optimization.
    decoder_weight_decay
        Weight decay (L2 penalty) for the AdamW decoder optimizer.
        Default is 0.0 (no penalty). Increase for decoder weight
        regularisation (typical range: 1e-5 to 1e-2).
    decoder_updates_per_iteration
        Number of decoder SGD steps per training iteration.
    max_iterations
        Maximum number of training iterations.
    enable_early_stopping
        If True (default), stop training early when the checkpoint-selection loss
        does not decrease for `early_stopping_patience` iterations. This is the
        decoder training MSE for ``disentanglement="none"`` and
        ``"leave_one_out"``; for ``"correlation"`` it additionally includes the
        weighted disentanglement penalty. If False, train for exactly
        `max_iterations`.
    early_stopping_patience
        Number of consecutive iterations without checkpoint-selection loss
        improvement required to trigger early stopping. Only used if
        enable_early_stopping is True. Default is 50.
    batch_size
        Minibatch size for decoder SGD updates.
    seed
        Random seed for reproducibility. If None, no seed is set.
        Controls train/val split, weight initialization, and SGD shuffling.
    device
        Device for computation ('cpu', 'cuda', or torch.device).
    diagnostics
        If True, collect per-iteration training diagnostics into a
        :class:`TrainingReport` (see `BAE.training_report`) and additionally show
        the relative encoder weight change (``dW``) and the number of selected
        genes (``n_sel``) in the progress bar. This adds full-data forward and
        backward passes per iteration, so it is off by default and
        `diagnostics=False` costs exactly what training cost before. Diagnostics
        are read-only: enabling them does not change the fitted model.
    """

    latent_dim: int = 10
    # Decoder MLP
    decoder_hidden_dims: tuple[int, ...] = (64,)
    decoder_activation: Literal["tanh", "relu", "leaky_relu", "elu"] = "tanh"
    split_softmax: bool = False
    # Boosting parameters
    boosting_stepno: int = 50
    boosting_nu: float = 0.1
    boosting_csf: float = 0.9
    boosting_independent: bool = True
    boosting_precompute_covcache: bool | Literal["auto"] = "auto"
    nuisance_ridge: float = 0.0
    prior_mode: Literal["frozen", "anchored"] = "frozen"
    disentanglement: Literal["none", "correlation", "leave_one_out"] = "none"
    disentanglement_lambda: float = 1e-4
    # Target computation
    target_optim_lr: float = 1.0
    # Training parameters
    decoder_lr: float = 1e-3
    decoder_weight_decay: float = 0.0
    decoder_updates_per_iteration: int = 10
    max_iterations: int = 1000
    enable_early_stopping: bool = True
    early_stopping_patience: int = 50
    batch_size: int = 2**9
    seed: int | None = None
    # Device
    device: Device = "cpu"
    # Diagnostics
    diagnostics: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.boosting_stepno < 1:
            raise ValueError("boosting_stepno must be >= 1")
        if not np.isfinite(self.nuisance_ridge) or self.nuisance_ridge < 0:
            raise ValueError("nuisance_ridge must be finite and >= 0")
        if self.boosting_precompute_covcache not in (True, False, "auto"):
            raise ValueError(
                "boosting_precompute_covcache must be True, False or 'auto', got "
                f"{self.boosting_precompute_covcache!r}"
            )
        if self.prior_mode not in {"frozen", "anchored"}:
            raise ValueError("prior_mode must be 'frozen' or 'anchored'")
        if not 0.0 < self.boosting_nu <= 1.0:
            raise ValueError("boosting_nu must be in (0.0, 1.0]")
        if self.disentanglement not in {"none", "correlation", "leave_one_out"}:
            raise ValueError(
                "disentanglement must be one of 'none', 'correlation', or 'leave_one_out'"
            )
        if not np.isfinite(self.disentanglement_lambda) or self.disentanglement_lambda < 0:
            raise ValueError("disentanglement_lambda must be finite and >= 0")
        if self.decoder_weight_decay < 0:
            raise ValueError("decoder_weight_decay must be >= 0")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if isinstance(self.device, str):
            try:
                import torch

                self.device = torch.device(self.device)
            except Exception:
                pass


@dataclass(frozen=True)
class TrainingReport:
    """Per-iteration training diagnostics collected by ``BAE.fit``.

    Produced only when ``diagnostics=True``. Every field is a NumPy array whose
    first axis indexes the training iteration; per-dimension fields have shape
    ``(n_iterations, latent_dim)``.

    Each BAE iteration alternates two optimizers, giving three points at which
    the full-data reconstruction loss can be measured:

    ``A`` (``loss_pre_boost``)
        before anything changes, i.e. encoder from the previous iteration.
    ``B`` (``loss_post_boost``)
        after the encoder is re-fit by boosting, before the decoder moves.
    ``C`` (``loss_post_decoder``)
        after the decoder SGD steps.

    The differences ``encoder_delta = B - A`` and ``decoder_delta = C - B``
    attribute the loss change of each iteration to the two halves of the
    alternation, which is otherwise invisible: a single loss curve cannot say
    whether the boosting step helps or whether the decoder is merely repairing
    the damage it does.

    Attributes
    ----------
    iteration
        0-based iteration index.
    train_loss
        Running average minibatch MSE during the decoder update. This is the
        historical metric and a noisier proxy for ``loss_post_decoder``. Early
        stopping uses it directly unless soft correlation disentanglement is
        enabled, in which case the corresponding
        ``training_history["selection_loss"]`` also includes that penalty.
    loss_pre_boost
        Full-data reconstruction MSE at point A above.
    loss_post_boost
        Full-data reconstruction MSE at point B above.
    loss_post_decoder
        Full-data reconstruction MSE at point C above.
    encoder_delta
        Loss change attributable to the boosting step; negative means it reduced
        the loss.
    decoder_delta
        Loss change attributable to the decoder step; negative means it reduced
        the loss.
    target_grad_norm
        Norm of the boosting-target gradient, ``||dL/dz||``. This is the signal
        the encoder is fitted against; if it collapses toward zero the boosting
        targets carry no information.
    decoder_grad_norm
        Norm of the full-data decoder parameter gradient.
    boosting_r2
        R^2 of the encoder output against the boosting targets, i.e. how well
        ``allboost`` fitted what it was asked to fit.
    encoder_weight_norm
        Mean absolute encoder weight. Typically grows by orders of magnitude
        during training, so weight *changes* must be read relative to it.
    weight_change_rel
        Mean absolute change of the encoder weights against the previous
        iteration, divided by ``encoder_weight_norm``. The primary convergence
        signal. ``NaN`` in the first iteration, which has no predecessor.
    support_jaccard
        Jaccard overlap of the selected-gene set against the previous iteration.
        Answers whether the *gene set* has settled, which is coarser and noisier
        than ``weight_change_rel`` but directly interpretable. ``NaN`` in the
        first iteration.
    min_dim_cosine
        Smallest per-latent-dimension cosine similarity between consecutive
        encoder weight matrices; scale-invariant, and identifies which dimension
        is still moving. ``NaN`` in the first iteration.
    n_selected
        Number of genes with a nonzero weight in any latent dimension.
    n_selected_per_dim
        Number of genes selected per latent dimension.
    latent_var_per_dim
        Variance of each latent dimension across cells. Compare dimensions with
        each other rather than against an absolute scale: the overall latent
        magnitude is set by the boosting shrinkage and is small by construction.
    """

    iteration: npt.NDArray[np.intp]
    train_loss: npt.NDArray[np.float64]
    loss_pre_boost: npt.NDArray[np.float64]
    loss_post_boost: npt.NDArray[np.float64]
    loss_post_decoder: npt.NDArray[np.float64]
    encoder_delta: npt.NDArray[np.float64]
    decoder_delta: npt.NDArray[np.float64]
    target_grad_norm: npt.NDArray[np.float64]
    decoder_grad_norm: npt.NDArray[np.float64]
    boosting_r2: npt.NDArray[np.float64]
    encoder_weight_norm: npt.NDArray[np.float64]
    weight_change_rel: npt.NDArray[np.float64]
    support_jaccard: npt.NDArray[np.float64]
    min_dim_cosine: npt.NDArray[np.float64]
    n_selected: npt.NDArray[np.intp]
    n_selected_per_dim: npt.NDArray[np.intp]
    latent_var_per_dim: npt.NDArray[np.float64]

    @property
    def n_iterations(self) -> int:
        """Number of recorded training iterations."""
        return int(self.iteration.shape[0])

    def to_dict(self) -> dict[str, npt.NDArray[Any]]:
        """Return the report as a plain dict of arrays, ready for ``adata.uns``."""
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, npt.NDArray[Any]]) -> TrainingReport:
        """Rebuild a report from :meth:`to_dict` output, e.g. read back from h5ad."""
        missing = [f for f in cls.__dataclass_fields__ if f not in data]
        if missing:
            raise KeyError(f"missing training report fields: {missing}")
        return cls(**{f: np.asarray(data[f]) for f in cls.__dataclass_fields__})
