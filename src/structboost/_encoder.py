"""BAE Encoder module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from ._types import BAEConfig


class SplitSoftmax(nn.Module):
    """Split-softmax transformation for compositional latent codes.

    Interleaves each latent dimension z_i with its negation −z_i, then
    applies softmax over the 2d-dimensional vector to produce a
    compositional representation h ∈ Δ^{2d−1}.

    σ_split(z_1, ..., z_d) = σ((z_1, −z_1, ..., z_d, −z_d))

    Reference: Supplementary Data of Brunn et al. (2025),
    https://academic.oup.com/bioinformatics/article/5/1/vbaf230/8262953
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map z ∈ ℝ^d to h ∈ Δ^{2d−1} via interleaved-negation softmax."""
        s = torch.stack([z, -z], dim=-1).reshape(z.shape[0], -1)  # (n, 2d)
        return torch.softmax(s, dim=-1)


class BAEEncoder(nn.Module):
    """Linear encoder for Boosting Autoencoder.

    Single linear projection from gene expression to latent space.
    Weights are optimized via componentwise boosting (not gradient descent).

    The encoder maps only gene expression to latent space: z = W @ x.
    Obs covariates never modify z. Depending on ``batch_integration_mode`` the
    encoded covariate enters allboost as a mandatory regressor, or the decoder as
    a conditioning input, or both, but never this layer. That is what keeps the
    encoder deployable on data carrying no covariate labels.

    Parameters
    ----------
    n_input
        Number of input features (genes).
    config
        BAE configuration object.
    """

    def __init__(self, n_input: int, config: BAEConfig) -> None:
        super().__init__()
        self.n_input = n_input
        self.config = config
        # Single linear layer: W ∈ ℝ^{n_genes × latent_dim}, no bias
        self.linear = nn.Linear(n_input, config.latent_dim, bias=False)
        self.reset_weights()

    def reset_weights(self) -> None:
        """Reset encoder weights to zero."""
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Parameters
        ----------
        x
            Input tensor of shape (n_cells, n_genes).

        Returns
        -------
        Latent representation of shape (n_cells, latent_dim).
        """
        return self.linear(x)

    def set_weights(self, W: torch.Tensor) -> None:
        """Set encoder weights directly (from boosting output).

        Parameters
        ----------
        W
            Weight tensor of shape (latent_dim, n_genes).
        """
        with torch.no_grad():
            self.linear.weight.copy_(W)
