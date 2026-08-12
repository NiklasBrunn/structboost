"""BAE Decoder module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from ._types import BAEConfig


class BAEDecoder(nn.Module):
    """MLP decoder for Boosting Autoencoder.

    Maps latent space back to gene expression via a multi-layer perceptron.

    Parameters
    ----------
    n_output
        Number of output features (genes).
    config
        BAE configuration object.
    input_dim_override
        If set, overrides ``config.latent_dim`` for the first layer input
        dimension.  Used when split-softmax doubles the latent size to 2d.
    n_covariates
        Number of encoded conditioning covariates. Zero disables conditioning;
        any positive value appends them to the decoder input.
    """

    def __init__(
        self,
        n_output: int,
        config: BAEConfig,
        *,
        input_dim_override: int | None = None,
        n_covariates: int = 0,
    ) -> None:
        super().__init__()
        self.n_output = n_output
        self.config = config
        self.n_covariates = n_covariates
        if n_covariates < 0:
            raise ValueError(f"n_covariates must be >= 0, got {n_covariates}")

        # Build activation function
        self.activation = self._get_activation(config.decoder_activation)

        # Build decoder MLP: effective_latent -> hidden_dims -> n_output
        effective_latent = (input_dim_override or config.latent_dim) + n_covariates
        layers: list[nn.Module] = []
        dims = [effective_latent] + list(config.decoder_hidden_dims)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(self.activation)

        self.hidden = nn.Sequential(*layers)
        # Final projection to output space
        self.output_layer = nn.Linear(dims[-1], n_output, bias=True)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]

    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, z: torch.Tensor, covariates: torch.Tensor | None = None) -> torch.Tensor:
        """Decode latent representation to reconstruction.

        Parameters
        ----------
        z
            Latent tensor of shape (n_cells, latent_dim).
        covariates
            Encoded conditioning covariates, required when conditioning is
            enabled.

        Returns
        -------
        Reconstruction of shape (n_cells, n_genes).
        """
        if self.n_covariates and covariates is None:
            raise ValueError("Decoder conditioning covariates are required")
        if not self.n_covariates and covariates is not None:
            raise ValueError("Decoder was built without conditioning covariates")
        decoder_input = torch.cat([z, covariates], dim=1) if self.n_covariates else z
        return self.output_layer(self.hidden(decoder_input))
