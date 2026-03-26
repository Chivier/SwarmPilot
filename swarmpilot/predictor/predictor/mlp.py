"""PyTorch MLP implementation for runtime prediction.

Provides a configurable multi-layer perceptron with flexible output dimensions.
"""

from __future__ import annotations

import torch
from torch import nn

torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
    """Multi-layer Perceptron for regression tasks.

    Supports configurable hidden layers and multiple outputs.

    Attributes:
        input_dim: Number of input features.
        output_dim: Number of output values.
        hidden_layers: List of hidden layer sizes.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_layers: list[int] | None = None,
    ) -> None:
        """Initialize MLP.

        Args:
            input_dim: Number of input features.
            output_dim: Number of output values (1 for expect_error,
                len(quantiles) for quantile).
            hidden_layers: List of hidden layer sizes (default: [64, 32]).
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        # Build network layers
        layers = []
        prev_dim = input_dim

        # Hidden layers with ReLU activation
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer (no activation, raw regression output)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)

    def get_config(self) -> dict[str, int | list[int]]:
        """Get model configuration for serialization.

        Returns:
            Dict with model architecture parameters.
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers,
        }

    @classmethod
    def from_config(cls, config: dict[str, int | list[int]]) -> MLP:
        """Create MLP instance from configuration dict.

        Args:
            config: Configuration dict from get_config().

        Returns:
            New MLP instance with the same architecture.
        """
        return cls(
            input_dim=config["input_dim"],
            output_dim=config["output_dim"],
            hidden_layers=config["hidden_layers"],
        )
