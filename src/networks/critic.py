"""
Critic (Value) Network for RQE-MAPPO

Estimates state value V(s) for computing advantages
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    """Value network that estimates V(s)"""

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list[int] = [64, 64],
        activation: str = "relu"
    ):
        """
        Args:
            obs_dim: Observation dimension (can be local or global state)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh')
        """
        super().__init__()

        self.obs_dim = obs_dim

        # Build MLP
        layers = []
        in_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            in_dim = hidden_dim

        # Output layer (scalar value)
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Orthogonal initialization (common in PPO)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            obs: Observations [batch, obs_dim]

        Returns:
            State values [batch, 1]
        """
        return self.mlp(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate (convenience method)

        Args:
            obs: Observations [batch, obs_dim]

        Returns:
            State values [batch] (squeezed)
        """
        return self.forward(obs).squeeze(-1)


if __name__ == "__main__":
    # Test critic network
    print("Testing CriticNetwork...")

    obs_dim = 10
    batch_size = 32

    critic = CriticNetwork(obs_dim)

    # Test forward pass
    obs = torch.randn(batch_size, obs_dim)
    values = critic(obs)
    print(f"Input shape: {obs.shape}")
    print(f"Output shape: {values.shape}")
    assert values.shape == (batch_size, 1)

    # Test get_value
    values_squeezed = critic.get_value(obs)
    print(f"Squeezed output shape: {values_squeezed.shape}")
    assert values_squeezed.shape == (batch_size,)
    assert torch.allclose(values.squeeze(), values_squeezed)

    print(f"Sample values: {values_squeezed[:5]}")
    print(f"Mean value: {values_squeezed.mean():.3f}")
    print(f"Std value: {values_squeezed.std():.3f}")

    print("\nAll tests passed!")
