"""
Actor (Policy) Network for RQE-MAPPO

Outputs action logits for discrete action spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """Policy network that outputs action logits"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [64, 64],
        activation: str = "relu"
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh')
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

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

        # Output layer
        layers.append(nn.Linear(in_dim, action_dim))

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
            Action logits [batch, action_dim]
        """
        return self.mlp(obs)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy

        Args:
            obs: Observations [batch, obs_dim]
            deterministic: If True, return argmax action

        Returns:
            action: Sampled actions [batch]
            log_prob: Log probabilities [batch]
            entropy: Entropy of distribution [batch]
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions

        Used during training to compute policy gradient

        Args:
            obs: Observations [batch, obs_dim]
            actions: Actions to evaluate [batch]

        Returns:
            log_probs: Log probabilities [batch]
            entropy: Entropy of distribution [batch]
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy


if __name__ == "__main__":
    # Test actor network
    print("Testing ActorNetwork...")

    obs_dim = 10
    action_dim = 5
    batch_size = 32

    actor = ActorNetwork(obs_dim, action_dim)

    # Test forward pass
    obs = torch.randn(batch_size, obs_dim)
    logits = actor(obs)
    print(f"Input shape: {obs.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, action_dim)

    # Test action sampling
    actions, log_probs, entropy = actor.get_action(obs)
    print(f"Actions shape: {actions.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print(f"Sample actions: {actions[:5]}")
    print(f"Sample log probs: {log_probs[:5]}")
    print(f"Mean entropy: {entropy.mean():.3f}")

    # Test deterministic action
    det_actions, _, _ = actor.get_action(obs, deterministic=True)
    print(f"Deterministic actions: {det_actions[:5]}")

    # Test evaluate_actions
    eval_log_probs, eval_entropy = actor.evaluate_actions(obs, actions)
    assert torch.allclose(log_probs, eval_log_probs, atol=1e-6)
    assert torch.allclose(entropy, eval_entropy, atol=1e-6)
    print("evaluate_actions test passed!")

    print("\nAll tests passed!")
