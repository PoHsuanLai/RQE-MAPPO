"""
Distributional Critic Network for RQE-MAPPO

Uses categorical distribution (C51-style) to represent value distribution.
Enables computation of risk measures (entropic, CVaR, mean-variance).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DistributionalCritic(nn.Module):
    """
    Distributional value network using categorical distribution (C51)

    Instead of predicting V(s) = E[return], predicts a distribution Z(s)
    over possible returns, enabling risk-aware value computation.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list[int] = [64, 64],
        activation: str = "relu",
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0
    ):
        """
        Args:
            obs_dim: Observation dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh')
            n_atoms: Number of atoms in categorical distribution
            v_min: Minimum value of support
            v_max: Maximum value of support
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Fixed support (never changes during training!)
        self.register_buffer(
            'support',
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

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

        # Output layer: logits for each atom
        layers.append(nn.Linear(in_dim, n_atoms))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Get probability distribution over returns

        Args:
            obs: Observations [batch, obs_dim]

        Returns:
            probs: Probabilities over atoms [batch, n_atoms]
        """
        logits = self.mlp(obs)  # [batch, n_atoms]
        probs = F.softmax(logits, dim=-1)  # [batch, n_atoms]
        return probs

    def get_distribution(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get full distribution (support + probabilities)

        Args:
            obs: Observations [batch, obs_dim]

        Returns:
            support: Fixed support values [n_atoms]
            probs: Probabilities [batch, n_atoms]
        """
        probs = self.forward(obs)
        return self.support, probs

    def get_expected_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get expected value E[Z(s)] (standard value estimate)

        Args:
            obs: Observations [batch, obs_dim]

        Returns:
            values: Expected values [batch]
        """
        probs = self.forward(obs)  # [batch, n_atoms]
        # E[Z] = Σ p_i * z_i
        values = (probs * self.support).sum(dim=-1)  # [batch]
        return values

    def get_risk_value(
        self,
        obs: torch.Tensor,
        tau: float,
        risk_type: str = "entropic"
    ) -> torch.Tensor:
        """
        Compute risk-adjusted value using risk measures

        Args:
            obs: Observations [batch, obs_dim]
            tau: Risk aversion parameter
            risk_type: Type of risk measure
                - "entropic": ρ_τ(Z) = -(1/τ) log E[exp(-τZ)]
                - "cvar": CVaR_α - average of worst α% outcomes
                - "mean_variance": E[Z] - (1/τ)Var[Z]

        Returns:
            risk_values: Risk-adjusted values [batch]
        """
        probs = self.forward(obs)  # [batch, n_atoms]

        if risk_type == "entropic":
            # Entropic risk: ρ_τ(Z) = -(1/τ) log E[exp(-τZ)]
            # E[exp(-τZ)] = Σ p_i * exp(-τ * z_i)

            # For large tau (risk-neutral), use expected value directly
            if tau > 100:
                # lim τ→∞ ρ_τ(Z) = E[Z]
                risk_value = (probs * self.support).sum(dim=-1)
            else:
                # Numerically stable computation using log-sum-exp trick
                # log E[exp(-τZ)] = log Σ p_i * exp(-τ * z_i)
                #                 = log Σ exp(log(p_i) - τ * z_i)
                # Use max for numerical stability
                neg_tau_z = -tau * self.support  # [n_atoms]
                max_val = neg_tau_z.max()

                # exp(neg_tau_z - max_val) is numerically stable
                exp_neg_tau_z = torch.exp(neg_tau_z - max_val)  # [n_atoms]
                expectation = (probs * exp_neg_tau_z).sum(dim=-1)  # [batch]

                # Risk value: -(1/τ) * [log(expectation) + max_val]
                risk_value = -(1.0 / tau) * (torch.log(expectation + 1e-8) + max_val)

        elif risk_type == "cvar":
            # CVaR: Average of worst α% outcomes
            alpha = 0.2  # Worst 20%
            n_worst = max(1, int(alpha * self.n_atoms))

            # For each batch element, compute CVaR
            batch_size = probs.shape[0]
            risk_value = torch.zeros(batch_size, device=probs.device)

            for i in range(batch_size):
                # Sort atoms by value (ascending)
                sorted_indices = torch.argsort(self.support)
                worst_atoms = sorted_indices[:n_worst]

                # Weighted average of worst atoms
                worst_probs = probs[i, worst_atoms]
                worst_values = self.support[worst_atoms]
                risk_value[i] = (worst_probs * worst_values).sum() / (worst_probs.sum() + 1e-8)

        elif risk_type == "mean_variance":
            # Mean-variance: E[Z] - (1/τ) * Var[Z]
            mean = (probs * self.support).sum(dim=-1)  # [batch]

            # Var[Z] = E[(Z - E[Z])²] = E[Z²] - E[Z]²
            mean_sq = (probs * self.support ** 2).sum(dim=-1)  # [batch]
            variance = mean_sq - mean ** 2

            risk_value = mean - (1.0 / tau) * variance

        else:
            raise ValueError(f"Unknown risk type: {risk_type}")

        return risk_value


def project_distribution(
    next_probs: torch.Tensor,
    rewards: torch.Tensor,
    support: torch.Tensor,
    v_min: float,
    delta_z: float,
    gamma: float = 0.99,
    dones: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Project Bellman-updated distribution onto fixed support

    This is the key operation in distributional RL that handles the
    misalignment between Bellman-updated values and fixed support.

    Args:
        next_probs: Probabilities at next state [batch, n_atoms]
        rewards: Immediate rewards [batch]
        support: Fixed support values [n_atoms]
        v_min: Minimum support value
        delta_z: Spacing between atoms
        gamma: Discount factor
        dones: Done flags [batch] (optional)

    Returns:
        target_probs: Projected probabilities [batch, n_atoms]
    """
    batch_size, n_atoms = next_probs.shape

    # Handle terminal states
    if dones is not None:
        gamma = gamma * (1 - dones.float())

    # Step 1: Compute Bellman targets Tz = r + γz
    # Shape: [batch, 1] + [batch, 1] * [1, n_atoms] = [batch, n_atoms]
    Tz = rewards.unsqueeze(-1) + gamma.unsqueeze(-1) * support.unsqueeze(0)

    # Step 2: Clamp to support range
    Tz = torch.clamp(Tz, v_min, v_min + (n_atoms - 1) * delta_z)

    # Step 3: Find fractional position on fixed support
    # b is the continuous index
    b = (Tz - v_min) / delta_z  # [batch, n_atoms]

    # Step 4: Find neighboring atoms
    l = b.floor().long()  # Lower neighbor
    u = b.ceil().long()   # Upper neighbor

    # Fix boundary conditions
    l = torch.clamp(l, 0, n_atoms - 1)
    u = torch.clamp(u, 0, n_atoms - 1)

    # Step 5: Distribute probability to neighbors (vectorized)
    target_probs = torch.zeros_like(next_probs)

    # Compute weights for interpolation
    # If b=1.3, then l=1, u=2
    # Weight for l: (u - b) = (2 - 1.3) = 0.7
    # Weight for u: (b - l) = (1.3 - 1) = 0.3
    ml = next_probs * (u.float() - b)  # Weight for lower
    mu = next_probs * (b - l.float())  # Weight for upper

    # Scatter-add to target distribution
    for i in range(batch_size):
        for j in range(n_atoms):
            target_probs[i, l[i, j]] += ml[i, j]
            target_probs[i, u[i, j]] += mu[i, j]

    return target_probs


if __name__ == "__main__":
    print("Testing DistributionalCritic...")

    obs_dim = 10
    batch_size = 32
    n_atoms = 51
    v_min, v_max = -10.0, 10.0

    critic = DistributionalCritic(
        obs_dim,
        hidden_dims=[64, 64],
        n_atoms=n_atoms,
        v_min=v_min,
        v_max=v_max
    )

    # Test forward pass
    obs = torch.randn(batch_size, obs_dim)
    probs = critic(obs)
    print(f"Input shape: {obs.shape}")
    print(f"Output shape (probs): {probs.shape}")
    assert probs.shape == (batch_size, n_atoms)

    # Check probabilities sum to 1
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5)
    print(f"Probabilities sum to 1: ✓")

    # Test expected value
    expected_values = critic.get_expected_value(obs)
    print(f"Expected values shape: {expected_values.shape}")
    assert expected_values.shape == (batch_size,)
    print(f"Sample expected values: {expected_values[:5]}")

    # Test risk values
    tau = 1.0
    for risk_type in ["entropic", "cvar", "mean_variance"]:
        risk_values = critic.get_risk_value(obs, tau, risk_type)
        print(f"\n{risk_type.upper()} risk values shape: {risk_values.shape}")
        print(f"Sample {risk_type} values: {risk_values[:5]}")

    # Test projection
    print("\n" + "="*50)
    print("Testing projection...")

    next_obs = torch.randn(batch_size, obs_dim)
    next_probs = critic(next_obs)
    rewards = torch.randn(batch_size)
    dones = torch.zeros(batch_size)

    target_probs = project_distribution(
        next_probs,
        rewards,
        critic.support,
        v_min,
        critic.delta_z,
        gamma=0.99,
        dones=dones
    )

    print(f"Target probs shape: {target_probs.shape}")
    assert target_probs.shape == (batch_size, n_atoms)

    # Check probabilities still sum to 1 after projection
    target_sums = target_probs.sum(dim=-1)
    assert torch.allclose(target_sums, torch.ones(batch_size), atol=1e-5)
    print(f"Projected probabilities sum to 1: ✓")

    print("\n" + "="*50)
    print("All tests passed! ✓")
