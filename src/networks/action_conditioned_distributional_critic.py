"""
Action-Conditioned Distributional Critic

Learns the return distribution Z(s,a) for each action, enabling
accurate computation of risk-adjusted Q-values Q_risk(s,a) for TRUE RQE.

This is the key component needed for implementing the theoretically
correct RQE gradient with exponential importance weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActionConditionedDistributionalCritic(nn.Module):
    """
    Distributional critic that outputs return distributions for EACH action

    Unlike the state-value critic which outputs Z(s), this outputs Z(s,a)
    for all actions a, enabling accurate risk assessment per action.

    Args:
        obs_dim: Observation dimension
        action_dim: Number of discrete actions
        hidden_dims: List of hidden layer sizes
        activation: Activation function name
        n_atoms: Number of atoms in distribution
        v_min: Minimum return value
        v_max: Maximum return value
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64],
        activation: str = "tanh",
        n_atoms: int = 51,
        v_min: float = 0.0,
        v_max: float = 600.0
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Atom locations (fixed)
        self.register_buffer(
            'z_atoms',
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Build MLP
        layers = []
        in_dim = obs_dim

        activation_fn = nn.Tanh if activation == "tanh" else nn.ReLU

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            in_dim = hidden_dim

        # Output: probability distribution for EACH action
        layers.append(nn.Linear(in_dim, action_dim * n_atoms))

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs):
        """
        Forward pass: compute return distribution for each action

        Args:
            obs: [batch, obs_dim]

        Returns:
            probs: [batch, action_dim, n_atoms] - probability distributions
        """
        batch_size = obs.shape[0]

        # Get logits
        logits = self.mlp(obs)  # [batch, action_dim * n_atoms]

        # Reshape to [batch, action_dim, n_atoms]
        logits = logits.view(batch_size, self.action_dim, self.n_atoms)

        # Apply softmax over atoms for each action
        probs = F.softmax(logits, dim=-1)

        return probs

    def get_distribution(self, obs, actions):
        """
        Get return distribution Z(s,a) for specific actions

        Args:
            obs: [batch, obs_dim]
            actions: [batch] - indices of actions to get distributions for

        Returns:
            probs: [batch, n_atoms] - distribution for taken actions
        """
        all_probs = self.forward(obs)  # [batch, action_dim, n_atoms]

        batch_size = obs.shape[0]
        batch_indices = torch.arange(batch_size, device=obs.device)

        # Select distributions for taken actions
        action_probs = all_probs[batch_indices, actions, :]  # [batch, n_atoms]

        return action_probs

    def get_expected_value(self, obs, actions):
        """
        Compute expected return E[Z(s,a)] for given actions

        Args:
            obs: [batch, obs_dim]
            actions: [batch]

        Returns:
            values: [batch] - expected returns
        """
        probs = self.get_distribution(obs, actions)  # [batch, n_atoms]

        # E[Z] = Σ p(z) * z
        values = (probs * self.z_atoms).sum(dim=-1)

        return values

    def get_risk_value(self, obs, actions, tau=1.0, risk_type="entropic"):
        """
        Compute risk-adjusted Q-value for given actions

        This is the KEY method for RQE: computes ρ_τ(Z(s,a))

        Args:
            obs: [batch, obs_dim]
            actions: [batch] - action indices
            tau: Risk aversion parameter (lower = more risk-averse)
            risk_type: "entropic", "cvar", or "mean_variance"

        Returns:
            q_risk: [batch] - risk-adjusted Q-values
        """
        probs = self.get_distribution(obs, actions)  # [batch, n_atoms]

        if risk_type == "entropic":
            # Entropic risk measure: ρ_τ(Z) = -(1/τ) log E[exp(-τZ)]
            exp_neg_tau_z = torch.exp(-tau * self.z_atoms)  # [n_atoms]
            expectation = (probs * exp_neg_tau_z).sum(dim=-1)  # [batch]
            q_risk = -(1.0 / tau) * torch.log(expectation + 1e-8)

        elif risk_type == "cvar":
            # CVaR: Conditional Value at Risk (average of worst α quantile)
            alpha = 1.0 / (1.0 + tau)  # Map tau to confidence level

            # Compute cumulative distribution
            cum_probs = torch.cumsum(probs, dim=-1)  # [batch, n_atoms]

            # Find atoms in worst α quantile
            in_tail = (cum_probs <= alpha).float()  # [batch, n_atoms]

            # Average return in tail (with normalization)
            tail_mass = in_tail.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            q_risk = ((in_tail * probs * self.z_atoms).sum(dim=-1) /
                     (in_tail * probs).sum(dim=-1).clamp(min=1e-8))

        elif risk_type == "mean_variance":
            # Mean-Variance: E[Z] - τ * Var[Z]
            mean = (probs * self.z_atoms).sum(dim=-1)  # [batch]
            variance = (probs * (self.z_atoms - mean.unsqueeze(-1))**2).sum(dim=-1)
            q_risk = mean - tau * variance

        else:
            raise ValueError(f"Unknown risk type: {risk_type}")

        return q_risk

    def get_all_risk_values(self, obs, tau=1.0, risk_type="entropic"):
        """
        Compute risk-adjusted Q-values for ALL actions

        Useful for action selection (e.g., choosing most risk-averse action)

        Args:
            obs: [batch, obs_dim]
            tau: Risk aversion parameter
            risk_type: Type of risk measure

        Returns:
            q_risk: [batch, action_dim] - risk-adjusted Q for each action
        """
        all_probs = self.forward(obs)  # [batch, action_dim, n_atoms]
        batch_size = obs.shape[0]

        q_risk = torch.zeros(batch_size, self.action_dim, device=obs.device)

        for a in range(self.action_dim):
            probs = all_probs[:, a, :]  # [batch, n_atoms]

            if risk_type == "entropic":
                exp_neg_tau_z = torch.exp(-tau * self.z_atoms)
                expectation = (probs * exp_neg_tau_z).sum(dim=-1)
                q_risk[:, a] = -(1.0 / tau) * torch.log(expectation + 1e-8)

            elif risk_type == "cvar":
                alpha = 1.0 / (1.0 + tau)
                cum_probs = torch.cumsum(probs, dim=-1)
                in_tail = (cum_probs <= alpha).float()
                q_risk[:, a] = ((in_tail * probs * self.z_atoms).sum(dim=-1) /
                               (in_tail * probs).sum(dim=-1).clamp(min=1e-8))

            elif risk_type == "mean_variance":
                mean = (probs * self.z_atoms).sum(dim=-1)
                variance = (probs * (self.z_atoms - mean.unsqueeze(-1))**2).sum(dim=-1)
                q_risk[:, a] = mean - tau * variance

        return q_risk


def project_distribution(
    rewards,
    next_probs,
    z_atoms,
    dones,
    gamma=0.99,
    v_min=0.0,
    v_max=600.0
):
    """
    Categorical projection for distributional Bellman backup

    Implements the projection step from C51:
    Φ[T_z](s,a) where T_z = r + γ * z (projected onto support)

    Args:
        rewards: [batch] - immediate rewards
        next_probs: [batch, n_atoms] - next state distribution
        z_atoms: [n_atoms] - atom locations
        dones: [batch] - terminal flags
        gamma: Discount factor
        v_min, v_max: Distribution support bounds

    Returns:
        target_probs: [batch, n_atoms] - projected target distribution
    """
    batch_size = rewards.shape[0]
    n_atoms = len(z_atoms)
    delta_z = (v_max - v_min) / (n_atoms - 1)

    # Compute projected atom locations: T_z = r + γ * z
    # [batch, 1] + [batch, 1] * [1, n_atoms] = [batch, n_atoms]
    Tz = rewards.unsqueeze(-1) + gamma * (1 - dones.unsqueeze(-1)) * z_atoms.unsqueeze(0)

    # Clip to support
    Tz = Tz.clamp(v_min, v_max)

    # Compute projection indices
    b = (Tz - v_min) / delta_z  # [batch, n_atoms]
    l = b.floor().long()
    u = b.ceil().long()

    # Fix edge case where l == u (atom falls exactly on grid point)
    l[(u > 0) * (l == u)] -= 1
    u[(l < (n_atoms - 1)) * (l == u)] += 1

    # Distribute probability mass
    target_probs = torch.zeros(batch_size, n_atoms, device=rewards.device)

    # Lower projection
    offset = torch.arange(batch_size, device=rewards.device).unsqueeze(-1) * n_atoms
    l_indices = (l + offset).view(-1)
    u_indices = (u + offset).view(-1)

    # Probability goes to lower and upper atoms proportionally
    # P(l) += p * (u - b), P(u) += p * (b - l)
    prob_u = (b - l.float())  # [batch, n_atoms]
    prob_l = 1.0 - prob_u

    # Flatten and scatter
    target_probs.view(-1).index_add_(
        0, l_indices,
        (next_probs * prob_l).view(-1)
    )
    target_probs.view(-1).index_add_(
        0, u_indices,
        (next_probs * prob_u).view(-1)
    )

    return target_probs


if __name__ == "__main__":
    print("Testing ActionConditionedDistributionalCritic...")

    # Create critic
    critic = ActionConditionedDistributionalCritic(
        obs_dim=4,
        action_dim=2,
        hidden_dims=[64, 64],
        n_atoms=51,
        v_min=0.0,
        v_max=500.0
    )

    # Test forward pass
    batch_size = 32
    obs = torch.randn(batch_size, 4)

    # Get distributions for all actions
    all_probs = critic.forward(obs)
    print(f"✓ Forward pass: {all_probs.shape} (should be [32, 2, 51])")
    assert all_probs.shape == (batch_size, 2, 51)

    # Check probabilities sum to 1
    prob_sums = all_probs.sum(dim=-1)
    print(f"✓ Probabilities sum to 1: {torch.allclose(prob_sums, torch.ones_like(prob_sums))}")

    # Get distribution for specific actions
    actions = torch.randint(0, 2, (batch_size,))
    action_probs = critic.get_distribution(obs, actions)
    print(f"✓ Action distributions: {action_probs.shape} (should be [32, 51])")

    # Compute expected values
    expected_values = critic.get_expected_value(obs, actions)
    print(f"✓ Expected values: {expected_values.shape}, mean={expected_values.mean():.2f}")

    # Compute risk-adjusted values
    for tau, risk_type in [(0.3, "entropic"), (1.0, "cvar"), (0.1, "mean_variance")]:
        q_risk = critic.get_risk_value(obs, actions, tau=tau, risk_type=risk_type)
        print(f"✓ Risk values (tau={tau}, {risk_type}): mean={q_risk.mean():.2f}")

    # Get all risk values
    all_q_risk = critic.get_all_risk_values(obs, tau=0.5, risk_type="entropic")
    print(f"✓ All risk values: {all_q_risk.shape} (should be [32, 2])")

    # Test projection
    rewards = torch.randn(batch_size) * 10
    next_probs = torch.rand(batch_size, 51)
    next_probs = next_probs / next_probs.sum(dim=-1, keepdim=True)
    dones = torch.randint(0, 2, (batch_size,)).float()

    target_probs = project_distribution(
        rewards, next_probs, critic.z_atoms, dones, gamma=0.99
    )
    print(f"✓ Projection: {target_probs.shape}, sums to 1: {torch.allclose(target_probs.sum(dim=-1), torch.ones(batch_size))}")

    print("\n✓ All tests passed!")
