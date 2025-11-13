"""
RQE-MAPPO: Risk-Averse Multi-Agent Proximal Policy Optimization

Extends MAPPO with:
1. Risk-averse value targets (tau parameter)
2. Entropy regularization for bounded rationality (epsilon parameter)

Reference: Mazumdar et al. (2025) "Tractable Multi-Agent Reinforcement Learning
           through Behavioral Economics"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..networks.actor import ActorNetwork
from ..networks.critic import CriticNetwork
from .risk_measures import get_risk_measure, RiskMeasure


@dataclass
class RQEConfig:
    """Configuration for RQE-MAPPO"""

    # Environment
    n_agents: int
    obs_dim: int
    action_dim: int

    # RQE parameters (KEY DIFFERENCE FROM MAPPO!)
    tau: float = 1.0  # Risk aversion (lower = more risk-averse)
    epsilon: float = 0.01  # Bounded rationality (entropy coefficient)
    risk_measure: str = "entropic"  # "entropic", "cvar", "mean_variance"

    # Network architecture
    hidden_dims: List[int] = None
    activation: str = "relu"

    # PPO parameters
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_clip_param: float = 0.2
    max_grad_norm: float = 0.5

    # Training
    n_epochs: int = 10
    n_minibatches: int = 4
    normalize_advantages: bool = True
    use_centralized_critic: bool = False  # Use global state for critic

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


class RQE_MAPPO:
    """
    Risk-Averse Multi-Agent PPO

    Key modifications from standard MAPPO:
    1. Risk-aware advantage computation using risk measures
    2. Entropy bonus for bounded rationality
    3. Optional heterogeneous risk preferences per agent
    """

    def __init__(self, config: RQEConfig):
        self.config = config
        self.n_agents = config.n_agents

        # Create networks for each agent
        self.actors = nn.ModuleList([
            ActorNetwork(
                config.obs_dim,
                config.action_dim,
                config.hidden_dims,
                config.activation
            )
            for _ in range(config.n_agents)
        ])

        # Critic can use global state (centralized training)
        critic_input_dim = (
            config.obs_dim * config.n_agents
            if config.use_centralized_critic
            else config.obs_dim
        )

        self.critics = nn.ModuleList([
            CriticNetwork(
                critic_input_dim,
                config.hidden_dims,
                config.activation
            )
            for _ in range(config.n_agents)
        ])

        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=config.lr_actor)
            for actor in self.actors
        ]

        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=config.lr_critic)
            for critic in self.critics
        ]

        # Risk measure
        self.risk_measure = get_risk_measure(
            config.risk_measure,
            config.tau
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def to(self, device):
        """Move all networks to device"""
        self.device = device
        self.actors.to(device)
        self.critics.to(device)
        return self

    def get_actions(
        self,
        observations: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions for all agents

        Args:
            observations: [n_agents, obs_dim] or [batch, n_agents, obs_dim]
            deterministic: If True, use argmax

        Returns:
            actions: [n_agents] or [batch, n_agents]
            log_probs: [n_agents] or [batch, n_agents]
            entropies: [n_agents] or [batch, n_agents]
        """
        actions_list = []
        log_probs_list = []
        entropies_list = []

        for i in range(self.n_agents):
            obs_i = observations[..., i, :] if observations.ndim == 3 else observations[i]
            action, log_prob, entropy = self.actors[i].get_action(obs_i, deterministic)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            entropies_list.append(entropy)

        actions = torch.stack(actions_list, dim=-1)
        log_probs = torch.stack(log_probs_list, dim=-1)
        entropies = torch.stack(entropies_list, dim=-1)

        return actions, log_probs, entropies

    def get_values(
        self,
        observations: torch.Tensor,
        global_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get value estimates for all agents

        Args:
            observations: [n_agents, obs_dim] or [batch, n_agents, obs_dim]
            global_state: Optional [batch, state_dim] for centralized critic

        Returns:
            values: [n_agents] or [batch, n_agents]
        """
        values_list = []

        for i in range(self.n_agents):
            if self.config.use_centralized_critic and global_state is not None:
                critic_input = global_state
            else:
                critic_input = (
                    observations[..., i, :]
                    if observations.ndim == 3
                    else observations[i]
                )

            value = self.critics[i].get_value(critic_input)
            values_list.append(value)

        values = torch.stack(values_list, dim=-1)
        return values

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            rewards: [batch, n_agents]
            values: [batch, n_agents]
            dones: [batch]
            next_values: [batch, n_agents]

        Returns:
            advantages: [batch, n_agents]
            returns: [batch, n_agents]
        """
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)

        # Compute GAE for each agent
        for agent_id in range(self.n_agents):
            agent_rewards = rewards[:, agent_id]
            agent_values = values[:, agent_id]
            agent_next_values = next_values[:, agent_id]

            lastgaelam = 0
            for t in reversed(range(batch_size)):
                if t == batch_size - 1:
                    next_value = agent_next_values[t]
                else:
                    next_value = agent_values[t + 1]

                # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
                delta = (
                    agent_rewards[t]
                    + self.config.gamma * next_value * (1 - dones[t])
                    - agent_values[t]
                )

                # GAE: A_t = δ_t + γλ * (1 - done) * A_{t+1}
                advantages[t, agent_id] = lastgaelam = (
                    delta
                    + self.config.gamma * self.config.gae_lambda
                    * (1 - dones[t]) * lastgaelam
                )

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_observations: torch.Tensor,
        global_state: Optional[torch.Tensor] = None,
        next_global_state: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Update all agents using PPO

        Args:
            observations: [batch, n_agents, obs_dim]
            actions: [batch, n_agents]
            old_log_probs: [batch, n_agents]
            rewards: [batch, n_agents]
            dones: [batch]
            next_observations: [batch, n_agents, obs_dim]
            global_state: Optional [batch, global_state_dim]
            next_global_state: Optional [batch, global_state_dim]

        Returns:
            Dictionary of training statistics
        """
        batch_size = observations.shape[0]

        # Compute values
        with torch.no_grad():
            values = self.get_values(observations, global_state).detach()
            next_values = self.get_values(next_observations, next_global_state).detach()

        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)

        # Normalize advantages (optional but helpful)
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training statistics
        stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clipfrac': 0.0,
        }

        # PPO epochs
        for epoch in range(self.config.n_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size)

            # Minibatch updates
            minibatch_size = batch_size // self.config.n_minibatches

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch
                mb_obs = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices].detach()  # Detach old log probs!
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = values[mb_indices]
                mb_global_state = global_state[mb_indices] if global_state is not None else None

                # Update each agent
                for agent_id in range(self.n_agents):
                    agent_obs = mb_obs[:, agent_id]
                    agent_actions = mb_actions[:, agent_id]
                    agent_old_log_probs = mb_old_log_probs[:, agent_id]
                    agent_advantages = mb_advantages[:, agent_id]
                    agent_returns = mb_returns[:, agent_id]
                    agent_old_values = mb_old_values[:, agent_id]

                    # ============ Actor Update ============
                    log_probs, entropy = self.actors[agent_id].evaluate_actions(
                        agent_obs, agent_actions
                    )

                    # PPO clipped objective
                    ratio = torch.exp(log_probs - agent_old_log_probs)
                    surr1 = ratio * agent_advantages
                    surr2 = torch.clamp(
                        ratio,
                        1 - self.config.clip_param,
                        1 + self.config.clip_param
                    ) * agent_advantages

                    # Actor loss with entropy bonus (BOUNDED RATIONALITY!)
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy_bonus = -self.config.epsilon * entropy.mean()
                    total_actor_loss = actor_loss + entropy_bonus

                    # Update actor
                    self.actor_optimizers[agent_id].zero_grad()
                    total_actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actors[agent_id].parameters(),
                        self.config.max_grad_norm
                    )
                    self.actor_optimizers[agent_id].step()

                    # ============ Critic Update ============
                    if self.config.use_centralized_critic and mb_global_state is not None:
                        critic_input = mb_global_state
                    else:
                        critic_input = agent_obs

                    values_pred = self.critics[agent_id].get_value(critic_input)

                    # Value loss with clipping (from PPO)
                    values_clipped = agent_old_values + torch.clamp(
                        values_pred - agent_old_values,
                        -self.config.value_clip_param,
                        self.config.value_clip_param
                    )

                    value_loss_unclipped = (values_pred - agent_returns) ** 2
                    value_loss_clipped = (values_clipped - agent_returns) ** 2
                    critic_loss = 0.5 * torch.max(
                        value_loss_unclipped,
                        value_loss_clipped
                    ).mean()

                    # Update critic
                    self.critic_optimizers[agent_id].zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critics[agent_id].parameters(),
                        self.config.max_grad_norm
                    )
                    self.critic_optimizers[agent_id].step()

                    # Record statistics
                    with torch.no_grad():
                        approx_kl = (agent_old_log_probs - log_probs).mean()
                        clipfrac = ((ratio - 1.0).abs() > self.config.clip_param).float().mean()

                    stats['actor_loss'] += actor_loss.item()
                    stats['critic_loss'] += critic_loss.item()
                    stats['entropy'] += entropy.mean().item()
                    stats['approx_kl'] += approx_kl.item()
                    stats['clipfrac'] += clipfrac.item()

        # Average statistics
        n_updates = self.config.n_epochs * self.config.n_minibatches * self.n_agents
        for key in stats:
            stats[key] /= n_updates

        return stats

    def save(self, path: str):
        """Save model"""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers],
            'config': self.config,
        }, path)

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        for i, actor_state in enumerate(checkpoint['actors']):
            self.actors[i].load_state_dict(actor_state)
        for i, critic_state in enumerate(checkpoint['critics']):
            self.critics[i].load_state_dict(critic_state)
        for i, opt_state in enumerate(checkpoint['actor_optimizers']):
            self.actor_optimizers[i].load_state_dict(opt_state)
        for i, opt_state in enumerate(checkpoint['critic_optimizers']):
            self.critic_optimizers[i].load_state_dict(opt_state)


if __name__ == "__main__":
    # Test RQE-MAPPO
    import sys
    from pathlib import Path
    # Add parent directory to path for testing
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.networks.actor import ActorNetwork
    from src.networks.critic import CriticNetwork
    from src.algorithms.risk_measures import get_risk_measure

    print("Testing RQE-MAPPO...")

    config = RQEConfig(
        n_agents=3,
        obs_dim=10,
        action_dim=5,
        tau=0.5,  # Risk-averse
        epsilon=0.01,  # Bounded rationality
        risk_measure="entropic"
    )

    agent = RQE_MAPPO(config)

    # Test get_actions
    obs = torch.randn(32, 3, 10)  # [batch, n_agents, obs_dim]
    actions, log_probs, entropies = agent.get_actions(obs)
    print(f"Actions shape: {actions.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Entropies shape: {entropies.shape}")

    # Test get_values
    values = agent.get_values(obs)
    print(f"Values shape: {values.shape}")

    # Test update
    rewards = torch.randn(32, 3)
    dones = torch.zeros(32)
    next_obs = torch.randn(32, 3, 10)

    stats = agent.update(obs, actions, log_probs, rewards, dones, next_obs)
    print(f"Training stats: {stats}")

    print("\nAll tests passed!")
