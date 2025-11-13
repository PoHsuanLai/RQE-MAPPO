"""
Distributional RQE-MAPPO: Risk-Averse Multi-Agent PPO with Distributional Critic

Uses distributional RL (C51-style) to learn full return distribution,
enabling precise computation of risk measures for risk-averse decision making.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..networks.actor import ActorNetwork
from ..networks.distributional_critic import DistributionalCritic, project_distribution


@dataclass
class DistributionalRQEConfig:
    """Configuration for Distributional RQE-MAPPO"""

    # Environment
    n_agents: int
    obs_dim: int
    action_dim: int

    # RQE parameters
    tau: float = 1.0  # Risk aversion (lower = more risk-averse)
    epsilon: float = 0.01  # Bounded rationality (entropy coefficient, FIXED!)
    risk_measure: str = "entropic"  # "entropic", "cvar", "mean_variance"

    # Distributional parameters
    n_atoms: int = 51  # Number of atoms in categorical distribution
    v_min: float = -10.0  # Minimum support value
    v_max: float = 10.0  # Maximum support value

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
    use_centralized_critic: bool = False

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


class DistributionalRQE_MAPPO:
    """
    Distributional Risk-Averse Multi-Agent PPO

    Key features:
    1. Distributional critic learns full return distribution Z(s)
    2. Risk measures applied to distribution for risk-aware values
    3. Fixed entropy bonus for bounded rationality
    4. Distributional Bellman updates with projection
    """

    def __init__(self, config: DistributionalRQEConfig):
        self.config = config
        self.n_agents = config.n_agents

        # Create actor networks (standard)
        self.actors = nn.ModuleList([
            ActorNetwork(
                config.obs_dim,
                config.action_dim,
                config.hidden_dims,
                config.activation
            )
            for _ in range(config.n_agents)
        ])

        # Critic input dimension
        critic_input_dim = (
            config.obs_dim * config.n_agents
            if config.use_centralized_critic
            else config.obs_dim
        )

        # Create DISTRIBUTIONAL critics (key difference!)
        self.critics = nn.ModuleList([
            DistributionalCritic(
                critic_input_dim,
                config.hidden_dims,
                config.activation,
                n_atoms=config.n_atoms,
                v_min=config.v_min,
                v_max=config.v_max
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

        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actors = self.actors.to(self.device)
        self.critics = self.critics.to(self.device)

    def select_actions(
        self,
        observations: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions for all agents

        Args:
            observations: Joint observations [n_agents, obs_dim]
            deterministic: If True, select argmax action

        Returns:
            actions: Selected actions [n_agents]
            log_probs: Log probabilities [n_agents]
            values: Risk-adjusted values [n_agents]
        """
        observations = torch.FloatTensor(observations).to(self.device)

        actions = []
        log_probs = []
        values = []

        for i in range(self.n_agents):
            obs_i = observations[i]

            # Get action from policy
            with torch.no_grad():
                action, log_prob = self.actors[i].sample_action(
                    obs_i.unsqueeze(0),
                    deterministic=deterministic
                )

                # Get risk-adjusted value
                value = self.critics[i].get_risk_value(
                    obs_i.unsqueeze(0),
                    tau=self.config.tau,
                    risk_type=self.config.risk_measure
                )

            actions.append(action.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
            values.append(value.cpu().numpy())

        return (
            np.array(actions),
            np.array(log_probs),
            np.array(values)
        )

    def compute_gae(
        self,
        agent_idx: int,
        rewards: torch.Tensor,
        observations: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation with RISK-ADJUSTED values

        This is where risk-aversion comes in! We use risk measures on the
        distribution to get conservative value estimates.

        Args:
            agent_idx: Which agent
            rewards: Rewards [T]
            observations: Observations [T, obs_dim]
            dones: Done flags [T]

        Returns:
            advantages: Advantages [T]
            returns: Risk-adjusted returns [T]
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        with torch.no_grad():
            # Get risk-adjusted values for all timesteps
            values = self.critics[agent_idx].get_risk_value(
                observations,
                tau=self.config.tau,
                risk_type=self.config.risk_measure
            )

            # GAE computation (same as standard, but with risk-adjusted values)
            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = 0.0
                else:
                    next_value = values[t + 1]

                # TD error
                delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]

                # GAE
                advantages[t] = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_advantage
                last_advantage = advantages[t]

            # Returns = advantages + values
            returns = advantages + values

        return advantages, returns

    def update(
        self,
        buffer: Dict[str, torch.Tensor],
        agent_idx: int
    ) -> Dict[str, float]:
        """
        Update one agent using PPO with distributional critic

        Args:
            buffer: Rollout buffer with keys:
                - observations: [T, obs_dim]
                - actions: [T]
                - rewards: [T]
                - dones: [T]
                - log_probs_old: [T]
                - values_old: [T]
            agent_idx: Which agent to update

        Returns:
            metrics: Training metrics
        """
        # Move buffer to device
        observations = buffer['observations'].to(self.device)
        actions = buffer['actions'].to(self.device)
        rewards = buffer['rewards'].to(self.device)
        dones = buffer['dones'].to(self.device)
        log_probs_old = buffer['log_probs_old'].to(self.device)

        # Compute advantages
        advantages, returns = self.compute_gae(
            agent_idx, rewards, observations, dones
        )

        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs of updates
        actor_losses = []
        critic_losses = []
        entropy_losses = []

        for _ in range(self.config.n_epochs):
            # Shuffle data
            indices = torch.randperm(len(observations))

            # Minibatch updates
            batch_size = len(observations) // self.config.n_minibatches

            for start in range(0, len(observations), batch_size):
                end = start + batch_size
                mb_indices = indices[start:end]

                mb_obs = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]
                mb_rewards = rewards[mb_indices]

                # ========== UPDATE CRITIC (Distributional!) ==========
                critic_loss = self._update_critic_distributional(
                    agent_idx,
                    mb_obs,
                    mb_rewards,
                    dones[mb_indices]
                )
                critic_losses.append(critic_loss)

                # ========== UPDATE ACTOR ==========
                actor_loss, entropy_loss = self._update_actor(
                    agent_idx,
                    mb_obs,
                    mb_actions,
                    mb_advantages,
                    mb_log_probs_old
                )
                actor_losses.append(actor_loss)
                entropy_losses.append(entropy_loss)

        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropy_losses)
        }

    def _update_critic_distributional(
        self,
        agent_idx: int,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """
        Update distributional critic using categorical cross-entropy loss

        This is the KEY difference from standard MAPPO:
        - Target is a DISTRIBUTION (projected Bellman update)
        - Loss is cross-entropy between distributions

        Args:
            agent_idx: Which agent
            observations: [batch, obs_dim]
            rewards: [batch]
            dones: [batch]

        Returns:
            loss: Scalar loss value
        """
        # Get next observations (shifted by 1)
        next_observations = torch.roll(observations, shifts=-1, dims=0)
        next_observations[-1] = observations[-1]  # Last next_obs doesn't matter

        # Get current distribution
        current_probs = self.critics[agent_idx](observations)  # [batch, n_atoms]

        # Compute target distribution (Bellman update with projection)
        with torch.no_grad():
            next_probs = self.critics[agent_idx](next_observations)  # [batch, n_atoms]

            # Project: T(Z) = r + γZ
            target_probs = project_distribution(
                next_probs,
                rewards,
                self.critics[agent_idx].support,
                self.config.v_min,
                self.critics[agent_idx].delta_z,
                gamma=self.config.gamma,
                dones=dones
            )

        # Cross-entropy loss: -Σ target * log(current)
        loss = -(target_probs * torch.log(current_probs + 1e-8)).sum(dim=-1).mean()

        # Update
        self.critic_optimizers[agent_idx].zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.critics[agent_idx].parameters(),
            self.config.max_grad_norm
        )
        self.critic_optimizers[agent_idx].step()

        return loss.item()

    def _update_actor(
        self,
        agent_idx: int,
        observations: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        log_probs_old: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Update actor using PPO clipped loss + FIXED entropy bonus

        Args:
            agent_idx: Which agent
            observations: [batch, obs_dim]
            actions: [batch]
            advantages: [batch]
            log_probs_old: [batch]

        Returns:
            actor_loss: Scalar loss
            entropy: Mean entropy
        """
        # Evaluate actions
        log_probs_new, entropy = self.actors[agent_idx].evaluate_actions(
            observations, actions
        )

        # PPO ratio
        ratio = torch.exp(log_probs_new - log_probs_old)

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # FIXED entropy bonus (KEY for RQE: don't anneal!)
        entropy_bonus = -self.config.epsilon * entropy.mean()

        # Total loss
        total_loss = actor_loss + entropy_bonus

        # Update
        self.actor_optimizers[agent_idx].zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actors[agent_idx].parameters(),
            self.config.max_grad_norm
        )
        self.actor_optimizers[agent_idx].step()

        return actor_loss.item(), entropy.mean().item()

    def save(self, path: str):
        """Save all networks"""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load all networks"""
        checkpoint = torch.load(path, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint['critics'][i])


if __name__ == "__main__":
    print("Testing DistributionalRQE_MAPPO...")

    # Create config
    config = DistributionalRQEConfig(
        n_agents=3,
        obs_dim=10,
        action_dim=5,
        tau=1.0,
        epsilon=0.01,
        risk_measure="entropic",
        n_atoms=51,
        v_min=-10.0,
        v_max=10.0
    )

    # Create algorithm
    algo = DistributionalRQE_MAPPO(config)

    print(f"Number of agents: {algo.n_agents}")
    print(f"Actors: {len(algo.actors)}")
    print(f"Critics: {len(algo.critics)}")
    print(f"Device: {algo.device}")

    # Test action selection
    obs = np.random.randn(config.n_agents, config.obs_dim)
    actions, log_probs, values = algo.select_actions(obs)

    print(f"\nActions shape: {actions.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Sample actions: {actions}")
    print(f"Sample values: {values}")

    # Test update
    T = 100
    buffer = {
        'observations': torch.randn(T, config.obs_dim),
        'actions': torch.randint(0, config.action_dim, (T,)),
        'rewards': torch.randn(T),
        'dones': torch.zeros(T),
        'log_probs_old': torch.randn(T),
        'values_old': torch.randn(T)
    }

    print(f"\nTesting update for agent 0...")
    metrics = algo.update(buffer, agent_idx=0)
    print(f"Metrics: {metrics}")

    print("\nAll tests passed! ✓")
