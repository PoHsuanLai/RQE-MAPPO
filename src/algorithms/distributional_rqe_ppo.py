"""
Distributional RQE-PPO: Single-Agent Risk-Averse PPO with Distributional Critic

Single-agent version for easier testing and debugging.
Uses distributional RL to learn full return distribution for precise risk measures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

from ..networks.actor import ActorNetwork
from ..networks.distributional_critic import DistributionalCritic, project_distribution


@dataclass
class DistributionalRQEPPOConfig:
    """Configuration for single-agent Distributional RQE-PPO"""

    # Environment
    obs_dim: int
    action_dim: int

    # RQE parameters
    tau: float = 1.0  # Risk aversion (lower = more risk-averse)
    epsilon: float = 0.01  # Bounded rationality (entropy coefficient)
    epsilon_decay: bool = False  # Whether to decay entropy over time (like standard PPO)
    epsilon_min: float = 0.0  # Minimum entropy coefficient (if decay enabled)
    epsilon_decay_rate: float = 0.99  # Decay rate per update (if decay enabled)
    risk_measure: str = "entropic"  # "entropic", "cvar", "mean_variance"

    # Distributional parameters
    n_atoms: int = 51  # Number of atoms in categorical distribution
    v_min: float = 0.0  # Minimum support value (task-dependent, adjust for your env)
    v_max: float = 600.0  # Maximum support value (task-dependent, adjust for your env)

    # Network architecture
    hidden_dims: list = field(default_factory=lambda: [64, 64])
    activation: str = "relu"

    # PPO parameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4  # Match actor LR for stability
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_clip_param: float = 0.2
    max_grad_norm: float = 0.5

    # Training
    n_epochs: int = 10
    n_minibatches: int = 4
    normalize_advantages: bool = True


class DistributionalRQE_PPO:
    """
    Single-Agent Distributional Risk-Averse PPO

    Key features:
    1. Distributional critic learns full return distribution Z(s)
    2. Risk measures applied to distribution for risk-aware values
    3. Fixed entropy bonus for bounded rationality
    4. Distributional Bellman updates with projection
    """

    def __init__(self, config: DistributionalRQEPPOConfig):
        self.config = config

        # Track current epsilon (for decay)
        self.current_epsilon = config.epsilon

        # Create actor network (standard)
        self.actor = ActorNetwork(
            config.obs_dim,
            config.action_dim,
            config.hidden_dims,
            config.activation
        )

        # Create DISTRIBUTIONAL critic (key difference!)
        self.critic = DistributionalCritic(
            config.obs_dim,
            config.hidden_dims,
            config.activation,
            n_atoms=config.n_atoms,
            v_min=config.v_min,
            v_max=config.v_max
        )

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action for single agent

        Args:
            observation: Observation [obs_dim]
            deterministic: If True, select argmax action

        Returns:
            action: Selected action (int)
            log_prob: Log probability (float)
            value: Risk-adjusted value (float)
        """
        observation = torch.FloatTensor(observation).to(self.device)

        with torch.no_grad():
            # Get action from policy
            action, log_prob, entropy = self.actor.get_action(
                observation.unsqueeze(0),
                deterministic=deterministic
            )

            # Get risk-adjusted value
            value = self.critic.get_risk_value(
                observation.unsqueeze(0),
                tau=self.config.tau,
                risk_type=self.config.risk_measure
            )

        return (
            action.item(),
            log_prob.item(),
            value.item()
        )

    def compute_gae(
        self,
        rewards: torch.Tensor,
        observations: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation with RISK-ADJUSTED values

        This is where risk-aversion comes in! We use risk measures on the
        distribution to get conservative value estimates.

        Args:
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
            values = self.critic.get_risk_value(
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

    def update(self, buffer: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update agent using PPO with distributional critic

        Args:
            buffer: Rollout buffer with keys:
                - observations: [T, obs_dim]
                - actions: [T]
                - rewards: [T]
                - dones: [T]
                - log_probs_old: [T]

        Returns:
            metrics: Training metrics
        """
        # Move buffer to device (do this ONCE, not per minibatch)
        observations = buffer['observations'].to(self.device, non_blocking=True)
        actions = buffer['actions'].to(self.device, non_blocking=True)
        rewards = buffer['rewards'].to(self.device, non_blocking=True)
        dones = buffer['dones'].to(self.device, non_blocking=True)
        log_probs_old = buffer['log_probs_old'].to(self.device, non_blocking=True)

        # Compute advantages
        advantages, returns = self.compute_gae(rewards, observations, dones)

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
                end = min(start + batch_size, len(observations))
                mb_indices = indices[start:end]

                mb_obs = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_log_probs_old = log_probs_old[mb_indices]
                mb_rewards = rewards[mb_indices]
                mb_dones = dones[mb_indices]

                # ========== UPDATE CRITIC (Distributional!) ==========
                critic_loss = self._update_critic_distributional(
                    mb_obs,
                    mb_rewards,
                    mb_dones
                )
                critic_losses.append(critic_loss)

                # ========== UPDATE ACTOR ==========
                actor_loss, entropy_loss = self._update_actor(
                    mb_obs,
                    mb_actions,
                    mb_advantages,
                    mb_log_probs_old
                )
                actor_losses.append(actor_loss)
                entropy_losses.append(entropy_loss)

        # Decay epsilon if enabled
        if self.config.epsilon_decay:
            self.current_epsilon = max(
                self.config.epsilon_min,
                self.current_epsilon * self.config.epsilon_decay_rate
            )

        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropy_losses),
            'epsilon': self.current_epsilon
        }

    def _update_critic_distributional(
        self,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """
        Update distributional critic using categorical cross-entropy loss

        This is the KEY difference from standard PPO:
        - Target is a DISTRIBUTION (projected Bellman update)
        - Loss is cross-entropy between distributions

        Args:
            observations: [batch, obs_dim]
            rewards: [batch]
            dones: [batch]

        Returns:
            loss: Scalar loss value
        """
        # Get next observations (properly handling episode boundaries)
        # VECTORIZED: No Python loops!

        # Shift observations by 1 to get next_observations
        next_observations = torch.roll(observations, shifts=-1, dims=0)

        # Detect episode boundaries: if dones[i] == 1, then next_observations[i]
        # should NOT be used for bootstrapping (gamma=0 handles this in project_distribution)
        # For terminal states and last position, set next_obs = obs (won't affect loss since gamma=0)
        terminal_mask = dones.bool()
        next_observations[terminal_mask] = observations[terminal_mask]
        next_observations[-1] = observations[-1]  # Last position has no valid next

        # Get current distribution
        current_probs = self.critic(observations)  # [batch, n_atoms]

        # Compute target distribution (Bellman update with projection)
        with torch.no_grad():
            next_probs = self.critic(next_observations)  # [batch, n_atoms]

            # Project: T(Z) = r + γZ
            # The project_distribution function will set gamma=0 for terminal states
            target_probs = project_distribution(
                next_probs,
                rewards,
                self.critic.support,
                self.config.v_min,
                self.critic.delta_z,
                gamma=self.config.gamma,
                dones=dones
            )

        # Cross-entropy loss: -Σ target * log(current)
        loss = -(target_probs * torch.log(current_probs + 1e-8)).sum(dim=-1).mean()

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite critic loss detected: {loss.item()}")
            return 0.0

        # Update
        self.critic_optimizer.zero_grad()
        loss.backward()

        # Check for NaN gradients
        grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.max_grad_norm
        )

        if not torch.isfinite(grad_norm):
            print(f"WARNING: Non-finite critic gradients detected")
            self.critic_optimizer.zero_grad()
            return 0.0

        self.critic_optimizer.step()

        return loss.item()

    def _update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        log_probs_old: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Update actor using PPO clipped loss + FIXED entropy bonus

        Args:
            observations: [batch, obs_dim]
            actions: [batch]
            advantages: [batch]
            log_probs_old: [batch]

        Returns:
            actor_loss: Scalar loss
            entropy: Mean entropy
        """
        # Check for NaN in inputs
        if not torch.isfinite(advantages).all():
            print(f"WARNING: Non-finite advantages detected")
            print(f"Advantages: min={advantages.min()}, max={advantages.max()}, mean={advantages.mean()}")
            return 0.0, 0.0

        # Evaluate actions
        log_probs_new, entropy = self.actor.evaluate_actions(
            observations, actions
        )

        # Check for NaN in log probs
        if not torch.isfinite(log_probs_new).all():
            print(f"WARNING: Non-finite log_probs_new detected")
            return 0.0, entropy.mean().item()

        # PPO ratio with numerical stability
        ratio = torch.exp(torch.clamp(log_probs_new - log_probs_old, -20, 20))

        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus (with optional decay if epsilon_decay=True)
        entropy_bonus = -self.current_epsilon * entropy.mean()

        # Total loss
        total_loss = actor_loss + entropy_bonus

        # Check for NaN loss
        if not torch.isfinite(total_loss):
            print(f"WARNING: Non-finite actor loss detected")
            return 0.0, entropy.mean().item()

        # Update
        self.actor_optimizer.zero_grad()
        total_loss.backward()

        # Check gradients
        grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.max_grad_norm
        )

        if not torch.isfinite(grad_norm):
            print(f"WARNING: Non-finite actor gradients detected")
            self.actor_optimizer.zero_grad()
            return 0.0, entropy.mean().item()

        self.actor_optimizer.step()

        return actor_loss.item(), entropy.mean().item()

    def save(self, path: str):
        """Save networks"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load networks"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


if __name__ == "__main__":
    print("Testing DistributionalRQE_PPO (single-agent)...")

    # Create config
    config = DistributionalRQEPPOConfig(
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
    algo = DistributionalRQE_PPO(config)

    print(f"Device: {algo.device}")
    print(f"Actor parameters: {sum(p.numel() for p in algo.actor.parameters())}")
    print(f"Critic parameters: {sum(p.numel() for p in algo.critic.parameters())}")

    # Test action selection
    obs = np.random.randn(config.obs_dim)
    action, log_prob, value = algo.select_action(obs)

    print(f"\nAction: {action}")
    print(f"Log prob: {log_prob:.4f}")
    print(f"Risk-adjusted value: {value:.4f}")

    # Test update
    T = 100
    buffer = {
        'observations': torch.randn(T, config.obs_dim),
        'actions': torch.randint(0, config.action_dim, (T,)),
        'rewards': torch.randn(T),
        'dones': torch.zeros(T),
        'log_probs_old': torch.randn(T)
    }

    print(f"\nTesting update...")
    metrics = algo.update(buffer)
    print(f"Metrics: {metrics}")

    print("\nAll tests passed! ✓")
