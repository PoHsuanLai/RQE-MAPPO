"""
RQE-PPO using Stable Baselines3 infrastructure

This wraps SB3's highly optimized PPO with our distributional critic
for risk-averse learning, keeping all the performance optimizations.

Key changes from standard PPO:
1. Replace scalar critic with distributional critic (51 atoms)
2. Compute risk-adjusted values using entropic risk measure
3. Use risk-adjusted values for GAE computation
"""

import sys
sys.path.insert(0, '/Users/pohsuanlai/Documents/rqe/stable-baselines3')

from typing import Any, ClassVar, Optional, Union
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.buffers import RolloutBuffer

from src.networks.distributional_critic import DistributionalCritic


class RiskAwareRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that computes risk-adjusted returns for GAE

    Only change: Use distributional critic to get risk-adjusted values
    """

    def __init__(self, *args, critic=None, tau=1.0, risk_measure="entropic", **kwargs):
        super().__init__(*args, **kwargs)
        self.critic = critic
        self.tau = tau
        self.risk_measure = risk_measure

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one for each env).
        """
        # Get risk-adjusted values for ALL observations in buffer
        # This is the KEY difference from standard PPO!
        if self.critic is not None:
            with th.no_grad():
                # Reshape: (buffer_size * n_envs, obs_dim)
                obs_flat = self.observations.reshape(-1, *self.observation_space.shape)

                # Convert to tensor if needed
                if isinstance(obs_flat, np.ndarray):
                    device = next(self.critic.parameters()).device
                    obs_flat = th.from_numpy(obs_flat).float().to(device)

                # Get risk-adjusted values
                risk_values = self.critic.get_risk_value(
                    obs_flat,
                    tau=self.tau,
                    risk_type=self.risk_measure
                )

                # Reshape back: (buffer_size, n_envs) and convert to numpy
                self.values = risk_values.reshape(self.buffer_size, self.n_envs).cpu().numpy()

        # Convert to numpy for GAE computation (same as standard)
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values


class RQE_PPO_SB3(PPO):
    """
    Risk-Averse PPO using SB3 infrastructure + distributional critic

    Args:
        tau: Risk aversion parameter (lower = more risk-averse)
        risk_measure: "entropic", "cvar", or "mean_variance"
        n_atoms: Number of atoms in distributional critic
        v_min: Minimum value support
        v_max: Maximum value support
        ... (all other PPO args)
    """

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        tau: float = 1.0,
        risk_measure: str = "entropic",
        n_atoms: int = 51,
        v_min: float = 0.0,
        v_max: float = 600.0,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        **kwargs
    ):
        # Store RQE parameters
        self.tau = tau
        self.risk_measure = risk_measure
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Initialize parent PPO
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rollout_buffer_class=RiskAwareRolloutBuffer,
            **kwargs
        )

        # Create distributional critic (replaces standard value network)
        obs_dim = self.observation_space.shape[0]
        self.distributional_critic = DistributionalCritic(
            obs_dim=obs_dim,
            hidden_dims=[64, 64],
            activation="tanh",  # Match SB3's default
            n_atoms=n_atoms,
            v_min=v_min,
            v_max=v_max
        ).to(self.device)

        # Create separate optimizer for distributional critic
        self.critic_optimizer = th.optim.Adam(
            self.distributional_critic.parameters(),
            lr=learning_rate
        )

        # Inject critic into rollout buffer
        self.rollout_buffer.critic = self.distributional_critic
        self.rollout_buffer.tau = tau
        self.rollout_buffer.risk_measure = risk_measure

    def train(self) -> None:
        """
        Override train to add distributional critic updates

        Keeps all SB3 optimizations, just adds one extra loss term
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        self.distributional_critic.train()

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        self._update_learning_rate(self.critic_optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses, critic_losses = [], [], []
        clip_fractions = []

        continue_training = True

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # ========== ACTOR UPDATE (same as PPO) ==========
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO clipped loss
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss (keep for stability, even though we use distributional critic)
                value_loss = F.mse_loss(rollout_data.returns, values)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Actor loss
                actor_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Update actor
                self.policy.optimizer.zero_grad()
                actor_loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # ========== DISTRIBUTIONAL CRITIC UPDATE (NEW!) ==========
                # This is where risk-aversion comes in!
                critic_loss = self._update_distributional_critic(rollout_data)
                critic_losses.append(critic_loss)

                # KL divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

            self._n_updates += 1
            if not continue_training:
                break

        # Logging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/clip_range", clip_range)

    def _update_distributional_critic(self, rollout_data) -> float:
        """
        Update distributional critic using categorical projection

        This is simplified - just trains on Monte Carlo returns
        (full distributional Bellman would be more complex in minibatch setting)
        """
        from src.networks.distributional_critic import project_distribution

        observations = rollout_data.observations
        returns = rollout_data.returns  # Monte Carlo returns from GAE

        # Get current distribution
        current_probs = self.distributional_critic(observations)

        # Target: Project returns onto categorical distribution
        with th.no_grad():
            # Convert scalar returns to target distribution (peaked at return value)
            target_probs = th.zeros_like(current_probs)
            batch_size = len(returns)

            for i in range(batch_size):
                # Find closest atom to return value
                ret = returns[i].item()
                ret = np.clip(ret, self.v_min, self.v_max)

                # Linear interpolation between atoms
                b = (ret - self.v_min) / self.distributional_critic.delta_z
                l = int(np.floor(b))
                u = int(np.ceil(b))

                # Handle edge cases
                l = np.clip(l, 0, self.n_atoms - 1)
                u = np.clip(u, 0, self.n_atoms - 1)

                # Distribute probability
                if l == u:
                    target_probs[i, l] = 1.0
                else:
                    prob_u = b - l
                    prob_l = 1.0 - prob_u
                    target_probs[i, l] = prob_l
                    target_probs[i, u] = prob_u

        # Cross-entropy loss
        loss = -(target_probs * th.log(current_probs + 1e-8)).sum(dim=-1).mean()

        # Update
        self.critic_optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(
            self.distributional_critic.parameters(),
            self.max_grad_norm
        )
        self.critic_optimizer.step()

        return loss.item()


if __name__ == "__main__":
    print("Testing RQE-PPO with SB3 infrastructure...")

    import gymnasium as gym
    from src.envs.risky_cartpole import register_risky_envs

    register_risky_envs()
    env = gym.make('RiskyCartPole-medium-v0')

    # Create risk-averse agent
    model = RQE_PPO_SB3(
        "MlpPolicy",
        env,
        tau=0.3,  # Risk-averse
        risk_measure="entropic",
        n_atoms=51,
        v_min=0.0,
        v_max=600.0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
    )

    print(f"Device: {model.device}")
    print(f"Distributional critic parameters: {sum(p.numel() for p in model.distributional_critic.parameters())}")

    # Test one training iteration
    print("\nTesting one training iteration...")
    model.learn(total_timesteps=4096, progress_bar=False)

    print("\n✓ All tests passed!")
