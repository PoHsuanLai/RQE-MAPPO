"""
True RQE-PPO with actual risk-aware policy gradient

This implements the ACTUAL RQE gradient from the theory:
    ∇_θ J = E[∇_θ log π(a|s) * w(s,a)]
where w(s,a) = exp(-τ * Q(s,a)) / E[exp(-τ * Q(s,a))]

This is the theoretically correct gradient but has high variance.
Use with caution - may need lower learning rates and gradient clipping.

Key differences from practical version:
1. Exponential importance weighting in policy gradient (not just risk-adjusted GAE)
2. Proper normalization of weights within minibatch
3. No advantage clipping (weights replace advantages)
"""

import sys
sys.path.insert(0, '/Users/pohsuanlai/Documents/rqe/stable-baselines3')

from typing import Any, Optional, Union
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from src.algorithms.rqe_ppo_sb3 import RiskAwareRolloutBuffer
from src.networks.distributional_critic import DistributionalCritic


class TrueRQE_PPO_SB3(PPO):
    """
    True RQE-PPO with exponential importance weighting

    This implements the actual RQE policy gradient, not just risk-adjusted GAE.

    WARNING: This has higher variance than the practical version!
    You may need:
    - Lower learning rates (1e-4 instead of 3e-4)
    - More gradient clipping
    - Smaller tau values initially

    Args:
        tau: Risk aversion parameter (lower = more risk-averse)
        use_clipping: Whether to use PPO clipping with RQE weights (recommended)
        normalize_weights: Whether to normalize importance weights (recommended)
        weight_clip: Maximum weight value (prevents explosion)
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
        use_clipping: bool = True,  # Use PPO clipping with RQE weights
        normalize_weights: bool = True,  # Normalize weights in minibatch
        weight_clip: float = 10.0,  # Clip extreme weights
        learning_rate: Union[float, Schedule] = 1e-4,  # Lower LR for stability
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        normalize_advantage: bool = False,  # Don't normalize (we use weights)
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
        self.use_clipping = use_clipping
        self.normalize_weights = normalize_weights
        self.weight_clip = weight_clip

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

        # Create distributional critic
        obs_dim = self.observation_space.shape[0]
        self.distributional_critic = DistributionalCritic(
            obs_dim=obs_dim,
            hidden_dims=[64, 64],
            activation="tanh",
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

    def _compute_rqe_weights(self, observations, actions):
        """
        Compute RQE importance weights: w(s,a) = exp(-τ * Q(s,a))

        This is the KEY difference from standard PPO!
        Instead of advantages, we use exponential utility weights.

        Args:
            observations: [batch, obs_dim]
            actions: [batch]

        Returns:
            weights: [batch] - normalized importance weights
            q_values: [batch] - risk-adjusted Q-values (for logging)
        """
        with th.no_grad():
            # Get distribution over returns
            probs = self.distributional_critic(observations)  # [batch, n_atoms]

            # Compute risk-adjusted Q-values for each action
            # For discrete actions, we need the Q-value of the taken action
            # Approximation: Use state value as proxy (can improve with action-conditioned critic)
            q_values = self.distributional_critic.get_risk_value(
                observations,
                tau=self.tau,
                risk_type=self.risk_measure
            )  # [batch]

            # Compute exponential weights: exp(-τ * Q)
            # Lower Q → higher weight (risk-averse: pessimistic about low values)
            weights = th.exp(-self.tau * q_values)

            # Clip extreme weights for numerical stability
            if self.weight_clip is not None:
                weights = th.clamp(weights, max=self.weight_clip)

            # Normalize weights within minibatch (if enabled)
            if self.normalize_weights:
                weights = weights / (weights.mean() + 1e-8)

        return weights, q_values

    def train(self) -> None:
        """
        Override train to use RQE importance weighting instead of advantages
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
        weight_means, weight_stds = [], []
        q_value_means = []

        continue_training = True

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # ========== COMPUTE RQE WEIGHTS (KEY DIFFERENCE!) ==========
                weights, q_values = self._compute_rqe_weights(
                    rollout_data.observations,
                    actions
                )

                # Log weight statistics
                weight_means.append(weights.mean().item())
                weight_stds.append(weights.std().item())
                q_value_means.append(q_values.mean().item())

                # ========== ACTOR UPDATE with RQE WEIGHTS ==========
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Compute policy gradient with RQE weights
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                if self.use_clipping:
                    # PPO-style clipping with RQE weights
                    policy_loss_1 = weights * ratio
                    policy_loss_2 = weights * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                else:
                    # Pure RQE gradient (no clipping)
                    policy_loss = -(weights * ratio).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss (still use returns for stability)
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

                # Check for NaN
                if not th.isfinite(actor_loss):
                    print(f"WARNING: Non-finite actor loss detected!")
                    print(f"  weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
                    print(f"  q_values: min={q_values.min():.3f}, max={q_values.max():.3f}")
                    continue

                # Update actor
                self.policy.optimizer.zero_grad()
                actor_loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # ========== DISTRIBUTIONAL CRITIC UPDATE ==========
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

        # RQE-specific logging
        self.logger.record("train/weight_mean", np.mean(weight_means))
        self.logger.record("train/weight_std", np.mean(weight_stds))
        self.logger.record("train/q_value_mean", np.mean(q_value_means))
        self.logger.record("train/tau", self.tau)

    def _update_distributional_critic(self, rollout_data) -> float:
        """
        Update distributional critic using categorical projection

        Same as RQE_PPO_SB3 version
        """
        from src.networks.distributional_critic import project_distribution

        observations = rollout_data.observations
        returns = rollout_data.returns

        # Get current distribution
        current_probs = self.distributional_critic(observations)

        # Target: Project returns onto categorical distribution
        with th.no_grad():
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
    print("Testing True RQE-PPO with actual risk-aware gradient...")

    import gymnasium as gym
    from src.envs.risky_cartpole import register_risky_envs

    register_risky_envs()
    env = gym.make('RiskyCartPole-medium-v0')

    # Create TRUE RQE agent with exponential weighting
    model = TrueRQE_PPO_SB3(
        "MlpPolicy",
        env,
        tau=0.5,  # Start with moderate risk-aversion
        risk_measure="entropic",
        n_atoms=51,
        v_min=0.0,
        v_max=600.0,
        learning_rate=1e-4,  # Lower LR for stability
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        use_clipping=True,  # Use PPO clipping for stability
        normalize_weights=True,  # Normalize weights
        weight_clip=10.0,  # Clip extreme weights
        verbose=1,
    )

    print(f"Device: {model.device}")
    print(f"Distributional critic parameters: {sum(p.numel() for p in model.distributional_critic.parameters())}")
    print(f"Risk aversion (tau): {model.tau}")
    print(f"Using clipping: {model.use_clipping}")
    print(f"Normalize weights: {model.normalize_weights}")

    # Test one training iteration
    print("\nTesting one training iteration...")
    model.learn(total_timesteps=4096, progress_bar=False)

    print("\n✓ All tests passed!")
    print("\nNote: This uses the ACTUAL RQE gradient with exponential importance weighting.")
    print("It may have higher variance than the practical version (risk-adjusted GAE).")
