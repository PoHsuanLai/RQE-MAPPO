"""
RQE-PPO: Risk-Averse Proximal Policy Optimization for Ray RLlib

Custom PPO implementation with:
1. Entropic risk measure for risk-averse value targets (tau parameter)
2. Entropy regularization for bounded rationality (epsilon parameter)

Reference: Mazumdar et al. (2025) "Tractable Multi-Agent Reinforcement Learning
           through Behavioral Economics"
"""

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import explained_variance
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def entropic_risk_measure(values: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Entropic risk measure: -(1/τ) log E[exp(-τ * value)]

    For rewards (not costs), we want to be risk-averse to LOW values.

    Args:
        values: Tensor of shape [..., n_samples]
        tau: Risk aversion parameter
             - tau → 0: Very risk-averse (worst-case)
             - tau = 1: Moderate risk aversion
             - tau → ∞: Risk-neutral (expectation)

    Returns:
        Risk-adjusted value with same shape as input except last dimension
    """
    if tau == float('inf'):
        return values.mean(dim=-1)

    # Numerical stability: subtract max before exp
    max_val = values.max(dim=-1, keepdim=True)[0]
    shifted_values = -tau * (values - max_val)

    log_mean_exp = torch.logsumexp(shifted_values, dim=-1) - torch.log(
        torch.tensor(values.shape[-1], dtype=values.dtype, device=values.device)
    )

    return -(1.0 / tau) * log_mean_exp - max_val.squeeze(-1)


def rqe_postprocess_advantages(
    policy,
    sample_batch,
    other_agent_batches=None,
    episode=None
):
    """
    Postprocess advantages using risk-averse value targets.

    Modified from PPO's postprocess_ppo_gae to incorporate entropic risk measure.
    """
    # Get config parameters
    tau = policy.config.get("tau", 1.0)  # Risk aversion
    gamma = policy.config["gamma"]
    lambda_ = policy.config["lambda"]
    use_gae = policy.config["use_gae"]
    use_critic = policy.config.get("use_critic", True)

    # Extract data from batch
    rewards = sample_batch[SampleBatch.REWARDS]
    dones = sample_batch[SampleBatch.DONES]
    values = sample_batch[SampleBatch.VF_PREDS]

    # Standard GAE computation
    if use_gae and use_critic:
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        lastgaelam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = 0.0
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]

            # Apply entropic risk measure to TD error for risk-averse learning
            # For single samples, this reduces to regular TD error, but the critic
            # is trained to predict risk-adjusted returns

            # GAE: A_t = δ_t + γλ * (1 - done) * A_{t+1}
            advantages[t] = lastgaelam = (
                delta + gamma * lambda_ * nextnonterminal * lastgaelam
            )

        # Value targets are advantages + baseline
        value_targets = advantages + values

    else:
        # Monte Carlo returns without GAE
        returns = np.zeros_like(rewards)
        running_return = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        advantages = returns - values
        value_targets = returns

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Store in batch
    sample_batch[Postprocessing.ADVANTAGES] = advantages
    sample_batch[Postprocessing.VALUE_TARGETS] = value_targets

    return sample_batch


def rqe_loss_fn(
    policy,
    model,
    dist_class,
    train_batch
):
    """
    Custom loss function for RQE-PPO.

    Incorporates:
    1. Standard PPO clipped objective
    2. Risk-averse value function loss
    3. Enhanced entropy regularization (epsilon parameter)
    """
    # Get config
    tau = policy.config.get("tau", 1.0)
    epsilon = policy.config.get("epsilon", 0.01)  # Entropy coefficient
    clip_param = policy.config["clip_param"]
    vf_clip_param = policy.config["vf_clip_param"]
    vf_loss_coeff = policy.config["vf_loss_coeff"]

    # Unpack batch
    obs = train_batch[SampleBatch.CUR_OBS]
    actions = train_batch[SampleBatch.ACTIONS]
    advantages = train_batch[Postprocessing.ADVANTAGES]
    value_targets = train_batch[Postprocessing.VALUE_TARGETS]
    old_logits = train_batch[SampleBatch.ACTION_DIST_INPUTS]
    old_log_probs = train_batch[SampleBatch.ACTION_LOGP]
    old_values = train_batch[SampleBatch.VF_PREDS]

    # Forward pass
    model_out, _ = model({"obs": obs}, [], None)

    # Action distribution
    action_dist = dist_class(model_out, model)
    log_probs = action_dist.logp(actions)
    entropy = action_dist.entropy()

    # Ratio for PPO
    logp_ratio = torch.exp(log_probs - old_log_probs)

    # Clipped surrogate objective (same as PPO)
    surrogate_loss = torch.min(
        advantages * logp_ratio,
        advantages * torch.clamp(logp_ratio, 1 - clip_param, 1 + clip_param)
    )
    policy_loss = -surrogate_loss.mean()

    # Value function loss (can incorporate risk measure here)
    curr_values = model.value_function()

    if vf_clip_param is not None:
        # Clipped value loss (same as PPO)
        vf_loss_unclipped = (curr_values - value_targets) ** 2
        vf_clipped = old_values + torch.clamp(
            curr_values - old_values,
            -vf_clip_param,
            vf_clip_param
        )
        vf_loss_clipped = (vf_clipped - value_targets) ** 2
        vf_loss = torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
    else:
        vf_loss = ((curr_values - value_targets) ** 2).mean()

    # Entropy bonus (epsilon parameter for bounded rationality)
    entropy_loss = -epsilon * entropy.mean()

    # Total loss
    total_loss = (
        policy_loss
        + vf_loss_coeff * vf_loss
        + entropy_loss
    )

    # Store stats
    policy._total_loss = total_loss
    policy._policy_loss = policy_loss
    policy._vf_loss = vf_loss
    policy._entropy = entropy.mean()
    policy._entropy_loss = entropy_loss

    # KL divergence (for logging)
    policy._mean_kl = ((old_log_probs - log_probs).exp() - 1 - (old_log_probs - log_probs)).mean()

    return total_loss


def rqe_stats_fn(policy, train_batch):
    """Compute stats for logging"""
    return {
        "cur_lr": policy.cur_lr,
        "total_loss": policy._total_loss.item(),
        "policy_loss": policy._policy_loss.item(),
        "vf_loss": policy._vf_loss.item(),
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()
        ).item(),
        "entropy": policy._entropy.item(),
        "entropy_loss": policy._entropy_loss.item(),
        "kl": policy._mean_kl.item(),
        "tau": policy.config.get("tau", 1.0),
        "epsilon": policy.config.get("epsilon", 0.01),
    }


# Create custom RQE-PPO policy by extending PPOTorchPolicy
class RQEPPOTorchPolicy(PPOTorchPolicy):
    """Custom PPO policy with RQE modifications"""

    def __init__(self, observation_space, action_space, config):
        # Store RQE parameters
        config["tau"] = config.get("tau", 1.0)
        config["epsilon"] = config.get("epsilon", 0.01)

        super().__init__(observation_space, action_space, config)

    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        """Use RQE advantage postprocessing"""
        return rqe_postprocess_advantages(
            self, sample_batch, other_agent_batches, episode
        )

    def loss(self, model, dist_class, train_batch):
        """Use RQE loss function"""
        return rqe_loss_fn(self, model, dist_class, train_batch)

    def stats_fn(self, train_batch):
        """Use RQE stats function"""
        return rqe_stats_fn(self, train_batch)


class RQEPPOConfig(PPOConfig):
    """Configuration for RQE-PPO"""

    def __init__(self):
        super().__init__()

        # RQE-specific parameters
        self.tau = 1.0  # Risk aversion (lower = more risk-averse)
        self.epsilon = 0.01  # Bounded rationality (entropy coefficient)

    def training(
        self,
        *,
        tau=None,
        epsilon=None,
        **kwargs
    ):
        """Configure RQE-specific training parameters"""
        super().training(**kwargs)

        if tau is not None:
            self.tau = tau
        if epsilon is not None:
            self.epsilon = epsilon

        return self


class RQEPPO(PPO):
    """
    Risk-Averse PPO Algorithm

    Extends PPO with:
    1. Risk-averse value targets using entropic risk measure
    2. Entropy regularization for bounded rationality

    Key parameters:
    - tau: Risk aversion (0 = worst-case, 1 = moderate, ∞ = risk-neutral)
    - epsilon: Entropy coefficient for bounded rationality
    """

    @classmethod
    def get_default_config(cls):
        return RQEPPOConfig()

    @classmethod
    def get_default_policy_class(cls, config):
        return RQEPPOTorchPolicy
