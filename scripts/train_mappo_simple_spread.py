#!/usr/bin/env python3
"""
Train TRUE MAPPO (Centralized Critic) on PettingZoo Simple Spread Environment

This implementation uses RLlib's centralized critic framework for true MAPPO:
- Centralized Training: Critic sees all agents' observations during training
- Decentralized Execution: Actors use only local observations at execution

Key Components:
1. CentralizedCriticModel: Critic V(o_1, o_2, ..., o_n) sees global state
2. Custom postprocessing: Collects other agents' observations
3. Custom loss: Uses centralized value function for advantage computation
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from gymnasium.spaces import Box, Discrete

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.tf_utils import explained_variance
from pettingzoo.mpe import simple_spread_v3

torch, nn = try_import_torch()

# Constants for storing other agents' observations
OTHER_AGENTS_OBS = "other_agents_obs"


class CentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model with centralized value function for MAPPO.

    The critic sees concatenated observations from all agents during training,
    while the actor only sees local observations during execution.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Actor network (uses only local observation)
        self.model = TorchFC(obs_space, action_space, num_outputs, model_config, name)

        # Centralized critic network
        # For Simple Spread: 3 agents * 18 obs_dim = 54 dimensional input
        num_agents = kwargs.get("num_agents", 3)
        single_obs_dim = obs_space.shape[0]
        central_vf_input_dim = num_agents * single_obs_dim

        self.central_vf = nn.Sequential(
            SlimFC(central_vf_input_dim, 256, activation_fn=nn.Tanh),
            SlimFC(256, 256, activation_fn=nn.Tanh),
            SlimFC(256, 1),
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """Forward pass for actor (uses only local obs)"""
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, all_agents_obs):
        """Centralized value function V(o_1, o_2, ..., o_n)"""
        return torch.reshape(self.central_vf(all_agents_obs), [-1])

    @override(TorchModelV2)
    def value_function(self):
        """Local value function (not used in centralized training)"""
        return self.model.value_function()


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


def centralized_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    """Collect other agents' observations and compute advantages with central VF."""

    if hasattr(policy, "compute_central_vf") and other_agent_batches is not None:
        # Collect observations from all agents
        if policy.config.get("enable_connectors", False):
            other_batches = [batch for _, _, batch in other_agent_batches.values()]
        else:
            other_batches = [batch for _, batch in other_agent_batches.values()]

        # Concatenate all agents' observations
        # Shape: [batch_size, num_agents * obs_dim]
        all_obs = [sample_batch[SampleBatch.CUR_OBS]]
        for other_batch in other_batches:
            all_obs.append(other_batch[SampleBatch.CUR_OBS])

        all_agents_obs = np.concatenate(all_obs, axis=1)
        sample_batch[OTHER_AGENTS_OBS] = all_agents_obs

        # Compute centralized value function predictions
        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(
                convert_to_torch_tensor(all_agents_obs, policy.device)
            )
            .cpu()
            .detach()
            .numpy()
        )
    else:
        # Policy hasn't been initialized yet, use zeros
        sample_batch[OTHER_AGENTS_OBS] = np.zeros(
            (len(sample_batch[SampleBatch.CUR_OBS]),
             sample_batch[SampleBatch.CUR_OBS].shape[1] * 3),  # 3 agents
            dtype=np.float32
        )
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32
        )

    # Compute advantages using GAE
    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch


def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    """Loss function that uses centralized value function."""
    # Save original value function
    vf_saved = model.value_function

    # Replace with central value function for loss computation
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[OTHER_AGENTS_OBS]
    )
    policy._central_value_out = model.value_function()

    # Compute PPO loss with centralized critic
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function
    model.value_function = vf_saved

    return loss


def central_vf_stats(policy, train_batch):
    """Report explained variance of the central value function."""
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out
        )
    }


def get_ccppo_policy(base):
    """Create a centralized critic PPO policy class."""
    class CCPPOTorchPolicy(CentralizedValueMixin, base):
        def __init__(self, observation_space, action_space, config):
            base.__init__(self, observation_space, action_space, config)
            CentralizedValueMixin.__init__(self)

        @override(base)
        def loss(self, model, dist_class, train_batch):
            return loss_with_central_critic(self, super(), model, dist_class, train_batch)

        @override(base)
        def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
            return centralized_critic_postprocessing(
                self, sample_batch, other_agent_batches, episode
            )

        @override(base)
        def stats_fn(self, train_batch):
            stats = super().stats_fn(train_batch)
            stats.update(central_vf_stats(self, train_batch))
            return stats

    return CCPPOTorchPolicy


# Create the centralized critic PPO policy
CCPPOTorchPolicy = get_ccppo_policy(PPOTorchPolicy)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Standard MAPPO on Simple Spread environment"
    )

    # Environment parameters
    parser.add_argument(
        "--num_agents",
        type=int,
        default=3,
        help="Number of agents (and landmarks)"
    )
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=25,
        help="Maximum cycles per episode"
    )

    # Training parameters
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4000,
        help="Training batch size"
    )
    parser.add_argument(
        "--sgd_minibatch_size",
        type=int,
        default=128,
        help="SGD minibatch size"
    )
    parser.add_argument(
        "--num_sgd_iter",
        type=int,
        default=10,
        help="Number of SGD iterations per training batch"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.95,
        dest="lambda_",
        help="GAE lambda"
    )
    parser.add_argument(
        "--entropy_coeff",
        type=float,
        default=0.01,
        help="Entropy coefficient"
    )

    # Experiment parameters
    parser.add_argument(
        "--stop_timesteps",
        type=int,
        default=1000000,
        help="Total timesteps to train"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=10,
        help="Checkpoint frequency (iterations)"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="/home/r13921098/RQE-MAPPO/results/simple_spread",
        help="Directory to save results"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (default: MAPPO_SimpleSpread)"
    )
    parser.add_argument(
        "--share_parameters",
        action="store_true",
        default=True,
        help="Share policy parameters across agents"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the centralized critic model
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

    # Environment name
    env_name = "simple_spread"

    # Register environment
    def env_creator(_):
        env = simple_spread_v3.parallel_env(
            N=args.num_agents,
            max_cycles=args.max_cycles,
            continuous_actions=False
        )
        return env

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    # Create dummy env to get observation/action spaces
    dummy_env = ParallelPettingZooEnv(env_creator(None))
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    print(f"Agents: {dummy_env.par_env.agents}")

    # Configure centralized critic policies
    # All agents use the same shared policy with centralized critic
    policies = {
        "shared_policy": (
            CCPPOTorchPolicy,
            obs_space,
            act_space,
            {
                "framework": "torch",
                "gamma": args.gamma,
                "lambda": args.lambda_,
                "use_gae": True,
                "model": {
                    "custom_model": "cc_model",
                    "custom_model_config": {
                        "num_agents": args.num_agents,
                    },
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "tanh",
                    "vf_share_layers": False,
                },
            }
        )
    }
    policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .rollouts(
            num_rollout_workers=args.num_workers,
            rollout_fragment_length=args.max_cycles,
        )
        .training(
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            clip_param=0.2,
            vf_clip_param=10.0,
            vf_loss_coeff=0.5,
            entropy_coeff=args.entropy_coeff,
            use_gae=True,
            use_critic=True,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    # Experiment name
    exp_name = args.exp_name or f"TRUE_MAPPO_SimpleSpread"

    # Run training
    print("="*70)
    print(f"Starting TRUE MAPPO (Centralized Critic) Training")
    print("="*70)
    print(f"Environment: Simple Spread (PettingZoo MPE)")
    print(f"Number of agents: {args.num_agents}")
    print(f"Max cycles per episode: {args.max_cycles}")
    print(f"Entropy coefficient: {args.entropy_coeff}")
    print(f"Total timesteps: {args.stop_timesteps}")
    print("="*70)
    print("✓ TRUE MAPPO: Using CENTRALIZED critic V(o_1, o_2, ..., o_n)")
    print("✓ Centralized Training: Critic sees all agents' observations")
    print("✓ Decentralized Execution: Actors use only local observations")
    print("✓ Parameter sharing across all agents")
    print("="*70)

    results = tune.run(
        PPO,
        name=exp_name,
        config=config.to_dict(),
        stop={"timesteps_total": args.stop_timesteps},
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        local_dir=args.local_dir,
        verbose=1,
    )

    print("="*70)
    print("Training completed!")
    print(f"Results saved to: {args.local_dir}/{exp_name}")
    print("="*70)

    # Get best checkpoint
    best_trial = results.get_best_trial(metric="episode_reward_mean", mode="max")
    if best_trial:
        print(f"Best checkpoint: {best_trial.checkpoint}")
        print(f"Best reward: {best_trial.last_result['episode_reward_mean']:.2f}")

    ray.shutdown()


if __name__ == "__main__":
    main()
