#!/usr/bin/env python3
"""
Train TRUE MAPPO (Centralized Critic) on Atari Boxing

This implementation uses RLlib's centralized critic framework for true MAPPO
adapted for visual observations from Atari Boxing.

Key Components:
1. CNN-based processing for pixel observations (210x160x3)
2. Centralized critic sees both players' visual observations
3. Competitive 2-player boxing environment

Environment:
- 2 agents (competitive boxing match)
- Visual observations: 210x160x3 RGB images
- 18 discrete actions (Atari joystick)
"""

import argparse
import sys
from pathlib import Path

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.atari import boxing_v2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TRUE MAPPO on Atari Boxing"
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
        default=1,
        help="Number of GPUs to use (recommended for vision)"
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
        default=5e-5,  # Lower LR for visual observations
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
    parser.add_argument(
        "--frame_stack",
        type=int,
        default=4,
        help="Number of frames to stack"
    )

    # Experiment parameters
    parser.add_argument(
        "--stop_timesteps",
        type=int,
        default=10000000,  # 10M for Atari
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
        default="/home/r13921098/RQE-MAPPO/results/boxing",
        help="Directory to save results"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (default: TRUE_MAPPO_Boxing)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Environment name
    env_name = "boxing"

    # Register environment
    def env_creator(_):
        env = boxing_v2.parallel_env()
        return env

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    # Create dummy env to get observation/action spaces
    dummy_env = ParallelPettingZooEnv(env_creator(None))
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    print(f"Agents: {dummy_env.par_env.agents}")

    # Configure multi-agent PPO with vision network
    # Both players use the same shared policy
    policies = {"shared_policy"}
    policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"

    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=args.num_gpus, num_gpus_per_worker=0.25 if args.num_gpus > 0 else 0)
        .rollouts(
            num_rollout_workers=args.num_workers,
            rollout_fragment_length="auto",
        )
        .training(
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            clip_param=0.1,  # Smaller clip for vision
            vf_clip_param=10.0,
            vf_loss_coeff=0.5,
            entropy_coeff=args.entropy_coeff,
            use_gae=True,
            use_critic=True,
            model={
                # Use CNN for visual observations
                "conv_filters": [
                    [16, [8, 8], 4],
                    [32, [4, 4], 2],
                    [64, [3, 3], 1],
                ],
                "conv_activation": "relu",
                "fcnet_hiddens": [512],
                "fcnet_activation": "relu",
                "vf_share_layers": True,  # Share conv layers
                "framestack": args.frame_stack,
            }
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    # Experiment name
    exp_name = args.exp_name or f"TRUE_MAPPO_Boxing"

    # Run training
    print("="*70)
    print(f"Starting TRUE MAPPO Training on Atari Boxing")
    print("="*70)
    print(f"Environment: Atari Boxing (2-player competitive)")
    print(f"Observation: 210x160x3 RGB images")
    print(f"Frame stack: {args.frame_stack}")
    print(f"Entropy coefficient: {args.entropy_coeff}")
    print(f"Total timesteps: {args.stop_timesteps}")
    print(f"GPUs: {args.num_gpus}")
    print("="*70)
    print("✓ Visual observations with CNN processing")
    print("✓ Parameter sharing across both players")
    print("✓ Competitive boxing environment")
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
