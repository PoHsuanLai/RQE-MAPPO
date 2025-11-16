#!/usr/bin/env python3
"""
Train Standard PPO on PettingZoo Simple Spread Environment

This is the baseline - standard PPO without risk-sensitive modifications.
"""

import argparse
import sys
from pathlib import Path

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_spread_v3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Standard PPO on Simple Spread environment"
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
        help="Experiment name (default: PPO_SimpleSpread)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

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

    # Configure Standard PPO
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
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
            }
        )
        .multi_agent(
            policies=dummy_env.par_env.agents,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
    )

    # Experiment name
    exp_name = args.exp_name or f"PPO_SimpleSpread"

    # Run training
    print("="*70)
    print(f"Starting Standard PPO Training on Simple Spread")
    print("="*70)
    print(f"Environment: Simple Spread (PettingZoo MPE)")
    print(f"Number of agents: {args.num_agents}")
    print(f"Max cycles per episode: {args.max_cycles}")
    print(f"Entropy coefficient: {args.entropy_coeff}")
    print(f"Total timesteps: {args.stop_timesteps}")
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
