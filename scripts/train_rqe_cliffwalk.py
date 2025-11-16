#!/usr/bin/env python3
"""
Train RQE-PPO (Approximation) on Cliff Walk Environment

This uses the simpler approximation that applies risk measures during GAE computation
instead of using an action-conditioned distributional critic.
"""

import argparse
import sys
from pathlib import Path

import ray
from ray import tune
from ray.tune.registry import register_env
import gymnasium as gym

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.cliff_walk import CliffWalkEnv

# Import RQE-PPO approximation from sumo baseline directory
sys.path.insert(0, str(Path(__file__).parent.parent / "sumo-rl" / "sumo_rl_baseline"))
from rqe_ppo import RQEPPO, RQEPPOConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RQE-PPO (approximation) on Cliff Walk environment"
    )

    # RQE parameters
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Risk aversion parameter (lower = more risk-averse)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Bounded rationality (entropy coefficient)"
    )

    # Training parameters
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
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

    # Experiment parameters
    parser.add_argument(
        "--stop_timesteps",
        type=int,
        default=100000,
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
        default="/home/r13921098/RQE-MAPPO/results/cliffwalk",
        help="Directory to save results"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name (default: RQE_PPO_tau{tau}_eps{epsilon})"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Environment name
    env_name = "cliffwalk"

    # Register environment
    # Use explicit import path to avoid pickling issues with Ray workers
    def env_creator(_):
        import sys
        from pathlib import Path
        # Ensure src is in path for Ray workers
        src_path = str(Path(__file__).parent.parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        from envs.cliff_walk import CliffWalkEnv
        return CliffWalkEnv()

    register_env(env_name, env_creator)

    # Create dummy env to get observation/action spaces
    dummy_env = env_creator(None)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")

    # Configure RQE-PPO (Approximation)
    config = (
        RQEPPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=args.num_gpus)
        .rollouts(
            num_rollout_workers=args.num_workers,
            rollout_fragment_length=200,
        )
        .rl_module(_enable_rl_module_api=False)
        .training(_enable_learner_api=False)
        .training(
            # RQE parameters
            tau=args.tau,
            epsilon=args.epsilon,
            # PPO parameters
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            clip_param=0.2,
            vf_clip_param=10.0,
            vf_loss_coeff=0.5,
            use_gae=True,
            use_critic=True,
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
                "vf_share_layers": False,
            }
        )
    )

    # Experiment name
    exp_name = args.exp_name or f"RQE_PPO_tau{args.tau}_eps{args.epsilon}"

    # Run training
    print("="*70)
    print(f"Starting RQE-PPO (Approximation) Training on Cliff Walk")
    print("="*70)
    print(f"Environment: {env_name}")
    print(f"Risk aversion (tau): {args.tau}")
    print(f"Bounded rationality (epsilon): {args.epsilon}")
    print(f"Total timesteps: {args.stop_timesteps}")
    print(f"Implementation: Approximation (risk-aware GAE)")
    print("="*70)

    results = tune.run(
        RQEPPO,
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
