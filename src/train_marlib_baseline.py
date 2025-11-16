"""
Train standard MAPPO baseline using MARLlib on SUMO environments

This serves as a baseline to compare against RQE-MAPPO implementation.
Standard MAPPO is risk-neutral (no risk measures, just expected value).

Usage:
    python -m src.train_marlib_baseline --net single-intersection --total_timesteps 500000
"""

import argparse
import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from pathlib import Path
import json
from datetime import datetime

# MARLlib imports
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAPPO baseline using MARLlib on SUMO")

    # SUMO Environment
    parser.add_argument("--net", type=str, default="single-intersection",
                        choices=["single-intersection", "2way-single-intersection",
                                "2x2grid", "3x3grid", "4x4-Lucas", "4x4loop"],
                        help="SUMO network")
    parser.add_argument("--route_variant", type=str, default="vhvh",
                        help="Route variant (e.g., vhvh, ns, ew)")
    parser.add_argument("--num_seconds", type=int, default=1000,
                        help="Simulation seconds per episode")
    parser.add_argument("--delta_time", type=int, default=5,
                        help="Seconds between actions")
    parser.add_argument("--yellow_time", type=int, default=2,
                        help="Yellow phase duration")
    parser.add_argument("--min_green", type=int, default=5,
                        help="Minimum green phase duration")
    parser.add_argument("--max_green", type=int, default=50,
                        help="Maximum green phase duration")

    # Training parameters
    parser.add_argument("--total_timesteps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Number of GPUs to use")
    parser.add_argument("--train_batch_size", type=int, default=2048,
                        help="Training batch size")
    parser.add_argument("--sgd_minibatch_size", type=int, default=512,
                        help="SGD minibatch size")
    parser.add_argument("--num_sgd_iter", type=int, default=10,
                        help="Number of SGD iterations")

    # PPO parameters
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="PPO clip parameter")
    parser.add_argument("--entropy_coeff", type=float, default=0.01,
                        help="Entropy coefficient")

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs/marlib_baseline",
                        help="Log directory")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--checkpoint_freq", type=int, default=50,
                        help="Checkpoint frequency (iterations)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def create_sumo_env_creator(args):
    """
    Create environment creator function for Ray/MARLlib

    Returns a function that creates SUMO environment instances
    """
    def env_creator(config=None):
        """Create SUMO environment compatible with PettingZoo API"""
        import sumo_rl
        from pathlib import Path

        # Set SUMO_HOME if not set
        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = '/usr/share/sumo'

        # Get sumo_rl package path
        sumo_rl_path = Path(sumo_rl.__file__).parent
        nets_path = sumo_rl_path / 'nets'

        # Construct file paths
        if args.net == "4x4-Lucas":
            file_prefix = "4x4"
        else:
            file_prefix = args.net

        net_file = str(nets_path / args.net / f"{file_prefix}.net.xml")

        # Find route file
        route_file_variant = nets_path / args.net / f"{file_prefix}-{args.route_variant}.rou.xml"

        if route_file_variant.exists():
            route_file = str(route_file_variant)
        else:
            # Try common patterns
            possible_routes = [
                nets_path / args.net / f"{file_prefix}.rou.xml",
                nets_path / args.net / f"{file_prefix}c1.rou.xml",
                nets_path / args.net / f"{file_prefix}-vhvh.rou.xml",
            ]

            route_file = None
            for r in possible_routes:
                if r.exists():
                    route_file = str(r)
                    break

            if route_file is None:
                route_files = list((nets_path / args.net).glob("*.rou.xml"))
                if route_files:
                    route_file = str(route_files[0])
                else:
                    raise FileNotFoundError(f"No route file found in {nets_path / args.net}")

        # Create PettingZoo parallel environment
        env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            num_seconds=args.num_seconds,
            delta_time=args.delta_time,
            yellow_time=args.yellow_time,
            min_green=args.min_green,
            max_green=args.max_green,
            reward_fn='diff-waiting-time',
        )

        return env

    return env_creator


def train_marlib_mappo(args):
    """
    Train MAPPO using MARLlib
    """

    # Set SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        os.environ['SUMO_HOME'] = '/usr/share/sumo'

    # Initialize Ray
    ray.init(
        num_cpus=args.num_workers + 1,
        num_gpus=args.num_gpus,
        ignore_reinit_error=True,
    )

    print("\n" + "=" * 80)
    print("Training Standard MAPPO using MARLlib")
    print("=" * 80)
    print(f"Network: {args.net}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Workers: {args.num_workers}")
    print(f"Learning rate: {args.lr}")
    print(f"Entropy coefficient: {args.entropy_coeff}")
    print()

    # Create environment creator
    env_creator = create_sumo_env_creator(args)

    # Register environment with Ray
    env_name = f"sumo_{args.net}"
    register_env(env_name, env_creator)

    # Get a sample environment to extract specs
    sample_env = env_creator()
    obs_space = sample_env.observation_space(sample_env.possible_agents[0])
    act_space = sample_env.action_space(sample_env.possible_agents[0])
    num_agents = len(sample_env.possible_agents)

    print(f"Environment registered: {env_name}")
    print(f"Number of agents: {num_agents}")
    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")
    print()

    # Configure MAPPO using Ray RLlib directly (more control than MARLlib wrapper)
    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.gae_lambda,
            clip_param=args.clip_param,
            entropy_coeff=args.entropy_coeff,
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            vf_loss_coeff=0.5,
            vf_clip_param=args.clip_param,
        )
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=1,
        )
        .resources(
            num_gpus=args.num_gpus,
        )
        .multi_agent(
            policies={f"agent_{i}": (None, obs_space, act_space, {}) for i in range(num_agents)},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: f"agent_{agent_id.split('_')[-1]}" if '_' in agent_id else "agent_0",
        )
    )

    # Setup experiment name and logging
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"mappo_baseline_{args.net}_seed{args.seed}_{timestamp}"

    log_dir = Path(args.log_dir) / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = vars(args)
    with open(log_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Experiment: {args.exp_name}")
    print(f"Log directory: {log_dir}")
    print()

    # Train using Ray Tune
    print("Starting training...")
    print("=" * 80)

    num_iterations = args.total_timesteps // args.train_batch_size

    results = tune.run(
        "PPO",
        name=args.exp_name,
        config=config.to_dict(),
        stop={"training_iteration": num_iterations},
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        local_dir=str(log_dir.parent),
        verbose=1,
    )

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    # Get best checkpoint
    best_checkpoint = results.get_best_checkpoint(
        trial=results.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean",
        mode="max"
    )

    print(f"\nBest checkpoint: {best_checkpoint}")
    print(f"Results saved to: {log_dir}")

    # Cleanup
    ray.shutdown()
    sample_env.close()

    return results


def main():
    args = parse_args()
    results = train_marlib_mappo(args)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Network: {args.net}")
    print("\nTo compare against RQE-MAPPO, run:")
    print(f"  ./scripts/train_sumo.sh {args.net} 0.5 0.01")
    print()


if __name__ == "__main__":
    main()
