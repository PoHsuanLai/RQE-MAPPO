"""
Train HARL algorithms on CliffWalk environment.

HARL (Heterogeneous-Agent Reinforcement Learning) provides state-of-the-art
multi-agent algorithms with theoretical guarantees.

Available algorithms:
    On-policy (CTDE):
        - happo: Heterogeneous-Agent PPO (state-of-the-art)
        - hatrpo: Heterogeneous-Agent TRPO
        - haa2c: Heterogeneous-Agent A2C
        - mappo: Multi-Agent PPO (baseline)

    Off-policy (CTDE):
        - haddpg: Heterogeneous-Agent DDPG
        - hatd3: Heterogeneous-Agent TD3
        - hasac: Heterogeneous-Agent SAC
        - had3qn: Heterogeneous-Agent Dueling DQN (for discrete actions)
        - maddpg: Multi-Agent DDPG (baseline)
        - matd3: Multi-Agent TD3 (baseline)

Usage:
    # Train with HAPPO (recommended)
    python scripts/train_harl_cliffwalk.py --algo happo

    # Train with MAPPO
    python scripts/train_harl_cliffwalk.py --algo mappo

    # Train with HAD3QN (off-policy DQN)
    python scripts/train_harl_cliffwalk.py --algo had3qn
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CRITICAL: Patch HARL modules BEFORE importing any runners
# The runners use "from harl.utils.envs_tools import make_train_env" at import
# time, so we must patch BEFORE the import happens
# ============================================================================

# Import the modules we need to patch
import harl.utils.envs_tools as envs_tools
import harl.utils.configs_tools as configs_tools
from harl.envs.env_wrappers import ShareDummyVecEnv

# Store original functions
_original_make_train_env = envs_tools.make_train_env
_original_make_eval_env = envs_tools.make_eval_env
_original_get_num_agents = envs_tools.get_num_agents
_original_get_task_name = configs_tools.get_task_name

# Global env args - will be set before runner creation
_global_env_args = {}


def _create_cliffwalk_env(env_args, seed):
    """Create a single CliffWalk environment."""
    from src.envs.harl_cliff_walk import HARLCliffWalkEnv
    env = HARLCliffWalkEnv(env_args)
    env.seed(seed)
    return env


def patched_make_train_env(env_name, seed, n_threads, env_args_unused):
    """Patched make_train_env that handles cliffwalk."""
    if env_name == "cliffwalk":
        def get_env_fn(rank):
            def init_env():
                return _create_cliffwalk_env(_global_env_args, seed + rank * 1000)
            return init_env
        return ShareDummyVecEnv([get_env_fn(i) for i in range(n_threads)])
    return _original_make_train_env(env_name, seed, n_threads, env_args_unused)


def patched_make_eval_env(env_name, seed, n_threads, env_args_unused):
    """Patched make_eval_env that handles cliffwalk."""
    if env_name == "cliffwalk":
        def get_env_fn(rank):
            def init_env():
                return _create_cliffwalk_env(_global_env_args, seed + rank * 1000 + 10000)
            return init_env
        return ShareDummyVecEnv([get_env_fn(i) for i in range(n_threads)])
    return _original_make_eval_env(env_name, seed, n_threads, env_args_unused)


def patched_get_num_agents(env_name, env_args_unused, envs):
    """Patched get_num_agents that handles cliffwalk."""
    if env_name == "cliffwalk":
        return 2
    return _original_get_num_agents(env_name, env_args_unused, envs)


def patched_get_task_name(env_name, env_args_unused):
    """Patched get_task_name that handles cliffwalk."""
    if env_name == "cliffwalk":
        return "cliff_walk_6x6"
    return _original_get_task_name(env_name, env_args_unused)


# Apply patches to the modules
envs_tools.make_train_env = patched_make_train_env
envs_tools.make_eval_env = patched_make_eval_env
envs_tools.get_num_agents = patched_get_num_agents
configs_tools.get_task_name = patched_get_task_name

# Also patch the logger registry
from harl.envs import LOGGER_REGISTRY
from harl.envs.pettingzoo_mpe.pettingzoo_mpe_logger import PettingZooMPELogger
LOGGER_REGISTRY["cliffwalk"] = PettingZooMPELogger

# NOW import runners - they will use our patched functions
from harl.runners import RUNNER_REGISTRY

# ============================================================================
# End of patching
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train HARL on CliffWalk")

    # Algorithm
    parser.add_argument("--algo", type=str, default="happo",
                        choices=["happo", "hatrpo", "haa2c", "mappo",
                                 "haddpg", "hatd3", "hasac", "had3qn",
                                 "maddpg", "matd3"],
                        help="Algorithm to use")

    # Environment
    parser.add_argument("--horizon", type=int, default=100,
                        help="Episode length")
    parser.add_argument("--reward_scale", type=float, default=1.0,
                        help="Reward scaling factor")
    parser.add_argument("--corner_reward", type=float, default=0.0,
                        help="Bonus for reaching safe corners")
    parser.add_argument("--enable_collision", action="store_true",
                        help="Enable collision dynamics")

    # Training
    parser.add_argument("--n_rollout_threads", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--n_eval_threads", type=int, default=4,
                        help="Number of evaluation environments")
    parser.add_argument("--episode_length", type=int, default=100,
                        help="Episode length for rollouts")
    parser.add_argument("--num_env_steps", type=int, default=1000000,
                        help="Total environment steps")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")

    # Model
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 64],
                        help="Hidden layer sizes")
    parser.add_argument("--share_param", action="store_true",
                        help="Share parameters between agents")

    # Logging
    parser.add_argument("--exp_name", type=str, default="cliffwalk",
                        help="Experiment name")
    parser.add_argument("--log_dir", type=str, default="results/harl",
                        help="Directory for logs")
    parser.add_argument("--use_eval", action="store_true",
                        help="Enable evaluation during training")
    parser.add_argument("--eval_interval", type=int, default=25,
                        help="Evaluation interval (episodes)")

    # Device
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")

    return parser.parse_args()


def main():
    global _global_env_args

    args = parse_args()

    print("=" * 70)
    print(f"Training HARL {args.algo.upper()} on CliffWalk")
    print("=" * 70)

    # Set global env args (used by patched functions)
    _global_env_args = {
        "grid_size": (6, 6),
        "horizon": args.horizon,
        "reward_scale": args.reward_scale,
        "corner_reward": args.corner_reward,
        "enable_collision": args.enable_collision,
    }

    # HARL configuration
    main_args = {
        "algo": args.algo,
        "env": "cliffwalk",
        "exp_name": args.exp_name,
    }

    algo_args = {
        "seed": {"seed": args.seed, "seed_specify": True},
        "device": {"cuda": args.cuda, "cuda_deterministic": True, "torch_threads": 4},
        "train": {
            "n_rollout_threads": args.n_rollout_threads,
            "num_env_steps": args.num_env_steps,
            "episode_length": args.episode_length,
            "log_interval": 5,
            "eval_interval": args.eval_interval,
            "use_linear_lr_decay": False,
            "save_interval": 100,
        },
        "eval": {
            "use_eval": args.use_eval,
            "n_eval_rollout_threads": args.n_eval_threads,
            "eval_episodes": 10,
        },
        "render": {
            "use_render": False,
            "render_episodes": 5,
        },
        "model": {
            "hidden_sizes": args.hidden_sizes,
            "activation_id": 1,  # ReLU
            "use_feature_normalization": True,
            "use_recurrent_policy": False,
            "recurrent_n": 1,
            "use_naive_recurrent_policy": False,
            "data_chunk_length": 10,
            # Policy model required fields
            "gain": 0.01,
            "initialization_method": "orthogonal_",
            "std_x_coef": 1.0,
            "std_y_coef": 0.5,
        },
        "algo": {
            "gamma": args.gamma,
            "use_gae": True,
            "gae_lambda": 0.95,
            "use_proper_time_limits": True,
            "use_valuenorm": True,
            "use_max_grad_norm": True,
            "max_grad_norm": 10.0,
            "share_param": args.share_param,
            "fixed_order": False,
            "action_aggregation": "prod",
            "use_policy_active_masks": False,
            # PPO-specific
            "ppo_epoch": 10,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "num_mini_batch": 1,
            "entropy_coef": 0.01,
            "value_loss_coef": 1.0,
            "lr": args.lr,
            "critic_lr": args.lr,
            "opti_eps": 1e-5,
            "weight_decay": 0,
            # Huber loss for critic
            "use_huber_loss": True,
            "huber_delta": 10.0,
        },
        "logger": {
            "log_dir": args.log_dir,
        },
    }

    print(f"\nEnvironment Configuration:")
    print(f"  Grid size: (6, 6)")
    print(f"  Horizon: {args.horizon}")
    print(f"  Reward scale: {args.reward_scale}")
    print(f"  Corner reward: {args.corner_reward}")
    print(f"  Collision: {args.enable_collision}")

    print(f"\nTraining Configuration:")
    print(f"  Algorithm: {args.algo}")
    print(f"  Rollout threads: {args.n_rollout_threads}")
    print(f"  Episode length: {args.episode_length}")
    print(f"  Total steps: {args.num_env_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Share params: {args.share_param}")
    print(f"  Seed: {args.seed}")

    print(f"\nModel:")
    print(f"  Hidden sizes: {args.hidden_sizes}")

    print("=" * 70)

    # Create runner
    runner_class = RUNNER_REGISTRY[args.algo]

    try:
        runner = runner_class(main_args, algo_args, _global_env_args)
        print("\nStarting training...")
        runner.run()
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
