"""
Train MAPPO baseline using Stable Baselines3

This implements Multi-Agent PPO by training independent PPO agents
(one per traffic light) with shared experiences.

Usage:
    python -m src.train_sb3_mappo --net single-intersection --total_timesteps 500000
"""

import argparse
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces

import sumo_rl


class SUMOMultiAgentEnv(gym.Env):
    """
    Wrapper to make SUMO multi-agent environment work with SB3

    Treats all agents as a single "super agent" with concatenated observations
    and actions. This allows us to use standard SB3 PPO.
    """

    def __init__(self, net='single-intersection', num_seconds=1000, delta_time=5,
                 use_gui=False, yellow_time=2, min_green=5, max_green=50):
        super().__init__()

        # Set SUMO_HOME
        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = '/usr/share/sumo'

        # Get network files
        import sumo_rl
        from pathlib import Path

        sumo_rl_path = Path(sumo_rl.__file__).parent
        nets_path = sumo_rl_path / 'nets'

        if net == "4x4-Lucas":
            file_prefix = "4x4"
        else:
            file_prefix = net

        net_file = str(nets_path / net / f"{file_prefix}.net.xml")

        # Find route file
        route_files = list((nets_path / net).glob(f"{file_prefix}*.rou.xml"))
        if not route_files:
            route_files = list((nets_path / net).glob("*.rou.xml"))
        route_file = str(route_files[0]) if route_files else None

        if route_file is None:
            raise FileNotFoundError(f"No route file found in {nets_path / net}")

        # Create SUMO environment
        from sumo_rl.environment.env import SumoEnvironment

        self.env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            num_seconds=num_seconds,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            reward_fn='diff-waiting-time',
        )

        # Get agent info
        obs = self.env.reset()
        self.agents = list(self.env.ts_ids)
        self.n_agents = len(self.agents)

        # Get dimensions
        first_agent = self.agents[0]
        single_obs_dim = len(obs[first_agent])
        single_action_dim = self.env.traffic_signals[first_agent].action_space.n

        # Define observation and action spaces (concatenated for all agents)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents * single_obs_dim,),
            dtype=np.float32
        )

        # Multi-Discrete action space (one action per agent)
        self.action_space = spaces.MultiDiscrete([single_action_dim] * self.n_agents)

        print(f"SUMO Multi-Agent Environment:")
        print(f"  Network: {net}")
        print(f"  Agents: {self.n_agents}")
        print(f"  Single agent obs dim: {single_obs_dim}")
        print(f"  Single agent action dim: {single_action_dim}")
        print(f"  Total obs dim: {self.n_agents * single_obs_dim}")
        print(f"  Total actions: {single_action_dim ** self.n_agents}")

    def reset(self, seed=None, options=None):
        """Reset environment"""
        obs_dict = self.env.reset()

        # Concatenate all agent observations
        obs_list = [obs_dict[agent] for agent in self.agents]
        concatenated_obs = np.concatenate(obs_list).astype(np.float32)

        return concatenated_obs, {}

    def step(self, actions):
        """Step environment with multi-agent actions"""
        # Convert concatenated actions to dict
        action_dict = {agent: int(actions[i]) for i, agent in enumerate(self.agents)}

        # Step environment
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)

        # Concatenate observations
        obs_list = [obs_dict[agent] for agent in self.agents]
        concatenated_obs = np.concatenate(obs_list).astype(np.float32)

        # Sum rewards (team reward)
        total_reward = sum(reward_dict.values())

        # Check if done
        done = all(done_dict.values())

        # Aggregate info
        info = {
            'episode': {
                'r': total_reward,
                'l': 1,
            }
        }

        return concatenated_obs, total_reward, done, False, info

    def close(self):
        """Close environment"""
        self.env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAPPO using Stable Baselines3")

    # SUMO Environment
    parser.add_argument("--net", type=str, default="single-intersection",
                        choices=["single-intersection", "2way-single-intersection",
                                "2x2grid", "3x3grid", "4x4-Lucas", "4x4loop"],
                        help="SUMO network")
    parser.add_argument("--num_seconds", type=int, default=1000,
                        help="Simulation seconds per episode")
    parser.add_argument("--delta_time", type=int, default=5,
                        help="Seconds between actions")
    parser.add_argument("--use_gui", action="store_true",
                        help="Use SUMO GUI")

    # Training parameters
    parser.add_argument("--total_timesteps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Steps per rollout")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of epochs")

    # PPO parameters
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient")

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs/sb3_mappo",
                        help="Log directory")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--save_freq", type=int, default=50000,
                        help="Save frequency")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto)")

    return parser.parse_args()


class LoggingCallback(BaseCallback):
    """Callback for logging training progress"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode info
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])

        # Print progress every 10k steps
        if self.n_calls % 10000 == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                print(f"[Step {self.n_calls}] Mean reward (last 100): {mean_reward:.2f}, "
                      f"Mean length: {mean_length:.1f}")

        return True


def main():
    args = parse_args()

    # Create log directory
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"sb3_mappo_{args.net}_seed{args.seed}_{timestamp}"

    log_dir = Path(args.log_dir) / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    import json
    with open(log_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "=" * 80)
    print("Training MAPPO using Stable Baselines3")
    print("=" * 80)
    print(f"Network: {args.net}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Log directory: {log_dir}")
    print()

    # Create environment
    def make_env():
        env = SUMOMultiAgentEnv(
            net=args.net,
            num_seconds=args.num_seconds,
            delta_time=args.delta_time,
            use_gui=args.use_gui,
        )
        env = Monitor(env)
        return env

    # Create vectorized environment
    env = DummyVecEnv([make_env])

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(log_dir / "tensorboard"),
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
    )

    print("Model created successfully!")
    print()

    # Create callback
    callback = LoggingCallback(verbose=1)

    # Train
    print("Starting training...")
    print("=" * 80)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # Save final model
    final_path = log_dir / "model_final.zip"
    model.save(str(final_path))
    print(f"\nModel saved: {final_path}")

    # Close environment
    env.close()

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"\nResults saved to: {log_dir}")
    print("\nTo compare with RQE-MAPPO, run:")
    print(f"  ./scripts/train_sumo.sh {args.net} 0.5 0.01")
    print()


if __name__ == "__main__":
    main()
