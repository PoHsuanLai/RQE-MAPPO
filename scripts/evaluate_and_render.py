#!/usr/bin/env python3
"""
Evaluate and render trained models on Simple Spread environment

Creates a GIF visualization of the agents' behavior
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import imageio
from pettingzoo.mpe import simple_spread_v3

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and render trained models"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["mappo", "true_mappo", "rllib"],
        help="Type of model (mappo, true_mappo, or rllib)"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--output_gif",
        type=str,
        default="/home/r13921098/RQE-MAPPO/results/evaluation.gif",
        help="Output path for GIF"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for GIF"
    )
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=25,
        help="Maximum cycles per episode"
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=3,
        help="Number of agents"
    )

    return parser.parse_args()


def load_mappo_checkpoint(checkpoint_path, config):
    """Load standalone MAPPO checkpoint"""
    from algorithms.rqe_mappo import RQE_MAPPO

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create agents
    loaded_config = checkpoint['config']
    loaded_config.device = device
    agents = RQE_MAPPO(loaded_config)

    # Load weights
    for i, actor in enumerate(agents.actors):
        actor.load_state_dict(checkpoint['actors'][i])
        actor.to(device)
        actor.eval()

    return agents, loaded_config


def load_true_mappo_checkpoint(checkpoint_path, config):
    """Load standalone True MAPPO checkpoint"""
    from algorithms.true_rqe_mappo import TrueRQE_MAPPO

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create agents
    loaded_config = checkpoint['config']
    loaded_config.device = device
    agents = TrueRQE_MAPPO(loaded_config)

    # Load weights
    for i, actor in enumerate(agents.actors):
        actor.load_state_dict(checkpoint['actors'][i])
        actor.to(device)
        actor.eval()

    return agents, loaded_config


def load_rllib_checkpoint(checkpoint_path):
    """Load RLlib checkpoint"""
    import os
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.tune.registry import register_env
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    # Initialize Ray with proper runtime environment for custom algorithms
    project_root = str(Path(__file__).parent.parent)
    src_path = os.path.join(project_root, "src")

    # Check if Ray is already initialized
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": f"{src_path}:{os.environ.get('PYTHONPATH', '')}"
                },
                "py_modules": [src_path]
            },
            ignore_reinit_error=True
        )

    # Register the environment that was used during training
    def env_creator(config):
        env = simple_spread_v3.parallel_env(
            N=config.get("N", 3),
            max_cycles=config.get("max_cycles", 25),
            continuous_actions=False
        )
        return env

    register_env("simple_spread", lambda config: ParallelPettingZooEnv(env_creator(config)))

    # Load the algorithm from checkpoint
    algo = Algorithm.from_checkpoint(checkpoint_path)

    return algo


def evaluate_mappo(agents, env, num_episodes, max_cycles, record_frames=True):
    """Evaluate standalone MAPPO agents"""
    n_agents = len(agents.actors)
    episode_rewards = []
    all_frames = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = {i: 0 for i in range(n_agents)}
        frames = []

        for step in range(max_cycles):
            # Render frame
            if record_frames:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            # Get observations
            obs_list = [obs[f"agent_{i}"] for i in range(n_agents)]
            obs_tensor = torch.FloatTensor(np.array(obs_list)).unsqueeze(0).to(agents.device)

            # Get actions (deterministic for evaluation)
            with torch.no_grad():
                actions_batch, _, _ = agents.get_actions(obs_tensor, deterministic=True)

            # Convert to environment format
            action_dict = {
                f"agent_{i}": actions_batch[0, i].item()
                for i in range(n_agents)
            }

            # Step environment
            next_obs, reward_dict, done_dict, trunc_dict, _ = env.step(action_dict)

            # Track rewards
            for i in range(n_agents):
                episode_reward[i] += reward_dict[f"agent_{i}"]

            obs = next_obs

            if all(done_dict.values()) or all(trunc_dict.values()):
                break

        avg_reward = np.mean([episode_reward[i] for i in range(n_agents)])
        episode_rewards.append(avg_reward)

        if record_frames and ep == 0:  # Only save frames from first episode for GIF
            all_frames = frames

        print(f"Episode {ep+1}/{num_episodes} - Avg Reward: {avg_reward:.2f}")

    return episode_rewards, all_frames


def evaluate_rllib(algo, env, num_episodes, max_cycles, record_frames=True):
    """Evaluate RLlib algorithm"""
    episode_rewards = []
    all_frames = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        frames = []

        for step in range(max_cycles):
            # Render frame
            if record_frames:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            # Get actions for all agents
            action_dict = {}
            for agent_id in env.agents:
                action = algo.compute_single_action(
                    obs[agent_id],
                    policy_id=agent_id,
                    explore=False
                )
                action_dict[agent_id] = action

            # Step environment
            next_obs, reward_dict, done_dict, trunc_dict, _ = env.step(action_dict)

            # Track total reward
            episode_reward += sum(reward_dict.values())

            obs = next_obs

            if all(done_dict.values()) or all(trunc_dict.values()):
                break

        episode_rewards.append(episode_reward)

        if record_frames and ep == 0:  # Only save frames from first episode for GIF
            all_frames = frames

        print(f"Episode {ep+1}/{num_episodes} - Total Reward: {episode_reward:.2f}")

    return episode_rewards, all_frames


def save_gif(frames, output_path, fps=10):
    """Save frames as GIF"""
    if not frames:
        print("No frames to save!")
        return

    # Convert frames to PIL Images if needed
    images = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            images.append(Image.fromarray(frame))
        else:
            images.append(frame)

    # Save as GIF
    imageio.mimsave(output_path, images, fps=fps)
    print(f"\nGIF saved to: {output_path}")
    print(f"Total frames: {len(frames)}")


def main():
    args = parse_args()

    print("=" * 70)
    print("Evaluating and Rendering Trained Model")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Episodes: {args.num_episodes}")
    print("=" * 70)

    # Create environment with rendering
    env = simple_spread_v3.parallel_env(
        N=args.num_agents,
        max_cycles=args.max_cycles,
        continuous_actions=False,
        render_mode="rgb_array"  # Enable rendering
    )

    # Load model and evaluate based on type
    if args.model_type == "mappo":
        print("\nLoading RQE-MAPPO model...")
        agents, config = load_mappo_checkpoint(args.checkpoint, None)
        episode_rewards, frames = evaluate_mappo(
            agents, env, args.num_episodes, args.max_cycles
        )

    elif args.model_type == "true_mappo":
        print("\nLoading True RQE-MAPPO model...")
        agents, config = load_true_mappo_checkpoint(args.checkpoint, None)
        episode_rewards, frames = evaluate_mappo(
            agents, env, args.num_episodes, args.max_cycles
        )

    elif args.model_type == "rllib":
        print("\nLoading RLlib model...")
        algo = load_rllib_checkpoint(args.checkpoint)
        episode_rewards, frames = evaluate_rllib(
            algo, env, args.num_episodes, args.max_cycles
        )

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Print statistics
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")

    # Save GIF
    if frames:
        save_gif(frames, args.output_gif, fps=args.fps)
    else:
        print("\nWarning: No frames captured for GIF!")

    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
