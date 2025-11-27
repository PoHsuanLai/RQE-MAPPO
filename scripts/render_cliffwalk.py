#!/usr/bin/env python3
"""
Render trained Cliff Walk models and save as GIF

Supports RQE-PPO and True RQE-PPO (RLlib checkpoints)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import imageio
import os

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Import environment
from envs.cliff_walk import CliffWalkEnv

# RLlib imports
import ray
from ray.rllib.algorithms.algorithm import Algorithm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render trained Cliff Walk models as GIFs"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to RLlib checkpoint directory"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["rqe_ppo", "true_rqe_ppo"],
        help="Type of model"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to render"
    )
    parser.add_argument(
        "--output_gif",
        type=str,
        default="results/cliffwalk_render.gif",
        help="Output path for GIF"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for GIF"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum steps per episode"
    )

    return parser.parse_args()


def load_rllib_checkpoint(checkpoint_path):
    """Load RLlib checkpoint"""

    # Check if Ray is already initialized
    if not ray.is_initialized():
        # Initialize Ray with custom algorithms path
        src_path = str(Path(__file__).parent.parent / "src")
        sumo_baseline_path = str(Path(__file__).parent.parent / "sumo-rl" / "sumo_rl_baseline")

        ray.init(
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": f"{src_path}:{sumo_baseline_path}:{os.environ.get('PYTHONPATH', '')}"
                },
                "py_modules": [src_path, sumo_baseline_path]
            },
            ignore_reinit_error=True
        )

    # Load the algorithm from checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)

    return algo


def evaluate_cliffwalk(algo, num_episodes, max_steps):
    """Evaluate agent on Cliff Walk and collect frames"""

    # Create environment with rendering
    env = CliffWalkEnv(render_mode="rgb_array", return_joint_reward=True)

    episode_rewards = []
    episode_lengths = []
    all_frames = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        frames = []
        step = 0

        for step in range(max_steps):
            # Render frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            # Get action from policy
            action = algo.compute_single_action(obs, explore=False)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                # Add final frame
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

        if ep == 0:  # Save first episode frames for GIF
            all_frames = frames

        # Print episode info
        agent1_reached = info.get('agent1_at_goal', False)
        agent2_reached = info.get('agent2_at_goal', False)

        print(f"Episode {ep+1}/{num_episodes}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {step + 1}")
        print(f"  Agent 1 reached goal: {agent1_reached}")
        print(f"  Agent 2 reached goal: {agent2_reached}")
        print(f"  Agent 1 reward: {info.get('agent1_reward', 0):.2f}")
        print(f"  Agent 2 reward: {info.get('agent2_reward', 0):.2f}")

    return episode_rewards, episode_lengths, all_frames


def save_gif(frames, output_path, fps=4):
    """Save frames as GIF"""
    if not frames:
        print("No frames to save!")
        return

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert frames to uint8 if needed
    images = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            images.append(frame)
        else:
            images.append(frame)

    # Save as GIF
    imageio.mimsave(output_path, images, fps=fps)
    print(f"\nGIF saved to: {output_path}")
    print(f"Total frames: {len(frames)}")


def main():
    args = parse_args()

    print("=" * 70)
    print("Rendering Cliff Walk Episode")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Episodes: {args.num_episodes}")
    print("=" * 70)

    # Load model
    algo = load_rllib_checkpoint(args.checkpoint)

    # Evaluate and collect frames
    episode_rewards, episode_lengths, frames = evaluate_cliffwalk(
        algo, args.num_episodes, args.max_steps
    )

    # Print statistics
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")

    # Save GIF
    if frames:
        save_gif(frames, args.output_gif, fps=args.fps)
    else:
        print("\nWarning: No frames captured for GIF!")

    print("=" * 70)

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
