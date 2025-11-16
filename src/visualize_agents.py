"""
Visualize trained agents and save GIFs

This script loads trained models and creates animated GIFs showing agent behavior.
Works with both .pt (our old format) and .zip (SB3 format) checkpoints.
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
from PIL import Image

from src.envs.risky_cartpole import register_risky_envs
from stable_baselines3 import PPO


def record_episode_as_gif(env, model, output_path, max_steps=500, is_sb3=True):
    """
    Record a single episode and save as GIF

    Args:
        env: Gymnasium environment (must be created with render_mode='rgb_array')
        model: Trained model (SB3 or custom)
        output_path: Path to save GIF file
        max_steps: Maximum steps per episode
        is_sb3: Whether model is SB3 format (uses .predict()) or custom (uses .select_action())

    Returns:
        episode_reward: Total reward
        episode_length: Episode length
    """
    frames = []
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(max_steps):
        # Render current frame
        frame = env.render()
        frames.append(frame)

        # Select action
        if is_sb3:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action, _, _ = model.select_action(obs, deterministic=True)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_length += 1

        obs = next_obs

        if terminated or truncated:
            # Add a few more frames at the end
            for _ in range(10):
                frame = env.render()
                frames.append(frame)
            break

    # Convert frames to PIL Images and save as GIF
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=50,  # 50ms per frame = 20 fps
        loop=0
    )

    return episode_reward, episode_length


def main():
    # Register environments
    register_risky_envs()

    # Checkpoint directory
    checkpoint_dir = Path('/Users/pohsuanlai/Documents/rqe/checkpoints/single_agent')

    # Models to visualize
    models_to_viz = [
        # SB3 baseline
        {
            'name': 'Standard PPO (SB3)',
            'path': 'agent_sb3_ppo_final.zip',
            'gif_name': 'demo_sb3_ppo.gif',
            'is_sb3': True,
        },
        # Practical RQE
        {
            'name': 'Practical RQE (tau=0.3)',
            'path': 'agent_tau0.3_practical_sb3.zip',
            'gif_name': 'demo_tau0.3_practical.gif',
            'is_sb3': True,
        },
        # True RQE
        {
            'name': 'True RQE (tau=0.5)',
            'path': 'agent_tau0.5_true_sb3.zip',
            'gif_name': 'demo_tau0.5_true.gif',
            'is_sb3': True,
        },
        # Risk-neutral
        {
            'name': 'Risk-Neutral RQE (tau=1000)',
            'path': 'agent_tau1000.0_practical_sb3.zip',
            'gif_name': 'demo_tau1000_practical.gif',
            'is_sb3': True,
        },
    ]

    print("=" * 80)
    print("Visualizing Trained Agents")
    print("=" * 80)
    print()

    for model_info in models_to_viz:
        checkpoint_path = checkpoint_dir / model_info['path']

        if not checkpoint_path.exists():
            print(f"✗ Skipping {model_info['name']}: checkpoint not found")
            print(f"  Expected: {checkpoint_path}")
            print()
            continue

        print(f"{model_info['name']}")
        print("-" * 80)

        # Create environment with rendering
        env = gym.make('RiskyCartPole-medium-v0', render_mode='rgb_array')

        # Load model
        if model_info['is_sb3']:
            model = PPO.load(checkpoint_path)
            print(f"✓ Loaded SB3 model: {checkpoint_path.name}")
        else:
            # Custom model loading would go here
            print(f"✗ Custom model loading not implemented yet")
            continue

        # Record GIF
        gif_path = checkpoint_dir / model_info['gif_name']
        print(f"  Recording episode...")

        episode_reward, episode_length = record_episode_as_gif(
            env, model, gif_path, max_steps=500, is_sb3=model_info['is_sb3']
        )

        env.close()

        print(f"  ✓ Saved GIF: {gif_path.name}")
        print(f"  Episode reward: {episode_reward:.1f}")
        print(f"  Episode length: {episode_length}")
        print()

    print("=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
