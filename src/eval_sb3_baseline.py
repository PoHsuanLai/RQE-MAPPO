"""
Evaluate trained Stable Baselines3 PPO model
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from src.envs.risky_cartpole import register_risky_envs


def main():
    checkpoint_dir = Path('/Users/pohsuanlai/Documents/rqe/checkpoints/single_agent')
    model_path = checkpoint_dir / 'agent_sb3_ppo_final.zip'

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    # Load model
    print("Loading Standard PPO model...")
    model = PPO.load(model_path)

    # Create environment
    register_risky_envs()
    env = gym.make('RiskyCartPole-medium-v0')

    # Evaluate
    print("Evaluating for 100 episodes...")
    returns = []
    lengths = []

    for episode in range(100):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

        returns.append(episode_reward)
        lengths.append(episode_length)

    env.close()

    # Print results
    print("\n" + "=" * 80)
    print("Standard PPO Evaluation Results")
    print("=" * 80)
    print(f"Mean return:     {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Min/Max return:  {np.min(returns):.2f} / {np.max(returns):.2f}")
    print(f"5th/95th %ile:   {np.percentile(returns, 5):.2f} / {np.percentile(returns, 95):.2f}")
    print(f"Mean length:     {np.mean(lengths):.1f}")
    print(f"Success rate:    {np.mean([l >= 500 for l in lengths])*100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
