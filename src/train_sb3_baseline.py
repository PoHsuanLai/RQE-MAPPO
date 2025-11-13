"""
Train standard PPO baseline using Stable Baselines3

This serves as a baseline to compare against our RQE-PPO implementation.
Standard PPO is risk-neutral (no risk measures, just expected value).
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.envs.risky_cartpole import register_risky_envs


class TrainingCallback(BaseCallback):
    """
    Callback for logging training progress
    """
    def __init__(self, eval_env, eval_freq=10, n_eval_episodes=10, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations = []
        self.iteration_returns = []
        self.iteration_lengths = []

    def _on_step(self) -> bool:
        # Check if it's time to evaluate
        if self.n_calls % (self.eval_freq * 2048) == 0:  # Every eval_freq rollouts
            iteration = self.n_calls // 2048

            # Evaluate
            episode_rewards = []
            episode_lengths = []

            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()  # Gym API returns (obs, info)
                terminated = False
                truncated = False
                episode_reward = 0
                episode_length = 0

                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)

            self.evaluations.append({
                'iteration': iteration,
                'mean_reward': mean_reward,
                'std_reward': np.std(episode_rewards),
                'mean_length': mean_length,
            })

            if self.verbose > 0:
                print(f"Iteration {iteration}: mean_reward={mean_reward:.2f}, mean_length={mean_length:.1f}")

        return True


def make_env():
    """Create and wrap environment"""
    register_risky_envs()
    env = gym.make('RiskyCartPole-medium-v0')
    env = Monitor(env)
    return env


def train_sb3_ppo(
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Entropy coefficient (standard value)
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=42,
):
    """
    Train standard PPO from Stable Baselines3

    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient (bounded rationality parameter)
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        seed: Random seed

    Returns:
        model: Trained PPO model
        callback: Training callback with evaluation history
    """
    # Create environments
    train_env = DummyVecEnv([make_env])
    eval_env = make_env()

    # Create model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,  # Standard entropy regularization
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        seed=seed,
        tensorboard_log=None,
        policy_kwargs=dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Match our architecture
        ),
    )

    # Create callback
    callback = TrainingCallback(
        eval_env=eval_env,
        eval_freq=10,  # Evaluate every 10 rollouts
        n_eval_episodes=10,
        verbose=1,
    )

    # Train
    print("=" * 80)
    print("Training Standard PPO (Stable Baselines3)")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"Environment: RiskyCartPole-medium-v0")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # Close environments
    eval_env.close()
    train_env.close()

    return model, callback


def plot_training_curve(callback, output_path):
    """Plot training curve from callback"""
    if not callback.evaluations:
        print("No evaluation data to plot")
        return

    iterations = [e['iteration'] for e in callback.evaluations]
    mean_rewards = [e['mean_reward'] for e in callback.evaluations]
    std_rewards = [e['std_reward'] for e in callback.evaluations]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mean_rewards, linewidth=2, label='Mean Return')
    plt.fill_between(
        iterations,
        np.array(mean_rewards) - np.array(std_rewards),
        np.array(mean_rewards) + np.array(std_rewards),
        alpha=0.3,
    )
    plt.xlabel('Iteration (×2048 steps)')
    plt.ylabel('Return')
    plt.title('Standard PPO Training Curve (Stable Baselines3)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curve: {output_path}")


def main():
    # Setup paths
    checkpoint_dir = Path('/Users/pohsuanlai/Documents/rqe/checkpoints/single_agent')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train
    model, callback = train_sb3_ppo(
        total_timesteps=1_000_000,  # 1M steps like our training
        seed=42,
    )

    # Save model
    model_path = checkpoint_dir / 'agent_sb3_ppo_final.zip'
    model.save(model_path)
    print(f"\n✓ Saved model: {model_path}")

    # Plot training curve
    plot_path = checkpoint_dir / 'training_curve_sb3_ppo.png'
    plot_training_curve(callback, plot_path)

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)

    register_risky_envs()
    eval_env = gym.make('RiskyCartPole-medium-v0')

    returns = []
    lengths = []

    for episode in range(10):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1

        returns.append(episode_reward)
        lengths.append(episode_length)

    print(f"Mean return:     {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Min/Max return:  {np.min(returns):.2f} / {np.max(returns):.2f}")
    print(f"5th/95th %ile:   {np.percentile(returns, 5):.2f} / {np.percentile(returns, 95):.2f}")
    print(f"Mean length:     {np.mean(lengths):.1f}")
    print(f"Success rate:    {np.mean([l >= 500 for l in lengths])*100:.1f}%")

    eval_env.close()

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
