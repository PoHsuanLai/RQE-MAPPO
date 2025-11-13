"""
Train single-agent Distributional RQE-PPO

Test script for comparing risk-averse vs risk-neutral policies
on risky environments.
"""

import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from src.algorithms.distributional_rqe_ppo import DistributionalRQE_PPO, DistributionalRQEPPOConfig
from src.envs.risky_cartpole import register_risky_envs


def collect_rollout(env, agent, max_steps=500):
    """
    Collect a single episode rollout

    Returns:
        buffer: Dictionary with trajectory data
        episode_reward: Total reward
        episode_length: Episode length
    """
    observations = []
    actions = []
    rewards = []
    dones = []
    log_probs = []

    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(max_steps):
        # Select action
        action, log_prob, value = agent.select_action(obs, deterministic=False)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store transition
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(terminated))
        log_probs.append(log_prob)

        episode_reward += reward
        episode_length += 1

        obs = next_obs

        if terminated or truncated:
            break

    # Convert to tensors
    buffer = {
        'observations': torch.FloatTensor(np.array(observations)),
        'actions': torch.LongTensor(actions),
        'rewards': torch.FloatTensor(rewards),
        'dones': torch.FloatTensor(dones),
        'log_probs_old': torch.FloatTensor(log_probs)
    }

    return buffer, episode_reward, episode_length


def evaluate(env, agent, n_episodes=10, deterministic=True):
    """
    Evaluate agent performance

    Returns:
        mean_reward: Average episode reward
        std_reward: Std of episode rewards
        mean_length: Average episode length
    """
    rewards = []
    lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            action, _, _ = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                break

        rewards.append(episode_reward)
        lengths.append(episode_length)

    return np.mean(rewards), np.std(rewards), np.mean(lengths)


def train(
    env_name: str = "RiskyCartPole-medium-v0",
    tau: float = 1.0,
    epsilon: float = 0.01,
    risk_measure: str = "entropic",
    n_iterations: int = 500,
    steps_per_iteration: int = 2048,
    eval_interval: int = 50,
    save_dir: str = "checkpoints/single_agent",
    seed: int = 42
):
    """
    Train single-agent Distributional RQE-PPO

    Args:
        env_name: Gym environment name
        tau: Risk aversion (lower = more risk-averse)
        epsilon: Bounded rationality (entropy coefficient)
        risk_measure: "entropic", "cvar", "mean_variance"
        n_iterations: Number of training iterations
        steps_per_iteration: Steps per iteration
        eval_interval: Evaluate every N iterations
        save_dir: Directory to save checkpoints
        seed: Random seed
    """
    print("=" * 60)
    print("Training Distributional RQE-PPO (Single-Agent)")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Risk aversion (tau): {tau}")
    print(f"Bounded rationality (epsilon): {epsilon}")
    print(f"Risk measure: {risk_measure}")
    print(f"Seed: {seed}")
    print("=" * 60)

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Register custom environments
    register_risky_envs()

    # Create environment
    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agent config
    config = DistributionalRQEPPOConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        tau=tau,
        epsilon=epsilon,
        risk_measure=risk_measure,
        n_atoms=51,
        v_min=-50.0,  # Adjust based on environment
        v_max=50.0,
        hidden_dims=[64, 64],
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        n_minibatches=4
    )

    # Create agent
    agent = DistributionalRQE_PPO(config)

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Training metrics
    train_rewards = []
    eval_rewards = []
    eval_stds = []
    eval_lengths = []

    print("\nStarting training...")
    print("-" * 60)

    # Training loop
    for iteration in range(n_iterations):
        # Collect rollouts
        iteration_rewards = []
        iteration_lengths = []
        all_buffers = []

        steps_collected = 0
        while steps_collected < steps_per_iteration:
            buffer, episode_reward, episode_length = collect_rollout(env, agent)
            all_buffers.append(buffer)
            iteration_rewards.append(episode_reward)
            iteration_lengths.append(episode_length)
            steps_collected += episode_length

        # Combine buffers
        combined_buffer = {
            key: torch.cat([buf[key] for buf in all_buffers], dim=0)
            for key in all_buffers[0].keys()
        }

        # Update agent
        metrics = agent.update(combined_buffer)

        # Log training progress
        mean_reward = np.mean(iteration_rewards)
        mean_length = np.mean(iteration_lengths)
        train_rewards.append(mean_reward)

        print(f"Iter {iteration:4d} | "
              f"Reward: {mean_reward:7.2f} | "
              f"Length: {mean_length:6.1f} | "
              f"Actor Loss: {metrics['actor_loss']:7.4f} | "
              f"Critic Loss: {metrics['critic_loss']:7.4f} | "
              f"Entropy: {metrics['entropy']:6.4f}")

        # Evaluate periodically
        if (iteration + 1) % eval_interval == 0:
            eval_mean, eval_std, eval_len = evaluate(eval_env, agent, n_episodes=10)
            eval_rewards.append(eval_mean)
            eval_stds.append(eval_std)
            eval_lengths.append(eval_len)

            print("-" * 60)
            print(f"EVAL | Reward: {eval_mean:.2f} ± {eval_std:.2f} | Length: {eval_len:.1f}")
            print("-" * 60)

            # Save checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = save_path / f"agent_tau{tau}_iter{iteration+1}_{timestamp}.pt"
            agent.save(str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    final_mean, final_std, final_len = evaluate(eval_env, agent, n_episodes=20)
    print(f"Final evaluation (20 episodes):")
    print(f"  Reward: {final_mean:.2f} ± {final_std:.2f}")
    print(f"  Length: {final_len:.1f}")
    print("=" * 60)

    # Save final model
    final_path = save_path / f"agent_tau{tau}_final.pt"
    agent.save(str(final_path))
    print(f"Saved final model: {final_path}")

    # Plot training curve
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_rewards, alpha=0.3, label='Train')
    if eval_rewards:
        eval_x = np.arange(eval_interval - 1, len(train_rewards), eval_interval)
        plt.errorbar(eval_x, eval_rewards, yerr=eval_stds, label='Eval', capsize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title(f'Training Curve (tau={tau})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if eval_lengths:
        plt.plot(eval_x, eval_lengths, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Episode Length')
        plt.title(f'Episode Length (tau={tau})')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = save_path / f"training_curve_tau{tau}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved training curve: {plot_path}")

    env.close()
    eval_env.close()

    return agent, train_rewards, eval_rewards


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="RiskyCartPole-medium-v0")
    parser.add_argument("--tau", type=float, default=1.0, help="Risk aversion")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Bounded rationality")
    parser.add_argument("--risk_measure", type=str, default="entropic", choices=["entropic", "cvar", "mean_variance"])
    parser.add_argument("--n_iterations", type=int, default=500)
    parser.add_argument("--steps_per_iteration", type=int, default=2048)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints/single_agent")

    args = parser.parse_args()

    train(
        env_name=args.env,
        tau=args.tau,
        epsilon=args.epsilon,
        risk_measure=args.risk_measure,
        n_iterations=args.n_iterations,
        steps_per_iteration=args.steps_per_iteration,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        seed=args.seed
    )
