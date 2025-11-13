"""
Evaluate trained RQE-PPO checkpoints

This script:
1. Loads trained models from checkpoints
2. Runs evaluation episodes
3. Computes statistics (mean, std, min/max returns, collision rates)
4. Compares different tau values
"""

import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

from src.algorithms.distributional_rqe_ppo import DistributionalRQE_PPO, DistributionalRQEPPOConfig
from src.envs.risky_cartpole import register_risky_envs


def evaluate_agent(env, agent, n_episodes=100, max_steps=500, render=False):
    """
    Evaluate an agent for multiple episodes

    Args:
        env: Gymnasium environment
        agent: Trained agent
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render (for visual inspection)

    Returns:
        stats: Dictionary with evaluation statistics
    """
    returns = []
    lengths = []
    successes = []  # Did not fail (for CartPole)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(max_steps):
            # Use deterministic policy for evaluation
            action, _, _ = agent.select_action(obs, deterministic=True)

            next_obs, reward, terminated, truncated, info = env.step(action)

            if render and episode == 0:  # Only render first episode
                env.render()

            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminated or truncated:
                break

        returns.append(episode_reward)
        lengths.append(episode_length)
        successes.append(1.0 if episode_length >= max_steps else 0.0)

    stats = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'percentile_5': np.percentile(returns, 5),
        'percentile_95': np.percentile(returns, 95),
        'mean_length': np.mean(lengths),
        'success_rate': np.mean(successes),
        'returns': returns,
        'lengths': lengths,
    }

    return stats


def load_checkpoint(checkpoint_path, env, tau):
    """Load a trained agent from checkpoint"""
    config = DistributionalRQEPPOConfig(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        tau=tau,
        n_atoms=51,
        v_min=-10.0,
        v_max=500.0,
    )

    agent = DistributionalRQE_PPO(config)

    # Load checkpoint (weights_only=False needed for custom classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])

    agent.actor.eval()
    agent.critic.eval()

    print(f"✓ Loaded checkpoint: {checkpoint_path.name}")
    print(f"  Training iteration: {checkpoint.get('iteration', 'N/A')}")

    return agent


def main():
    # Setup
    register_risky_envs()
    env = gym.make('RiskyCartPole-medium-v0')  # Use medium risk version

    checkpoint_dir = Path('/Users/pohsuanlai/Documents/rqe/checkpoints/single_agent')

    # Find final checkpoints for each tau
    tau_configs = {
        'Risk-Averse (τ=0.3)': ('agent_tau0.3_final.pt', 0.3),
        'Moderate (τ=1.0)': ('agent_tau1.0_final.pt', 1.0),
        'Risk-Neutral (τ=1000.0)': ('agent_tau1000.0_final.pt', 1000.0),
    }

    results = {}

    print("=" * 80)
    print("Evaluating Trained RQE-PPO Agents")
    print("=" * 80)
    print(f"Environment: {env.spec.id}")
    print(f"Evaluation episodes: 100")
    print()

    for name, (checkpoint_name, tau) in tau_configs.items():
        checkpoint_path = checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            print(f"✗ Checkpoint not found: {checkpoint_name}")
            continue

        print(f"\n{name}")
        print("-" * 40)

        # Load agent
        agent = load_checkpoint(checkpoint_path, env, tau)

        # Evaluate
        stats = evaluate_agent(env, agent, n_episodes=100, max_steps=500)
        results[name] = stats

        # Print statistics
        print(f"  Mean return:     {stats['mean_return']:.2f} ± {stats['std_return']:.2f}")
        print(f"  Min/Max return:  {stats['min_return']:.2f} / {stats['max_return']:.2f}")
        print(f"  5th/95th %ile:   {stats['percentile_5']:.2f} / {stats['percentile_95']:.2f}")
        print(f"  Mean length:     {stats['mean_length']:.1f}")
        print(f"  Success rate:    {stats['success_rate']*100:.1f}%")

    # Create comparison plots
    print("\n" + "=" * 80)
    print("Generating comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RQE-PPO: Risk-Averse vs Risk-Neutral Comparison', fontsize=16)

    # Plot 1: Return distributions
    ax = axes[0, 0]
    for name, stats in results.items():
        ax.hist(stats['returns'], bins=20, alpha=0.5, label=name)
    ax.set_xlabel('Episode Return')
    ax.set_ylabel('Frequency')
    ax.set_title('Return Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Box plot comparison
    ax = axes[0, 1]
    data = [stats['returns'] for stats in results.values()]
    labels = list(results.keys())
    bp = ax.boxplot(data, labels=[l.split('(')[0].strip() for l in labels])
    ax.set_ylabel('Episode Return')
    ax.set_title('Return Statistics')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Mean ± Std comparison
    ax = axes[1, 0]
    names = list(results.keys())
    means = [results[n]['mean_return'] for n in names]
    stds = [results[n]['std_return'] for n in names]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split('(')[0].strip() for n in names], rotation=45, ha='right')
    ax.set_ylabel('Mean Return')
    ax.set_title('Mean Return ± Std Dev')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Risk metrics comparison
    ax = axes[1, 1]
    metrics = {
        'Mean': [results[n]['mean_return'] for n in names],
        '5th %ile': [results[n]['percentile_5'] for n in names],
        'Min': [results[n]['min_return'] for n in names],
    }
    x = np.arange(len(names))
    width = 0.25
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax.bar(x + i*width, values, width, label=metric_name, alpha=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels([n.split('(')[0].strip() for n in names], rotation=45, ha='right')
    ax.set_ylabel('Return')
    ax.set_title('Risk Metrics Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = Path('/Users/pohsuanlai/Documents/rqe/checkpoints/single_agent/evaluation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Config':<25} {'Mean±Std':<20} {'5th %ile':<12} {'Success%':<10}")
    print("-" * 80)
    for name, stats in results.items():
        config_name = name.split('(')[0].strip()
        mean_std = f"{stats['mean_return']:.1f} ± {stats['std_return']:.1f}"
        percentile_5 = f"{stats['percentile_5']:.1f}"
        success = f"{stats['success_rate']*100:.1f}%"
        print(f"{config_name:<25} {mean_std:<20} {percentile_5:<12} {success:<10}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)

    env.close()
    plt.show()


if __name__ == "__main__":
    main()
