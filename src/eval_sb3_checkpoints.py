"""
Evaluate SB3-based RQE-PPO checkpoints and compare with baseline
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from stable_baselines3 import PPO
from src.envs.risky_cartpole import register_risky_envs


def evaluate_model(model, env, n_episodes=100, deterministic=True):
    """
    Evaluate a model for n episodes

    Returns:
        returns: List of episode returns
        lengths: List of episode lengths
    """
    returns = []
    lengths = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

        returns.append(episode_reward)
        lengths.append(episode_length)

    return returns, lengths


def print_statistics(name, returns, lengths):
    """Print evaluation statistics"""
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")
    print(f"Mean return:     {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Min/Max return:  {np.min(returns):.2f} / {np.max(returns):.2f}")
    print(f"5th/95th %ile:   {np.percentile(returns, 5):.2f} / {np.percentile(returns, 95):.2f}")
    print(f"Mean length:     {np.mean(lengths):.1f}")
    print(f"Success rate:    {np.mean([l >= 500 for l in lengths])*100:.1f}%")
    print(f"{'=' * 80}")


def record_episode_as_gif(env, model, output_path, max_steps=500):
    """Record a single episode and save as GIF"""
    frames = []
    obs, _ = env.reset()

    for step in range(max_steps):
        # Render frame
        frame = env.render()
        frames.append(frame)

        # Get action
        action, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Save as GIF
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=50,  # 50ms per frame = 20 fps
        loop=0
    )
    print(f"✓ Saved demonstration: {output_path}")


def main():
    checkpoint_dir = Path('/Users/pohsuanlai/Documents/rqe/checkpoints/single_agent')

    # Register environment
    register_risky_envs()
    env = gym.make('RiskyCartPole-medium-v0', render_mode='rgb_array')

    # Models to evaluate
    models = {
        'Standard PPO': checkpoint_dir / 'agent_sb3_ppo_final.zip',
        'RQE-PPO (τ=1000, Risk-Neutral)': checkpoint_dir / 'agent_tau1000.0_practical_sb3.zip',
        'RQE-PPO (τ=0.3, Risk-Averse)': checkpoint_dir / 'agent_tau0.3_practical_sb3.zip',
    }

    results = {}

    print("\n" + "=" * 80)
    print("EVALUATING MODELS (100 episodes each)")
    print("=" * 80)

    for name, path in models.items():
        if not path.exists():
            print(f"✗ Model not found: {path}")
            continue

        print(f"\nLoading: {name}")
        model = PPO.load(path)

        # Evaluate
        returns, lengths = evaluate_model(model, env, n_episodes=100)

        # Store results
        results[name] = {
            'returns': returns,
            'lengths': lengths,
            'mean': np.mean(returns),
            'std': np.std(returns),
            'percentile_5': np.percentile(returns, 5),
            'percentile_95': np.percentile(returns, 95),
            'success_rate': np.mean([l >= 500 for l in lengths]) * 100,
        }

        # Print statistics
        print_statistics(name, returns, lengths)

        # Record demo GIF
        gif_name = path.stem + '_demo.gif'
        gif_path = checkpoint_dir / gif_name
        record_episode_as_gif(env, model, gif_path, max_steps=500)

    env.close()

    # Create comparison plots
    if results:
        create_comparison_plots(results, checkpoint_dir)
        print_comparison_table(results)


def create_comparison_plots(results, output_dir):
    """Create bar chart comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison: Standard PPO vs RQE-PPO (Fixed Implementation)',
                 fontsize=16, fontweight='bold')

    methods = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Plot 1: Mean Return
    ax = axes[0, 0]
    means = [results[m]['mean'] for m in methods]
    stds = [results[m]['std'] for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Mean Return', fontsize=12)
    ax.set_title('Mean Return ± Std Dev', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Standard Deviation (Lower is Better)
    ax = axes[0, 1]
    stds = [results[m]['std'] for m in methods]
    bars = ax.bar(x, stds, alpha=0.7, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Standard Deviation', fontsize=12)
    ax.set_title('Return Variance (Lower = More Consistent)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, std in zip(bars, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{std:.1f}', ha='center', va='bottom', fontsize=10)

    # Plot 3: 5th Percentile (Worst Case)
    ax = axes[1, 0]
    percentiles = [results[m]['percentile_5'] for m in methods]
    bars = ax.bar(x, percentiles, alpha=0.7, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('5th Percentile Return', fontsize=12)
    ax.set_title('Worst-Case Performance (Higher = Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, p5 in zip(bars, percentiles):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{p5:.1f}', ha='center', va='bottom', fontsize=10)

    # Plot 4: Success Rate
    ax = axes[1, 1]
    success_rates = [results[m]['success_rate'] for m in methods]
    bars = ax.bar(x, success_rates, alpha=0.7, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate (Episode Length ≥ 500)', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')

    for bar, sr in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{sr:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / 'comparison_final_performance_fixed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_path}")
    plt.close()


def print_comparison_table(results):
    """Print comparison table"""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Method':<40} {'Mean±Std':<20} {'5th %ile':<12} {'Success%':<10}")
    print("-" * 80)

    for method, stats in results.items():
        mean_std = f"{stats['mean']:.1f} ± {stats['std']:.1f}"
        p5 = f"{stats['percentile_5']:.1f}"
        success = f"{stats['success_rate']:.1f}%"
        print(f"{method:<40} {mean_std:<20} {p5:<12} {success:<10}")

    print("=" * 80)


if __name__ == "__main__":
    main()
