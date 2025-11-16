"""
Visualize Practical RQE vs True RQE on Cliff Walk

Creates visualizations matching the paper's style (Figure 2)
"""

import sys
sys.path.insert(0, '/Users/pohsuanlai/Documents/rqe/stable-baselines3')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.envs.cliff_walk import CliffWalkEnv
from src.algorithms.rqe_ppo_sb3 import RQE_PPO_SB3
from src.algorithms.true_rqe_ppo_sb3 import TrueRQE_PPO_SB3


def visualize_trajectory(env, model, model_name, tau, ax, max_steps=200):
    """
    Visualize a single trajectory on the cliff walk grid

    Color scheme matching paper:
    - Black: Cliff
    - Pink: Agent 1 goal
    - Blue: Agent 2 goal
    - Red arrows: Agent 1 path
    - Blue arrows: Agent 2 path
    """
    obs, _ = env.reset(seed=42)

    # Collect trajectory
    agent1_path = [env.agent1_pos.copy()]
    agent2_path = [env.agent2_pos.copy()]

    done = False
    step = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        agent1_path.append(env.agent1_pos.copy())
        agent2_path.append(env.agent2_pos.copy())

        done = terminated or truncated
        step += 1

        # Stop if both agents reached goals
        if info.get('agent1_at_goal') and info.get('agent2_at_goal'):
            break

    # Draw grid
    grid_height, grid_width = env.height, env.width

    # Set up the plot
    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)
    ax.set_aspect('equal')

    # Draw grid lines
    for i in range(grid_height + 1):
        ax.plot([0, grid_width], [i, i], 'k-', linewidth=0.5, alpha=0.3)
    for j in range(grid_width + 1):
        ax.plot([j, j], [0, grid_height], 'k-', linewidth=0.5, alpha=0.3)

    # Draw cliffs (black)
    for (r, c) in env.cliff_cells:
        rect = patches.Rectangle(
            (c, grid_height - r - 1), 1, 1,
            linewidth=0.5,
            edgecolor='black',
            facecolor='black'
        )
        ax.add_patch(rect)

    # Draw Agent 1 goal (pink/red)
    gr, gc = env.agent1_goal
    rect = patches.Rectangle(
        (gc, grid_height - gr - 1), 1, 1,
        linewidth=1,
        edgecolor='darkred',
        facecolor='#FFB6C1',  # Light pink
        alpha=0.5
    )
    ax.add_patch(rect)
    ax.text(gc + 0.5, grid_height - gr - 0.5, 'Ag 1 Goal',
            ha='center', va='center', fontsize=8, weight='bold')

    # Draw Agent 2 goal (blue)
    gr, gc = env.agent2_goal
    rect = patches.Rectangle(
        (gc, grid_height - gr - 1), 1, 1,
        linewidth=1,
        edgecolor='darkblue',
        facecolor='#ADD8E6',  # Light blue
        alpha=0.5
    )
    ax.add_patch(rect)
    ax.text(gc + 0.5, grid_height - gr - 0.5, 'Ag 2 Goal',
            ha='center', va='center', fontsize=8, weight='bold')

    # Draw Agent 1 path (red/orange arrows)
    for i in range(len(agent1_path) - 1):
        r1, c1 = agent1_path[i]
        r2, c2 = agent1_path[i + 1]

        # Convert to plot coordinates
        x1, y1 = c1 + 0.5, grid_height - r1 - 0.5
        x2, y2 = c2 + 0.5, grid_height - r2 - 0.5

        # Draw arrow
        ax.arrow(x1, y1, x2 - x1, y2 - y1,
                head_width=0.15, head_length=0.1,
                fc='red', ec='darkred',
                alpha=0.7, linewidth=1.5,
                length_includes_head=True)

    # Draw Agent 2 path (blue arrows)
    for i in range(len(agent2_path) - 1):
        r1, c1 = agent2_path[i]
        r2, c2 = agent2_path[i + 1]

        # Convert to plot coordinates
        x1, y1 = c1 + 0.5, grid_height - r1 - 0.5
        x2, y2 = c2 + 0.5, grid_height - r2 - 0.5

        # Draw arrow
        ax.arrow(x1, y1, x2 - x1, y2 - y1,
                head_width=0.15, head_length=0.1,
                fc='blue', ec='darkblue',
                alpha=0.7, linewidth=1.5,
                length_includes_head=True)

    # Mark starting positions
    r1, c1 = agent1_path[0]
    ax.plot(c1 + 0.5, grid_height - r1 - 0.5, 'o',
           color='red', markersize=10, markeredgecolor='darkred',
           markeredgewidth=2, label='Agent 1 start')

    r2, c2 = agent2_path[0]
    ax.plot(c2 + 0.5, grid_height - r2 - 0.5, 's',
           color='blue', markersize=10, markeredgecolor='darkblue',
           markeredgewidth=2, label='Agent 2 start')

    # Mark final positions with X if not at goal
    if not info.get('agent1_at_goal'):
        rf, cf = agent1_path[-1]
        ax.plot(cf + 0.5, grid_height - rf - 0.5, 'x',
               color='darkred', markersize=15, markeredgewidth=3)

    if not info.get('agent2_at_goal'):
        rf, cf = agent2_path[-1]
        ax.plot(cf + 0.5, grid_height - rf - 0.5, 'x',
               color='darkblue', markersize=15, markeredgewidth=3)

    # Title with parameters
    ax.set_title(f'{model_name}\nτ₁ = {tau:.2f}, ε₁ = {0.01:.2f}',
                fontsize=12, weight='bold')

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    # Add info text
    reached_goal = info.get('agent1_at_goal') or info.get('agent2_at_goal')
    hit_cliff = done and not reached_goal

    status_text = []
    if info.get('agent1_at_goal'):
        status_text.append('Agent 1: ✓ Goal')
    else:
        status_text.append('Agent 1: ✗')

    if info.get('agent2_at_goal'):
        status_text.append('Agent 2: ✓ Goal')
    else:
        status_text.append('Agent 2: ✗')

    status_text.append(f'Steps: {step}')

    ax.text(0.02, 0.02, '\n'.join(status_text),
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    return step, reached_goal, hit_cliff


def create_comparison_figure():
    """Create figure comparing Practical RQE vs True RQE"""

    print("Training models...")

    # Parameters
    tau = 0.5
    timesteps = 30000

    # Train Practical RQE
    print("\nTraining Practical RQE...")
    env_practical = CliffWalkEnv()
    practical_model = RQE_PPO_SB3(
        "MlpPolicy",
        env_practical,
        tau=tau,
        risk_measure="entropic",
        n_atoms=51,
        v_min=-10.0,
        v_max=50.0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
    )
    practical_model.learn(total_timesteps=timesteps, progress_bar=True)

    # Train True RQE
    print("\nTraining True RQE...")
    env_true = CliffWalkEnv()
    true_model = TrueRQE_PPO_SB3(
        "MlpPolicy",
        env_true,
        tau=tau,
        risk_measure="entropic",
        n_atoms=51,
        v_min=-10.0,
        v_max=50.0,
        learning_rate=1e-4,
        critic_learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        critic_epochs=5,
        gamma=0.99,
        verbose=0,
    )
    true_model.learn(total_timesteps=timesteps, progress_bar=True)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    print("\nGenerating visualizations...")

    # Visualize Practical RQE
    env1 = CliffWalkEnv()
    visualize_trajectory(env1, practical_model, "Practical RQE-PPO", tau, ax1)

    # Visualize True RQE
    env2 = CliffWalkEnv()
    visualize_trajectory(env2, true_model, "True RQE-PPO", tau, ax2)

    plt.tight_layout()

    # Save figure
    output_path = 'cliff_walk_comparison.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")

    plt.show()

    return fig


def create_multi_tau_comparison():
    """Compare different tau values for both methods"""

    tau_values = [0.01, 0.5, 1.0]  # Low (risk-averse), medium, high (risk-neutral)
    timesteps = 30000

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, tau in enumerate(tau_values):
        print(f"\n{'='*60}")
        print(f"Training with tau={tau}")
        print(f"{'='*60}")

        # Train Practical RQE
        print(f"\nPractical RQE (tau={tau})...")
        env_practical = CliffWalkEnv()
        practical_model = RQE_PPO_SB3(
            "MlpPolicy",
            env_practical,
            tau=tau,
            risk_measure="entropic",
            n_atoms=51,
            v_min=-10.0,
            v_max=50.0,
            learning_rate=3e-4,
            verbose=0,
        )
        practical_model.learn(total_timesteps=timesteps, progress_bar=False)

        # Train True RQE
        print(f"True RQE (tau={tau})...")
        env_true = CliffWalkEnv()
        true_model = TrueRQE_PPO_SB3(
            "MlpPolicy",
            env_true,
            tau=tau,
            risk_measure="entropic",
            n_atoms=51,
            v_min=-10.0,
            v_max=50.0,
            learning_rate=1e-4,
            critic_learning_rate=3e-4,
            critic_epochs=5,
            verbose=0,
        )
        true_model.learn(total_timesteps=timesteps, progress_bar=False)

        # Visualize
        env1 = CliffWalkEnv()
        visualize_trajectory(env1, practical_model, "Practical RQE", tau, axes[0, col])

        env2 = CliffWalkEnv()
        visualize_trajectory(env2, true_model, "True RQE", tau, axes[1, col])

    plt.tight_layout()

    output_path = 'cliff_walk_multi_tau.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved multi-tau comparison to: {output_path}")

    plt.show()

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'multi'],
                       help='Single comparison or multi-tau comparison')
    args = parser.parse_args()

    if args.mode == 'single':
        create_comparison_figure()
    else:
        create_multi_tau_comparison()
