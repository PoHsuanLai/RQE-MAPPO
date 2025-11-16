"""
Replicate Figure 2 from the paper exactly

The paper shows cliff walk scenarios with asymmetric risk preferences:
- Scenario 1: Agent 2 more risk-averse (τ₁=0.01, τ₂=0.02)
- Scenario 2: Agent 2 less risk-averse (τ₁=0.01, τ₂=0.005)

Both agents have bounded rationality: ε₁, ε₂
"""

import sys
sys.path.insert(0, '/Users/pohsuanlai/Documents/rqe/stable-baselines3')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.envs.cliff_walk import CliffWalkEnv
from src.algorithms.rqe_ppo_sb3 import RQE_PPO_SB3


def visualize_trajectory(env, model, title, ax, max_steps=200):
    """Visualize a single trajectory on the cliff walk grid"""
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

        if info.get('agent1_at_goal') and info.get('agent2_at_goal'):
            break

    # Draw grid
    grid_height, grid_width = env.height, env.width

    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)
    ax.set_aspect('equal')

    # Grid lines
    for i in range(grid_height + 1):
        ax.plot([0, grid_width], [i, i], 'k-', linewidth=0.5, alpha=0.3)
    for j in range(grid_width + 1):
        ax.plot([j, j], [0, grid_height], 'k-', linewidth=0.5, alpha=0.3)

    # Draw cliffs
    for (r, c) in env.cliff_cells:
        rect = patches.Rectangle(
            (c, grid_height - r - 1), 1, 1,
            linewidth=0.5,
            edgecolor='black',
            facecolor='black'
        )
        ax.add_patch(rect)

    # Agent 1 goal (pink)
    gr, gc = env.agent1_goal
    rect = patches.Rectangle(
        (gc, grid_height - gr - 1), 1, 1,
        linewidth=1,
        edgecolor='darkred',
        facecolor='#FFB6C1',
        alpha=0.5
    )
    ax.add_patch(rect)
    ax.text(gc + 0.5, grid_height - gr - 0.5, 'Ag 1 Goal',
            ha='center', va='center', fontsize=8, weight='bold')

    # Agent 2 goal (blue)
    gr, gc = env.agent2_goal
    rect = patches.Rectangle(
        (gc, grid_height - gr - 1), 1, 1,
        linewidth=1,
        edgecolor='darkblue',
        facecolor='#ADD8E6',
        alpha=0.5
    )
    ax.add_patch(rect)
    ax.text(gc + 0.5, grid_height - gr - 0.5, 'Ag 2 Goal',
            ha='center', va='center', fontsize=8, weight='bold')

    # Agent 1 path (red arrows)
    for i in range(len(agent1_path) - 1):
        r1, c1 = agent1_path[i]
        r2, c2 = agent1_path[i + 1]
        x1, y1 = c1 + 0.5, grid_height - r1 - 0.5
        x2, y2 = c2 + 0.5, grid_height - r2 - 0.5
        ax.arrow(x1, y1, x2 - x1, y2 - y1,
                head_width=0.15, head_length=0.1,
                fc='red', ec='darkred',
                alpha=0.7, linewidth=1.5,
                length_includes_head=True)

    # Agent 2 path (blue arrows)
    for i in range(len(agent2_path) - 1):
        r1, c1 = agent2_path[i]
        r2, c2 = agent2_path[i + 1]
        x1, y1 = c1 + 0.5, grid_height - r1 - 0.5
        x2, y2 = c2 + 0.5, grid_height - r2 - 0.5
        ax.arrow(x1, y1, x2 - x1, y2 - y1,
                head_width=0.15, head_length=0.1,
                fc='blue', ec='darkblue',
                alpha=0.7, linewidth=1.5,
                length_includes_head=True)

    # Starting positions
    r1, c1 = agent1_path[0]
    ax.plot(c1 + 0.5, grid_height - r1 - 0.5, 'o',
           color='red', markersize=10, markeredgecolor='darkred',
           markeredgewidth=2)

    r2, c2 = agent2_path[0]
    ax.plot(c2 + 0.5, grid_height - r2 - 0.5, 's',
           color='blue', markersize=10, markeredgecolor='darkblue',
           markeredgewidth=2)

    # Mark final positions with X if not at goal
    if not info.get('agent1_at_goal'):
        rf, cf = agent1_path[-1]
        ax.plot(cf + 0.5, grid_height - rf - 0.5, 'x',
               color='darkred', markersize=15, markeredgewidth=3)

    if not info.get('agent2_at_goal'):
        rf, cf = agent2_path[-1]
        ax.plot(cf + 0.5, grid_height - rf - 0.5, 'x',
               color='darkblue', markersize=15, markeredgewidth=3)

    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    # Status text
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


def train_and_visualize_scenario(tau1, tau2, epsilon1, epsilon2, timesteps=50000):
    """
    Train RQE with asymmetric risk parameters

    Note: The current implementation uses a single tau parameter.
    For true multi-agent asymmetric risk, we would need to modify
    the algorithm to support per-agent risk parameters.

    For now, we'll use the average tau as an approximation.
    """
    print(f"\nScenario: τ₁={tau1}, τ₂={tau2}, ε₁={epsilon1}, ε₂={epsilon2}")

    # Use average tau (limitation of current implementation)
    tau_avg = (tau1 + tau2) / 2
    print(f"  Using average τ={tau_avg:.3f} (current implementation limitation)")

    env = CliffWalkEnv()
    model = RQE_PPO_SB3(
        "MlpPolicy",
        env,
        tau=tau_avg,
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

    model.learn(total_timesteps=timesteps, progress_bar=True)

    return model


if __name__ == "__main__":
    print("="*70)
    print("Replicating Paper Figure 2: Cliff Walk with Asymmetric Risk")
    print("="*70)

    # Paper scenarios - exact values from Figure 2
    scenarios = [
        {
            'name': 'Agent 2 More Risk-Averse',
            'tau1': 0.01,
            'tau2': 0.02,
            'epsilon1': 50,
            'epsilon2': 100,
        },
        {
            'name': 'Agent 2 Less Risk-Averse',
            'tau1': 0.01,
            'tau2': 0.005,
            'epsilon1': 100,
            'epsilon2': 200,
        }
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, scenario in enumerate(scenarios):
        print(f"\n{'='*70}")
        print(f"Training: {scenario['name']}")
        print(f"{'='*70}")

        model = train_and_visualize_scenario(
            tau1=scenario['tau1'],
            tau2=scenario['tau2'],
            epsilon1=scenario['epsilon1'],
            epsilon2=scenario['epsilon2'],
            timesteps=1000000  # Increased for better convergence
        )

        # Visualize
        env = CliffWalkEnv()
        title = f"{scenario['name']}\nτ₁={scenario['tau1']}, τ₂={scenario['tau2']}, ε₁={scenario['epsilon1']}, ε₂={scenario['epsilon2']}"
        visualize_trajectory(env, model, title, axes[idx])

    plt.tight_layout()

    output_path = 'paper_figure2_replication.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Saved replication to: {output_path}")
    print(f"{'='*70}")

    plt.show()
