"""
Generate visualization figures for presentation

Creates comparison plots between risk-averse and risk-neutral agents
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from src.algorithms.distributional_rqe_ppo import DistributionalRQE_PPO
from src.envs.risky_cartpole import register_risky_envs

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12


def load_agent(checkpoint_path):
    """Load trained agent from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    agent = DistributionalRQE_PPO(config)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])

    return agent, config


def visualize_trajectory(env, agent, ax, color, label, max_steps=100):
    """
    Visualize a single trajectory in state space

    For CartPole: plot (position, angle) trajectory
    """
    obs, _ = env.reset(seed=42)

    positions = []
    angles = []

    for step in range(max_steps):
        action, _, _ = agent.select_action(obs, deterministic=True)

        # Store state
        positions.append(obs[0])
        angles.append(obs[2])

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Plot trajectory
    ax.plot(positions, angles, color=color, linewidth=2, label=label, alpha=0.8)
    ax.scatter(positions[0], angles[0], color=color, s=100, marker='o', zorder=5, edgecolors='black')
    ax.scatter(positions[-1], angles[-1], color=color, s=100, marker='X', zorder=5, edgecolors='black')

    return len(positions)


def generate_trajectory_comparison():
    """
    Figure 1: Trajectory comparison showing safe vs risky paths
    """
    print("Generating trajectory comparison...")

    # Register environments
    register_risky_envs()
    env = gym.make('RiskyCartPole-medium-v0')

    # Load agents
    risk_averse_agent, _ = load_agent('checkpoints/single_agent/agent_tau0.3_final.pt')
    risk_neutral_agent, _ = load_agent('checkpoints/single_agent/agent_tau1000.0_final.pt')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot trajectories
    len_averse = visualize_trajectory(env, risk_averse_agent, ax, 'blue', 'Risk-Averse (τ=0.3)')
    len_neutral = visualize_trajectory(env, risk_neutral_agent, ax, 'red', 'Risk-Neutral (τ=∞)', max_steps=100)

    # Add danger zones (large angles)
    theta_threshold = env.unwrapped.theta_threshold_radians
    ax.axhspan(theta_threshold * 0.6, theta_threshold, alpha=0.2, color='red', label='Risky Region')
    ax.axhspan(-theta_threshold, -theta_threshold * 0.6, alpha=0.2, color='red')

    # Labels
    ax.set_xlabel('Cart Position', fontsize=14)
    ax.set_ylabel('Pole Angle (radians)', fontsize=14)
    ax.set_title('Trajectory Comparison: Risk-Averse vs Risk-Neutral', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add text annotations
    ax.text(0.02, 0.98, f'Risk-Averse: {len_averse} steps',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.2))
    ax.text(0.02, 0.90, f'Risk-Neutral: {len_neutral} steps',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))

    plt.tight_layout()

    # Save
    output_path = Path('proposal/figures/trajectory_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")

    env.close()


def generate_policy_visualization():
    """
    Figure 2: Policy visualization showing action probabilities
    """
    print("Generating policy visualization...")

    # Register environments
    register_risky_envs()
    env = gym.make('RiskyCartPole-medium-v0')

    # Load agents
    risk_averse_agent, _ = load_agent('checkpoints/single_agent/agent_tau0.3_final.pt')
    risk_neutral_agent, _ = load_agent('checkpoints/single_agent/agent_tau1000.0_final.pt')

    # Create state grid (angle vs angular velocity)
    angles = np.linspace(-0.2, 0.2, 30)
    ang_vels = np.linspace(-2, 2, 30)

    def get_policy_map(agent, angles, ang_vels):
        """Get action probabilities over state space"""
        action_probs = np.zeros((len(ang_vels), len(angles)))

        for i, ang_vel in enumerate(ang_vels):
            for j, angle in enumerate(angles):
                # Create observation [position, velocity, angle, angular_velocity]
                obs = np.array([0.0, 0.0, angle, ang_vel], dtype=np.float32)

                # Get action probabilities
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    logits = agent.actor(obs_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    # Probability of moving right (action 1)
                    action_probs[i, j] = probs[0, 1].item()

        return action_probs

    # Get policy maps
    policy_averse = get_policy_map(risk_averse_agent, angles, ang_vels)
    policy_neutral = get_policy_map(risk_neutral_agent, angles, ang_vels)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot risk-averse policy
    im1 = axes[0].imshow(policy_averse, extent=[angles[0], angles[-1], ang_vels[0], ang_vels[-1]],
                         origin='lower', cmap='RdBu', vmin=0, vmax=1, aspect='auto')
    axes[0].set_xlabel('Pole Angle (radians)', fontsize=12)
    axes[0].set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    axes[0].set_title('Risk-Averse Policy (τ=0.3)', fontsize=14, fontweight='bold')
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.colorbar(im1, ax=axes[0], label='P(Right)')

    # Plot risk-neutral policy
    im2 = axes[1].imshow(policy_neutral, extent=[angles[0], angles[-1], ang_vels[0], ang_vels[-1]],
                         origin='lower', cmap='RdBu', vmin=0, vmax=1, aspect='auto')
    axes[1].set_xlabel('Pole Angle (radians)', fontsize=12)
    axes[1].set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    axes[1].set_title('Risk-Neutral Policy (τ=∞)', fontsize=14, fontweight='bold')
    axes[1].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.colorbar(im2, ax=axes[1], label='P(Right)')

    plt.tight_layout()

    # Save
    output_path = Path('proposal/figures/policy_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")

    env.close()


def generate_distribution_comparison():
    """
    Figure 3: Return distribution comparison
    """
    print("Generating distribution comparison...")

    # Register environments
    register_risky_envs()
    env = gym.make('RiskyCartPole-medium-v0')

    # Load agents
    risk_averse_agent, _ = load_agent('checkpoints/single_agent/agent_tau0.3_final.pt')
    risk_neutral_agent, _ = load_agent('checkpoints/single_agent/agent_tau1000.0_final.pt')

    # Get initial state
    obs, _ = env.reset(seed=42)
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

    # Get distributions
    with torch.no_grad():
        # Risk-averse
        probs_averse = risk_averse_agent.critic(obs_tensor)[0].numpy()
        support = risk_averse_agent.critic.support.numpy()
        value_averse = risk_averse_agent.critic.get_risk_value(obs_tensor, tau=0.3, risk_type='entropic').item()
        expected_averse = risk_averse_agent.critic.get_expected_value(obs_tensor).item()

        # Risk-neutral
        probs_neutral = risk_neutral_agent.critic(obs_tensor)[0].numpy()
        value_neutral = risk_neutral_agent.critic.get_expected_value(obs_tensor).item()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot distributions
    axes[0].bar(support, probs_averse, width=0.8, alpha=0.7, color='blue', label='Risk-Averse')
    axes[0].axvline(expected_averse, color='blue', linestyle='--', linewidth=2, label=f'E[Z] = {expected_averse:.2f}')
    axes[0].axvline(value_averse, color='darkblue', linestyle='-', linewidth=2, label=f'ρ_τ(Z) = {value_averse:.2f}')
    axes[0].set_xlabel('Return Value', fontsize=12)
    axes[0].set_ylabel('Probability', fontsize=12)
    axes[0].set_title('Risk-Averse Distribution (τ=0.3)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(support, probs_neutral, width=0.8, alpha=0.7, color='red', label='Risk-Neutral')
    axes[1].axvline(value_neutral, color='darkred', linestyle='-', linewidth=2, label=f'E[Z] = {value_neutral:.2f}')
    axes[1].set_xlabel('Return Value', fontsize=12)
    axes[1].set_ylabel('Probability', fontsize=12)
    axes[1].set_title('Risk-Neutral Distribution (τ=∞)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path('proposal/figures/distribution_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")

    env.close()


def generate_simple_path_diagram():
    """
    Figure 4: Simple schematic showing safe vs risky path
    """
    print("Generating simple path diagram...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw grid
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')

    # Start and goal
    start = patches.Circle((0.5, 0.5), 0.2, color='green', alpha=0.7, zorder=3)
    goal = patches.Circle((3.5, 2.5), 0.2, color='gold', alpha=0.7, zorder=3)
    ax.add_patch(start)
    ax.add_patch(goal)

    # Obstacle (hazard)
    hazard = patches.Rectangle((1.5, 0.2), 1, 1.5, color='red', alpha=0.3, zorder=1)
    ax.add_patch(hazard)
    ax.text(2, 0.95, 'Hazard', ha='center', va='center', fontsize=12, fontweight='bold')

    # Risk-neutral path (optimal but closer to hazard)
    risky_x = [0.5, 1.2, 2.7, 3.5]
    risky_y = [0.5, 1.8, 2.2, 2.5]
    ax.plot(risky_x, risky_y, 'r--', linewidth=3, label='Risk-Neutral τ=∞', alpha=0.8)
    ax.arrow(2.5, 2.15, 0.3, 0.1, head_width=0.15, head_length=0.1, fc='red', ec='red', alpha=0.8)

    # Moderate risk-aversion path (middle ground)
    moderate_x = [0.5, 1.0, 2.0, 3.0, 3.5]
    moderate_y = [0.5, 2.0, 2.2, 2.4, 2.5]
    ax.plot(moderate_x, moderate_y, 'g-.', linewidth=3, label='Moderate τ=1.0', alpha=0.8)
    ax.arrow(2.8, 2.35, 0.3, 0.05, head_width=0.15, head_length=0.1, fc='green', ec='green', alpha=0.8)

    # Risk-averse path (safe, wide margin around hazard)
    safe_x = [0.5, 0.5, 3.5]
    safe_y = [0.5, 2.5, 2.5]
    ax.plot(safe_x, safe_y, 'b-', linewidth=3, label='Risk-Averse τ=0.3', alpha=0.8)
    # Arrow on the horizontal part of the blue line
    ax.arrow(1.5, 2.5, 0.5, 0, head_width=0.15, head_length=0.1, fc='blue', ec='blue', alpha=0.8)

    # Labels
    ax.text(0.5, 0.2, 'Start', ha='center', fontsize=11, fontweight='bold')
    ax.text(3.5, 2.8, 'Goal', ha='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Path Comparison: Safe vs Risky', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.axis('off')

    plt.tight_layout()

    # Save (only one version needed)
    output_path = Path('proposal/figures/path_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Presentation Figures")
    print("=" * 60)

    # Generate all figures
    try:
        generate_simple_path_diagram()
        print()

        generate_trajectory_comparison()
        print()

        generate_policy_visualization()
        print()

        generate_distribution_comparison()
        print()

        print("=" * 60)
        print("✓ All figures generated successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - proposal/figures/path_comparison.png")
        print("  - proposal/figures/trajectory_comparison.png")
        print("  - proposal/figures/policy_comparison.png")
        print("  - proposal/figures/distribution_comparison.png")

    except Exception as e:
        print(f"\n✗ Error generating figures: {e}")
        import traceback
        traceback.print_exc()
