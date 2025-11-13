"""
Visualization script for RQE-MAPPO training results

Usage:
    python -m src.visualize --exp_name traffic_merge_tau0.1_eps0.01
    python -m src.visualize --exp_name traffic_merge_tau0.1_eps0.01 --render
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns

from src.algorithms import RQE_MAPPO, RQEConfig
from src.envs import TrafficGridEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RQE-MAPPO results")

    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name (directory in logs/)")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--render", action="store_true",
                        help="Render episodes with trained policy")
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Number of episodes to render/evaluate")
    parser.add_argument("--save_fig", action="store_true",
                        help="Save figures instead of showing")

    return parser.parse_args()


def load_metrics(exp_dir: Path) -> Dict[str, List]:
    """Load metrics from JSON files"""
    metrics = {}

    metrics_file = exp_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            # Load entire JSON array
            data_list = json.load(f)

            # Convert to dict of lists
            for data in data_list:
                for key, value in data.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)

    return metrics


def plot_training_curves(metrics: Dict[str, List], save_path: Optional[Path] = None):
    """Plot training curves"""

    # Set style
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("RQE-MAPPO Training Results", fontsize=16, fontweight='bold')

    # Helper function to plot metric
    def plot_metric(ax, key, ylabel, color='blue', window=10):
        if key in metrics and len(metrics[key]) > 0:
            data = np.array(metrics[key])
            steps = np.arange(len(data))

            # Plot raw data
            ax.plot(steps, data, alpha=0.3, color=color, linewidth=0.5)

            # Plot smoothed data
            if len(data) >= window:
                smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
                smooth_steps = steps[:len(smoothed)]
                ax.plot(smooth_steps, smoothed, color=color, linewidth=2, label='Smoothed')

            ax.set_xlabel('Training Steps')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.grid(True, alpha=0.3)
            if len(data) >= window:
                ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for {key}',
                   ha='center', va='center', transform=ax.transAxes)

    # Plot each metric (try with and without train/ prefix)
    plot_metric(axes[0, 0], 'episode_reward', 'Episode Reward', 'blue')
    plot_metric(axes[0, 1], 'collision_rate', 'Collision Rate', 'red')
    plot_metric(axes[0, 2], 'actor_loss', 'Actor Loss', 'green')
    plot_metric(axes[1, 0], 'critic_loss', 'Critic Loss', 'orange')
    plot_metric(axes[1, 1], 'entropy', 'Policy Entropy', 'purple')
    plot_metric(axes[1, 2], 'total_goals_reached', 'Goals Reached (Cumulative)', 'cyan')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_evaluation_results(eval_metrics: Dict[str, np.ndarray],
                            save_path: Optional[Path] = None):
    """Plot evaluation results"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Evaluation Results", fontsize=16, fontweight='bold')

    # Episode rewards
    if 'rewards' in eval_metrics:
        rewards = eval_metrics['rewards']
        axes[0].bar(range(len(rewards)), rewards, color='blue', alpha=0.7)
        axes[0].axhline(rewards.mean(), color='red', linestyle='--',
                       label=f'Mean: {rewards.mean():.2f}')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Episode Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Collisions
    if 'collisions' in eval_metrics:
        collisions = eval_metrics['collisions']
        axes[1].bar(range(len(collisions)), collisions, color='red', alpha=0.7)
        axes[1].axhline(collisions.mean(), color='darkred', linestyle='--',
                       label=f'Mean: {collisions.mean():.2f}')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Number of Collisions')
        axes[1].set_title('Collisions per Episode')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Goals reached
    if 'goals_reached' in eval_metrics:
        goals = eval_metrics['goals_reached']
        axes[2].bar(range(len(goals)), goals, color='green', alpha=0.7)
        axes[2].axhline(goals.mean(), color='darkgreen', linestyle='--',
                       label=f'Mean: {goals.mean():.2f}')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Number of Goals Reached')
        axes[2].set_title('Goals Reached per Episode')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def render_episode(agent: RQE_MAPPO, env: TrafficGridEnv, episode_num: int = 0):
    """Render a single episode with the trained agent"""

    print(f"\n{'='*70}")
    print(f"Rendering Episode {episode_num + 1}")
    print(f"{'='*70}\n")

    obs = env.reset()
    episode_reward = torch.zeros(env.n_vehicles, device=obs.device)
    done = False
    step = 0

    trajectory = {
        'positions': [],
        'velocities': [],
        'rewards': [],
        'actions': []
    }

    while not done:
        with torch.no_grad():
            actions, _, _ = agent.get_actions(obs.unsqueeze(0), deterministic=True)
            actions = actions.squeeze(0)

        # Store trajectory
        # Extract positions and velocities from vehicle states
        positions = np.array([[v.x, v.y] for v in env.vehicles])
        velocities = np.array([[v.vx, v.vy] for v in env.vehicles])
        trajectory['positions'].append(positions.copy())
        trajectory['velocities'].append(velocities.copy())
        trajectory['actions'].append(actions.cpu().numpy().copy())

        obs, rewards, done, info = env.step(actions)
        episode_reward += rewards
        trajectory['rewards'].append(rewards.cpu().numpy().copy())

        step += 1

        # Print step info
        print(f"Step {step:3d} | Reward: {rewards.mean().item():7.3f} | "
              f"Positions: {positions}")

    print(f"\n{'='*70}")
    print(f"Episode Summary:")
    print(f"  Total Reward: {episode_reward.mean().item():.2f}")
    print(f"  Episode Length: {step}")
    print(f"  Collisions: {info['collisions']}")
    print(f"  Goals Reached: {info['goals_reached']}")
    print(f"{'='*70}\n")

    return trajectory, episode_reward.mean().item(), info


def plot_trajectory(trajectory: Dict, info: Dict, save_path: Optional[Path] = None):
    """Plot vehicle trajectories"""

    positions = np.array(trajectory['positions'])  # (steps, n_vehicles, 2)
    n_steps, n_vehicles, _ = positions.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Trajectory plot
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_vehicles))

    for i in range(n_vehicles):
        traj = positions[:, i, :]
        ax.plot(traj[:, 0], traj[:, 1], '-o', color=colors[i],
               label=f'Vehicle {i}', markersize=2, linewidth=1.5)
        # Start position
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=colors[i],
               markersize=10, markeredgecolor='black', markeredgewidth=2)
        # End position
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=colors[i],
               markersize=10, markeredgecolor='black', markeredgewidth=2)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Vehicle Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Reward over time
    ax = axes[1]
    rewards = np.array(trajectory['rewards'])  # (steps, n_vehicles)
    for i in range(n_vehicles):
        ax.plot(rewards[:, i], color=colors[i], label=f'Vehicle {i}', linewidth=1.5)

    cumulative = rewards.mean(axis=1).cumsum()
    ax2 = ax.twinx()
    ax2.plot(cumulative, 'k--', label='Cumulative', linewidth=2, alpha=0.5)
    ax2.set_ylabel('Cumulative Reward', fontsize=10)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Reward')
    ax.set_title(f'Rewards (Collisions: {info["collisions"]}, Goals: {info["goals_reached"]})')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()


def evaluate_agent(agent: RQE_MAPPO, env: TrafficGridEnv, n_episodes: int = 10) -> Dict:
    """Evaluate agent and return metrics"""

    print(f"\nEvaluating agent over {n_episodes} episodes...")

    episode_rewards = []
    episode_collisions = []
    episode_goals = []

    for i in range(n_episodes):
        obs = env.reset()
        episode_reward = torch.zeros(env.n_vehicles, device=obs.device)
        done = False

        while not done:
            with torch.no_grad():
                actions, _, _ = agent.get_actions(obs.unsqueeze(0), deterministic=True)
                actions = actions.squeeze(0)

            obs, rewards, done, info = env.step(actions)
            episode_reward += rewards

        episode_rewards.append(episode_reward.mean().item())
        episode_collisions.append(info['collisions'])
        episode_goals.append(info['goals_reached'])

        print(f"  Episode {i+1:2d}: Reward={episode_reward.mean().item():7.2f}, "
              f"Collisions={info['collisions']}, Goals={info['goals_reached']}")

    metrics = {
        'rewards': np.array(episode_rewards),
        'collisions': np.array(episode_collisions),
        'goals_reached': np.array(episode_goals)
    }

    print(f"\nEvaluation Summary:")
    print(f"  Mean Reward: {metrics['rewards'].mean():.2f} ± {metrics['rewards'].std():.2f}")
    print(f"  Mean Collisions: {metrics['collisions'].mean():.2f}")
    print(f"  Mean Goals Reached: {metrics['goals_reached'].mean():.2f}")
    print(f"  Collision Rate: {metrics['collisions'].mean() / env.n_vehicles * 100:.1f}%")
    print(f"  Goal Rate: {metrics['goals_reached'].mean() / env.n_vehicles * 100:.1f}%\n")

    return metrics


def main():
    args = parse_args()

    # Paths
    exp_dir = Path(args.log_dir) / args.exp_name
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Visualizing experiment: {args.exp_name}")
    print(f"{'='*70}\n")

    # Load config
    config_file = exp_dir / "config.json"
    if not config_file.exists():
        print(f"Warning: config.json not found in {exp_dir}")
        config_data = {}
    else:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        print("Configuration:")
        for key, value in config_data.items():
            print(f"  {key}: {value}")
        print()

    # Load and plot training metrics
    print("Loading training metrics...")
    metrics = load_metrics(exp_dir)

    if metrics:
        print(f"Found {len(metrics)} metric types with {len(metrics.get('train/episode_reward', []))} data points")

        save_path = exp_dir / "training_curves.png" if args.save_fig else None
        plot_training_curves(metrics, save_path)
    else:
        print("No training metrics found.")

    # If render flag is set, load model and render episodes
    if args.render:
        print("\nLoading trained model...")

        # Find checkpoint
        checkpoint_dir = exp_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                # Use final model or latest checkpoint
                final_model = checkpoint_dir / "model_final.pt"
                model_path = final_model if final_model.exists() else checkpoints[-1]
                print(f"Loading model from: {model_path}")
            else:
                print("No model checkpoints found.")
                return
        else:
            print("No checkpoints directory found.")
            return

        # Create environment
        scenario = config_data.get('scenario', 'intersection')
        n_vehicles = config_data.get('n_vehicles', 3)
        grid_size = config_data.get('grid_size', 20.0)
        max_steps = config_data.get('max_steps', 100)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        env = TrafficGridEnv(
            n_vehicles=n_vehicles,
            grid_size=grid_size,
            max_steps=max_steps,
            scenario=scenario,
            device=device
        )

        # Load agent
        tau = config_data.get('tau', 1.0)
        epsilon = config_data.get('epsilon', 0.01)
        hidden_dims = config_data.get('hidden_dims', [64, 64])

        config = RQEConfig(
            n_agents=env.n_vehicles,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            tau=tau,
            epsilon=epsilon,
            hidden_dims=hidden_dims
        )

        agent = RQE_MAPPO(config).to(device)
        agent.load(str(model_path))

        print(f"Model loaded successfully!")
        print(f"Environment: {scenario} with {n_vehicles} vehicles")
        print(f"Parameters: tau={tau}, epsilon={epsilon}\n")

        # Evaluate
        eval_metrics = evaluate_agent(agent, env, args.n_episodes)

        save_path = exp_dir / "evaluation_results.png" if args.save_fig else None
        plot_evaluation_results(eval_metrics, save_path)

        # Render a few episodes with trajectory plots
        print("\nRendering sample episodes with trajectory plots...")
        for i in range(min(3, args.n_episodes)):
            trajectory, reward, info = render_episode(agent, env, i)

            save_path = exp_dir / f"trajectory_ep{i+1}.png" if args.save_fig else None
            plot_trajectory(trajectory, info, save_path)

    print(f"\n{'='*70}")
    print("Visualization complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
