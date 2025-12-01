#!/usr/bin/env python3
"""
Train RQE-MAPPO on Cliff Walk Environment

This script trains RQE-MAPPO on the two-agent Cliff Walk environment
to debug and visualize what values and policies the algorithm learns
for different risk-aversion parameters (tau).

Key insight: Lower tau should make agents more risk-averse, preferring
safer paths that avoid cliffs even at the cost of lower expected reward.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.cliff_walk import CliffWalkEnv
from src.algorithms.rqe_mappo import RQE_MAPPO, RQEConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQE-MAPPO on Cliff Walk")

    # RQE parameters (supports per-agent values: --tau 0.5 2.0 or single value --tau 1.0)
    parser.add_argument("--tau", type=float, nargs="+", default=[1.0],
                        help="Risk aversion parameter(s) (lower = more risk-averse). "
                             "Single value for both agents or two values for [agent1, agent2]")
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.1],
                        help="Entropy coefficient(s) (bounded rationality). "
                             "Single value for both agents or two values for [agent1, agent2]")
    parser.add_argument("--risk_measure", type=str, default="entropic",
                        choices=["entropic", "cvar", "mean_variance"],
                        help="Risk measure type")

    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--horizon", type=int, default=100,
                        help="Max steps per episode")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for updates")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")

    # Network
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 64],
                        help="Hidden layer dimensions")

    # Distributional critic
    parser.add_argument("--n_atoms", type=int, default=51,
                        help="Number of atoms for distributional critic")
    parser.add_argument("--v_min", type=float, default=-50.0,
                        help="Minimum value for distributional critic")
    parser.add_argument("--v_max", type=float, default=50.0,
                        help="Maximum value for distributional critic")

    # Logging
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log every N episodes")
    parser.add_argument("--save_interval", type=int, default=200,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--output_dir", type=str, default="results/rqe_mappo_cliffwalk",
                        help="Output directory")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use more deterministic environment (easier)")
    parser.add_argument("--standard_mappo", action="store_true",
                        help="Use standard MAPPO (scalar critic) instead of RQE-MAPPO")
    parser.add_argument("--reward_scale", type=float, default=50.0,
                        help="Reward scaling factor (paper uses 50.0)")
    parser.add_argument("--corner_reward", type=float, default=0.0,
                        help="One-time corner reward (e.g., 25.0 for shaping)")

    return parser.parse_args()


def create_env(horizon, deterministic=False, reward_scale=50.0, corner_reward=0.0):
    """Create Cliff Walk environment"""
    # Use return_joint_reward=False to get per-agent rewards
    env = CliffWalkEnv(
        grid_size=(6, 6),
        horizon=horizon,
        return_joint_reward=False,  # Get (reward1, reward2) tuple
        reward_scale=reward_scale,
        corner_reward=corner_reward,
    )
    # For debugging, make environment more deterministic
    if deterministic:
        env.pd_close = 0.95  # Nearly deterministic when close
        env.pd_far = 0.85    # More deterministic when far (instead of 0.5)
    return env


def obs_to_tensor(obs, n_agents=2):
    """
    Convert environment observation to tensor format for RQE-MAPPO

    Env obs: [agent1_row, agent1_col, agent2_row, agent2_col]
    RQE-MAPPO expects: [batch, n_agents, obs_dim]

    Each agent gets observation with THEIR position first, then opponent's position.
    This helps each agent learn from their own perspective.
    """
    # Normalize positions to [0, 1]
    obs_normalized = obs / 5.0  # Grid is 6x6, so max is 5

    # Agent 1's view: [my_row, my_col, opp_row, opp_col] = [a1_row, a1_col, a2_row, a2_col]
    agent1_obs = obs_normalized  # Already in correct order

    # Agent 2's view: [my_row, my_col, opp_row, opp_col] = [a2_row, a2_col, a1_row, a1_col]
    agent2_obs = np.array([obs_normalized[2], obs_normalized[3],
                           obs_normalized[0], obs_normalized[1]], dtype=np.float32)

    agent_obs = np.stack([agent1_obs, agent2_obs], axis=0)  # [2, 4]

    return torch.FloatTensor(agent_obs).unsqueeze(0)  # [1, 2, 4]


def train(args):
    """Main training loop"""

    # Parse per-agent tau and epsilon
    # Support both single value (applied to both) and per-agent values
    if len(args.tau) == 1:
        tau = [args.tau[0], args.tau[0]]
    elif len(args.tau) == 2:
        tau = args.tau
    else:
        raise ValueError(f"tau must have 1 or 2 values, got {len(args.tau)}")

    if len(args.epsilon) == 1:
        epsilon = [args.epsilon[0], args.epsilon[0]]
    elif len(args.epsilon) == 2:
        epsilon = args.epsilon
    else:
        raise ValueError(f"epsilon must have 1 or 2 values, got {len(args.epsilon)}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.standard_mappo:
        run_name = "standard_mappo"
    else:
        run_name = f"tau{tau}_eps{epsilon}"
    run_dir = os.path.join(args.output_dir, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    print("=" * 70)
    print("Training RQE-MAPPO on Cliff Walk")
    print("=" * 70)
    print(f"Output directory: {run_dir}")
    print(f"\nRQE Parameters (per-agent):")
    print(f"  tau (risk aversion): Agent1={tau[0]}, Agent2={tau[1]}")
    print(f"  epsilon (entropy coef): Agent1={epsilon[0]}, Agent2={epsilon[1]}")
    print(f"  risk_measure: {args.risk_measure}")
    print(f"\nTraining:")
    print(f"  episodes: {args.episodes}")
    print(f"  horizon: {args.horizon}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning rate: {args.lr}")
    print(f"\nDistributional Critic:")
    print(f"  n_atoms: {args.n_atoms}")
    print(f"  v_min: {args.v_min}")
    print(f"  v_max: {args.v_max}")
    print(f"\nReward Shaping:")
    print(f"  reward_scale: {args.reward_scale}")
    print(f"  corner_reward: {args.corner_reward}")
    print("=" * 70)

    # Create environment
    env = create_env(args.horizon, deterministic=args.deterministic,
                     reward_scale=args.reward_scale, corner_reward=args.corner_reward)
    if args.deterministic:
        print(f"  pd_close: {env.pd_close}")
        print(f"  pd_far: {env.pd_far}")

    # Create RQE-MAPPO config (stores per-agent tau/epsilon)
    config = RQEConfig(
        n_agents=2,
        action_dim=4,  # up, down, left, right
        tau=tau[0],  # Default tau for config (agent-specific handled separately)
        epsilon=epsilon[0],  # Default epsilon for config
        risk_measure=args.risk_measure,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        actor_lr=args.lr,
        critic_lr=args.lr,
        hidden_dims=args.hidden_dims,
        use_self_play=False,  # Two agents trained together
    )
    # Store per-agent values for later use
    config.tau_list = tau
    config.epsilon_list = epsilon

    # Create RQE-MAPPO agent
    # Need to provide obs_dim for default network creation
    obs_dim = 4  # [row1, col1, row2, col2]

    # Create actors and critics with proper dimensions
    from src.algorithms.rqe_mappo import Actor, DistributionalCritic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    actors = [
        Actor(obs_dim, config.action_dim, config.hidden_dims).to(device)
        for _ in range(config.n_agents)
    ]

    # Standard MAPPO uses scalar critic, RQE-MAPPO uses distributional critic
    if args.standard_mappo:
        print("\n*** Using STANDARD MAPPO (scalar critic) ***")
        # Simple scalar critic that mimics DistributionalCritic interface
        class ScalarCritic(nn.Module):
            def __init__(self, obs_dim, hidden_dims, v_min=-50, v_max=50, n_atoms=51):
                super().__init__()
                layers = []
                last_dim = obs_dim
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(last_dim, hidden_dim))
                    layers.append(nn.Tanh())
                    last_dim = hidden_dim
                layers.append(nn.Linear(last_dim, 1))
                self.network = nn.Sequential(*layers)

                # Dummy attributes to match DistributionalCritic interface
                self.n_atoms = n_atoms
                self.v_min = v_min
                self.v_max = v_max
                self.delta_z = (v_max - v_min) / (n_atoms - 1)
                self.register_buffer('z_atoms', torch.linspace(v_min, v_max, n_atoms))
                self.is_scalar = True  # Flag to identify scalar critic

            def forward(self, obs):
                # Return "probabilities" peaked at the predicted value
                # This is a hack to work with the distributional update
                value = self.network(obs).squeeze(-1)  # [batch]
                return value

            def get_risk_value(self, obs, tau=None, risk_type=None):
                # Standard MAPPO: just return expected value (ignore risk params)
                return self.network(obs).squeeze(-1)

        critics = [
            ScalarCritic(obs_dim, config.hidden_dims, config.v_min, config.v_max, config.n_atoms).to(device)
            for _ in range(config.n_agents)
        ]
    else:
        print(f"\n*** Using RQE-MAPPO (distributional critic, tau={args.tau}) ***")
        critics = [
            DistributionalCritic(
                obs_dim, config.hidden_dims, config.n_atoms, config.v_min, config.v_max
            ).to(device)
            for _ in range(config.n_agents)
        ]

    # Create RQE-MAPPO with custom networks
    agent = RQE_MAPPO(actors, critics, config)

    # Training tracking
    all_rewards = []
    agent1_rewards = []
    agent2_rewards = []
    cliff_episodes = []  # Track episodes that hit cliff
    goal_episodes = []   # Track episodes where both reach goal

    # Collect experiences
    batch_obs = []
    batch_actions = []
    batch_log_probs = []
    batch_rewards = []
    batch_dones = []
    batch_next_obs = []

    print(f"\nStarting training for {args.episodes} episodes...")

    for episode in tqdm(range(args.episodes), desc="Training"):
        obs, info = env.reset()
        obs_tensor = obs_to_tensor(obs)

        episode_reward = [0.0, 0.0]
        hit_cliff = False
        reached_goal = [False, False]

        for step in range(args.horizon):
            # Get actions
            with torch.no_grad():
                actions, log_probs, _ = agent.get_actions(obs_tensor)

            # Convert to env format
            action_np = actions[0].cpu().numpy()  # [n_agents]

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # Track rewards
            episode_reward[0] += info['agent1_reward']
            episode_reward[1] += info['agent2_reward']

            # Track cliff hits (check for scaled cliff penalty)
            if info['agent1_reward'] <= -2.0 * args.reward_scale or info['agent2_reward'] <= -2.0 * args.reward_scale:
                hit_cliff = True
            if info['agent1_at_goal']:
                reached_goal[0] = True
            if info['agent2_at_goal']:
                reached_goal[1] = True

            # Store experience
            next_obs_tensor = obs_to_tensor(next_obs)

            # Convert reward tuple to tensor [n_agents]
            reward_tensor = torch.FloatTensor([[info['agent1_reward'], info['agent2_reward']]])

            batch_obs.append(obs_tensor)
            batch_actions.append(actions)
            batch_log_probs.append(log_probs)
            batch_rewards.append(reward_tensor)
            batch_dones.append(torch.FloatTensor([[float(done)]]))
            batch_next_obs.append(next_obs_tensor)

            obs_tensor = next_obs_tensor

            if done:
                break

        # Track episode stats
        all_rewards.append(sum(episode_reward))
        agent1_rewards.append(episode_reward[0])
        agent2_rewards.append(episode_reward[1])
        cliff_episodes.append(1 if hit_cliff else 0)
        goal_episodes.append(1 if all(reached_goal) else 0)

        # Update when we have enough experiences
        if len(batch_obs) >= args.batch_size:
            # Stack batch
            obs_batch = torch.cat(batch_obs, dim=0)
            actions_batch = torch.cat(batch_actions, dim=0)
            log_probs_batch = torch.cat(batch_log_probs, dim=0)
            rewards_batch = torch.cat(batch_rewards, dim=0)
            dones_batch = torch.cat(batch_dones, dim=0).squeeze(-1)
            next_obs_batch = torch.cat(batch_next_obs, dim=0)

            # Update
            stats = agent.update(
                obs_batch, actions_batch, log_probs_batch,
                rewards_batch, dones_batch, next_obs_batch
            )

            # Clear batch
            batch_obs = []
            batch_actions = []
            batch_log_probs = []
            batch_rewards = []
            batch_dones = []
            batch_next_obs = []

        # Logging
        if (episode + 1) % args.log_interval == 0:
            recent_rewards = all_rewards[-args.log_interval:]
            recent_cliff = cliff_episodes[-args.log_interval:]
            recent_goal = goal_episodes[-args.log_interval:]

            print(f"\nEpisode {episode + 1}/{args.episodes}")
            print(f"  Avg Reward: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
            print(f"  Agent 1 Avg: {np.mean(agent1_rewards[-args.log_interval:]):.2f}")
            print(f"  Agent 2 Avg: {np.mean(agent2_rewards[-args.log_interval:]):.2f}")
            print(f"  Cliff Rate: {np.mean(recent_cliff) * 100:.1f}%")
            print(f"  Goal Rate: {np.mean(recent_goal) * 100:.1f}%")

        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                run_dir, "checkpoints", f"checkpoint_{episode + 1:06d}.pt"
            )
            agent.save(checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

    # Final save
    final_path = os.path.join(run_dir, "checkpoints", "final_checkpoint.pt")
    agent.save(final_path)
    print(f"\nFinal checkpoint saved to {final_path}")

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Smooth rewards
    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Total reward
    axes[0, 0].plot(smooth(all_rewards), label='Smoothed')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title(f'Total Reward (tau={args.tau})')
    axes[0, 0].legend()

    # Per-agent rewards
    axes[0, 1].plot(smooth(agent1_rewards), label='Agent 1', color='red')
    axes[0, 1].plot(smooth(agent2_rewards), label='Agent 2', color='blue')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Per-Agent Rewards')
    axes[0, 1].legend()

    # Cliff rate
    axes[1, 0].plot(smooth(cliff_episodes), color='black')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Cliff Rate')
    axes[1, 0].set_title('Cliff Hit Rate')

    # Goal rate
    axes[1, 1].plot(smooth(goal_episodes), color='green')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Goal Rate')
    axes[1, 1].set_title('Both Agents Reach Goal Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to {run_dir}/training_curves.png")

    # Visualize learned value function
    visualize_values(agent, config, run_dir)

    # Compare with RQE solver
    visualize_comparison(agent, config, run_dir, args.reward_scale, args.corner_reward)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    return agent, run_dir


def visualize_values(agent, config, output_dir):
    """Visualize the learned value function on a grid"""

    print("\nVisualizing learned value function...")

    device = agent.device

    # Create a grid of states
    grid_size = 6
    cliff_cells = [(1, 0), (2, 0), (3, 0), (4, 0), (2, 2), (2, 3), (3, 2), (3, 3)]
    agent1_goal = (0, 0)
    agent2_goal = (5, 0)
    agent1_start = (4, 2)
    agent2_start = (1, 2)

    # Show values for both agents, each with opponent at their start position
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for agent_idx in range(2):
        ax = axes[agent_idx]

        # Fix the other agent at their start
        if agent_idx == 0:
            fixed_pos = agent2_start
            my_goal = agent1_goal
            title = f'Agent 1 Value (Goal at {agent1_goal})\n(Agent 2 at {fixed_pos})'
        else:
            fixed_pos = agent1_start
            my_goal = agent2_goal
            title = f'Agent 2 Value (Goal at {agent2_goal})\n(Agent 1 at {fixed_pos})'

        # Get per-agent tau/epsilon
        agent_tau = config.tau_list[agent_idx] if hasattr(config, 'tau_list') else config.tau
        agent_eps = config.epsilon_list[agent_idx] if hasattr(config, 'epsilon_list') else config.epsilon

        values = np.zeros((grid_size, grid_size))

        for r in range(grid_size):
            for c in range(grid_size):
                # Create observation: [my_row, my_col, opp_row, opp_col]
                obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

                # Get risk value for this agent using their specific tau
                with torch.no_grad():
                    value = agent.critics[agent_idx].get_risk_value(
                        obs_tensor, agent_tau, config.risk_measure
                    )
                    values[r, c] = value.item()

        # Plot heatmap
        im = ax.imshow(values, cmap='RdYlGn', origin='upper')
        ax.set_title(f'{title}\n(tau={agent_tau}, eps={agent_eps})')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        # Add value text in each cell with path effects for visibility
        import matplotlib.patheffects as pe
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) not in cliff_cells:
                    ax.text(c, r, f'{values[r, c]:.2f}', ha='center', va='center',
                           fontsize=7, color='white', fontweight='bold',
                           path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        # Mark cliff cells
        for (cr, cc) in cliff_cells:
            ax.add_patch(plt.Rectangle((cc - 0.5, cr - 0.5), 1, 1,
                                       fill=True, facecolor='black', edgecolor='white', linewidth=2))

        # Mark goals
        ax.plot(agent1_goal[1], agent1_goal[0], 'r^', markersize=15, markeredgecolor='white', markeredgewidth=2, label='Goal 1')
        ax.plot(agent2_goal[1], agent2_goal[0], 'bs', markersize=15, markeredgecolor='white', markeredgewidth=2, label='Goal 2')

        # Mark my goal with a star
        ax.plot(my_goal[1], my_goal[0], 'y*', markersize=20, markeredgecolor='black', markeredgewidth=1)

        plt.colorbar(im, ax=ax)

    tau_str = config.tau_list if hasattr(config, 'tau_list') else config.tau
    eps_str = config.epsilon_list if hasattr(config, 'epsilon_list') else config.epsilon
    plt.suptitle(f'RQE-MAPPO Learned Value Function (tau={tau_str}, epsilon={eps_str})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "value_function.png"), dpi=150)
    plt.close()

    print(f"Value function visualization saved to {output_dir}/value_function.png")


def compute_rqe_solver_values(reward_scale, corner_reward, horizon=30, tau=None, epsilon=None):
    """Compute RQE solver values for comparison

    Args:
        tau: Can be float (same for both) or list [tau1, tau2] for per-agent
        epsilon: Can be float (same for both) or list [eps1, eps2] for per-agent
    """
    from src.visualize_rqe_solver_cliffwalk import create_cliffwalk_markov_game
    from algorithms.markov_rqe_solver import MarkovRQESolver

    # Default values
    if tau is None:
        tau = [1.0, 1.0]
    if epsilon is None:
        epsilon = [0.1, 0.1]

    # Convert to list if single value
    if isinstance(tau, (int, float)):
        tau = [float(tau), float(tau)]
    if isinstance(epsilon, (int, float)):
        epsilon = [float(epsilon), float(epsilon)]

    games, grid_size, cliff_cells, agent1_goal, agent2_goal = create_cliffwalk_markov_game(
        tau=tau, epsilon=epsilon, horizon=horizon, deterministic=True,
        reward_scale=reward_scale, corner_reward=corner_reward
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_values = []

    for player_idx, (config, game, goal) in enumerate(games):
        solver = MarkovRQESolver(config)
        policies, values = solver.solve(game, device)

        V0_full = values[0][0].cpu().numpy()
        # n_flags = 8 if corner_reward > 0 else 4 (track goal + cliff, optionally corner)
        n_flags = 8 if corner_reward > 0 else 4
        V0 = V0_full[::n_flags]  # Extract initial state (flags=0)

        all_values.append(V0.reshape(grid_size, grid_size))

    return all_values, cliff_cells, agent1_goal, agent2_goal, tau, epsilon


def visualize_comparison(agent, config, output_dir, reward_scale, corner_reward):
    """Visualize learned values vs RQE solver values side by side"""
    # Get per-agent tau/epsilon
    tau_list = config.tau_list if hasattr(config, 'tau_list') else [config.tau, config.tau]
    eps_list = config.epsilon_list if hasattr(config, 'epsilon_list') else [config.epsilon, config.epsilon]

    print("\nComputing RQE solver values for comparison...")
    print(f"  Using tau={tau_list}, epsilon={eps_list}")
    rqe_values, cliff_cells, agent1_goal, agent2_goal, tau, epsilon = compute_rqe_solver_values(
        reward_scale, corner_reward, tau=tau_list, epsilon=eps_list
    )

    grid_size = 6
    agent1_start = (4, 2)
    agent2_start = (1, 2)
    device = agent.device

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    import matplotlib.patheffects as pe

    for agent_idx in range(2):
        fixed_pos = agent2_start if agent_idx == 0 else agent1_start
        my_goal = agent1_goal if agent_idx == 0 else agent2_goal

        # Get per-agent tau/epsilon
        agent_tau = tau_list[agent_idx]
        agent_eps = eps_list[agent_idx]

        # Compute learned values
        learned_values = np.zeros((grid_size, grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

                with torch.no_grad():
                    value = agent.critics[agent_idx].get_risk_value(
                        obs_tensor, agent_tau, config.risk_measure
                    )
                    learned_values[r, c] = value.item()

        # Plot learned values (top row)
        ax_learned = axes[0, agent_idx]
        im1 = ax_learned.imshow(learned_values, cmap='RdYlGn', origin='upper')
        ax_learned.set_title(f'Agent {agent_idx+1} RQE-MAPPO Learned\n(tau={agent_tau}, eps={agent_eps})')

        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) not in cliff_cells:
                    val = learned_values[r, c]
                    ax_learned.text(c, r, f'{val:.2f}', ha='center', va='center',
                                   fontsize=7, color='white', fontweight='bold',
                                   path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        for (cr, cc) in cliff_cells:
            ax_learned.add_patch(plt.Rectangle((cc - 0.5, cr - 0.5), 1, 1,
                                              fill=True, facecolor='black', edgecolor='white', linewidth=2))
        ax_learned.plot(my_goal[1], my_goal[0], 'y*', markersize=15, markeredgecolor='black')
        plt.colorbar(im1, ax=ax_learned)

        # Plot RQE solver values (bottom row)
        # Add entry rewards to terminal states for intuitive display
        ax_rqe = axes[1, agent_idx]
        rqe_vals = rqe_values[agent_idx].copy()
        # Goal: add entry reward so goal shows as highest value
        rqe_vals[my_goal[0], my_goal[1]] += reward_scale
        # Cliff: show penalty directly
        for (cr, cc) in cliff_cells:
            rqe_vals[cr, cc] = -2 * reward_scale

        im2 = ax_rqe.imshow(rqe_vals, cmap='RdYlGn', origin='upper')
        ax_rqe.set_title(f'Agent {agent_idx+1} RQE Solver (Optimal)\n(tau={agent_tau}, eps={agent_eps})')

        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) not in cliff_cells:
                    val = rqe_vals[r, c]
                    ax_rqe.text(c, r, f'{val:.2f}', ha='center', va='center',
                               fontsize=7, color='white', fontweight='bold',
                               path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        for (cr, cc) in cliff_cells:
            ax_rqe.add_patch(plt.Rectangle((cc - 0.5, cr - 0.5), 1, 1,
                                          fill=True, facecolor='black', edgecolor='white', linewidth=2))
            ax_rqe.text(cc, cr, f'{-2*reward_scale:.2f}', ha='center', va='center',
                       fontsize=7, color='white', fontweight='bold',
                       path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        ax_rqe.plot(my_goal[1], my_goal[0], 'y*', markersize=15, markeredgecolor='black')
        plt.colorbar(im2, ax=ax_rqe)

    plt.suptitle(f'RQE-MAPPO Learned vs RQE Solver (tau={tau_list}, eps={eps_list}, scale={reward_scale})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "value_comparison.png"), dpi=150)
    plt.close()
    print(f"Value comparison saved to {output_dir}/value_comparison.png")


if __name__ == "__main__":
    args = parse_args()
    train(args)
