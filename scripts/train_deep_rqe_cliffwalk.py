#!/usr/bin/env python3
"""
Train Deep RQE Q-Learning on Cliff Walk Environment

This script trains Deep RQE Q-Learning on the two-agent Cliff Walk environment
and visualizes what Q-values and policies the algorithm learns for different
risk-aversion parameters (tau).

Key features:
- Per-agent tau/epsilon support
- RQE solver comparison visualization
- Reward scaling and corner reward shaping
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import copy

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.cliff_walk import CliffWalkEnv
from src.algorithms.deep_rqe import DeepRQEConfig, QNetwork
from src.algorithms.rqe_solver import RQESolver, RQEConfig as RQESolverConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep RQE Q-Learning on Cliff Walk")

    # RQE parameters (supports per-agent values: --tau 0.5 2.0 or single value --tau 1.0)
    parser.add_argument("--tau", type=float, nargs="+", default=[1.0],
                        help="Risk aversion parameter(s) (lower = more risk-averse). "
                             "Single value for both agents or two values for [agent1, agent2]")
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.1],
                        help="Entropy coefficient(s) (bounded rationality). "
                             "Single value for both agents or two values for [agent1, agent2]")

    # Training parameters
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of training episodes")
    parser.add_argument("--horizon", type=int, default=100,
                        help="Max steps per episode")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for updates")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon_greedy", type=float, default=0.1,
                        help="Epsilon-greedy exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                        help="Epsilon decay rate per episode")
    parser.add_argument("--epsilon_min", type=float, default=0.01,
                        help="Minimum epsilon value")

    # Network
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 64],
                        help="Hidden layer dimensions")

    # RQE solver
    parser.add_argument("--rqe_iterations", type=int, default=10,
                        help="RQE solver iterations")
    parser.add_argument("--rqe_lr", type=float, default=0.5,
                        help="RQE solver learning rate")

    # Buffer
    parser.add_argument("--buffer_size", type=int, default=50000,
                        help="Replay buffer size")
    parser.add_argument("--update_frequency", type=int, default=4,
                        help="Update every N steps")
    parser.add_argument("--target_update_freq", type=int, default=100,
                        help="Target network update frequency (episodes)")

    # Logging
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log every N episodes")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--output_dir", type=str, default="results/deep_rqe_cliffwalk",
                        help="Output directory")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use more deterministic environment (easier)")
    parser.add_argument("--reward_scale", type=float, default=50.0,
                        help="Reward scaling factor (paper uses 50.0)")
    parser.add_argument("--corner_reward", type=float, default=0.0,
                        help="One-time corner reward (e.g., 25.0 for shaping)")

    return parser.parse_args()


class ReplayBuffer:
    """Simple replay buffer"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, actions, rewards, next_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, actions, rewards, next_obs, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return {
            'obs': [b[0] for b in batch],
            'actions': [b[1] for b in batch],
            'rewards': [b[2] for b in batch],
            'next_obs': [b[3] for b in batch],
            'done': [b[4] for b in batch],
        }

    def __len__(self):
        return len(self.buffer)


class DeepRQE_QLearning_CliffWalk:
    """
    Deep RQE Q-Learning for CliffWalk environment

    Each agent has a Q-network that outputs Q(s, a_i, a_j) matrix.
    Actions are selected by solving RQE on the Q-matrices.
    """

    def __init__(self, config: DeepRQEConfig, tau_list, epsilon_list):
        self.config = config
        self.tau_list = tau_list
        self.epsilon_list = epsilon_list
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create Q-networks for each agent
        # Q_i(s) outputs [my_actions, opponent_actions] matrix
        self.q_networks = nn.ModuleList([
            QNetwork(
                my_action_dim=config.action_dims[i],
                opponent_action_dim=config.action_dims[1-i],
                obs_dim=config.obs_dim,
                hidden_dims=config.hidden_dims,
                activation="relu"
            ).to(self.device)
            for i in range(config.n_agents)
        ])

        # Target networks
        self.target_networks = nn.ModuleList([
            copy.deepcopy(self.q_networks[i]).to(self.device)
            for i in range(config.n_agents)
        ])

        # Optimizers
        self.optimizers = [
            torch.optim.Adam(self.q_networks[i].parameters(), lr=config.lr_critic)
            for i in range(config.n_agents)
        ]

        # RQE Solver
        solver_config = RQESolverConfig(
            action_dims=config.action_dims,
            tau=tau_list,
            epsilon=epsilon_list,
            max_iterations=config.rqe_iterations,
            learning_rate=config.rqe_lr,
            momentum=0.9,
            tolerance=1e-4,
            entropy_reg=True,
        )
        self.rqe_solver = RQESolver(solver_config)

        # Replay buffer
        self.buffer = ReplayBuffer(config.buffer_size)
        self.update_counter = 0

    def select_action(self, obs, epsilon_greedy=0.0):
        """Select actions using RQE policy (with epsilon-greedy exploration)"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Get Q-matrices for all agents
            Q_matrices = []
            for i in range(self.config.n_agents):
                Q = self.q_networks[i](obs_tensor)  # [1, my_actions, opp_actions]
                Q_matrices.append(Q)

            # Solve for RQE policies
            policies = self.rqe_solver.solve(Q_matrices)

            # Sample actions
            actions = []
            for i in range(self.config.n_agents):
                if np.random.rand() < epsilon_greedy:
                    # Random exploration
                    action = np.random.randint(self.config.action_dims[i])
                else:
                    # Sample from RQE policy
                    probs = policies[i][0].cpu().numpy()
                    # Ensure valid probabilities
                    probs = np.clip(probs, 0, 1)
                    probs = probs / (probs.sum() + 1e-8)
                    action = np.random.choice(self.config.action_dims[i], p=probs)
                actions.append(action)

            return actions

    def update(self, obs, actions, rewards, next_obs, done):
        """Store transition and update Q-networks"""
        self.buffer.push(obs, actions, rewards, next_obs, done)

        if len(self.buffer) < self.config.batch_size:
            return {}

        # Update every N steps
        self.update_counter += 1
        if self.update_counter % self.config.update_frequency != 0:
            return {}

        # Sample batch
        batch = self.buffer.sample(self.config.batch_size)
        obs_batch = torch.FloatTensor(np.array(batch['obs'])).to(self.device)
        next_obs_batch = torch.FloatTensor(np.array(batch['next_obs'])).to(self.device)
        actions_batch = [torch.LongTensor([b[i] for b in batch['actions']]).to(self.device)
                        for i in range(self.config.n_agents)]
        rewards_batch = [torch.FloatTensor([b[i] for b in batch['rewards']]).to(self.device)
                        for i in range(self.config.n_agents)]
        done_batch = torch.FloatTensor(batch['done']).to(self.device)

        # Compute target Q-values using RQE on next state
        with torch.no_grad():
            Q_next = [self.target_networks[i](next_obs_batch) for i in range(self.config.n_agents)]

            # Solve for RQE at next state
            policies_rqe = self.rqe_solver.solve(Q_next)

            # Compute expected values under RQE
            targets = []
            for i in range(self.config.n_agents):
                j = 1 - i
                # Expected Q-value: sum over (a_i, a_j) of pi_i(a_i) * Q_i(a_i, a_j) * pi_j(a_j)
                q_next_expected = torch.einsum('ba,bac,bc->b',
                                              policies_rqe[i],
                                              Q_next[i],
                                              policies_rqe[j])
                target = rewards_batch[i] + self.config.gamma * (1 - done_batch) * q_next_expected
                targets.append(target)

        # Update Q-networks
        losses = []
        for i in range(self.config.n_agents):
            j = 1 - i

            # Get current Q-values
            Q_current = self.q_networks[i](obs_batch)  # [batch, my_actions, opp_actions]

            # Extract Q-values for taken actions
            batch_indices = torch.arange(self.config.batch_size).to(self.device)
            q_values = Q_current[batch_indices, actions_batch[i], actions_batch[j]]

            # Compute loss
            loss = F.mse_loss(q_values, targets[i])
            losses.append(loss.item())

            # Update
            self.optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_networks[i].parameters(), 1.0)
            self.optimizers[i].step()

        return {'loss_agent0': losses[0], 'loss_agent1': losses[1]}

    def update_target_networks(self):
        """Hard update of target networks"""
        for i in range(self.config.n_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())

    def get_q_value(self, obs, agent_idx):
        """Get Q-value matrix for visualization"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            Q = self.q_networks[agent_idx](obs_tensor)
            return Q[0].cpu().numpy()

    def get_rqe_value(self, obs, agent_idx):
        """Get RQE value for visualization"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            Q_matrices = [self.q_networks[i](obs_tensor) for i in range(self.config.n_agents)]
            policies = self.rqe_solver.solve(Q_matrices)

            # Compute RQE value: V_i = pi_i^T @ Q_i @ pi_j
            i = agent_idx
            j = 1 - i
            V = torch.einsum('ba,bac,bc->b', policies[i], Q_matrices[i], policies[j])
            return V[0].item()


def create_env(horizon, deterministic=False, reward_scale=50.0, corner_reward=0.0):
    """Create Cliff Walk environment"""
    env = CliffWalkEnv(
        grid_size=(6, 6),
        horizon=horizon,
        return_joint_reward=False,
        reward_scale=reward_scale,
        corner_reward=corner_reward,
    )
    if deterministic:
        env.pd_close = 0.95
        env.pd_far = 0.85
    return env


def obs_to_agent_obs(obs, agent_idx):
    """
    Convert environment observation to agent-specific observation

    Env obs: [agent1_row, agent1_col, agent2_row, agent2_col]
    Agent obs: [my_row, my_col, opp_row, opp_col] (normalized)
    """
    obs_normalized = obs / 5.0

    if agent_idx == 0:
        # Agent 1: [a1_row, a1_col, a2_row, a2_col]
        return obs_normalized
    else:
        # Agent 2: [a2_row, a2_col, a1_row, a1_col]
        return np.array([obs_normalized[2], obs_normalized[3],
                        obs_normalized[0], obs_normalized[1]], dtype=np.float32)


def train(args):
    """Main training loop"""

    # Parse per-agent tau and epsilon
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
    run_name = f"tau{tau}_eps{epsilon}"
    run_dir = os.path.join(args.output_dir, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    print("=" * 70)
    print("Training Deep RQE Q-Learning on Cliff Walk")
    print("=" * 70)
    print(f"Output directory: {run_dir}")
    print(f"\nRQE Parameters (per-agent):")
    print(f"  tau (risk aversion): Agent1={tau[0]}, Agent2={tau[1]}")
    print(f"  epsilon (entropy coef): Agent1={epsilon[0]}, Agent2={epsilon[1]}")
    print(f"\nTraining:")
    print(f"  episodes: {args.episodes}")
    print(f"  horizon: {args.horizon}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning rate: {args.lr}")
    print(f"  gamma: {args.gamma}")
    print(f"\nReward Shaping:")
    print(f"  reward_scale: {args.reward_scale}")
    print(f"  corner_reward: {args.corner_reward}")
    print("=" * 70)

    # Create environment
    env = create_env(args.horizon, deterministic=args.deterministic,
                     reward_scale=args.reward_scale, corner_reward=args.corner_reward)
    if args.deterministic:
        print(f"Deterministic mode: pd_close={env.pd_close}, pd_far={env.pd_far}")

    # Create Deep RQE config
    config = DeepRQEConfig(
        n_agents=2,
        action_dims=[4, 4],  # up, down, left, right for each agent
        tau=tau,
        epsilon=epsilon,
        obs_dim=4,  # [my_row, my_col, opp_row, opp_col]
        hidden_dims=args.hidden_dims,
        lr_critic=args.lr,
        gamma=args.gamma,
        rqe_iterations=args.rqe_iterations,
        rqe_lr=args.rqe_lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        update_frequency=args.update_frequency,
    )

    # Create agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    agent = DeepRQE_QLearning_CliffWalk(config, tau, epsilon)

    # Training tracking
    all_rewards = []
    agent1_rewards = []
    agent2_rewards = []
    cliff_episodes = []
    goal_episodes = []
    current_epsilon = args.epsilon_greedy

    print(f"\nStarting training for {args.episodes} episodes...")

    for episode in tqdm(range(args.episodes), desc="Training"):
        obs, info = env.reset()
        episode_reward = [0.0, 0.0]
        hit_cliff = False
        reached_goal = [False, False]

        for step in range(args.horizon):
            # Get agent-specific observations
            obs1 = obs_to_agent_obs(obs, 0)
            obs2 = obs_to_agent_obs(obs, 1)

            # Select actions (use agent1's observation for now - they see the same grid)
            actions = agent.select_action(obs1, epsilon_greedy=current_epsilon)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # Track rewards
            episode_reward[0] += info['agent1_reward']
            episode_reward[1] += info['agent2_reward']

            # Track cliff hits
            if info['agent1_reward'] <= -2.0 * args.reward_scale or info['agent2_reward'] <= -2.0 * args.reward_scale:
                hit_cliff = True
            if info['agent1_at_goal']:
                reached_goal[0] = True
            if info['agent2_at_goal']:
                reached_goal[1] = True

            # Store and update
            next_obs1 = obs_to_agent_obs(next_obs, 0)
            rewards = [info['agent1_reward'], info['agent2_reward']]
            agent.update(obs1, actions, rewards, next_obs1, float(done))

            obs = next_obs

            if done:
                break

        # Track episode stats
        all_rewards.append(sum(episode_reward))
        agent1_rewards.append(episode_reward[0])
        agent2_rewards.append(episode_reward[1])
        cliff_episodes.append(1 if hit_cliff else 0)
        goal_episodes.append(1 if all(reached_goal) else 0)

        # Decay epsilon
        current_epsilon = max(args.epsilon_min, current_epsilon * args.epsilon_decay)

        # Update target networks
        if (episode + 1) % args.target_update_freq == 0:
            agent.update_target_networks()

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
            print(f"  Epsilon: {current_epsilon:.4f}")

        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(run_dir, "checkpoints", f"checkpoint_{episode + 1:06d}.pt")
            torch.save({
                'q_networks': [net.state_dict() for net in agent.q_networks],
                'target_networks': [net.state_dict() for net in agent.target_networks],
                'episode': episode + 1,
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

    # Final save
    final_path = os.path.join(run_dir, "checkpoints", "final_checkpoint.pt")
    torch.save({
        'q_networks': [net.state_dict() for net in agent.q_networks],
        'target_networks': [net.state_dict() for net in agent.target_networks],
        'episode': args.episodes,
    }, final_path)
    print(f"\nFinal checkpoint saved to {final_path}")

    # Plot training curves
    plot_training_curves(all_rewards, agent1_rewards, agent2_rewards,
                        cliff_episodes, goal_episodes, tau, run_dir)

    # Visualize learned values
    visualize_values(agent, tau, epsilon, run_dir)

    # Compare with RQE solver
    visualize_comparison(agent, tau, epsilon, run_dir, args.reward_scale, args.corner_reward)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    return agent, run_dir


def plot_training_curves(all_rewards, agent1_rewards, agent2_rewards,
                         cliff_episodes, goal_episodes, tau, output_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def smooth(data, window=50):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Total reward
    axes[0, 0].plot(smooth(all_rewards), label='Smoothed')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title(f'Total Reward (tau={tau})')
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
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to {output_dir}/training_curves.png")


def visualize_values(agent, tau, epsilon, output_dir):
    """Visualize the learned RQE values on a grid"""

    print("\nVisualizing learned value function...")

    grid_size = 6
    cliff_cells = [(1, 0), (2, 0), (3, 0), (4, 0), (2, 2), (2, 3), (3, 2), (3, 3)]
    agent1_goal = (0, 0)
    agent2_goal = (5, 0)
    agent1_start = (4, 2)
    agent2_start = (1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for agent_idx in range(2):
        ax = axes[agent_idx]

        if agent_idx == 0:
            fixed_pos = agent2_start
            my_goal = agent1_goal
            title = f'Agent 1 RQE Value (Goal at {agent1_goal})\n(Agent 2 at {fixed_pos})'
        else:
            fixed_pos = agent1_start
            my_goal = agent2_goal
            title = f'Agent 2 RQE Value (Goal at {agent2_goal})\n(Agent 1 at {fixed_pos})'

        agent_tau = tau[agent_idx]
        agent_eps = epsilon[agent_idx]

        values = np.zeros((grid_size, grid_size))

        for r in range(grid_size):
            for c in range(grid_size):
                # Create observation: [my_row, my_col, opp_row, opp_col]
                obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0
                values[r, c] = agent.get_rqe_value(obs, agent_idx)

        # Plot heatmap
        im = ax.imshow(values, cmap='RdYlGn', origin='upper')
        ax.set_title(f'{title}\n(tau={agent_tau}, eps={agent_eps})')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        # Add value text with path effects
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
        ax.plot(agent1_goal[1], agent1_goal[0], 'r^', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.plot(agent2_goal[1], agent2_goal[0], 'bs', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.plot(my_goal[1], my_goal[0], 'y*', markersize=20, markeredgecolor='black', markeredgewidth=1)

        plt.colorbar(im, ax=ax)

    plt.suptitle(f'Deep RQE Q-Learning - Learned RQE Values (tau={tau}, epsilon={epsilon})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "value_function.png"), dpi=150)
    plt.close()

    print(f"Value function visualization saved to {output_dir}/value_function.png")


def compute_rqe_solver_values(reward_scale, corner_reward, horizon=30, tau=None, epsilon=None):
    """Compute RQE solver values for comparison"""
    from src.visualize_rqe_solver_cliffwalk import create_cliffwalk_markov_game
    from algorithms.markov_rqe_solver import MarkovRQESolver

    if tau is None:
        tau = [1.0, 1.0]
    if epsilon is None:
        epsilon = [0.1, 0.1]

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
        n_flags = 8 if corner_reward > 0 else 4
        V0 = V0_full[::n_flags]

        all_values.append(V0.reshape(grid_size, grid_size))

    return all_values, cliff_cells, agent1_goal, agent2_goal, tau, epsilon


def visualize_comparison(agent, tau, epsilon, output_dir, reward_scale, corner_reward):
    """Visualize learned values vs RQE solver values side by side"""
    print("\nComputing RQE solver values for comparison...")
    print(f"  Using tau={tau}, epsilon={epsilon}")

    rqe_values, cliff_cells, agent1_goal, agent2_goal, _, _ = compute_rqe_solver_values(
        reward_scale, corner_reward, tau=tau, epsilon=epsilon
    )

    grid_size = 6
    agent1_start = (4, 2)
    agent2_start = (1, 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    import matplotlib.patheffects as pe

    for agent_idx in range(2):
        fixed_pos = agent2_start if agent_idx == 0 else agent1_start
        my_goal = agent1_goal if agent_idx == 0 else agent2_goal

        agent_tau = tau[agent_idx]
        agent_eps = epsilon[agent_idx]

        # Compute learned values
        learned_values = np.zeros((grid_size, grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0
                learned_values[r, c] = agent.get_rqe_value(obs, agent_idx)

        # Plot learned values (top row)
        ax_learned = axes[0, agent_idx]
        im1 = ax_learned.imshow(learned_values, cmap='RdYlGn', origin='upper')
        ax_learned.set_title(f'Agent {agent_idx+1} Deep RQE Learned\n(tau={agent_tau}, eps={agent_eps})')

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
        ax_rqe = axes[1, agent_idx]
        rqe_vals = rqe_values[agent_idx].copy()
        rqe_vals[my_goal[0], my_goal[1]] += reward_scale
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

    plt.suptitle(f'Deep RQE Learned vs RQE Solver (tau={tau}, eps={epsilon}, scale={reward_scale})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "value_comparison.png"), dpi=150)
    plt.close()
    print(f"Value comparison saved to {output_dir}/value_comparison.png")


if __name__ == "__main__":
    args = parse_args()
    train(args)
