#!/usr/bin/env python3
"""
Train Standard PPO (MAPPO) on Cliff Walk Environment

Simple PPO implementation without any RQE/risk components.
Used as baseline comparison for RQE-MAPPO.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.cliff_walk import CliffWalkEnv
from src.visualize_rqe_solver_cliffwalk import create_cliffwalk_markov_game
from algorithms.markov_rqe_solver import MarkovRQESolver


class Actor(nn.Module):
    """Simple actor network"""
    def __init__(self, obs_dim, action_dim, hidden_dims=[64, 64]):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(last_dim, action_dim)

    def forward(self, obs):
        features = self.network(obs)
        logits = self.action_head(features)
        return logits

    def get_action(self, obs, deterministic=False):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


class Critic(nn.Module):
    """Centralized critic network - takes global state (all agents' observations)"""
    def __init__(self, obs_dim, n_agents, hidden_dims=[64, 64]):
        super().__init__()
        # Centralized critic takes concatenated observations from all agents
        global_obs_dim = obs_dim * n_agents
        layers = []
        last_dim = global_obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        # Output value for each agent
        layers.append(nn.Linear(last_dim, n_agents))
        self.network = nn.Sequential(*layers)
        self.n_agents = n_agents

    def forward(self, global_obs):
        # global_obs: [batch, n_agents * obs_dim]
        return self.network(global_obs)  # [batch, n_agents]


class PPOAgent:
    """Simple PPO agent for multi-agent setting"""
    def __init__(self, obs_dim, action_dim, n_agents, hidden_dims=[64, 64],
                 lr=3e-4, gamma=0.99, clip_param=0.2, vf_coef=0.5, ent_coef=0.01):
        self.n_agents = n_agents
        self.gamma = gamma
        self.clip_param = clip_param
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim

        # Create actor for each agent (decentralized execution)
        self.actors = [Actor(obs_dim, action_dim, hidden_dims).to(self.device)
                      for _ in range(n_agents)]

        # Single centralized critic (CTDE paradigm)
        self.critic = Critic(obs_dim, n_agents, hidden_dims).to(self.device)

        # Optimizers
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=lr)
                                for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def get_actions(self, obs):
        """Get actions for all agents"""
        # obs: [batch, n_agents, obs_dim]
        obs = obs.to(self.device)
        actions = []
        log_probs = []
        entropies = []

        for i in range(self.n_agents):
            agent_obs = obs[:, i, :]
            action, log_prob, entropy = self.actors[i].get_action(agent_obs)
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)

        actions = torch.stack(actions, dim=1)  # [batch, n_agents]
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1)

        return actions, log_probs, entropies

    def get_values(self, obs):
        """Get values for all agents using centralized critic"""
        obs = obs.to(self.device)
        # Flatten all agents' observations into global state
        batch_size = obs.shape[0]
        global_obs = obs.view(batch_size, -1)  # [batch, n_agents * obs_dim]
        values = self.critic(global_obs)  # [batch, n_agents]
        return values

    def update(self, obs, actions, old_log_probs, rewards, dones, next_obs):
        """PPO update with centralized critic"""
        obs = obs.to(self.device).float()
        actions = actions.to(self.device).long()
        old_log_probs = old_log_probs.to(self.device).float()
        rewards = rewards.to(self.device).float()
        dones = dones.to(self.device).float()
        next_obs = next_obs.to(self.device).float()

        stats = {}

        # Compute values and advantages using centralized critic
        with torch.no_grad():
            values = self.get_values(obs)  # [batch, n_agents]
            next_values = self.get_values(next_obs)  # [batch, n_agents]
            dones_expanded = dones.unsqueeze(1).expand(-1, self.n_agents)
            # TD target
            targets = rewards + self.gamma * next_values * (1 - dones_expanded)
            advantages = targets - values
            # Normalize advantages per agent
            advantages = (advantages - advantages.mean(dim=0)) / (advantages.std(dim=0) + 1e-8)

        # Update each agent's actor (decentralized)
        for i in range(self.n_agents):
            agent_obs = obs[:, i, :]
            agent_actions = actions[:, i]
            agent_old_log_probs = old_log_probs[:, i]
            agent_advantages = advantages[:, i]

            # Actor update
            logits = self.actors[i](agent_obs)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(agent_actions)
            entropy = dist.entropy().mean()

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - agent_old_log_probs)
            surr1 = ratio * agent_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * agent_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.actor_optimizers[i].step()

            stats[f'actor_loss_{i}'] = actor_loss.item()
            stats[f'entropy_{i}'] = entropy.item()

        # Update centralized critic (single update for all agents)
        batch_size = obs.shape[0]
        global_obs = obs.view(batch_size, -1)  # [batch, n_agents * obs_dim]
        values_pred = self.critic(global_obs)  # [batch, n_agents]
        critic_loss = F.mse_loss(values_pred, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        stats['critic_loss'] = critic_loss.item()

        return stats

    def save(self, path):
        torch.save({
            'actors': [a.state_dict() for a in self.actors],
            'critic': self.critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        for i, state in enumerate(checkpoint['actors']):
            self.actors[i].load_state_dict(state)
        self.critic.load_state_dict(checkpoint['critic'])


def parse_args():
    parser = argparse.ArgumentParser(description="Train Standard PPO on Cliff Walk")

    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results/ppo_cliffwalk")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--reward_scale", type=float, default=50.0,
                        help="Reward scaling factor (paper uses 50.0)")
    parser.add_argument("--corner_reward", type=float, default=0.0,
                        help="One-time corner reward (e.g., 25.0 for shaping)")

    return parser.parse_args()


def obs_to_tensor(obs):
    """Convert env obs to tensor format"""
    obs_normalized = obs / 5.0
    # Agent 1: [my_row, my_col, opp_row, opp_col]
    agent1_obs = obs_normalized
    # Agent 2: [my_row, my_col, opp_row, opp_col] (swapped)
    agent2_obs = np.array([obs_normalized[2], obs_normalized[3],
                           obs_normalized[0], obs_normalized[1]], dtype=np.float32)
    agent_obs = np.stack([agent1_obs, agent2_obs], axis=0)
    return torch.FloatTensor(agent_obs).unsqueeze(0)


def train(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    print("=" * 70)
    print("Training MAPPO (Centralized Critic) on Cliff Walk")
    print("=" * 70)
    print(f"Output: {run_dir}")
    print(f"Episodes: {args.episodes}, LR: {args.lr}, Gamma: {args.gamma}")
    print(f"Reward scale: {args.reward_scale}")
    print(f"Corner reward: {args.corner_reward}")
    print("=" * 70)

    # Create environment with reward shaping built-in
    env = CliffWalkEnv(
        grid_size=(6, 6),
        horizon=args.horizon,
        return_joint_reward=False,
        reward_scale=args.reward_scale,
        corner_reward=args.corner_reward,
    )
    if args.deterministic:
        env.pd_close = 0.95
        env.pd_far = 0.85
        print(f"Deterministic mode: pd_close={env.pd_close}, pd_far={env.pd_far}")

    # Create agent
    agent = PPOAgent(
        obs_dim=4,
        action_dim=4,
        n_agents=2,
        hidden_dims=args.hidden_dims,
        lr=args.lr,
        gamma=args.gamma,
        clip_param=args.clip_param,
        ent_coef=args.ent_coef
    )
    print(f"Using device: {agent.device}")

    # Training
    all_rewards = []
    agent1_rewards = []
    agent2_rewards = []
    cliff_rates = []
    goal_rates = []

    batch_obs = []
    batch_actions = []
    batch_log_probs = []
    batch_rewards = []
    batch_dones = []
    batch_next_obs = []

    for episode in tqdm(range(args.episodes), desc="Training"):
        obs, _ = env.reset()
        obs_tensor = obs_to_tensor(obs)
        ep_reward = [0.0, 0.0]
        hit_cliff = False
        reached_goal = [False, False]

        for step in range(args.horizon):
            with torch.no_grad():
                actions, log_probs, _ = agent.get_actions(obs_tensor)

            action_np = actions[0].cpu().numpy()
            next_obs, _, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # Rewards come directly from env (already scaled with corner shaping)
            r1 = info['agent1_reward']
            r2 = info['agent2_reward']

            ep_reward[0] += r1
            ep_reward[1] += r2

            # Track cliff hits (check for scaled cliff penalty)
            if r1 <= -2.0 * args.reward_scale or r2 <= -2.0 * args.reward_scale:
                hit_cliff = True
            if info['agent1_at_goal']:
                reached_goal[0] = True
            if info['agent2_at_goal']:
                reached_goal[1] = True

            next_obs_tensor = obs_to_tensor(next_obs)
            reward_tensor = torch.FloatTensor([[r1, r2]])

            batch_obs.append(obs_tensor)
            batch_actions.append(actions)
            batch_log_probs.append(log_probs)
            batch_rewards.append(reward_tensor)
            batch_dones.append(torch.FloatTensor([[float(done)]]))
            batch_next_obs.append(next_obs_tensor)

            obs_tensor = next_obs_tensor
            if done:
                break

        all_rewards.append(sum(ep_reward))
        agent1_rewards.append(ep_reward[0])
        agent2_rewards.append(ep_reward[1])
        cliff_rates.append(1 if hit_cliff else 0)
        goal_rates.append(1 if all(reached_goal) else 0)

        # Update
        if len(batch_obs) >= args.batch_size:
            obs_batch = torch.cat(batch_obs, dim=0)
            actions_batch = torch.cat(batch_actions, dim=0)
            log_probs_batch = torch.cat(batch_log_probs, dim=0)
            rewards_batch = torch.cat(batch_rewards, dim=0)
            dones_batch = torch.cat(batch_dones, dim=0).squeeze(-1)
            next_obs_batch = torch.cat(batch_next_obs, dim=0)

            agent.update(obs_batch, actions_batch, log_probs_batch,
                        rewards_batch, dones_batch, next_obs_batch)

            batch_obs = []
            batch_actions = []
            batch_log_probs = []
            batch_rewards = []
            batch_dones = []
            batch_next_obs = []

        # Logging
        if (episode + 1) % args.log_interval == 0:
            recent = args.log_interval
            print(f"\nEpisode {episode + 1}")
            print(f"  Reward: {np.mean(all_rewards[-recent:]):.2f}")
            print(f"  Cliff: {np.mean(cliff_rates[-recent:])*100:.1f}%")
            print(f"  Goal: {np.mean(goal_rates[-recent:])*100:.1f}%")

        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(run_dir, "checkpoints", f"ckpt_{episode+1}.pt"))

    # Final save
    agent.save(os.path.join(run_dir, "checkpoints", "final.pt"))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    window = 50

    def smooth(x):
        return np.convolve(x, np.ones(window)/window, mode='valid') if len(x) >= window else x

    axes[0, 0].plot(smooth(all_rewards))
    axes[0, 0].set_title('Total Reward')
    axes[0, 1].plot(smooth(agent1_rewards), label='Agent 1')
    axes[0, 1].plot(smooth(agent2_rewards), label='Agent 2')
    axes[0, 1].legend()
    axes[0, 1].set_title('Per-Agent Rewards')
    axes[1, 0].plot(smooth(cliff_rates))
    axes[1, 0].set_title('Cliff Rate')
    axes[1, 1].plot(smooth(goal_rates))
    axes[1, 1].set_title('Goal Rate')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training.png"), dpi=150)
    plt.close()

    # Visualize values
    visualize_values(agent, run_dir)

    # Compare with RQE solver
    visualize_comparison(agent, run_dir, args.reward_scale, args.corner_reward)

    print(f"\nDone! Results in {run_dir}")


def visualize_values(agent, output_dir):
    """Visualize learned value function from centralized critic"""
    grid_size = 6
    cliff_cells = [(1, 0), (2, 0), (3, 0), (4, 0), (2, 2), (2, 3), (3, 2), (3, 3)]
    agent1_goal = (0, 0)
    agent2_goal = (5, 0)
    agent1_start = (4, 2)
    agent2_start = (1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for agent_idx in range(2):
        ax = axes[agent_idx]
        fixed_pos = agent2_start if agent_idx == 0 else agent1_start
        my_goal = agent1_goal if agent_idx == 0 else agent2_goal

        values = np.zeros((grid_size, grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                # Create observation for both agents
                if agent_idx == 0:
                    # Agent 1's position varies, Agent 2 fixed
                    agent1_obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0
                    agent2_obs = np.array([fixed_pos[0], fixed_pos[1], r, c], dtype=np.float32) / 5.0
                else:
                    # Agent 2's position varies, Agent 1 fixed
                    agent1_obs = np.array([fixed_pos[0], fixed_pos[1], r, c], dtype=np.float32) / 5.0
                    agent2_obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0

                # Create global observation for centralized critic
                global_obs = np.concatenate([agent1_obs, agent2_obs])  # [8]
                global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(agent.device)

                with torch.no_grad():
                    all_values = agent.critic(global_obs_tensor)  # [1, 2]
                    values[r, c] = all_values[0, agent_idx].item()

        im = ax.imshow(values, cmap='RdYlGn', origin='upper')

        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) not in cliff_cells:
                    color = 'white' if values[r, c] < (values.min() + values.max()) / 2 else 'black'
                    ax.text(c, r, f'{values[r, c]:.1f}', ha='center', va='center',
                           fontsize=8, color=color, fontweight='bold')

        for (cr, cc) in cliff_cells:
            ax.add_patch(plt.Rectangle((cc - 0.5, cr - 0.5), 1, 1,
                                       fill=True, facecolor='black', edgecolor='white', linewidth=2))

        ax.plot(agent1_goal[1], agent1_goal[0], 'r^', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.plot(agent2_goal[1], agent2_goal[0], 'bs', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.plot(my_goal[1], my_goal[0], 'y*', markersize=20, markeredgecolor='black', markeredgewidth=1)

        title = f'Agent {agent_idx+1} Value (Goal at {my_goal})'
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.suptitle('MAPPO (Centralized Critic) - Learned Values', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "value_function.png"), dpi=150)
    plt.close()
    print(f"Value function saved to {output_dir}/value_function.png")


def compute_rqe_solver_values(reward_scale, corner_reward, horizon=30, tau=1.0, epsilon=0.1):
    """Compute RQE solver values for comparison"""
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

    return all_values, cliff_cells, agent1_goal, agent2_goal


def visualize_comparison(agent, output_dir, reward_scale, corner_reward):
    """Visualize learned values vs RQE solver values side by side"""
    print("\nComputing RQE solver values for comparison...")
    rqe_values, cliff_cells, agent1_goal, agent2_goal = compute_rqe_solver_values(
        reward_scale, corner_reward
    )

    grid_size = 6
    agent1_start = (4, 2)
    agent2_start = (1, 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for agent_idx in range(2):
        fixed_pos = agent2_start if agent_idx == 0 else agent1_start
        my_goal = agent1_goal if agent_idx == 0 else agent2_goal

        # Compute learned values
        learned_values = np.zeros((grid_size, grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                if agent_idx == 0:
                    agent1_obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0
                    agent2_obs = np.array([fixed_pos[0], fixed_pos[1], r, c], dtype=np.float32) / 5.0
                else:
                    agent1_obs = np.array([fixed_pos[0], fixed_pos[1], r, c], dtype=np.float32) / 5.0
                    agent2_obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0

                global_obs = np.concatenate([agent1_obs, agent2_obs])
                global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(agent.device)

                with torch.no_grad():
                    all_vals = agent.critic(global_obs_tensor)
                    learned_values[r, c] = all_vals[0, agent_idx].item()

        # Plot learned values (top row)
        ax_learned = axes[0, agent_idx]
        im1 = ax_learned.imshow(learned_values, cmap='RdYlGn', origin='upper')
        ax_learned.set_title(f'Agent {agent_idx+1} MAPPO Learned')

        import matplotlib.patheffects as pe
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
        ax_rqe.set_title(f'Agent {agent_idx+1} RQE Solver (Optimal)')

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

    plt.suptitle(f'MAPPO Learned vs RQE Solver (scale={reward_scale}, corner={corner_reward})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "value_comparison.png"), dpi=150)
    plt.close()
    print(f"Value comparison saved to {output_dir}/value_comparison.png")


if __name__ == "__main__":
    args = parse_args()
    train(args)
