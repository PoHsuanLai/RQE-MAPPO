"""
Multi-Agent DQN with Centralized Critic for CliffWalk Environment

This implements MADDPG-style centralized training for discrete action spaces:
- Centralized Q-function: Q_i(o_1, o_2, a_i) sees all observations
- Decentralized execution: Each agent selects actions based on local observations
- Individual rewards: Each agent optimizes its own reward

This is the discrete-action equivalent of MADDPG, sometimes called:
- MA-DQN with centralized critic
- Central-V DQN
- QMIX without mixing (individual Q-functions)

Key insight: The centralized Q-function helps because the other agent's position
affects the dynamics (proximity changes stochasticity).
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.cliff_walk import CliffWalkEnv, simulate_trajectory, visualize_trajectory, get_normalized_obs


# =============================================================================
# Neural Network Models
# =============================================================================

class CentralizedQNetwork(nn.Module):
    """
    Centralized Q-network for MADDPG-style learning.

    Q_i(o_1, o_2, a_i) - takes all observations and outputs Q-values for agent i's actions.

    Architecture:
    - Input: concatenated observations from all agents
    - Output: Q-values for each action of the specific agent
    """

    def __init__(self, obs_dim_per_agent, n_agents, n_actions, hidden_sizes=[64, 64]):
        super().__init__()

        self.obs_dim_per_agent = obs_dim_per_agent
        self.n_agents = n_agents
        self.n_actions = n_actions

        # Total input: all agents' observations concatenated
        total_obs_dim = obs_dim_per_agent * n_agents

        layers = []
        prev_dim = total_obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, n_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, all_obs):
        """
        Forward pass.

        Args:
            all_obs: (batch, n_agents * obs_dim) concatenated observations

        Returns:
            q_values: (batch, n_actions) Q-values for each action
        """
        return self.network(all_obs)


class DecentralizedPolicy(nn.Module):
    """
    Decentralized policy network for action selection.

    π_i(o_i) - takes only local observation and outputs action probabilities.
    Used for decentralized execution (epsilon-greedy over Q-values).
    """

    def __init__(self, obs_dim, n_actions, hidden_sizes=[64, 64]):
        super().__init__()

        layers = []
        prev_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, n_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim) local observation

        Returns:
            q_values: (batch, n_actions) Q-values for action selection
        """
        return self.network(obs)


# =============================================================================
# Replay Buffer
# =============================================================================

class MultiAgentReplayBuffer:
    """
    Replay buffer for multi-agent experience.

    Stores transitions with all agents' observations, actions, and rewards.
    """

    def __init__(self, capacity, n_agents):
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_all, actions, rewards, next_obs_all, done):
        """
        Store a transition.

        Args:
            obs_all: list of observations for each agent
            actions: list of actions for each agent
            rewards: list of rewards for each agent
            next_obs_all: list of next observations for each agent
            done: whether episode terminated
        """
        self.buffer.append((obs_all, actions, rewards, next_obs_all, done))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        obs_all = [[] for _ in range(self.n_agents)]
        actions = [[] for _ in range(self.n_agents)]
        rewards = [[] for _ in range(self.n_agents)]
        next_obs_all = [[] for _ in range(self.n_agents)]
        dones = []

        for transition in batch:
            for i in range(self.n_agents):
                obs_all[i].append(transition[0][i])
                actions[i].append(transition[1][i])
                rewards[i].append(transition[2][i])
                next_obs_all[i].append(transition[3][i])
            dones.append(transition[4])

        return (
            [np.array(obs) for obs in obs_all],
            [np.array(act) for act in actions],
            [np.array(rew) for rew in rewards],
            [np.array(next_obs) for next_obs in next_obs_all],
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# MA-DQN Agent
# =============================================================================

class MADQNAgent:
    """
    Multi-Agent DQN with Centralized Critic.

    Each agent has:
    - A centralized Q-network Q_i(o_1, o_2, a_i) for training
    - A decentralized policy π_i(o_i) for execution
    - A target Q-network for stable learning
    """

    def __init__(
        self,
        agent_id,
        obs_dim_per_agent,
        n_agents,
        n_actions,
        hidden_sizes=[64, 64],
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        device="cpu"
    ):
        self.agent_id = agent_id
        self.obs_dim_per_agent = obs_dim_per_agent
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Centralized Q-network (sees all observations)
        self.q_network = CentralizedQNetwork(
            obs_dim_per_agent, n_agents, n_actions, hidden_sizes
        ).to(device)

        self.target_q_network = CentralizedQNetwork(
            obs_dim_per_agent, n_agents, n_actions, hidden_sizes
        ).to(device)

        # Copy weights to target
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Decentralized policy (sees only local observation)
        # For DQN, we use the centralized Q-values but can also have a local policy
        self.local_policy = DecentralizedPolicy(
            obs_dim_per_agent, n_actions, hidden_sizes
        ).to(device)

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.local_policy.parameters(), lr=lr)

    def select_action(self, local_obs, all_obs, epsilon=0.1, use_centralized=True):
        """
        Select action using epsilon-greedy.

        Args:
            local_obs: agent's local observation
            all_obs: all agents' observations (for centralized Q)
            epsilon: exploration rate
            use_centralized: whether to use centralized Q for action selection
        """
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            if use_centralized:
                # Concatenate all observations
                all_obs_tensor = torch.FloatTensor(
                    np.concatenate(all_obs)
                ).unsqueeze(0).to(self.device)
                q_values = self.q_network(all_obs_tensor)
            else:
                # Use local policy
                local_obs_tensor = torch.FloatTensor(local_obs).unsqueeze(0).to(self.device)
                q_values = self.local_policy(local_obs_tensor)

            return q_values.argmax(dim=1).item()

    def update(self, batch, other_agents):
        """
        Update Q-network using batch from replay buffer.

        Args:
            batch: (obs_all, actions, rewards, next_obs_all, dones)
            other_agents: list of other MADQNAgent instances
        """
        obs_all, actions, rewards, next_obs_all, dones = batch

        # Convert to tensors
        # Concatenate all observations
        all_obs = np.concatenate([obs_all[i] for i in range(self.n_agents)], axis=1)
        all_next_obs = np.concatenate([next_obs_all[i] for i in range(self.n_agents)], axis=1)

        all_obs_tensor = torch.FloatTensor(all_obs).to(self.device)
        all_next_obs_tensor = torch.FloatTensor(all_next_obs).to(self.device)

        actions_tensor = torch.LongTensor(actions[self.agent_id]).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards[self.agent_id]).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.q_network(all_obs_tensor)
        current_q = current_q.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN style)
        with torch.no_grad():
            # Use online network to select actions
            next_q_online = self.q_network(all_next_obs_tensor)
            next_actions = next_q_online.argmax(dim=1)

            # Use target network to evaluate
            next_q_target = self.target_q_network(all_next_obs_tensor)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q

        # Q-network loss
        q_loss = F.mse_loss(current_q, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.q_optimizer.step()

        # Update local policy to match centralized Q (distillation)
        local_obs_tensor = torch.FloatTensor(obs_all[self.agent_id]).to(self.device)

        with torch.no_grad():
            target_q_values = self.q_network(all_obs_tensor)

        local_q_values = self.local_policy(local_obs_tensor)
        policy_loss = F.mse_loss(local_q_values, target_q_values)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return q_loss.item(), policy_loss.item()

    def soft_update_target(self):
        """Soft update target network."""
        for target_param, param in zip(
            self.target_q_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


# =============================================================================
# Training Loop
# =============================================================================

def train_madqn(
    env,
    n_episodes=1000,
    max_steps=100,
    batch_size=64,
    buffer_size=100000,
    lr=1e-3,
    gamma=0.99,
    tau=0.005,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    hidden_sizes=[64, 64],
    update_freq=4,
    target_update_freq=100,
    device="cpu"
):
    """
    Train MA-DQN agents on the environment.
    """
    n_agents = 2
    obs_dim = 4  # (r1, c1, r2, c2) normalized
    n_actions = 4  # up, down, left, right

    # Create agents
    agents = [
        MADQNAgent(
            agent_id=i,
            obs_dim_per_agent=obs_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            hidden_sizes=hidden_sizes,
            lr=lr,
            gamma=gamma,
            tau=tau,
            device=device
        )
        for i in range(n_agents)
    ]

    # Create replay buffer
    replay_buffer = MultiAgentReplayBuffer(buffer_size, n_agents)

    # Training metrics
    episode_rewards = [[] for _ in range(n_agents)]
    episode_lengths = []
    epsilon = epsilon_start
    total_steps = 0

    print("=" * 70)
    print("Starting MA-DQN Training (Centralized Critic)")
    print("=" * 70)

    # Running averages for tqdm display
    avg_window = 100
    recent_rewards_0 = deque(maxlen=avg_window)
    recent_rewards_1 = deque(maxlen=avg_window)
    recent_lengths = deque(maxlen=avg_window)

    pbar = tqdm(range(n_episodes), desc="Training")
    for episode in pbar:
        obs_dict, _ = env.reset()

        # Get observations for each agent
        obs_all = [
            get_normalized_obs(env, i) for i in range(n_agents)
        ]

        episode_reward = [0.0 for _ in range(n_agents)]

        for step in range(max_steps):
            # Select actions
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(
                    obs_all[i], obs_all, epsilon, use_centralized=True
                )
                actions.append(action)

            # Create action tuple for environment (CliffWalkEnv expects tuple, not dict)
            action_tuple = (actions[0], actions[1])

            # Step environment
            next_obs_raw, rewards_tuple, terminated, truncated, info = env.step(action_tuple)
            done = terminated or truncated

            # Get next observations
            next_obs_all = [
                get_normalized_obs(env, i) for i in range(n_agents)
            ]

            # Get rewards (CliffWalkEnv returns tuple (r1, r2) when return_joint_reward=False)
            rewards = [rewards_tuple[0], rewards_tuple[1]]

            # Store transition
            replay_buffer.push(obs_all, actions, rewards, next_obs_all, done)

            # Update episode rewards
            for i in range(n_agents):
                episode_reward[i] += rewards[i]

            obs_all = next_obs_all
            total_steps += 1

            # Update agents
            if len(replay_buffer) >= batch_size and total_steps % update_freq == 0:
                batch = replay_buffer.sample(batch_size)
                for i, agent in enumerate(agents):
                    agent.update(batch, [agents[j] for j in range(n_agents) if j != i])

            # Soft update targets
            if total_steps % target_update_freq == 0:
                for agent in agents:
                    agent.soft_update_target()

            if done:
                break

        # Record metrics
        for i in range(n_agents):
            episode_rewards[i].append(episode_reward[i])
        episode_lengths.append(step + 1)

        # Update running averages
        recent_rewards_0.append(episode_reward[0])
        recent_rewards_1.append(episode_reward[1])
        recent_lengths.append(step + 1)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update tqdm progress bar
        pbar.set_postfix({
            'R0': f'{np.mean(recent_rewards_0):.2f}',
            'R1': f'{np.mean(recent_rewards_1):.2f}',
            'Len': f'{np.mean(recent_lengths):.1f}',
            'Eps': f'{epsilon:.3f}'
        })

    return agents, episode_rewards, episode_lengths


def visualize_q_values(agents, output_dir, exp_name, device):
    """Visualize learned Q-values (max over actions) on the grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid_size = 6
    cliff_cells = [(1, 0), (2, 0), (3, 0), (4, 0), (2, 2), (2, 3), (3, 2), (3, 3)]
    agent1_goal = (0, 0)
    agent2_goal = (5, 0)
    agent1_start = (4, 2)
    agent2_start = (1, 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for agent_idx, agent in enumerate(agents):
        ax = axes[agent_idx]
        fixed_pos = agent2_start if agent_idx == 0 else agent1_start
        my_goal = agent1_goal if agent_idx == 0 else agent2_goal

        values = np.zeros((grid_size, grid_size))
        for r in range(grid_size):
            for c in range(grid_size):
                # Create observation for both agents (normalized)
                if agent_idx == 0:
                    # Agent 1's position varies, Agent 2 fixed
                    agent1_obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0
                    agent2_obs = np.array([fixed_pos[0], fixed_pos[1], r, c], dtype=np.float32) / 5.0
                else:
                    # Agent 2's position varies, Agent 1 fixed
                    agent1_obs = np.array([fixed_pos[0], fixed_pos[1], r, c], dtype=np.float32) / 5.0
                    agent2_obs = np.array([r, c, fixed_pos[0], fixed_pos[1]], dtype=np.float32) / 5.0

                # Create global observation for centralized Q-network
                global_obs = np.concatenate([agent1_obs, agent2_obs])
                global_obs_tensor = torch.FloatTensor(global_obs).unsqueeze(0).to(device)

                with torch.no_grad():
                    q_values = agent.q_network(global_obs_tensor)  # [1, n_actions]
                    # Use max Q-value as state value
                    values[r, c] = q_values.max(dim=1)[0].item()

        im = ax.imshow(values, cmap='RdYlGn', origin='upper')

        # Add value annotations
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) not in cliff_cells:
                    color = 'white' if values[r, c] < (values.min() + values.max()) / 2 else 'black'
                    ax.text(c, r, f'{values[r, c]:.1f}', ha='center', va='center',
                           fontsize=8, color=color, fontweight='bold')

        # Mark cliff cells
        for (cr, cc) in cliff_cells:
            ax.add_patch(plt.Rectangle((cc - 0.5, cr - 0.5), 1, 1,
                                       fill=True, facecolor='black', edgecolor='white', linewidth=2))

        # Mark goals
        ax.plot(agent1_goal[1], agent1_goal[0], 'r^', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.plot(agent2_goal[1], agent2_goal[0], 'bs', markersize=15, markeredgecolor='white', markeredgewidth=2)
        ax.plot(my_goal[1], my_goal[0], 'y*', markersize=20, markeredgecolor='black', markeredgewidth=1)

        title = f'Agent {agent_idx+1} Q-Value (Goal at {my_goal})'
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'MA-DQN (Centralized Critic) - Learned Q-Values\n{exp_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "value_function.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Value function saved to {output_dir / 'value_function.png'}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train MA-DQN on CliffWalk")

    # Environment
    parser.add_argument("--horizon", type=int, default=100,
                        help="Episode length")
    parser.add_argument("--reward_scale", type=float, default=1.0,
                        help="Reward scaling")
    parser.add_argument("--enable_collision", action="store_true",
                        help="Enable collision dynamics")

    # Training
    parser.add_argument("--n_episodes", type=int, default=2000,
                        help="Number of episodes")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=100000,
                        help="Replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft update coefficient")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Initial exploration rate")
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                        help="Final exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                        help="Epsilon decay rate")

    # Model
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 64],
                        help="Hidden layer sizes")

    # Logging
    parser.add_argument("--local_dir", type=str, default="results/madqn_cliffwalk",
                        help="Directory to save results")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")

    # Device
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA")

    # Checkpoint loading
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to load (for visualization or continued training)")

    return parser.parse_args()


def load_agents_from_checkpoint(checkpoint_path, hidden_sizes, device):
    """Load trained agents from a checkpoint directory."""
    checkpoint_path = Path(checkpoint_path)

    n_agents = 2
    obs_dim = 4  # (r1, c1, r2, c2) normalized
    n_actions = 4  # up, down, left, right

    agents = [
        MADQNAgent(
            agent_id=i,
            obs_dim_per_agent=obs_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            hidden_sizes=hidden_sizes,
            device=device
        )
        for i in range(n_agents)
    ]

    # Load Q-network and policy weights
    for i, agent in enumerate(agents):
        q_network_path = checkpoint_path / f"agent_{i}_q_network.pt"
        policy_path = checkpoint_path / f"agent_{i}_policy.pt"

        if q_network_path.exists():
            agent.q_network.load_state_dict(torch.load(q_network_path, map_location=device))
            agent.target_q_network.load_state_dict(agent.q_network.state_dict())
            print(f"  Loaded Q-network for agent {i} from {q_network_path}")
        else:
            print(f"  WARNING: Q-network not found at {q_network_path}")

        if policy_path.exists():
            agent.local_policy.load_state_dict(torch.load(policy_path, map_location=device))
            print(f"  Loaded policy for agent {i} from {policy_path}")
        else:
            print(f"  WARNING: Policy not found at {policy_path}")

    return agents


def main():
    args = parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # Experiment name
    exp_name = args.exp_name or "MADQN_CliffWalk"

    # Determine save path
    if args.load_checkpoint:
        # Use checkpoint path as save path for visualizations
        save_path = Path(args.load_checkpoint).resolve()
        if not save_path.exists():
            print(f"ERROR: Checkpoint path does not exist: {save_path}")
            return
    else:
        # Create save path (same style as MAPPO)
        save_path = Path(args.local_dir).resolve() / exp_name
        save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    if args.load_checkpoint:
        print("Loading MA-DQN from Checkpoint")
    else:
        print("Starting MA-DQN (Centralized Critic) Training")
    print("=" * 70)
    print(f"Environment: Cliff Walk")
    print(f"  Grid size: (6, 6)")
    print(f"  Horizon: {args.horizon}")
    print(f"  Reward scale: {args.reward_scale}")
    print(f"  Collision dynamics: {args.enable_collision}")
    print()
    if args.load_checkpoint:
        print(f"Checkpoint: {args.load_checkpoint}")
        print(f"  Episodes to train: {args.n_episodes} (0 = visualization only)")
    else:
        print(f"Training:")
        print(f"  Episodes: {args.n_episodes}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Buffer size: {args.buffer_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Gamma: {args.gamma}")
        print(f"  Tau: {args.tau}")
        print(f"  Epsilon: {args.epsilon_start} -> {args.epsilon_end}")
    print(f"  Device: {device}")
    print()
    print(f"Model:")
    print(f"  Hidden sizes: {args.hidden_sizes}")
    print()
    print(f"Output: {save_path}")
    print("=" * 70)
    print("✓ Centralized Training: Q_i(o_1, o_2, a_i) sees all observations")
    print("✓ Decentralized Execution: π_i(o_i) uses only local observations")
    print("✓ Individual Rewards: Each agent optimizes its own reward")
    print("=" * 70)

    # Create environment
    env = CliffWalkEnv(
        grid_size=(6, 6),
        horizon=args.horizon,
        reward_scale=args.reward_scale,
        return_joint_reward=False,
        enable_collision=args.enable_collision
    )

    # Load from checkpoint or train from scratch
    if args.load_checkpoint:
        print(f"\nLoading agents from checkpoint: {args.load_checkpoint}")
        agents = load_agents_from_checkpoint(args.load_checkpoint, args.hidden_sizes, device)
        rewards = [[], []]
        lengths = []

        # Optionally continue training
        if args.n_episodes > 0:
            print(f"\nContinuing training for {args.n_episodes} episodes...")
            agents, rewards, lengths = train_madqn(
                env,
                n_episodes=args.n_episodes,
                max_steps=args.horizon,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                lr=args.lr,
                gamma=args.gamma,
                tau=args.tau,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay=args.epsilon_decay,
                hidden_sizes=args.hidden_sizes,
                device=device
            )

            # Save updated models
            for i, agent in enumerate(agents):
                torch.save(agent.q_network.state_dict(), save_path / f"agent_{i}_q_network.pt")
                torch.save(agent.local_policy.state_dict(), save_path / f"agent_{i}_policy.pt")
            print(f"\nUpdated models saved to {save_path}")
    else:
        # Train from scratch
        agents, rewards, lengths = train_madqn(
            env,
            n_episodes=args.n_episodes,
            max_steps=args.horizon,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            lr=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            hidden_sizes=args.hidden_sizes,
            device=device
        )

        # Save models
        for i, agent in enumerate(agents):
            torch.save(agent.q_network.state_dict(), save_path / f"agent_{i}_q_network.pt")
            torch.save(agent.local_policy.state_dict(), save_path / f"agent_{i}_policy.pt")

        # Save training data
        np.save(save_path / "episode_rewards.npy", np.array(rewards))
        np.save(save_path / "episode_lengths.npy", np.array(lengths))

        # Save config
        config = {
            "horizon": args.horizon,
            "reward_scale": args.reward_scale,
            "enable_collision": args.enable_collision,
            "n_episodes": args.n_episodes,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "lr": args.lr,
            "gamma": args.gamma,
            "tau": args.tau,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "hidden_sizes": args.hidden_sizes,
        }
        import json
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nResults saved to {save_path}")

    # Plot learning curves (only if we have training data)
    if len(rewards[0]) > 0:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            window = min(50, len(rewards[0]) // 10) if len(rewards[0]) > 10 else 1

            # Plot 1: Individual agent rewards (smoothed)
            for i in range(2):
                if len(rewards[i]) > window:
                    smoothed = np.convolve(rewards[i], np.ones(window)/window, mode='valid')
                    axes[0, 0].plot(smoothed, label=f'Agent {i+1}', alpha=0.8)
                else:
                    axes[0, 0].plot(rewards[i], label=f'Agent {i+1}', alpha=0.8)
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title(f'Episode Rewards (smoothed, window={window})')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Episode lengths (smoothed)
            if len(lengths) > window:
                smoothed_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(smoothed_lengths, color='green', alpha=0.8)
            else:
                axes[0, 1].plot(lengths, color='green', alpha=0.8)
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Length')
            axes[0, 1].set_title(f'Episode Lengths (smoothed, window={window})')
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Combined reward (sum of both agents)
            combined_rewards = [rewards[0][i] + rewards[1][i] for i in range(len(rewards[0]))]
            if len(combined_rewards) > window:
                smoothed_combined = np.convolve(combined_rewards, np.ones(window)/window, mode='valid')
                axes[1, 0].plot(smoothed_combined, color='purple', alpha=0.8)
            else:
                axes[1, 0].plot(combined_rewards, color='purple', alpha=0.8)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Combined Reward')
            axes[1, 0].set_title('Combined Episode Reward (Agent 1 + Agent 2)')
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Raw rewards (for comparison)
            axes[1, 1].plot(rewards[0], label='Agent 1', alpha=0.3, linewidth=0.5)
            axes[1, 1].plot(rewards[1], label='Agent 2', alpha=0.3, linewidth=0.5)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].set_title('Raw Episode Rewards')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle(f'MA-DQN Training Results - {exp_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path / "learning_curves.png", dpi=150, bbox_inches='tight')
            print(f"Learning curves saved to {save_path / 'learning_curves.png'}")
            plt.close()

        except Exception as e:
            print(f"Could not plot learning curves: {e}")

    # Visualize learned Q-values on the grid
    try:
        visualize_q_values(agents, save_path, exp_name, device)
    except Exception as e:
        print(f"Could not plot Q-values: {e}")

    # Simulate and visualize trajectory
    try:
        print("\nSimulating trajectory with trained agents...")

        # Create policy function for trajectory simulation
        def madqn_policy(obs, env):
            obs_all = [get_normalized_obs(env, i) for i in range(2)]
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(
                    obs_all[i], obs_all, epsilon=0.0, use_centralized=True
                )
                actions.append(action)
            return actions

        trajectory = simulate_trajectory(env, madqn_policy, max_steps=args.horizon)
        visualize_trajectory(
            trajectory,
            save_path=str(save_path / "trajectory.png"),
            title=f"MA-DQN Trajectory - {exp_name}"
        )
    except Exception as e:
        print(f"Could not simulate/visualize trajectory: {e}")

    # Print summary statistics
    print("\n" + "=" * 70)
    if args.load_checkpoint and args.n_episodes == 0:
        print("Visualization Complete!")
    else:
        print("Training Complete!")
    print("=" * 70)
    if len(rewards[0]) > 0:
        print(f"\nFinal Statistics (last 100 episodes):")
        last_n = min(100, len(rewards[0]))
        print(f"  Agent 1 Avg Reward: {np.mean(rewards[0][-last_n:]):.2f}")
        print(f"  Agent 2 Avg Reward: {np.mean(rewards[1][-last_n:]):.2f}")
        print(f"  Avg Episode Length: {np.mean(lengths[-last_n:]):.1f}")
    print(f"\nOutput directory: {save_path}")


if __name__ == "__main__":
    main()
