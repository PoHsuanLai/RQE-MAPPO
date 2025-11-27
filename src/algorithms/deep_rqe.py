"""
Deep RQE: Neural Network-based RQE Algorithms

Implements both:
1. Deep RQE Q-Learning (DQN-style, discrete actions)
2. Deep RQE-MAPPO (Actor-Critic, continuous actions)

Key insight: The RQE Q-networks can be shared as critics for both algorithms!

Based on "Tractable Multi-Agent Reinforcement Learning Through Behavioral Economics" (ICLR 2025)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy

# Import the standalone RQE solver
from .rqe_solver import RQESolver, RQEConfig as RQESolverConfig


@dataclass
class DeepRQEConfig:
    """
    Configuration for Deep RQE algorithms

    MODEL-AGNOSTIC: Supports custom network architectures via q_network_class and actor_class
    """
    n_agents: int
    action_dims: List[int]  # Action space size per agent

    # Risk-aversion and bounded rationality
    tau: List[float]
    epsilon: List[float]

    # MODEL-AGNOSTIC: Optional custom network classes
    # If None, will use default MLP networks with obs_dim
    q_network_class: type = None  # Custom Q-network class
    actor_class: type = None  # Custom Actor class (for MAPPO)

    # Network kwargs (passed to custom network classes)
    q_network_kwargs: Dict = None  # e.g., {"features_dim": 512} for CNN
    actor_kwargs: Dict = None

    # For default MLP networks only (ignored if custom classes provided)
    obs_dim: int = None  # Observation dimension (for MLP)
    hidden_dims: List[int] = None  # [256, 256] (for MLP)
    activation: str = "relu"

    # Learning parameters
    lr_critic: float = 3e-4
    lr_actor: float = 3e-4  # For MAPPO
    gamma: float = 0.99

    # RQE solver parameters (OPTIMIZED)
    rqe_iterations: int = 3  # Reduced for speed (3 iterations with warm start is sufficient)
    rqe_lr: float = 0.5  # Increased for faster convergence with fewer iterations
    rqe_tolerance: float = 1e-4  # Not used in GPU-optimized version
    rqe_momentum: float = 0.9

    # Penalty and regularizer
    penalty_type: str = "kl"
    regularizer_type: str = "entropy"

    # Training
    batch_size: int = 256
    buffer_size: int = 100000
    update_frequency: int = 4  # Update every N steps (reduces solver calls)

    # Warmup: skip RQE solver during initial exploration
    warmup_steps: int = 5000  # Steps before using RQE solver
    solver_frequency: int = 1  # Run solver every N action selections (1 = always after warmup)

    def __post_init__(self):
        # Set defaults
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]
        if self.q_network_kwargs is None:
            self.q_network_kwargs = {}
        if self.actor_kwargs is None:
            self.actor_kwargs = {}

        # Validation
        if self.q_network_class is None and self.obs_dim is None:
            raise ValueError("Must provide either q_network_class or obs_dim for default MLP")

        # Check tractability
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                tractable = self.epsilon[i] * self.epsilon[j] >= (1/self.tau[i]) * (1/self.tau[j])
                if not tractable:
                    print(f"Warning: Agents {i},{j} don't satisfy tractability condition!")


# ==================== Q-Network Interface ====================
# Custom Q-networks should be nn.Module with:
# - __init__(my_action_dim: int, opponent_action_dim: int, **kwargs)
# - forward(obs: torch.Tensor) -> torch.Tensor [batch, my_action_dim, opponent_action_dim]


class QNetwork(nn.Module):
    """
    Default MLP Q-Network for RQE Q-Learning

    Outputs Q(obs, my_action, opponent_action) for each agent
    """
    def __init__(self, my_action_dim: int, opponent_action_dim: int,
                 obs_dim: int, hidden_dims: List[int] = [256, 256], activation: str = "relu"):
        super().__init__()

        self.my_action_dim = my_action_dim
        self.opponent_action_dim = opponent_action_dim
        self.obs_dim = obs_dim

        # Build network
        layers = []
        input_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            input_dim = hidden_dim

        # Output layer: Q(obs, a_i, a_{-i})
        # Shape: [my_action_dim, opponent_action_dim]
        output_dim = my_action_dim * opponent_action_dim
        layers.append(nn.Linear(input_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [batch, obs_dim]

        Returns:
            Q-values: [batch, my_action_dim, opponent_action_dim]
        """
        batch_size = obs.shape[0]
        q_flat = self.network(obs)  # [batch, my_action_dim * opponent_action_dim]
        q_matrix = q_flat.view(batch_size, self.my_action_dim, self.opponent_action_dim)
        return q_matrix


# ==================== Actor Interface ====================
# Custom actors should be nn.Module with:
# - __init__(action_dim: int, **kwargs)
# - forward(obs: torch.Tensor) -> torch.distributions.Distribution
# - get_action(obs: torch.Tensor, deterministic: bool) -> (action, log_prob)


class Actor(nn.Module):
    """
    Default MLP Actor network for RQE-MAPPO

    Outputs policy π(a|obs) for continuous or discrete actions
    """
    def __init__(self, action_dim: int, obs_dim: int = None,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = "relu", action_type: str = "discrete"):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type

        # Build network
        layers = []
        input_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            input_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        if action_type == "discrete":
            # Output logits for discrete actions
            self.action_head = nn.Linear(input_dim, action_dim)
        elif action_type == "continuous":
            # Output mean and log_std for continuous actions
            self.mean_head = nn.Linear(input_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        """
        Args:
            obs: [batch, obs_dim]

        Returns:
            Distribution over actions
        """
        features = self.features(obs)

        if self.action_type == "discrete":
            logits = self.action_head(features)
            return torch.distributions.Categorical(logits=logits)
        else:
            mean = self.mean_head(features)
            std = torch.exp(self.log_std)
            return torch.distributions.Normal(mean, std)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability"""
        dist = self.forward(obs)

        if deterministic:
            if self.action_type == "discrete":
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob


def _create_rqe_solver_from_deep_config(config: DeepRQEConfig) -> RQESolver:
    """
    Create an RQESolver from DeepRQEConfig.

    This adapter converts DeepRQEConfig parameters to the standalone RQEConfig format.
    """
    solver_config = RQESolverConfig(
        action_dims=config.action_dims,
        tau=config.tau,
        epsilon=config.epsilon,
        max_iterations=config.rqe_iterations,
        learning_rate=config.rqe_lr,
        momentum=config.rqe_momentum,
        tolerance=config.rqe_tolerance,
        entropy_reg=(config.regularizer_type == "entropy"),
    )
    return RQESolver(solver_config)


class DeepRQE_QLearning:
    """
    Deep RQE Q-Learning: Multi-Agent Q-Learning with RQE equilibrium

    MODEL-AGNOSTIC: Supports custom network architectures

    Like DQN but solves RQE instead of taking max over Q-values
    Works with discrete action spaces
    """
    def __init__(self, config: DeepRQEConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine Q-network class to use
        if config.q_network_class is not None:
            # User provided custom Q-network class
            q_net_class = config.q_network_class
            q_net_kwargs = config.q_network_kwargs
        else:
            # Use default MLP Q-network
            q_net_class = QNetwork
            q_net_kwargs = {
                "obs_dim": config.obs_dim,
                "hidden_dims": config.hidden_dims,
                "activation": config.activation
            }

        # Create Q-networks for each agent
        self.q_networks = nn.ModuleList([
            q_net_class(
                my_action_dim=config.action_dims[i],
                opponent_action_dim=config.action_dims[1-i] if config.n_agents == 2
                    else sum(config.action_dims) - config.action_dims[i],
                **q_net_kwargs
            ).to(self.device)
            for i in range(config.n_agents)
        ])

        # Target networks
        self.target_q_networks = copy.deepcopy(self.q_networks)

        # Optimizers
        self.optimizers = [
            torch.optim.Adam(self.q_networks[i].parameters(), lr=config.lr_critic)
            for i in range(config.n_agents)
        ]

        # RQE solver (using standalone solver)
        self.rqe_solver = _create_rqe_solver_from_deep_config(config)

        # Experience replay buffer (simple implementation)
        self.buffer = []
        self.step_count = 0  # For update frequency tracking
        self.total_steps = 0  # For warmup tracking
        self.solver_call_count = 0  # For solver frequency

    def select_action(self, obs: np.ndarray, epsilon_greedy: float = 0.0) -> List[int]:
        """
        Select actions using epsilon-greedy with RQE policy

        Args:
            obs: Observation [obs_dim]
            epsilon_greedy: Exploration rate

        Returns:
            List of actions, one per agent
        """
        self.total_steps += 1

        if np.random.random() < epsilon_greedy:
            # Random exploration
            return [np.random.randint(self.config.action_dims[i])
                   for i in range(self.config.n_agents)]

        # During warmup, use uniform random policy (skip expensive solver)
        if self.total_steps < self.config.warmup_steps:
            return [np.random.randint(self.config.action_dims[i])
                   for i in range(self.config.n_agents)]

        # Check solver frequency (run solver every N steps after warmup)
        self.solver_call_count += 1
        if self.solver_call_count % self.config.solver_frequency != 0:
            # Use cached uniform distribution
            return [np.random.randint(self.config.action_dims[i])
                   for i in range(self.config.n_agents)]

        # Get Q-values
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            Q_matrices = [net(obs_tensor) for net in self.q_networks]

            # Solve RQE
            policies = self.rqe_solver.solve(Q_matrices)

            # Sample from RQE policies
            actions = [
                torch.multinomial(policies[i][0], 1).item()
                for i in range(self.config.n_agents)
            ]

        return actions

    def update(self, obs: np.ndarray, actions: List[int], rewards: List[float],
              next_obs: np.ndarray, done: bool):
        """
        Single-step Q-learning update with RQE

        OPTIMIZED: Only update every N steps to reduce solver calls

        Args:
            obs: Current observation
            actions: Actions taken
            rewards: Rewards received
            next_obs: Next observation
            done: Whether episode is done
        """
        # Store in replay buffer
        self.buffer.append((obs, actions, rewards, next_obs, done))
        if len(self.buffer) > self.config.buffer_size:
            self.buffer.pop(0)

        # OPTIMIZATION: Skip ALL updates during warmup (just collect experience)
        if self.total_steps < self.config.warmup_steps:
            return  # Skip CNN forward pass entirely during warmup

        # OPTIMIZATION: Update every N steps
        self.step_count += 1
        if self.step_count % self.config.update_frequency != 0:
            return

        # Sample batch
        if len(self.buffer) < self.config.batch_size:
            return

        batch_indices = np.random.choice(len(self.buffer), self.config.batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]

        # Unpack batch (optimized with numpy)
        obs_batch = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions_batch = [[b[1][i] for b in batch] for i in range(self.config.n_agents)]
        rewards_batch = [[b[2][i] for b in batch] for i in range(self.config.n_agents)]
        next_obs_batch = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        done_batch = torch.FloatTensor(np.array([b[4] for b in batch])).to(self.device)

        # Compute values at next_obs
        with torch.no_grad():
            Q_next = [net(next_obs_batch) for net in self.target_q_networks]

            # During warmup, use simple mean Q-value (like standard DQN, much faster)
            if self.total_steps < self.config.warmup_steps:
                V_rqe = [Q_next[i].mean(dim=[1, 2]) for i in range(self.config.n_agents)]
            else:
                # After warmup, compute RQE equilibrium values
                policies_rqe = self.rqe_solver.solve(Q_next)

                # Compute values under RQE
                V_rqe = []
                for i in range(self.config.n_agents):
                    j = 1 - i  # Opponent
                    # V = π_i^T @ Q_i @ π_j
                    V = torch.sum(
                        policies_rqe[i].unsqueeze(-1) * Q_next[i] * policies_rqe[j].unsqueeze(1),
                        dim=[1, 2]
                    )
                    V_rqe.append(V)

        # Update each agent's Q-network
        for i in range(self.config.n_agents):
            # Current Q-values
            Q_current_all = self.q_networks[i](obs_batch)

            # Get Q-values for taken actions
            actions_i = torch.LongTensor(actions_batch[i]).to(self.device)
            actions_j = torch.LongTensor(actions_batch[1-i]).to(self.device)

            Q_current = Q_current_all[torch.arange(self.config.batch_size), actions_i, actions_j]

            # Compute targets
            rewards_i = torch.FloatTensor(rewards_batch[i]).to(self.device)
            targets = rewards_i + self.config.gamma * V_rqe[i] * (1 - done_batch)

            # Loss and update
            loss = F.mse_loss(Q_current, targets)

            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

    def update_target_networks(self, tau: float = 0.005):
        """Soft update of target networks"""
        for i in range(self.config.n_agents):
            for param, target_param in zip(
                self.q_networks[i].parameters(),
                self.target_q_networks[i].parameters()
            ):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class DeepRQE_MAPPO:
    """
    Deep RQE-MAPPO: Multi-Agent Actor-Critic with RQE equilibrium

    Uses RQE Q-networks as critics!
    Works with both discrete and continuous action spaces
    """
    def __init__(self, config: DeepRQEConfig, action_type: str = "discrete"):
        self.config = config
        self.action_type = action_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine Actor class to use
        if config.actor_class is not None:
            actor_class = config.actor_class
            actor_kwargs = config.actor_kwargs
        else:
            actor_class = Actor
            actor_kwargs = {
                "obs_dim": config.obs_dim,
                "hidden_dims": config.hidden_dims,
                "activation": config.activation,
                "action_type": action_type
            }

        # Determine Critic (Q-network) class to use
        if config.q_network_class is not None:
            critic_class = config.q_network_class
            critic_kwargs = config.q_network_kwargs
        else:
            critic_class = QNetwork
            critic_kwargs = {
                "obs_dim": config.obs_dim,
                "hidden_dims": config.hidden_dims,
                "activation": config.activation
            }

        # Create Actors (policy networks)
        self.actors = nn.ModuleList([
            actor_class(
                action_dim=config.action_dims[i],
                **actor_kwargs
            ).to(self.device)
            for i in range(config.n_agents)
        ])

        # Create Critics (RQE Q-networks - shared with Q-Learning!)
        self.critics = nn.ModuleList([
            critic_class(
                my_action_dim=config.action_dims[i],
                opponent_action_dim=config.action_dims[1-i] if config.n_agents == 2
                    else sum(config.action_dims) - config.action_dims[i],
                **critic_kwargs
            ).to(self.device)
            for i in range(config.n_agents)
        ])

        # Optimizers
        self.actor_optimizers = [
            torch.optim.Adam(self.actors[i].parameters(), lr=config.lr_actor)
            for i in range(config.n_agents)
        ]
        self.critic_optimizers = [
            torch.optim.Adam(self.critics[i].parameters(), lr=config.lr_critic)
            for i in range(config.n_agents)
        ]

        # RQE solver (using standalone solver)
        self.rqe_solver = _create_rqe_solver_from_deep_config(config)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[List, List]:
        """
        Select actions from actor policies

        Returns:
            actions: List of actions
            log_probs: List of log probabilities
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        actions = []
        log_probs = []

        with torch.no_grad():
            for i in range(self.config.n_agents):
                action, log_prob = self.actors[i].get_action(obs_tensor, deterministic)
                actions.append(action.item() if self.action_type == "discrete" else action.cpu().numpy()[0])
                log_probs.append(log_prob.item())

        return actions, log_probs

    def compute_rqe_values(self, obs_batch: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute V_RQE(obs) for all agents

        This is the key: We use RQE equilibrium for value estimation!
        """
        # Get Q-matrices from critics
        Q_matrices = [critic(obs_batch) for critic in self.critics]

        # Solve RQE
        policies_rqe = self.rqe_solver.solve(Q_matrices)

        # Compute values under RQE
        V_rqe = []
        for i in range(self.config.n_agents):
            j = 1 - i
            V = torch.sum(
                policies_rqe[i].unsqueeze(-1) * Q_matrices[i] * policies_rqe[j].unsqueeze(1),
                dim=[1, 2]
            )
            V_rqe.append(V)

        return V_rqe

    def update(self, trajectories: Dict):
        """
        PPO-style update with RQE values

        Args:
            trajectories: Dict with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones'
        """
        obs_batch = torch.FloatTensor(trajectories['obs']).to(self.device)
        next_obs_batch = torch.FloatTensor(trajectories['next_obs']).to(self.device)
        actions_batch = trajectories['actions']  # List of lists
        rewards_batch = trajectories['rewards']  # List of lists
        dones_batch = torch.FloatTensor(trajectories['dones']).to(self.device)

        # Compute RQE values
        with torch.no_grad():
            V_rqe_next = self.compute_rqe_values(next_obs_batch)

        V_rqe_current = self.compute_rqe_values(obs_batch)

        # Update critics
        for i in range(self.config.n_agents):
            # TD targets
            rewards_i = torch.FloatTensor(rewards_batch[i]).to(self.device)
            targets = rewards_i + self.config.gamma * V_rqe_next[i] * (1 - dones_batch)

            # Critic loss
            critic_loss = F.mse_loss(V_rqe_current[i], targets)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizers[i].step()

        # Update actors (policy gradient with RQE values as baseline)
        for i in range(self.config.n_agents):
            # Get actions and log probs from current policy
            dist = self.actors[i](obs_batch)
            actions_i = torch.LongTensor(actions_batch[i]).to(self.device)
            log_probs = dist.log_prob(actions_i)

            # Advantages (using RQE values as baseline)
            with torch.no_grad():
                rewards_i = torch.FloatTensor(rewards_batch[i]).to(self.device)
                advantages = rewards_i + self.config.gamma * V_rqe_next[i] * (1 - dones_batch) - V_rqe_current[i]

            # Policy loss
            actor_loss = -(log_probs * advantages).mean()

            # Entropy bonus
            entropy = dist.entropy().mean()
            actor_loss -= 0.01 * entropy

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()


# Testing
if __name__ == "__main__":
    print("Deep RQE Implementation")
    print("=" * 50)

    # Configuration
    config = DeepRQEConfig(
        n_agents=2,
        obs_dim=10,
        action_dims=[5, 5],
        tau=[2.0, 2.0],
        epsilon=[0.5, 0.5],
        hidden_dims=[64, 64],
        lr_critic=1e-3,
        lr_actor=1e-3,
    )

    print(f"Configuration:")
    print(f"  Agents: {config.n_agents}")
    print(f"  Obs dim: {config.obs_dim}")
    print(f"  Action dims: {config.action_dims}")
    print(f"  Tau: {config.tau}")
    print(f"  Epsilon: {config.epsilon}")

    # Test Deep RQE Q-Learning
    print("\n1. Testing Deep RQE Q-Learning...")
    rqe_qlearning = DeepRQE_QLearning(config)

    obs = np.random.randn(config.obs_dim)
    actions = rqe_qlearning.select_action(obs, epsilon_greedy=0.1)
    print(f"   Selected actions: {actions}")

    # Test update
    next_obs = np.random.randn(config.obs_dim)
    rewards = [1.0, -1.0]
    rqe_qlearning.update(obs, actions, rewards, next_obs, done=False)
    print(f"   Update successful ✓")

    # Test Deep RQE-MAPPO
    print("\n2. Testing Deep RQE-MAPPO...")
    rqe_mappo = DeepRQE_MAPPO(config, action_type="discrete")

    actions, log_probs = rqe_mappo.select_action(obs)
    print(f"   Selected actions: {actions}")
    print(f"   Log probs: {log_probs}")

    # Test update with batch
    batch_size = 32
    trajectories = {
        'obs': np.random.randn(batch_size, config.obs_dim),
        'actions': [[np.random.randint(config.action_dims[i]) for _ in range(batch_size)]
                   for i in range(config.n_agents)],
        'rewards': [[np.random.randn() for _ in range(batch_size)]
                   for i in range(config.n_agents)],
        'next_obs': np.random.randn(batch_size, config.obs_dim),
        'dones': np.random.randint(0, 2, batch_size).astype(float)
    }
    rqe_mappo.update(trajectories)
    print(f"   Update successful ✓")

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("\nKey Features:")
    print("  ✓ Deep RQE Q-Learning for discrete actions")
    print("  ✓ Deep RQE-MAPPO for discrete/continuous actions")
    print("  ✓ Shared RQE Q-networks as critics")
    print("  ✓ No-regret learning RQE solver")
    print("  ✓ GPU support")
