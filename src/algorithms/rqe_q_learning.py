"""
RQE Q-Learning: Multi-Agent Q-Learning with Risk-Averse Quantal Response Equilibrium

Based on "Tractable Multi-Agent Reinforcement Learning Through Behavioral Economics" (ICLR 2025)
Combines risk-aversion (tau) and bounded rationality (epsilon) in a game-theoretic framework.

Key idea: Instead of computing Nash equilibrium like Nash Q-learning, compute RQE which is:
1. More tractable (guaranteed convergence via no-regret learning)
2. More realistic (models human-like risk-aversion and bounded rationality)
3. Provably efficient under tractability condition: ε_i * ε_j >= (1/τ_i) * (1/τ_j)

Uses the standalone RQE solver from rqe_solver.py for equilibrium computation.
"""

import numpy as np
import torch
from typing import List
from dataclasses import dataclass

# Import the standalone RQE solver
from .rqe_solver import RQESolver, RQEConfig as RQESolverConfig


@dataclass
class RQEQLearningConfig:
    """Configuration for RQE Q-Learning"""
    n_agents: int
    n_states: int
    n_actions: List[int]  # Actions per agent

    # Risk-aversion parameters (higher tau = less risk-averse)
    tau: List[float]  # One per agent

    # Bounded rationality parameters (higher = more mistakes)
    epsilon: List[float]  # One per agent

    # Learning parameters
    alpha: float = 0.1  # Learning rate
    gamma: float = 0.99  # Discount factor

    # RQE solver parameters
    rqe_iterations: int = 50  # Solver iterations
    rqe_lr: float = 0.3  # Learning rate for RQE solver
    rqe_momentum: float = 0.9  # Momentum for faster convergence

    # Warmup: skip solver during initial random exploration
    warmup_steps: int = 1000  # Steps before using RQE solver
    use_solver_for_update: bool = True  # Use solver for Q-value updates

    def __post_init__(self):
        """Validate configuration"""
        assert len(self.tau) == self.n_agents, "Need tau for each agent"
        assert len(self.epsilon) == self.n_agents, "Need epsilon for each agent"
        assert len(self.n_actions) == self.n_agents, "Need n_actions for each agent"

        # Check tractability condition for all pairs
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                tractable = self.epsilon[i] * self.epsilon[j] >= (1/self.tau[i]) * (1/self.tau[j])
                if not tractable:
                    print(f"Warning: Agents {i},{j} don't satisfy tractability condition!")
                    print(f"  ε{i}*ε{j} = {self.epsilon[i] * self.epsilon[j]:.4f}")
                    print(f"  1/τ{i} * 1/τ{j} = {(1/self.tau[i]) * (1/self.tau[j]):.4f}")


# Backwards compatibility alias
RQEConfig = RQEQLearningConfig


class RQE_QLearning:
    """
    RQE Q-Learning for Multi-Agent Systems

    Extends standard Q-learning by computing Risk-Averse Quantal Response Equilibrium
    at each state instead of Nash equilibrium.

    Uses the standalone RQE solver for equilibrium computation.
    """

    def __init__(self, config: RQEQLearningConfig):
        self.config = config
        self.n_agents = config.n_agents
        self.n_states = config.n_states
        self.n_actions = config.n_actions

        # Initialize Q-tables
        self.Q = self._initialize_Q_tables()

        # Create the standalone RQE solver
        solver_config = RQESolverConfig(
            action_dims=config.n_actions,
            tau=config.tau,
            epsilon=config.epsilon,
            max_iterations=config.rqe_iterations,
            learning_rate=config.rqe_lr,
            momentum=config.rqe_momentum,
        )
        self.rqe_solver = RQESolver(solver_config)

        # Track learning statistics
        self.update_count = 0
        self.total_steps = 0

    def _initialize_Q_tables(self) -> List[np.ndarray]:
        """
        Initialize Q-tables for all agents

        For 2 agents: Q[i] is shape [n_states, n_actions[i], n_actions[j]]
        """
        Q_tables = []
        for i in range(self.n_agents):
            if self.n_agents == 2:
                j = 1 - i
                shape = (self.n_states, self.n_actions[i], self.n_actions[j])
            else:
                # For n-player, flatten opponent actions
                opponent_actions = [self.n_actions[j] for j in range(self.n_agents) if j != i]
                opponent_space_size = int(np.prod(opponent_actions))
                shape = (self.n_states, self.n_actions[i], opponent_space_size)

            Q_tables.append(np.zeros(shape))

        return Q_tables

    def update(self, state: int, actions: List[int], rewards: List[float],
               next_state: int, done: bool = False):
        """
        Single-step Q-learning update with RQE

        Args:
            state: Current state index
            actions: List of actions taken by each agent
            rewards: List of rewards received by each agent
            next_state: Next state index
            done: Whether episode is done
        """
        self.total_steps += 1

        # Compute RQE values for next state
        if done:
            V_next = [0.0] * self.n_agents
        else:
            # During warmup, use simple mean Q-value instead of solving RQE
            if self.total_steps < self.config.warmup_steps or not self.config.use_solver_for_update:
                V_next = self._compute_mean_values(next_state)
            else:
                V_next = self.compute_RQE_values(next_state)

        # Update Q-values for all agents
        for i in range(self.n_agents):
            if self.n_agents == 2:
                j = 1 - i
                opponent_idx = actions[j]
            else:
                opponent_actions = [actions[j] for j in range(self.n_agents) if j != i]
                opponent_idx = self._actions_to_index(opponent_actions, i)

            # Current Q-value
            current_Q = self.Q[i][state, actions[i], opponent_idx]

            # TD target
            target = rewards[i] + self.config.gamma * V_next[i]

            # Q-learning update
            self.Q[i][state, actions[i], opponent_idx] = \
                (1 - self.config.alpha) * current_Q + self.config.alpha * target

        self.update_count += 1

    def _compute_mean_values(self, state: int) -> List[float]:
        """Compute mean Q-values (used during warmup)"""
        return [self.Q[i][state].mean() for i in range(self.n_agents)]

    def compute_RQE_values(self, state: int) -> List[float]:
        """
        Solve RQE at given state to get V_i^RQE(s) for all agents

        Args:
            state: State index

        Returns:
            List of RQE values, one per agent
        """
        # Extract Q-matrices for this state and convert to torch tensors
        Q_matrices = [
            torch.FloatTensor(self.Q[i][state]).unsqueeze(0)  # Add batch dim
            for i in range(self.n_agents)
        ]

        # Solve RQE to get equilibrium policies
        equilibrium_policies = self.rqe_solver.solve(Q_matrices)

        # Compute expected values under equilibrium
        V = []
        for i in range(self.n_agents):
            if self.n_agents == 2:
                j = 1 - i
                # V = πi^T Q πj
                pi_i = equilibrium_policies[i][0]  # Remove batch dim
                pi_j = equilibrium_policies[j][0]
                Q_i = Q_matrices[i][0]
                V_i = (pi_i @ Q_i @ pi_j).item()
            else:
                # General case (not fully implemented)
                V_i = self.Q[i][state].mean()
            V.append(V_i)

        return V

    def get_action(self, state: int, epsilon_greedy: float = 0.0) -> List[int]:
        """
        Sample actions from current RQE policy at given state

        Args:
            state: Current state index
            epsilon_greedy: Exploration rate (0 = pure RQE policy)

        Returns:
            List of actions, one per agent
        """
        # Random exploration
        if np.random.random() < epsilon_greedy:
            return [np.random.randint(self.n_actions[i]) for i in range(self.n_agents)]

        # During warmup, use uniform random
        if self.total_steps < self.config.warmup_steps:
            return [np.random.randint(self.n_actions[i]) for i in range(self.n_agents)]

        # Extract Q-matrices for this state
        Q_matrices = [
            torch.FloatTensor(self.Q[i][state]).unsqueeze(0)
            for i in range(self.n_agents)
        ]

        # Solve RQE
        policies = self.rqe_solver.solve(Q_matrices)

        # Sample from RQE policies
        actions = []
        for i in range(self.n_agents):
            pi = policies[i][0].numpy()  # Remove batch dim
            action = np.random.choice(self.n_actions[i], p=pi)
            actions.append(action)

        return actions

    def _actions_to_index(self, opponent_actions: List[int], agent_idx: int) -> int:
        """Convert opponent action tuple to flat index"""
        opponent_action_sizes = [self.n_actions[j] for j in range(self.n_agents) if j != agent_idx]

        idx = 0
        multiplier = 1
        for i in range(len(opponent_actions) - 1, -1, -1):
            idx += opponent_actions[i] * multiplier
            multiplier *= opponent_action_sizes[i]

        return idx

    def save(self, filepath: str):
        """Save Q-tables to file"""
        np.savez(filepath,
                 Q=[self.Q[i] for i in range(self.n_agents)],
                 config=self.config.__dict__,
                 total_steps=self.total_steps)

    def load(self, filepath: str):
        """Load Q-tables from file"""
        data = np.load(filepath, allow_pickle=True)
        self.Q = list(data['Q'])
        if 'total_steps' in data:
            self.total_steps = int(data['total_steps'])


# For testing
if __name__ == "__main__":
    # Simple 2-player grid world test
    config = RQEQLearningConfig(
        n_agents=2,
        n_states=10,
        n_actions=[4, 4],
        tau=[2.0, 2.0],
        epsilon=[0.5, 0.5],
        alpha=0.1,
        gamma=0.99,
        rqe_iterations=50,
        rqe_lr=0.3,
        warmup_steps=100,
    )

    print("RQE Q-Learning Configuration:")
    print(f"  Agents: {config.n_agents}")
    print(f"  States: {config.n_states}")
    print(f"  Actions: {config.n_actions}")
    print(f"  Tau (risk-aversion): {config.tau}")
    print(f"  Epsilon (bounded rationality): {config.epsilon}")
    print(f"  Warmup steps: {config.warmup_steps}")

    # Initialize algorithm
    agent = RQE_QLearning(config)

    # Test update (during warmup - no solver)
    print("\nTesting updates during warmup...")
    for _ in range(50):
        state = np.random.randint(10)
        actions = [np.random.randint(4), np.random.randint(4)]
        rewards = [np.random.randn(), np.random.randn()]
        next_state = np.random.randint(10)
        agent.update(state, actions, rewards, next_state)

    print(f"  Updates completed: {agent.update_count}")

    # Force past warmup
    agent.total_steps = config.warmup_steps + 1

    # Test action selection with solver
    print("\nTesting action selection with RQE solver...")
    selected_actions = agent.get_action(0)
    print(f"  Selected actions: {selected_actions}")

    # Test RQE value computation
    print("\nTesting RQE value computation...")
    values = agent.compute_RQE_values(0)
    print(f"  RQE values: {values}")

    print("\nRQE Q-Learning test complete!")
