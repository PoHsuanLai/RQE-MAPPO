"""
PSRO with RQE (Policy Space Response Oracles with Risk-Averse Quantal Response Equilibrium)

Model-agnostic implementation following the same design pattern as Deep RQE.

Key Components:
1. Policy Population: Maintain a set of learned policies per agent
2. Meta-Game: Empirical game matrix from policy evaluations
3. RQE Solver: Solve for equilibrium in meta-game (tractable!)
4. Best Response Oracle: Train new policies via RL

References:
- PSRO: "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning" (Lanctot et al., 2017)
- RQE: "Tractable Equilibrium Computation in Markov Games through Risk Aversion" (Mazumdar et al., 2024)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from .rqe_solver import RQESolver, RQEConfig


@dataclass
class PSRORQEConfig:
    """
    Configuration for PSRO with RQE meta-solver

    MODEL-AGNOSTIC: Supports custom network architectures
    """
    n_agents: int

    # Risk-aversion and bounded rationality for RQE
    tau: List[float]
    epsilon: List[float]

    # RQE solver config
    rqe_iterations: int = 20
    rqe_lr: float = 0.5
    lr_schedule: str = "sqrt"

    # PSRO parameters
    psro_iterations: int = 10  # Number of PSRO iterations
    oracle_episodes: int = 1000  # Episodes to train best response
    eval_episodes: int = 100  # Episodes to evaluate policy pairs

    # MODEL-AGNOSTIC: Optional custom network classes
    policy_network_class: type = None  # Custom policy network class
    value_network_class: type = None  # Custom value network class (for oracle)

    # Network kwargs (passed to custom network classes)
    policy_kwargs: Dict = None
    value_kwargs: Dict = None

    # For default MLP networks only (ignored if custom classes provided)
    obs_dim: int = None
    action_dims: List[int] = None
    hidden_dims: Tuple[int, ...] = (128, 128)
    activation: str = "relu"

    # Best response oracle parameters
    oracle_type: str = "ppo"  # "ppo", "dqn", "a2c"
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    gamma: float = 0.99

    # PPO-specific
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95

    # DQN-specific
    dqn_batch_size: int = 64
    dqn_buffer_size: int = 10000
    dqn_update_frequency: int = 4
    dqn_target_update: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1000

    # Population management
    max_population_size: int = 20  # Max policies per agent
    initialization_policies: int = 2  # Random policies to start with

    def __post_init__(self):
        if self.policy_kwargs is None:
            self.policy_kwargs = {}
        if self.value_kwargs is None:
            self.value_kwargs = {}


class PolicyNetwork(nn.Module):
    """Default MLP policy network"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (128, 128),
                 activation: str = "relu"):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

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

        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        """
        Args:
            obs: [batch_size, obs_dim]
        Returns:
            action_probs: [batch_size, action_dim]
        """
        return self.network(obs)


class ValueNetwork(nn.Module):
    """Default MLP value network for PPO/A2C oracle"""

    def __init__(self, obs_dim: int, hidden_dims: Tuple[int, ...] = (128, 128),
                 activation: str = "relu"):
        super().__init__()

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

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        """
        Args:
            obs: [batch_size, obs_dim]
        Returns:
            value: [batch_size, 1]
        """
        return self.network(obs)


class PSRO_RQE:
    """
    Policy Space Response Oracles with RQE meta-solver

    MODEL-AGNOSTIC: Works with any policy/value network architecture

    Algorithm:
    1. Initialize population with random policies
    2. For each PSRO iteration:
        a. Evaluate all policy pairs → build empirical meta-game
        b. Solve RQE on meta-game → get meta-strategy
        c. Train best response policies against meta-strategy
        d. Add new policies to population
    3. Return final meta-strategy
    """

    def __init__(self, config: PSRORQEConfig, env_factory: Callable):
        """
        Args:
            config: PSRO-RQE configuration
            env_factory: Function that creates a new environment instance
                        (needed because we'll create multiple envs for parallel evaluation)
        """
        self.config = config
        self.env_factory = env_factory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy populations (list of neural networks per agent)
        self.policy_populations = [[] for _ in range(config.n_agents)]

        # Meta-game payoff matrices (updated each iteration)
        # meta_payoffs[i][j][k] = payoff for agent i when using policy j against opponent using policy k
        self.meta_payoffs = [None for _ in range(config.n_agents)]

        # RQE solver for meta-game (will be recreated when action_dims change)
        # Initially set action_dims based on initialization_policies
        initial_action_dims = [config.initialization_policies for _ in range(config.n_agents)]
        rqe_config = RQEConfig(
            action_dims=initial_action_dims,
            tau=config.tau,
            epsilon=config.epsilon,
            max_iterations=config.rqe_iterations,
            learning_rate=config.rqe_lr,
        )
        self.rqe_solver = RQESolver(rqe_config)

        # Current meta-strategy (distribution over policies)
        self.meta_strategy = None

        # Initialize population with random policies
        self._initialize_population()

    def _create_policy_network(self):
        """Create a policy network (model-agnostic)"""
        if self.config.policy_network_class is not None:
            # Custom network class
            return self.config.policy_network_class(**self.config.policy_kwargs).to(self.device)
        else:
            # Default MLP
            return PolicyNetwork(
                self.config.obs_dim,
                self.config.action_dims[0],  # Assume symmetric action spaces for now
                self.config.hidden_dims,
                self.config.activation
            ).to(self.device)

    def _create_value_network(self):
        """Create a value network for oracle (model-agnostic)"""
        if self.config.value_network_class is not None:
            # Custom network class
            return self.config.value_network_class(**self.config.value_kwargs).to(self.device)
        else:
            # Default MLP
            return ValueNetwork(
                self.config.obs_dim,
                self.config.hidden_dims,
                self.config.activation
            ).to(self.device)

    def _initialize_population(self):
        """Initialize policy populations with random policies"""
        print(f"Initializing populations with {self.config.initialization_policies} random policies per agent")

        for agent_idx in range(self.config.n_agents):
            for _ in range(self.config.initialization_policies):
                policy = self._create_policy_network()
                # Random initialization is fine (default PyTorch init)
                self.policy_populations[agent_idx].append(policy)

    def evaluate_policy_pair(self, policies: List[nn.Module], n_episodes: int) -> List[float]:
        """
        Evaluate a fixed set of policies for n_episodes

        Args:
            policies: List of policy networks (one per agent)
            n_episodes: Number of episodes to evaluate

        Returns:
            avg_returns: Average return per agent over episodes
        """
        env = self.env_factory()
        total_returns = [0.0 for _ in range(self.config.n_agents)]

        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_returns = [0.0 for _ in range(self.config.n_agents)]

            while not done:
                # Get actions from policies
                actions = []
                with torch.no_grad():
                    for agent_idx, policy in enumerate(policies):
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        action_probs = policy(obs_tensor)
                        action = torch.multinomial(action_probs, 1).item()
                        actions.append(action)

                # Step environment
                next_obs, reward, terminated, truncated, _ = env.step(actions)
                done = terminated or truncated

                # Accumulate returns (assume shared reward for simplicity)
                for agent_idx in range(self.config.n_agents):
                    episode_returns[agent_idx] += reward

                obs = next_obs

            for agent_idx in range(self.config.n_agents):
                total_returns[agent_idx] += episode_returns[agent_idx]

        avg_returns = [total / n_episodes for total in total_returns]
        return avg_returns

    def build_meta_game(self):
        """
        Build empirical meta-game by evaluating all policy pairs

        Returns:
            meta_payoffs: List of payoff matrices (one per agent)
                         meta_payoffs[i][j][k] = payoff for agent i with policy j vs opponent policy k
        """
        print("\nBuilding empirical meta-game...")

        pop_sizes = [len(pop) for pop in self.policy_populations]
        print(f"Population sizes: {pop_sizes}")

        # Initialize payoff matrices
        meta_payoffs = []
        for i in range(self.config.n_agents):
            j = 1 - i  # Assume 2-player for now
            payoff_matrix = np.zeros((pop_sizes[i], pop_sizes[j]))
            meta_payoffs.append(payoff_matrix)

        # Evaluate all policy pairs
        total_evaluations = pop_sizes[0] * pop_sizes[1]
        eval_count = 0

        for idx_0, policy_0 in enumerate(self.policy_populations[0]):
            for idx_1, policy_1 in enumerate(self.policy_populations[1]):
                eval_count += 1
                print(f"Evaluating pair ({idx_0}, {idx_1}) [{eval_count}/{total_evaluations}]", end='\r')

                policies = [policy_0, policy_1]
                avg_returns = self.evaluate_policy_pair(policies, self.config.eval_episodes)

                # Store payoffs
                meta_payoffs[0][idx_0, idx_1] = avg_returns[0]
                meta_payoffs[1][idx_1, idx_0] = avg_returns[1]  # Transpose for agent 1

        print()  # New line after progress

        # Convert to torch tensors
        self.meta_payoffs = [torch.FloatTensor(payoff).unsqueeze(0).to(self.device)
                             for payoff in meta_payoffs]

        return self.meta_payoffs

    def solve_meta_game(self) -> List[torch.Tensor]:
        """
        Solve RQE on meta-game

        Returns:
            meta_strategy: Distribution over policies per agent
        """
        print("\nSolving meta-game with RQE...")

        # Recreate RQE solver with current population sizes
        pop_sizes = [len(pop) for pop in self.policy_populations]
        rqe_config = RQEConfig(
            action_dims=pop_sizes,
            tau=self.config.tau,
            epsilon=self.config.epsilon,
            max_iterations=self.config.rqe_iterations,
            learning_rate=self.config.rqe_lr,
        )
        self.rqe_solver = RQESolver(rqe_config)

        # Solve RQE (warm_start=False to ensure clean initialization)
        meta_strategy = self.rqe_solver.solve(self.meta_payoffs, warm_start=False)

        # Print meta-strategy
        for i, strategy in enumerate(meta_strategy):
            print(f"Agent {i} meta-strategy: {strategy[0].cpu().numpy()}")

        self.meta_strategy = meta_strategy
        return meta_strategy

    def train_best_response_ppo(self, agent_idx: int, opponent_mixture) -> nn.Module:
        """
        Train best response policy using PPO

        Args:
            agent_idx: Index of agent to train
            opponent_mixture: Distribution over opponent policies

        Returns:
            best_response: Trained policy network
        """
        print(f"\nTraining best response for agent {agent_idx} (PPO)...")

        # Create networks
        policy = self._create_policy_network()
        value_net = self._create_value_network()

        # Optimizers
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=self.config.lr_policy)
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=self.config.lr_value)

        # Environment
        env = self.env_factory()

        # Sample opponent policy index for this episode
        opponent_idx = 1 - agent_idx
        opponent_probs = opponent_mixture[0].cpu().numpy()

        episode_rewards = []

        # Batch multiple episodes before updating (more stable)
        batch_size = 10
        batch_obs, batch_actions, batch_logprobs, batch_advantages, batch_returns = [], [], [], [], []

        for episode in range(self.config.oracle_episodes):
            # Sample opponent policy from mixture
            opponent_policy_idx = np.random.choice(len(opponent_probs), p=opponent_probs)
            opponent_policy = self.policy_populations[opponent_idx][opponent_policy_idx]

            # Collect trajectory
            obs_list, action_list, reward_list, value_list, logprob_list = [], [], [], [], []

            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                # Get action from training policy
                with torch.no_grad():
                    action_probs = policy(obs_tensor)
                    value = value_net(obs_tensor)

                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                logprob = action_dist.log_prob(action)

                # Get opponent action
                with torch.no_grad():
                    opponent_action_probs = opponent_policy(obs_tensor)
                    opponent_action = torch.multinomial(opponent_action_probs, 1).item()

                # Construct joint action
                if agent_idx == 0:
                    actions = [action.item(), opponent_action]
                else:
                    actions = [opponent_action, action.item()]

                # Step environment
                next_obs, reward, terminated, truncated, _ = env.step(actions)
                done = terminated or truncated

                # Store transition
                obs_list.append(obs)
                action_list.append(action)
                reward_list.append(reward)
                value_list.append(value)
                logprob_list.append(logprob)

                episode_reward += reward
                obs = next_obs

            episode_rewards.append(episode_reward)

            # Skip if episode was empty (shouldn't happen but be safe)
            if len(obs_list) == 0:
                continue

            # Compute GAE advantages
            advantages, returns = self._compute_gae(reward_list, value_list, done)

            # Add to batch
            batch_obs.extend(obs_list)
            batch_actions.extend(action_list)
            batch_logprobs.extend(logprob_list)
            batch_advantages.extend(advantages.cpu().tolist())
            batch_returns.extend(returns.cpu().tolist())

            # PPO update every batch_size episodes
            if (episode + 1) % batch_size == 0 and len(batch_obs) > 0:
                # Convert to tensors
                batch_advantages_tensor = torch.FloatTensor(batch_advantages).to(self.device)
                batch_returns_tensor = torch.FloatTensor(batch_returns).to(self.device)

                self._ppo_update(policy, value_net, policy_optimizer, value_optimizer,
                               batch_obs, batch_actions, batch_logprobs,
                               batch_advantages_tensor, batch_returns_tensor)

                # Clear batch
                batch_obs, batch_actions, batch_logprobs, batch_advantages, batch_returns = [], [], [], [], []

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}/{self.config.oracle_episodes}: avg_reward={avg_reward:.2f}")

        print(f"Best response training complete. Final avg reward: {np.mean(episode_rewards[-100:]):.2f}")
        return policy

    def _compute_gae(self, rewards, values, done):
        """Compute GAE advantages"""
        advantages = []
        returns = []

        gae = 0
        next_value = 0 if done else values[-1].item()

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.gamma * next_value - values[t].item()
            gae = delta + self.config.gamma * self.config.gae_lambda * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t].item())

            next_value = values[t].item()

        return torch.FloatTensor(advantages).to(self.device), torch.FloatTensor(returns).to(self.device)

    def _ppo_update(self, policy, value_net, policy_optimizer, value_optimizer,
                    obs_list, action_list, old_logprobs, advantages, returns):
        """PPO update step"""
        # Convert to tensors
        obs_batch = torch.FloatTensor(np.array(obs_list)).to(self.device)
        action_batch = torch.stack(action_list)
        old_logprobs_batch = torch.stack(old_logprobs)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.config.ppo_epochs):
            # Policy loss
            action_probs = policy(obs_batch)
            dist = torch.distributions.Categorical(action_probs)
            new_logprobs = dist.log_prob(action_batch)

            ratio = torch.exp(new_logprobs - old_logprobs_batch)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = value_net(obs_batch).squeeze()
            value_loss = nn.functional.mse_loss(values, returns)

            # Update
            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
            value_optimizer.step()

    def run(self):
        """
        Run PSRO algorithm

        Returns:
            final_meta_strategy: Final equilibrium distribution over policies
            policy_populations: Learned policy populations
        """
        print("=" * 80)
        print("Running PSRO with RQE")
        print("=" * 80)
        print(f"PSRO iterations: {self.config.psro_iterations}")
        print(f"Oracle episodes per iteration: {self.config.oracle_episodes}")
        print(f"Evaluation episodes: {self.config.eval_episodes}")
        print()

        for iteration in range(self.config.psro_iterations):
            print("=" * 80)
            print(f"PSRO Iteration {iteration + 1}/{self.config.psro_iterations}")
            print("=" * 80)

            # 1. Build empirical meta-game
            self.build_meta_game()

            # 2. Solve RQE on meta-game
            meta_strategy = self.solve_meta_game()

            # 3. Train best response policies
            new_policies = []
            for agent_idx in range(self.config.n_agents):
                opponent_idx = 1 - agent_idx
                opponent_mixture = meta_strategy[opponent_idx]

                if self.config.oracle_type == "ppo":
                    br_policy = self.train_best_response_ppo(agent_idx, opponent_mixture)
                else:
                    raise NotImplementedError(f"Oracle type {self.config.oracle_type} not implemented")

                new_policies.append(br_policy)

            # 4. Add to populations
            for agent_idx, policy in enumerate(new_policies):
                self.policy_populations[agent_idx].append(policy)
                print(f"Agent {agent_idx} population size: {len(self.policy_populations[agent_idx])}")

            # Optional: Prune populations if too large
            if self.config.max_population_size is not None:
                for agent_idx in range(self.config.n_agents):
                    if len(self.policy_populations[agent_idx]) > self.config.max_population_size:
                        # Keep most recent policies
                        self.policy_populations[agent_idx] = self.policy_populations[agent_idx][-self.config.max_population_size:]
                        print(f"Pruned agent {agent_idx} population to {self.config.max_population_size}")

        # Final meta-game and solution
        print("\n" + "=" * 80)
        print("Computing final meta-game and solution")
        print("=" * 80)
        self.build_meta_game()
        final_meta_strategy = self.solve_meta_game()

        return final_meta_strategy, self.policy_populations

    def get_policy_from_meta_strategy(self, agent_idx: int) -> nn.Module:
        """
        Sample a single policy from the meta-strategy for deployment

        Args:
            agent_idx: Agent index

        Returns:
            policy: Sampled policy network
        """
        if self.meta_strategy is None:
            raise ValueError("Meta-strategy not computed. Run PSRO first.")

        probs = self.meta_strategy[agent_idx][0].cpu().numpy()
        policy_idx = np.random.choice(len(probs), p=probs)
        return self.policy_populations[agent_idx][policy_idx]
