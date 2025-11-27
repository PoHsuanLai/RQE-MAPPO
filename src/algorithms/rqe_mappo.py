"""
RQE-MAPPO: Multi-Agent PPO with Risk-Averse Quantal Response Equilibrium

Implements multi-agent PPO with:
1. Risk-averse learning via distributional critics
2. Bounded rationality via entropy regularization
3. Self-play for equilibrium convergence

Based on Mazumdar et al. (2025) "Tractable Multi-Agent Reinforcement Learning
through Behavioral Economics"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class RQEConfig:
    """Configuration for RQE-MAPPO (model-agnostic)"""

    # Environment
    n_agents: int
    action_dim: int

    # Risk-aversion parameters
    tau: float = 1.0  # Risk aversion (lower = more risk-averse)
    risk_measure: str = "entropic"  # "entropic", "cvar", or "mean_variance"

    # Bounded rationality
    epsilon: float = 0.01  # Entropy coefficient

    # Distributional critic
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 50.0

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    vf_clip_param: Optional[float] = 10.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Network architecture
    hidden_dims: List[int] = None

    # Self-play
    use_self_play: bool = True
    population_size: int = 5  # Keep last N policies for opponent sampling
    update_population_every: int = 10  # Updates between population additions

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


class DistributionalCritic(nn.Module):
    """
    Multi-agent distributional critic

    Each agent has its own critic that learns Z(s) - distribution of returns
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int],
        n_atoms: int,
        v_min: float,
        v_max: float
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support atoms
        self.register_buffer(
            "z_atoms",
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Network
        layers = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, n_atoms))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get distribution over returns

        Args:
            obs: [batch, n_agents, obs_dim] or [batch, obs_dim]

        Returns:
            probs: [batch, n_agents, n_atoms] or [batch, n_atoms]
        """
        original_shape = obs.shape

        # Handle both batched and unbatched
        if len(original_shape) == 3:  # [batch, n_agents, obs_dim]
            batch, n_agents, obs_dim = original_shape
            obs_flat = obs.reshape(-1, obs_dim)  # [batch * n_agents, obs_dim]
        else:  # [batch, obs_dim]
            obs_flat = obs

        # Get logits
        logits = self.network(obs_flat)  # [batch * n_agents, n_atoms] or [batch, n_atoms]
        probs = F.softmax(logits, dim=-1)

        # Reshape back if needed
        if len(original_shape) == 3:
            probs = probs.reshape(batch, n_agents, self.n_atoms)

        return probs

    def get_risk_value(
        self,
        obs: torch.Tensor,
        tau: float,
        risk_type: str = "entropic"
    ) -> torch.Tensor:
        """
        Compute risk-adjusted value

        Args:
            obs: [batch, n_agents, obs_dim] or [batch, obs_dim]
            tau: Risk aversion parameter
            risk_type: "entropic", "cvar", or "mean_variance"

        Returns:
            risk_values: [batch, n_agents] or [batch]
        """
        probs = self.forward(obs)  # [batch, n_agents, n_atoms] or [batch, n_atoms]

        if risk_type == "entropic":
            # Entropic risk: -(1/τ) log E[exp(-τ * Z)]
            # For numerical stability: -(1/τ) * (log_sum_exp(-τ * z * p) - log(sum(p)))
            weighted_values = -tau * self.z_atoms  # [n_atoms]

            # Expand dimensions for broadcasting
            if len(probs.shape) == 3:  # [batch, n_agents, n_atoms]
                weighted_values = weighted_values.unsqueeze(0).unsqueeze(0)  # [1, 1, n_atoms]
            else:  # [batch, n_atoms]
                weighted_values = weighted_values.unsqueeze(0)  # [1, n_atoms]

            # Log-sum-exp for numerical stability
            log_exp_sum = torch.logsumexp(
                weighted_values + torch.log(probs + 1e-8),
                dim=-1
            )
            risk_value = -(1.0 / tau) * log_exp_sum

        elif risk_type == "cvar":
            # CVaR at level tau
            cumsum = torch.cumsum(probs, dim=-1)
            mask = (cumsum <= tau).float()
            cvar_probs = mask * probs
            cvar_probs = cvar_probs / (cvar_probs.sum(dim=-1, keepdim=True) + 1e-8)

            if len(probs.shape) == 3:
                z_atoms_expanded = self.z_atoms.unsqueeze(0).unsqueeze(0)
            else:
                z_atoms_expanded = self.z_atoms.unsqueeze(0)

            risk_value = (cvar_probs * z_atoms_expanded).sum(dim=-1)

        else:  # mean_variance
            # Mean-variance: E[Z] - τ * Var[Z]
            if len(probs.shape) == 3:
                z_atoms_expanded = self.z_atoms.unsqueeze(0).unsqueeze(0)
            else:
                z_atoms_expanded = self.z_atoms.unsqueeze(0)

            mean = (probs * z_atoms_expanded).sum(dim=-1)
            variance = (probs * (z_atoms_expanded - mean.unsqueeze(-1)) ** 2).sum(dim=-1)
            risk_value = mean - tau * variance

        return risk_value


class Actor(nn.Module):
    """Multi-agent actor network"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        layers = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get action logits

        Args:
            obs: [batch, n_agents, obs_dim] or [batch, obs_dim]

        Returns:
            logits: [batch, n_agents, action_dim] or [batch, action_dim]
        """
        original_shape = obs.shape

        if len(original_shape) == 3:  # [batch, n_agents, obs_dim]
            batch, n_agents, obs_dim = original_shape
            obs_flat = obs.reshape(-1, obs_dim)
            logits = self.network(obs_flat)
            action_dim = logits.shape[-1]
            logits = logits.reshape(batch, n_agents, action_dim)
        else:
            logits = self.network(obs)

        return logits

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample actions from policy

        Args:
            obs: [batch, n_agents, obs_dim]
            deterministic: If True, take argmax instead of sampling

        Returns:
            actions: [batch, n_agents]
            log_probs: [batch, n_agents]
            entropies: [batch, n_agents]
        """
        logits = self.forward(obs)  # [batch, n_agents, action_dim]

        if deterministic:
            actions = logits.argmax(dim=-1)  # [batch, n_agents]
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
        else:
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()

        return actions, log_probs, entropies


class RQE_MAPPO:
    """
    Multi-Agent PPO with Risk-Averse Quantal Response Equilibrium

    Features:
    - Each agent has its own actor and critic
    - Distributional critics for risk-averse learning
    - Self-play for equilibrium convergence
    - Centralized training, decentralized execution (CTDE)

    This class is MODEL-AGNOSTIC - it accepts any actor and critic networks.
    """

    def __init__(
        self,
        actors: List[nn.Module],
        critics: List[nn.Module],
        config: RQEConfig
    ):
        """
        Initialize RQE-MAPPO with custom actor and critic networks

        Args:
            actors: List of actor networks, one per agent
                    Each must implement: forward(obs) -> logits
                                       get_action(obs, deterministic) -> (actions, log_probs, entropies)
            critics: List of critic networks, one per agent
                     Each must implement: get_risk_value(obs, tau, risk_type) -> values
                                        forward(obs) -> probs [batch, n_atoms]
            config: RQEConfig with algorithm hyperparameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store provided networks
        self.actors = nn.ModuleList(actors).to(self.device)
        self.critics = nn.ModuleList(critics).to(self.device)

        # Optimizers
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
            for actor in self.actors
        ]

        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
            for critic in self.critics
        ]

        # Self-play population (store past actor policies)
        if config.use_self_play:
            self.policy_population = []
            self.update_counter = 0

    def get_actions(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions for all agents

        Args:
            obs: [batch, n_agents, obs_dim]
            deterministic: If True, take argmax

        Returns:
            actions: [batch, n_agents]
            log_probs: [batch, n_agents]
            entropies: [batch, n_agents]
        """
        # Move to device
        obs = obs.to(self.device)

        actions_list = []
        log_probs_list = []
        entropies_list = []

        for agent_id in range(self.config.n_agents):
            agent_obs = obs[:, agent_id, :]  # [batch, obs_dim]

            actions, log_probs, entropies = self.actors[agent_id].get_action(
                agent_obs, deterministic
            )

            actions_list.append(actions)
            log_probs_list.append(log_probs)
            entropies_list.append(entropies)

        # Stack along agent dimension
        actions = torch.stack(actions_list, dim=1)  # [batch, n_agents]
        log_probs = torch.stack(log_probs_list, dim=1)
        entropies = torch.stack(entropies_list, dim=1)

        return actions, log_probs, entropies

    def get_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get risk-adjusted values for all agents

        Args:
            obs: [batch, n_agents, obs_dim]

        Returns:
            values: [batch, n_agents]
        """
        # Move to device
        obs = obs.to(self.device)

        values_list = []

        for agent_id in range(self.config.n_agents):
            agent_obs = obs[:, agent_id, :]  # [batch, obs_dim]

            value = self.critics[agent_id].get_risk_value(
                agent_obs,
                tau=self.config.tau,
                risk_type=self.config.risk_measure
            )

            values_list.append(value)

        values = torch.stack(values_list, dim=1)  # [batch, n_agents]
        return values

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages

        Args:
            rewards: [batch, n_agents]
            values: [batch, n_agents]
            dones: [batch] - shared done signal
            next_values: [batch, n_agents]

        Returns:
            advantages: [batch, n_agents]
            returns: [batch, n_agents]
        """
        batch_size = rewards.shape[0]

        # Expand dones to match agent dimension
        dones_expanded = dones.unsqueeze(1).expand(-1, self.config.n_agents)  # [batch, n_agents]

        # TD error: δ = r + γ * V(s') * (1 - done) - V(s)
        delta = (
            rewards
            + self.config.gamma * next_values * (1 - dones_expanded)
            - values
        )

        # For single-step batch, GAE reduces to TD error
        advantages = delta

        # Returns = advantages + baseline
        returns = advantages + values

        return advantages, returns

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_obs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update actors and critics

        Args:
            obs: [batch, n_agents, obs_dim]
            actions: [batch, n_agents]
            old_log_probs: [batch, n_agents]
            rewards: [batch, n_agents]
            dones: [batch]
            next_obs: [batch, n_agents, obs_dim]

        Returns:
            stats: Dictionary of training statistics
        """
        # Move to device
        obs = obs.to(self.device).float()
        actions = actions.to(self.device).long()
        old_log_probs = old_log_probs.to(self.device).float()
        rewards = rewards.to(self.device).float()
        dones = dones.to(self.device).float()
        next_obs = next_obs.to(self.device).float()

        # Compute values
        with torch.no_grad():
            values = self.get_values(obs)
            next_values = self.get_values(next_obs)

            # Compute advantages using GAE
            advantages, returns = self.compute_gae(rewards, values, dones, next_values)

            # Normalize advantages per agent
            advantages = (advantages - advantages.mean(dim=0)) / (advantages.std(dim=0) + 1e-8)

        # Update each agent's actor and critic
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0

        for agent_id in range(self.config.n_agents):
            agent_obs = obs[:, agent_id, :]  # [batch, obs_dim]
            agent_actions = actions[:, agent_id]  # [batch]
            agent_old_log_probs = old_log_probs[:, agent_id]  # [batch]
            agent_advantages = advantages[:, agent_id]  # [batch]
            agent_returns = returns[:, agent_id]  # [batch]

            # ========== Actor Update ==========
            logits = self.actors[agent_id](agent_obs)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(agent_actions)
            entropy = dist.entropy()

            # PPO clipped loss
            ratio = torch.exp(new_log_probs - agent_old_log_probs)
            surr1 = ratio * agent_advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * agent_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Add entropy bonus (bounded rationality)
            actor_loss = actor_loss - self.config.epsilon * entropy.mean()

            # Update actor
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.config.max_grad_norm)
            self.actor_optimizers[agent_id].step()

            # ========== Critic Update ==========
            # Get current distribution
            current_probs = self.critics[agent_id](agent_obs)  # [batch, n_atoms]

            # Target: Project returns onto distribution
            with torch.no_grad():
                target_probs = torch.zeros_like(current_probs)
                batch_size = len(agent_returns)

                for i in range(batch_size):
                    ret = agent_returns[i].item()
                    ret = np.clip(ret, self.config.v_min, self.config.v_max)

                    # Linear interpolation
                    b = (ret - self.config.v_min) / self.critics[agent_id].delta_z
                    l = int(np.floor(b))
                    u = int(np.ceil(b))

                    l = np.clip(l, 0, self.config.n_atoms - 1)
                    u = np.clip(u, 0, self.config.n_atoms - 1)

                    if l == u:
                        target_probs[i, l] = 1.0
                    else:
                        prob_u = b - l
                        prob_l = 1.0 - prob_u
                        target_probs[i, l] = prob_l
                        target_probs[i, u] = prob_u

            # Cross-entropy loss
            critic_loss = -(target_probs * torch.log(current_probs + 1e-8)).sum(dim=-1).mean()

            # Update critic
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.config.max_grad_norm)
            self.critic_optimizers[agent_id].step()

            # Accumulate stats
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()

            # KL and clip fraction
            with torch.no_grad():
                kl = (agent_old_log_probs - new_log_probs).mean().item()
                clipfrac = ((ratio - 1.0).abs() > self.config.clip_param).float().mean().item()
                total_kl += kl
                total_clipfrac += clipfrac

        # Average stats across agents
        n = self.config.n_agents
        stats = {
            "actor_loss": total_actor_loss / n,
            "critic_loss": total_critic_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_kl / n,
            "clipfrac": total_clipfrac / n,
        }

        # Update self-play population
        if self.config.use_self_play:
            self.update_counter += 1
            if self.update_counter % self.config.update_population_every == 0:
                self._add_to_population()

        return stats

    def _add_to_population(self):
        """Add current policies to population for self-play"""
        # Deep copy current actors
        import copy
        policy_snapshot = copy.deepcopy([actor.state_dict() for actor in self.actors])
        self.policy_population.append(policy_snapshot)

        # Limit population size
        if len(self.policy_population) > self.config.population_size:
            self.policy_population.pop(0)

    def sample_opponent_from_population(self, agent_id: int) -> Optional[nn.Module]:
        """
        Sample an opponent policy from the population for self-play

        Args:
            agent_id: Which agent to get opponent for

        Returns:
            opponent_actor: Actor network, or None if population empty
        """
        if not self.config.use_self_play or len(self.policy_population) == 0:
            return None

        # Sample random past policy
        policy_snapshot = self.policy_population[np.random.randint(len(self.policy_population))]

        # Create opponent by cloning one of the existing actors
        import copy
        opponent_agent_id = np.random.choice([i for i in range(self.config.n_agents) if i != agent_id])
        opponent = copy.deepcopy(self.actors[opponent_agent_id])
        opponent.to(self.device)

        # Load the opponent's policy (could be any agent from the population)
        # For simplicity, sample another agent's policy
        opponent_agent_id = np.random.choice([i for i in range(self.config.n_agents) if i != agent_id])
        opponent.load_state_dict(policy_snapshot[opponent_agent_id])
        opponent.eval()

        return opponent

    def save(self, path: str):
        """Save all agents' policies and critics"""
        torch.save({
            "actors": [actor.state_dict() for actor in self.actors],
            "critics": [critic.state_dict() for critic in self.critics],
            "config": self.config,
            "population": self.policy_population if self.config.use_self_play else None
        }, path)

    def load(self, path: str):
        """Load all agents' policies and critics"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        for i, actor_state in enumerate(checkpoint["actors"]):
            self.actors[i].load_state_dict(actor_state)

        for i, critic_state in enumerate(checkpoint["critics"]):
            self.critics[i].load_state_dict(critic_state)

        if self.config.use_self_play and checkpoint["population"] is not None:
            self.policy_population = checkpoint["population"]
