"""
True RQE-MAPPO: Multi-Agent PPO with True Risk-Averse Quantal Response Equilibrium

Uses action-conditioned distributional critics to compute true Q_risk(s,a) and
applies exponential importance weighting in the policy gradient.

This is the theoretically correct implementation as described in:
Mazumdar et al. (2025) "Tractable Multi-Agent Reinforcement Learning
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
class TrueRQEConfig:
    """Configuration for True RQE-MAPPO"""

    # Environment
    n_agents: int
    obs_dim: int
    action_dim: int

    # Risk-aversion parameters
    tau: float = 1.0  # Risk aversion (lower = more risk-averse)
    risk_measure: str = "entropic"  # "entropic", "cvar", or "mean_variance"

    # Bounded rationality
    epsilon: float = 0.01  # Entropy coefficient

    # Distributional critic (action-conditioned)
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
    actor_lr: float = 1e-4  # Lower for True RQE (more stable)
    critic_lr: float = 3e-4

    # Critic training
    critic_epochs: int = 5  # Extra epochs for critic

    # Network architecture
    hidden_dims: List[int] = None

    # Self-play
    use_self_play: bool = True
    population_size: int = 5
    update_population_every: int = 10

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


class ActionConditionedDistributionalCritic(nn.Module):
    """
    Action-conditioned distributional critic for True RQE

    Learns Z(s,a) = distribution of returns for EACH action
    This allows computing true Q_risk(s,a) = ρ_τ(Z(s,a))
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        n_atoms: int,
        v_min: float,
        v_max: float
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support atoms
        self.register_buffer(
            "z_atoms",
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Network: obs → hidden → [action_dim × n_atoms] logits
        layers = []
        last_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, action_dim * n_atoms))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get distribution logits for all actions

        Args:
            obs: [batch, obs_dim]

        Returns:
            probs: [batch, action_dim, n_atoms]
        """
        obs = obs.float()
        logits = self.network(obs)  # [batch, action_dim * n_atoms]
        logits = logits.view(-1, self.action_dim, self.n_atoms)  # [batch, action_dim, n_atoms]
        probs = F.softmax(logits, dim=-1)  # Softmax over atoms
        return probs

    def get_distribution(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get return distribution for specific actions

        Args:
            obs: [batch, obs_dim]
            actions: [batch] - action indices

        Returns:
            probs: [batch, n_atoms] - distribution over returns
        """
        all_probs = self.forward(obs)  # [batch, action_dim, n_atoms]

        # Gather distributions for selected actions
        actions_expanded = actions.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
        actions_expanded = actions_expanded.expand(-1, 1, self.n_atoms)  # [batch, 1, n_atoms]

        probs = torch.gather(all_probs, 1, actions_expanded).squeeze(1)  # [batch, n_atoms]

        return probs

    def get_risk_value(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        tau: float = 1.0,
        risk_type: str = "entropic"
    ) -> torch.Tensor:
        """
        Compute risk-adjusted Q-values

        Args:
            obs: [batch, obs_dim]
            actions: [batch] - if provided, get Q_risk(s,a) for specific actions
                     if None, get Q_risk(s,a) for all actions
            tau: Risk aversion parameter
            risk_type: "entropic", "cvar", or "mean_variance"

        Returns:
            risk_values: [batch] if actions provided, else [batch, action_dim]
        """
        if actions is not None:
            # Get distribution for specific actions
            probs = self.get_distribution(obs, actions)  # [batch, n_atoms]
        else:
            # Get distributions for all actions
            probs = self.forward(obs)  # [batch, action_dim, n_atoms]

        if risk_type == "entropic":
            # Entropic risk: -(1/τ) log E[exp(-τ * Z)]
            weighted_values = -tau * self.z_atoms  # [n_atoms]

            if actions is not None:
                # [batch, n_atoms]
                weighted_values = weighted_values.unsqueeze(0)  # [1, n_atoms]
                log_exp_sum = torch.logsumexp(
                    weighted_values + torch.log(probs + 1e-8),
                    dim=-1
                )  # [batch]
            else:
                # [batch, action_dim, n_atoms]
                weighted_values = weighted_values.unsqueeze(0).unsqueeze(0)  # [1, 1, n_atoms]
                log_exp_sum = torch.logsumexp(
                    weighted_values + torch.log(probs + 1e-8),
                    dim=-1
                )  # [batch, action_dim]

            risk_value = -(1.0 / tau) * log_exp_sum

        elif risk_type == "cvar":
            # CVaR at level tau
            cumsum = torch.cumsum(probs, dim=-1)
            mask = (cumsum <= tau).float()
            cvar_probs = mask * probs
            cvar_probs = cvar_probs / (cvar_probs.sum(dim=-1, keepdim=True) + 1e-8)

            if actions is not None:
                z_atoms_expanded = self.z_atoms.unsqueeze(0)  # [1, n_atoms]
            else:
                z_atoms_expanded = self.z_atoms.unsqueeze(0).unsqueeze(0)  # [1, 1, n_atoms]

            risk_value = (cvar_probs * z_atoms_expanded).sum(dim=-1)

        else:  # mean_variance
            if actions is not None:
                z_atoms_expanded = self.z_atoms.unsqueeze(0)
            else:
                z_atoms_expanded = self.z_atoms.unsqueeze(0).unsqueeze(0)

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
        """Get action logits"""
        return self.network(obs)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample actions from policy"""
        logits = self.forward(obs)

        if deterministic:
            actions = logits.argmax(dim=-1)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
        else:
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()

        return actions, log_probs, entropies


class TrueRQE_MAPPO:
    """
    True Multi-Agent PPO with Risk-Averse Quantal Response Equilibrium

    Key differences from Practical RQE-MAPPO:
    1. Uses action-conditioned critics Q_risk(s,a) instead of V_risk(s)
    2. Applies exponential importance weighting in policy gradient
    3. More computationally expensive but theoretically correct
    """

    def __init__(self, config: TrueRQEConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create actors and action-conditioned critics for each agent
        self.actors = nn.ModuleList([
            Actor(config.obs_dim, config.action_dim, config.hidden_dims)
            for _ in range(config.n_agents)
        ]).to(self.device)

        self.critics = nn.ModuleList([
            ActionConditionedDistributionalCritic(
                config.obs_dim,
                config.action_dim,
                config.hidden_dims,
                config.n_atoms,
                config.v_min,
                config.v_max
            )
            for _ in range(config.n_agents)
        ]).to(self.device)

        # Optimizers
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
            for actor in self.actors
        ]

        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
            for critic in self.critics
        ]

        # Self-play population
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

        actions = torch.stack(actions_list, dim=1)  # [batch, n_agents]
        log_probs = torch.stack(log_probs_list, dim=1)
        entropies = torch.stack(entropies_list, dim=1)

        return actions, log_probs, entropies

    def get_values(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get risk-adjusted Q-values for all agents

        Args:
            obs: [batch, n_agents, obs_dim]
            actions: [batch, n_agents]

        Returns:
            q_values: [batch, n_agents]
        """
        q_values_list = []

        for agent_id in range(self.config.n_agents):
            agent_obs = obs[:, agent_id, :]  # [batch, obs_dim]
            agent_actions = actions[:, agent_id]  # [batch]

            q_value = self.critics[agent_id].get_risk_value(
                agent_obs,
                agent_actions,
                tau=self.config.tau,
                risk_type=self.config.risk_measure
            )

            q_values_list.append(q_value)

        q_values = torch.stack(q_values_list, dim=1)  # [batch, n_agents]
        return q_values

    def compute_gae(
        self,
        rewards: torch.Tensor,
        q_values: torch.Tensor,
        dones: torch.Tensor,
        next_q_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages using Q-values

        Args:
            rewards: [batch, n_agents]
            q_values: [batch, n_agents]
            dones: [batch]
            next_q_values: [batch, n_agents]

        Returns:
            advantages: [batch, n_agents]
            returns: [batch, n_agents]
        """
        dones_expanded = dones.unsqueeze(1).expand(-1, self.config.n_agents)

        # TD error: δ = r + γ * Q(s',a') * (1 - done) - Q(s,a)
        delta = (
            rewards
            + self.config.gamma * next_q_values * (1 - dones_expanded)
            - q_values
        )

        # For single-step, GAE = TD error
        advantages = delta
        returns = advantages + q_values

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
        Update actors and critics with TRUE RQE

        Key difference: Uses exponential importance weighting based on Q_risk(s,a)

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

        # Get next actions for Q(s',a')
        with torch.no_grad():
            next_actions, _, _ = self.get_actions(next_obs, deterministic=False)

        # Compute Q-values and advantages
        with torch.no_grad():
            q_values = self.get_values(obs, actions)
            next_q_values = self.get_values(next_obs, next_actions)

            advantages, returns = self.compute_gae(rewards, q_values, dones, next_q_values)

            # Normalize advantages per agent
            advantages = (advantages - advantages.mean(dim=0)) / (advantages.std(dim=0) + 1e-8)

        # Update each agent
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0

        for agent_id in range(self.config.n_agents):
            agent_obs = obs[:, agent_id, :]
            agent_actions = actions[:, agent_id]
            agent_old_log_probs = old_log_probs[:, agent_id]
            agent_advantages = advantages[:, agent_id]
            agent_returns = returns[:, agent_id]

            # ========== TRUE RQE ACTOR UPDATE ==========
            # Compute exponential importance weights based on Q_risk(s,a)
            with torch.no_grad():
                q_risk_values = self.critics[agent_id].get_risk_value(
                    agent_obs,
                    agent_actions,
                    tau=self.config.tau,
                    risk_type=self.config.risk_measure
                )
                # Exponential weighting: exp(-τ * Q_risk(s,a))
                importance_weights = torch.exp(-self.config.tau * q_risk_values)
                # Normalize for stability
                importance_weights = importance_weights / (importance_weights.mean() + 1e-8)

            logits = self.actors[agent_id](agent_obs)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(agent_actions)
            entropy = dist.entropy()

            # PPO ratio with TRUE RQE importance weighting
            ratio = torch.exp(new_log_probs - agent_old_log_probs)

            # Apply importance weights to advantages
            weighted_advantages = importance_weights * agent_advantages

            surr1 = ratio * weighted_advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * weighted_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus
            actor_loss = actor_loss - self.config.epsilon * entropy.mean()

            # Update actor
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.config.max_grad_norm)
            self.actor_optimizers[agent_id].step()

            # ========== CRITIC UPDATE (multiple epochs) ==========
            # Train critic for multiple epochs on the same data
            critic_loss_sum = 0.0
            for _ in range(self.config.critic_epochs):
                # Get current distribution for all actions
                all_probs = self.critics[agent_id](agent_obs)  # [batch, action_dim, n_atoms]

                # Target: Project returns onto distribution
                with torch.no_grad():
                    target_probs = torch.zeros(
                        len(agent_returns), self.config.n_atoms,
                        device=self.device
                    )

                    for i in range(len(agent_returns)):
                        ret = agent_returns[i].item()
                        ret = np.clip(ret, self.config.v_min, self.config.v_max)

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

                # Gather probs for taken actions
                actions_expanded = agent_actions.unsqueeze(-1).unsqueeze(-1)
                actions_expanded = actions_expanded.expand(-1, 1, self.config.n_atoms)
                current_probs = torch.gather(all_probs, 1, actions_expanded).squeeze(1)

                # Cross-entropy loss
                critic_loss = -(target_probs * torch.log(current_probs + 1e-8)).sum(dim=-1).mean()

                # Update critic
                self.critic_optimizers[agent_id].zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.config.max_grad_norm)
                self.critic_optimizers[agent_id].step()

                critic_loss_sum += critic_loss.item()

            # Average critic loss over epochs
            critic_loss_avg = critic_loss_sum / self.config.critic_epochs

            # Accumulate stats
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss_avg
            total_entropy += entropy.mean().item()

            with torch.no_grad():
                kl = (agent_old_log_probs - new_log_probs).mean().item()
                clipfrac = ((ratio - 1.0).abs() > self.config.clip_param).float().mean().item()
                total_kl += kl
                total_clipfrac += clipfrac

        # Average stats
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
        import copy
        policy_snapshot = copy.deepcopy([actor.state_dict() for actor in self.actors])
        self.policy_population.append(policy_snapshot)

        if len(self.policy_population) > self.config.population_size:
            self.policy_population.pop(0)

    def sample_opponent_from_population(self, agent_id: int) -> Optional[nn.Module]:
        """Sample an opponent policy from the population for self-play"""
        if not self.config.use_self_play or len(self.policy_population) == 0:
            return None

        policy_snapshot = self.policy_population[np.random.randint(len(self.policy_population))]

        opponent = Actor(
            self.config.obs_dim,
            self.config.action_dim,
            self.config.hidden_dims
        ).to(self.device)

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
        checkpoint = torch.load(path, map_location=self.device)

        for i, actor_state in enumerate(checkpoint["actors"]):
            self.actors[i].load_state_dict(actor_state)

        for i, critic_state in enumerate(checkpoint["critics"]):
            self.critics[i].load_state_dict(critic_state)

        if self.config.use_self_play and checkpoint["population"] is not None:
            self.policy_population = checkpoint["population"]
