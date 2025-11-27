#!/usr/bin/env python3
"""
Train True RQE-MAPPO on Atari Boxing

Uses standalone True RQE-MAPPO with CNN networks for visual observations.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from pettingzoo.atari import boxing_v2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from algorithms.true_rqe_mappo import TrueRQE_MAPPO, TrueRQEConfig


# ==================== CNN Network Definitions ====================

class CNNActor(nn.Module):
    """CNN-based actor for visual observations"""

    def __init__(self, action_dim: int):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 210, 160)
            cnn_out_size = self.cnn(dummy).shape[1]

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get action logits

        Args:
            obs: [batch, 3, 210, 160]

        Returns:
            logits: [batch, action_dim]
        """
        features = self.cnn(obs)
        logits = self.fc(features)
        return torch.clamp(logits, min=-10.0, max=10.0)

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


class CNNDistributionalCritic(nn.Module):
    """CNN-based action-conditioned distributional critic"""

    def __init__(
        self,
        action_dim: int,
        n_atoms: int,
        v_min: float,
        v_max: float
    ):
        super().__init__()

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

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 210, 160)
            cnn_out_size = self.cnn(dummy).shape[1]

        # FC layers to distributional outputs
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * n_atoms)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get distribution logits for all actions

        Args:
            obs: [batch, 3, 210, 160]

        Returns:
            probs: [batch, action_dim, n_atoms]
        """
        features = self.cnn(obs)
        logits = self.fc(features)  # [batch, action_dim * n_atoms]
        logits = logits.view(-1, self.action_dim, self.n_atoms)
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_distribution(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get return distribution for specific actions"""
        all_probs = self.forward(obs)  # [batch, action_dim, n_atoms]

        # Gather distributions for selected actions
        actions_expanded = actions.unsqueeze(-1).unsqueeze(-1)
        actions_expanded = actions_expanded.expand(-1, 1, self.n_atoms)
        probs = torch.gather(all_probs, 1, actions_expanded).squeeze(1)

        return probs

    def get_risk_value(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor = None,
        tau: float = 1.0,
        risk_type: str = "entropic"
    ) -> torch.Tensor:
        """Compute risk-adjusted Q-values"""
        if actions is not None:
            probs = self.get_distribution(obs, actions)
        else:
            probs = self.forward(obs)

        if risk_type == "entropic":
            weighted_values = -tau * self.z_atoms

            if actions is not None:
                weighted_values = weighted_values.unsqueeze(0)
                log_exp_sum = torch.logsumexp(
                    weighted_values + torch.log(probs + 1e-8),
                    dim=-1
                )
            else:
                weighted_values = weighted_values.unsqueeze(0).unsqueeze(0)
                log_exp_sum = torch.logsumexp(
                    weighted_values + torch.log(probs + 1e-8),
                    dim=-1
                )

            risk_value = -(1.0 / tau) * log_exp_sum

        elif risk_type == "cvar":
            cumsum = torch.cumsum(probs, dim=-1)
            mask = (cumsum <= tau).float()
            cvar_probs = mask * probs
            cvar_probs = cvar_probs / (cvar_probs.sum(dim=-1, keepdim=True) + 1e-8)

            if actions is not None:
                z_atoms_expanded = self.z_atoms.unsqueeze(0)
            else:
                z_atoms_expanded = self.z_atoms.unsqueeze(0).unsqueeze(0)

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


# ==================== Training Code ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train True RQE-MAPPO on Boxing"
    )

    # RQE parameters
    parser.add_argument("--tau", type=float, default=1.0,
                       help="Risk aversion parameter")
    parser.add_argument("--epsilon", type=float, default=0.01,
                       help="Bounded rationality (entropy coefficient)")
    parser.add_argument("--n_atoms", type=int, default=51,
                       help="Number of atoms in distributional critic")
    parser.add_argument("--v_min", type=float, default=-200.0,
                       help="Minimum value for distributional critic")
    parser.add_argument("--v_max", type=float, default=200.0,
                       help="Maximum value for distributional critic")

    # Training parameters
    parser.add_argument("--total_timesteps", type=int, default=10000000,
                       help="Total timesteps to train")
    parser.add_argument("--batch_size", type=int, default=2048,
                       help="Batch size for training")
    parser.add_argument("--actor_lr", type=float, default=1e-4,
                       help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4,
                       help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda")

    # Self-play parameters
    parser.add_argument("--use_self_play", action="store_true", default=True,
                       help="Use self-play")
    parser.add_argument("--population_size", type=int, default=5,
                       help="Self-play population size")

    # Logging
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log every N updates")
    parser.add_argument("--save_dir", type=str,
                       default="/home/r13921098/RQE-MAPPO/results/boxing",
                       help="Directory to save results")
    parser.add_argument("--exp_name", type=str, default=None,
                       help="Experiment name")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                       help="Save checkpoint every N updates")

    return parser.parse_args()


def preprocess_obs(obs):
    """Preprocess visual observation: transpose and normalize"""
    # obs comes as (210, 160, 3), need (3, 210, 160)
    if isinstance(obs, dict):
        obs = list(obs.values())[0]

    obs = np.array(obs, dtype=np.float32)
    if obs.ndim == 3:
        obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
    obs = obs / 255.0  # Normalize to [0, 1]
    return obs


def collect_rollout(env, agents, batch_size, max_cycles=10000):
    """Collect a batch of experience from the environment"""

    n_agents = 2  # Boxing has 2 agents
    observations = []
    actions = []
    rewards = []
    dones = []
    next_observations = []
    log_probs = []

    timesteps = 0
    total_reward = 0
    episode_lengths = []

    while timesteps < batch_size:
        obs, _ = env.reset()
        episode_length = 0

        for step in range(max_cycles):
            # Preprocess observations
            obs_list = [preprocess_obs(obs[agent]) for agent in env.agents]
            obs_tensor = torch.FloatTensor(np.array(obs_list)).unsqueeze(0).to(agents.device)

            # Get actions
            actions_batch, log_probs_batch, _ = agents.get_actions(
                obs_tensor, deterministic=False
            )

            # Convert to environment format
            action_dict = {
                agent: actions_batch[0, i].item()
                for i, agent in enumerate(env.agents)
            }

            # Step environment
            next_obs, reward_dict, done_dict, trunc_dict, _ = env.step(action_dict)

            # Store experience
            next_obs_list = [preprocess_obs(next_obs[agent]) for agent in env.agents]
            reward_list = [reward_dict[agent] for agent in env.agents]
            done_any = any(done_dict.values()) or any(trunc_dict.values())

            observations.append(obs_list)
            actions.append([actions_batch[0, i].item() for i in range(n_agents)])
            rewards.append(reward_list)
            dones.append(done_any)
            next_observations.append(next_obs_list)
            log_probs.append([log_probs_batch[0, i].item() for i in range(n_agents)])

            total_reward += np.mean(reward_list)

            obs = next_obs
            timesteps += 1
            episode_length += 1

            if done_any:
                break

        episode_lengths.append(episode_length)

    # Convert to tensors [batch, n_agents, C, H, W]
    obs_tensor = torch.FloatTensor(np.array(observations))
    actions_tensor = torch.LongTensor(np.array(actions))
    rewards_tensor = torch.FloatTensor(np.array(rewards))
    dones_tensor = torch.FloatTensor(np.array(dones))
    next_obs_tensor = torch.FloatTensor(np.array(next_observations))
    log_probs_tensor = torch.FloatTensor(np.array(log_probs))

    avg_reward = total_reward / len(observations)
    avg_length = np.mean(episode_lengths)

    return obs_tensor, actions_tensor, log_probs_tensor, rewards_tensor, dones_tensor, next_obs_tensor, avg_reward, avg_length


def main():
    args = parse_args()

    # Create environment
    env = boxing_v2.parallel_env()
    env.reset()

    # Get action dimension
    action_dim = env.action_space(env.agents[0]).n
    n_agents = len(env.agents)

    print(f"Action dim: {action_dim}")
    print(f"Number of agents: {n_agents}")
    print(f"Observation shape: (3, 210, 160)")

    # Create CNN networks
    actors = [CNNActor(action_dim) for _ in range(n_agents)]
    critics = [
        CNNDistributionalCritic(
            action_dim,
            args.n_atoms,
            args.v_min,
            args.v_max
        )
        for _ in range(n_agents)
    ]

    # Create config
    config = TrueRQEConfig(
        n_agents=n_agents,
        action_dim=action_dim,
        tau=args.tau,
        epsilon=args.epsilon,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        use_self_play=args.use_self_play,
        population_size=args.population_size,
    )

    # Create agents with custom networks
    agents = TrueRQE_MAPPO(actors, critics, config)

    # Experiment name
    exp_name = args.exp_name or f"TrueRQE_MAPPO_Boxing_tau{args.tau}_eps{args.epsilon}"

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.save_dir, exp_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print(f"Starting True RQE-MAPPO Training on Atari Boxing")
    print("=" * 70)
    print(f"Risk aversion (tau): {args.tau}")
    print(f"Bounded rationality (epsilon): {args.epsilon}")
    print(f"Self-play: {args.use_self_play}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("=" * 70)

    # Training loop
    total_timesteps = 0
    update = 0
    best_reward = float('-inf')

    while total_timesteps < args.total_timesteps:
        # Collect rollout
        obs, actions, log_probs, rewards, dones, next_obs, avg_reward, avg_length = collect_rollout(
            env, agents, args.batch_size
        )

        # Update agents
        metrics = agents.update(obs, actions, log_probs, rewards, dones, next_obs)

        total_timesteps += len(obs)
        update += 1

        # Logging
        if update % args.log_interval == 0:
            print(f"Update {update} | Timesteps {total_timesteps}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Episode Length: {avg_length:.1f}")
            print(f"  Actor Loss: {metrics['actor_loss']:.4f}")
            print(f"  Critic Loss: {metrics['critic_loss']:.4f}")
            print("-" * 70)

        # Save checkpoint
        if update % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{update:06d}.pt")
            checkpoint = {
                'update': update,
                'total_timesteps': total_timesteps,
                'actors': [actor.state_dict() for actor in agents.actors],
                'critics': [critic.state_dict() for critic in agents.critics],
                'actor_optimizers': [opt.state_dict() for opt in agents.actor_optimizers],
                'critic_optimizers': [opt.state_dict() for opt in agents.critic_optimizers],
                'config': config,
                'avg_reward': avg_reward,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # Save best checkpoint
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
                torch.save(checkpoint, best_checkpoint_path)
                print(f"Best checkpoint updated: {best_checkpoint_path} (reward: {best_reward:.2f})")

    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    final_checkpoint = {
        'update': update,
        'total_timesteps': total_timesteps,
        'actors': [actor.state_dict() for actor in agents.actors],
        'critics': [critic.state_dict() for critic in agents.critics],
        'actor_optimizers': [opt.state_dict() for opt in agents.actor_optimizers],
        'critic_optimizers': [opt.state_dict() for opt in agents.critic_optimizers],
        'config': config,
        'avg_reward': avg_reward,
    }
    torch.save(final_checkpoint, final_checkpoint_path)

    print("=" * 70)
    print("Training completed!")
    print(f"Final checkpoint saved: {final_checkpoint_path}")
    print(f"Best checkpoint: {os.path.join(checkpoint_dir, 'best_checkpoint.pt')}")
    print(f"Best reward: {best_reward:.2f}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
