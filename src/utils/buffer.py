"""
Rollout buffer for storing trajectories

Collects experience for on-policy algorithms like PPO
"""

import torch
import numpy as np
from typing import Dict, Optional


class RolloutBuffer:
    """
    Buffer for storing rollouts for on-policy algorithms

    Stores:
    - observations
    - actions
    - log_probs
    - rewards
    - dones
    - values (optional, for logging)
    """

    def __init__(
        self,
        buffer_size: int,
        n_agents: int,
        obs_dim: int,
        device: str = "cpu"
    ):
        """
        Args:
            buffer_size: Maximum number of transitions to store
            n_agents: Number of agents
            obs_dim: Observation dimension
            device: torch device
        """
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.device = device

        # Initialize buffers
        self.observations = torch.zeros(
            (buffer_size, n_agents, obs_dim),
            dtype=torch.float32,
            device=device
        )
        self.next_observations = torch.zeros(
            (buffer_size, n_agents, obs_dim),
            dtype=torch.float32,
            device=device
        )
        self.actions = torch.zeros(
            (buffer_size, n_agents),
            dtype=torch.long,
            device=device
        )
        self.log_probs = torch.zeros(
            (buffer_size, n_agents),
            dtype=torch.float32,
            device=device
        )
        self.rewards = torch.zeros(
            (buffer_size, n_agents),
            dtype=torch.float32,
            device=device
        )
        self.dones = torch.zeros(
            buffer_size,
            dtype=torch.float32,
            device=device
        )

        # Pointer and size
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: bool,
        next_obs: torch.Tensor
    ):
        """
        Add a transition to the buffer

        Args:
            obs: [n_agents, obs_dim]
            action: [n_agents]
            log_prob: [n_agents]
            reward: [n_agents]
            done: bool
            next_obs: [n_agents, obs_dim]
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.next_observations[self.ptr] = next_obs

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get all data from buffer

        Returns:
            Dictionary containing all transitions
        """
        return {
            'observations': self.observations[:self.size],
            'actions': self.actions[:self.size],
            'log_probs': self.log_probs[:self.size],
            'rewards': self.rewards[:self.size],
            'dones': self.dones[:self.size],
            'next_observations': self.next_observations[:self.size]
        }

    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size


if __name__ == "__main__":
    # Test rollout buffer
    print("Testing RolloutBuffer...")

    buffer = RolloutBuffer(
        buffer_size=100,
        n_agents=3,
        obs_dim=10
    )

    # Add some transitions
    for i in range(50):
        obs = torch.randn(3, 10)
        action = torch.randint(0, 5, (3,))
        log_prob = torch.randn(3)
        reward = torch.randn(3)
        done = (i % 10 == 9)
        next_obs = torch.randn(3, 10)

        buffer.add(obs, action, log_prob, reward, done, next_obs)

    print(f"Buffer size: {len(buffer)}")

    # Get all data
    data = buffer.get()
    print(f"Observations shape: {data['observations'].shape}")
    print(f"Actions shape: {data['actions'].shape}")
    print(f"Rewards shape: {data['rewards'].shape}")
    print(f"Dones shape: {data['dones'].shape}")

    # Test clear
    buffer.clear()
    print(f"Buffer size after clear: {len(buffer)}")

    print("\n✓ All tests passed!")
